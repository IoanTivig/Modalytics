"""
Pixel sources: the read-only, tiled access to image data used by the canvas.

A ``PixelSource`` exposes one or more resolution levels (finest first), each
split into fixed-size tiles. The canvas only ever asks for whole tiles, so the
same code displays a 150 000 px whole-slide pyramid and a 750 px phase map.

Sources know nothing about µm or placement; that lives on the ``ImageNode``.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Level:
    """One resolution level of a source.

    Attributes:
        shape: (height, width) in pixels of this level.
        tile: (tile_height, tile_width) in pixels.
        downsample: (dy, dx) size of one pixel of this level in level-0 pixels,
            computed from the shapes so tiles line up exactly.
    """

    shape: tuple[int, int]
    tile: tuple[int, int]
    downsample: tuple[float, float]

    @property
    def grid(self) -> tuple[int, int]:
        """Number of tiles (rows, cols) covering the level."""
        return (-(-self.shape[0] // self.tile[0]), -(-self.shape[1] // self.tile[1]))


class PixelSource(ABC):
    """Abstract tiled, multi-resolution image.

    Subclasses fill ``levels``, ``dtype`` and ``channels`` and implement
    ``read_tile``. ``read_tile`` must be thread-safe: the canvas calls it from
    a pool of loader threads.
    """

    def __init__(self) -> None:
        self.uid: str = uuid.uuid4().hex   # identity used in GPU tile-cache keys
        self.levels: list[Level] = []
        self.dtype = np.dtype(np.uint8)
        self.channels: int = 1

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of level 0."""
        return self.levels[0].shape

    @property
    def is_rgb(self) -> bool:
        """True for 3- or 4-channel colour data."""
        return self.channels >= 3

    @abstractmethod
    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        """Read one tile.

        Args:
            level: Level index (0 = finest).
            ty: Tile row.
            tx: Tile column.

        Returns:
            np.ndarray: (h, w) or (h, w, c) array; edge tiles are cropped to the
            level's shape (so may be smaller than ``Level.tile``).
        """

    def value_range(self) -> tuple[float, float]:
        """Default display range (contrast limits).

        Returns:
            tuple: (low, high). For 8-bit data (0, 255); subclasses with float
            data should override with robust percentiles.
        """
        if self.dtype == np.uint8:
            return 0.0, 255.0
        return 0.0, 1.0

    def thumbnail(self) -> np.ndarray:
        """Return the whole coarsest level as one array (used by the minimap)."""
        lv = len(self.levels) - 1
        level = self.levels[lv]
        gy, gx = level.grid
        rows = [np.concatenate([self.read_tile(lv, ty, tx) for tx in range(gx)], axis=1)
                for ty in range(gy)]
        return np.concatenate(rows, axis=0)


class ArraySource(PixelSource):
    """An in-memory array (e.g. a DHM phase map), optionally with a pyramid.

    Args:
        data: (H, W) or (H, W, C) array.
        tile: Tile edge length in pixels; small images become a single tile.
        robust_range: For float data, use the 0.5–99.5 percentiles as the
            default display range instead of min/max.
    """

    def __init__(self, data: np.ndarray, tile: int = 2048, robust_range: bool = True) -> None:
        super().__init__()
        self.data = np.ascontiguousarray(data)
        self.dtype = self.data.dtype
        self.channels = 1 if self.data.ndim == 2 else self.data.shape[2]
        h, w = self.data.shape[:2]
        t = (min(tile, h), min(tile, w))
        self.levels = [Level(shape=(h, w), tile=t, downsample=(1.0, 1.0))]
        self._robust = robust_range

    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        th, tw = self.levels[level].tile
        return self.data[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]

    def value_range(self) -> tuple[float, float]:
        if self.dtype == np.uint8:
            return 0.0, 255.0
        finite = self.data[np.isfinite(self.data)]
        if finite.size == 0:
            return 0.0, 1.0
        if self._robust:
            lo, hi = np.percentile(finite, [0.5, 99.5])
        else:
            lo, hi = finite.min(), finite.max()
        return float(lo), float(hi if hi > lo else lo + 1.0)


class MissingSource(PixelSource):
    """Stands in for an image whose file was not found when the project was opened.

    The image keeps its size, placement and masks, and is drawn as grey
    stripes. The original reference (``kind``, ``path``, ``rel``) is kept, so
    saving the project writes it back unchanged: nothing is lost while the file
    is away, and Properties › Relink… points the image at the file again.
    Analysis tools refuse it (``missing``) instead of measuring the stripes.

    Args:
        kind: How the image was stored ("zarr", "tiff").
        path: Its absolute path.
        rel: Its project-relative path (or None).
        shape: (height, width) of level 0, in pixels.
    """

    missing = True

    def __init__(self, kind: str, path: str | None, rel: str | None, shape: tuple[int, int]) -> None:
        super().__init__()
        self.kind, self.path, self.rel = kind, path, rel
        self.dtype = np.dtype(np.uint8)
        self.channels = 3
        h, w = int(shape[0]), int(shape[1])
        f = 1
        while True:                                      # a pyramid (÷4), so a whole slide draws quickly
            lh, lw = max(1, -(-h // f)), max(1, -(-w // f))
            self.levels.append(Level(shape=(lh, lw), tile=(1024, 1024), downsample=(h / lh, w / lw)))
            if max(lh, lw) <= 1024:
                break
            f *= 4

    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        lv = self.levels[level]
        th, tw = lv.tile
        h, w = min(th, lv.shape[0] - ty * th), min(tw, lv.shape[1] - tx * tw)
        yy, xx = np.mgrid[ty * th:ty * th + h, tx * tw:tx * tw + w]
        stripes = ((xx + yy) // 24) % 2 == 0                 # diagonal stripes: clearly not an image
        tile = np.where(stripes, 92, 70).astype(np.uint8)
        return np.repeat(tile[..., None], 3, axis=2)


def is_missing(image) -> bool:
    """Whether an image's file was not found (its pixels are a placeholder)."""
    return bool(getattr(image.source, "missing", False))
