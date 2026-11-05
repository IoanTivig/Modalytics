"""
Minimap thumbnails (CPU side, no GPU and no Qt).

The minimap shows whatever the selection refers to:

* an image (WSI, phase map, …) → that image's thumbnail;
* a mask → its image, with the masked area(s) at full brightness and the rest
  dimmed, so what is still unannotated stands out;
* nothing → the largest visible image.

Thumbnails come from the coarsest pyramid level, downsampled to at most
``MAX_PX`` pixels on the long side. The mask highlight tests every thumbnail
pixel centre against the exact vector geometry (``shapely.contains_xy``),
which takes a few tens of ms at this size.
"""

from __future__ import annotations

import math

import numpy as np
import shapely

MAX_PX = 640          # long side of a minimap thumbnail (px)
DIM = 0.28            # brightness of unmasked areas in mask mode


def thumbnail(image, colormaps: dict) -> tuple[np.ndarray, float, float]:
    """RGBA thumbnail of an image and its pixel size in the image's local µm.

    Args:
        image: ImageNode.
        colormaps: name → pygfx TextureMap (used for single-channel data, with
            the image's own contrast limits, so the minimap matches the canvas).

    Returns:
        tuple: (uint8 RGBA array, µm per thumbnail pixel along x, along y).
    """
    src = image.view_source
    lv = src.levels[-1]
    data = src.thumbnail()
    k = max(1, math.ceil(max(data.shape[:2]) / MAX_PX))
    data = data[::k, ::k]
    if data.ndim == 3:
        rgba = np.empty(data.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = data[..., :3]
        rgba[..., 3] = 255
    else:
        lo, hi = image.view_display.clim
        t = np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-12), 0, 1)
        lut = np.asarray(colormaps[image.view_display.colormap].texture.data)        # (256, 4) floats
        rgba = (lut[(t * 255).astype(np.int32)] * 255).astype(np.uint8)
        rgba[..., 3] = 255
    sx, sy = image.pixel_size_um
    return np.ascontiguousarray(rgba), sx * lv.downsample[1] * k, sy * lv.downsample[0] * k


def highlight(rgba: np.ndarray, geometry, px: float, py: float, dim: float = DIM) -> np.ndarray:
    """Dim everything outside a mask geometry.

    Args:
        rgba: Thumbnail (H, W, 4) uint8.
        geometry: shapely geometry in the image's local µm.
        px, py: µm per thumbnail pixel.
        dim: Brightness factor applied outside the mask.

    Returns:
        np.ndarray: New RGBA array.
    """
    h, w = rgba.shape[:2]
    out = rgba.copy()
    if geometry is None or geometry.is_empty:
        out[..., :3] = (out[..., :3] * dim).astype(np.uint8)
        return out
    xs = (np.arange(w) + 0.5) * px
    ys = (np.arange(h) + 0.5) * py
    # Only test pixels inside the mask's bounding box; everything else is outside
    x0, y0, x1, y1 = geometry.bounds
    cols = np.nonzero((xs >= x0) & (xs <= x1))[0]
    rows = np.nonzero((ys >= y0) & (ys <= y1))[0]
    inside = np.zeros((h, w), bool)
    if len(cols) and len(rows):
        shapely.prepare(geometry)
        gx, gy = np.meshgrid(xs[cols], ys[rows])
        inside[np.ix_(rows, cols)] = shapely.contains_xy(geometry, gx.ravel(), gy.ravel()).reshape(gy.shape)
    out[~inside, :3] = (out[~inside, :3] * dim).astype(np.uint8)
    return out
