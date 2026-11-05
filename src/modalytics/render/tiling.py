"""
Which tiles to show: pure functions, no GPU and no Qt (unit-tested).

For each image and each frame the canvas asks:

1. Which pyramid level matches the current zoom?  → ``choose_level``
2. Which tiles of that level intersect the view? → ``visible_tiles``

The view is expressed in world µm; it is mapped into the image's level-0
pixel grid with the inverse of ``ImageNode.pixel_to_world`` (this handles
rotated and flipped images too).
"""

from __future__ import annotations

import numpy as np

from modalytics.core import geometry as geo
from modalytics.io.sources import Level


def choose_level(screen_um_per_px: float, level_um_per_px: list[float], max_magnification: float = 1.5) -> int:
    """Pick the coarsest level that is still sharp enough on screen.

    A level is acceptable if one of its pixels covers at most
    ``max_magnification`` screen pixels. With the usual ×4 pyramid steps,
    1.5 trades a little softness for far fewer tiles (a level 4× too fine
    would need 16× more tiles).

    Args:
        screen_um_per_px: World µm covered by one screen pixel.
        level_um_per_px: µm per pixel of each level (finest first, increasing).
        max_magnification: Largest allowed texel-to-screen-pixel ratio.

    Returns:
        int: Level index (0 = finest).
    """
    best = 0
    for i, s in enumerate(level_um_per_px):
        if s <= screen_um_per_px * max_magnification:
            best = i
    return best


def visible_tiles(level: Level, view_corners_px: np.ndarray, margin_tiles: float = 0.25) -> list[tuple[int, int]]:
    """List the tiles of a level that intersect the view, centre first.

    Args:
        level: The level.
        view_corners_px: (4, 2) view corners in level-0 pixel coordinates (x, y).
        margin_tiles: Extra border (in tiles) loaded around the view, so small
            pans show detail immediately.

    Returns:
        list: (ty, tx) tuples sorted by distance to the view centre, so the
        middle of the screen sharpens first.
    """
    dy, dx = level.downsample
    th, tw = level.tile
    gy, gx = level.grid
    # 1. View bounding box in this level's tile units
    x0, y0, x1, y1 = geo.bounds(view_corners_px)
    tx0 = int(np.floor(x0 / dx / tw - margin_tiles))
    tx1 = int(np.floor(x1 / dx / tw + margin_tiles))
    ty0 = int(np.floor(y0 / dy / th - margin_tiles))
    ty1 = int(np.floor(y1 / dy / th + margin_tiles))
    # 2. Clamp to the grid
    tx0, ty0 = max(tx0, 0), max(ty0, 0)
    tx1, ty1 = min(tx1, gx - 1), min(ty1, gy - 1)
    if tx1 < tx0 or ty1 < ty0:
        return []
    # 3. Centre-first order
    cx, cy = (x0 + x1) / 2 / dx / tw - 0.5, (y0 + y1) / 2 / dy / th - 0.5
    tiles = [(ty, tx) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    tiles.sort(key=lambda t: (t[1] - cx) ** 2 + (t[0] - cy) ** 2)
    return tiles


def tile_rect_px(level: Level, ty: int, tx: int) -> tuple[float, float, float, float]:
    """Return the tile's (x0, y0, width, height) in level-0 pixel units."""
    dy, dx = level.downsample
    th, tw = level.tile
    h = min(th, level.shape[0] - ty * th)
    w = min(tw, level.shape[1] - tx * tw)
    return tx * tw * dx, ty * th * dy, w * dx, h * dy
