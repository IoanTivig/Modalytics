"""
The 2D view: what part of the µm world is on screen.

Kept separate from the GPU camera so the mouse logic can be unit-tested and
so the minimap, scale bar and tile selection all use the same numbers.

Screen coordinates are logical pixels, origin top-left, y down; world
coordinates are µm, y down. ``um_per_px`` is the zoom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class View:
    """Centre and zoom of the canvas.

    Attributes:
        cx, cy: World µm at the centre of the canvas.
        um_per_px: World µm per logical screen pixel (smaller = more zoomed in).
        width, height: Canvas size in logical pixels.
    """

    cx: float = 0.0
    cy: float = 0.0
    um_per_px: float = 1.0
    width: int = 800
    height: int = 600

    MIN_UM_PER_PX = 0.01     # 100 screen px per µm: far beyond any objective
    MAX_UM_PER_PX = 2000.0   # whole slides of several cm still fit

    def screen_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Logical screen pixel → world µm."""
        return (self.cx + (x - self.width / 2) * self.um_per_px,
                self.cy + (y - self.height / 2) * self.um_per_px)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """World µm → logical screen pixel."""
        return ((x - self.cx) / self.um_per_px + self.width / 2,
                (y - self.cy) / self.um_per_px + self.height / 2)

    def corners(self) -> np.ndarray:
        """(4, 2) world corners of the visible area, clockwise from top-left."""
        return np.array([self.screen_to_world(0, 0), self.screen_to_world(self.width, 0),
                         self.screen_to_world(self.width, self.height),
                         self.screen_to_world(0, self.height)])

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Move the content by (dx, dy) screen pixels (drag direction)."""
        self.cx -= dx * self.um_per_px
        self.cy -= dy * self.um_per_px

    def zoom_at(self, factor: float, x: float, y: float) -> None:
        """Zoom by ``factor`` (>1 zooms in) keeping the world point under (x, y) fixed.

        Args:
            factor: Magnification change.
            x, y: Anchor in logical screen pixels (usually the cursor).
        """
        wx, wy = self.screen_to_world(x, y)
        self.um_per_px = float(np.clip(self.um_per_px / factor, self.MIN_UM_PER_PX, self.MAX_UM_PER_PX))
        # Re-centre so (wx, wy) stays under the cursor
        self.cx = wx - (x - self.width / 2) * self.um_per_px
        self.cy = wy - (y - self.height / 2) * self.um_per_px

    def reframe(self, dx_left: float, dy_top: float, d_width: float, d_height: float) -> None:
        """Shift the view so the content stays still on the monitor during a resize.

        When a dock is dragged, the canvas' top-left corner moves by
        (dx_left, dy_top) screen pixels and its size changes by
        (d_width, d_height). Keeping the centre fixed would slide the image by
        half of every resize step (a visible wobble); instead the world point
        under each *monitor* pixel stays the same. The new size itself is set
        separately (``width``/``height``), so the call order does not matter.

        Args:
            dx_left, dy_top: Movement of the canvas' top-left corner (px).
            d_width, d_height: Change of the canvas size (px).
        """
        self.cx += (dx_left + d_width / 2) * self.um_per_px
        self.cy += (dy_top + d_height / 2) * self.um_per_px

    def fit(self, bounds: tuple[float, float, float, float], margin: float = 0.05) -> None:
        """Show the world rectangle (xmin, ymin, xmax, ymax) with a margin."""
        x0, y0, x1, y1 = bounds
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        self.um_per_px = max(w / max(self.width, 1), h / max(self.height, 1)) * (1 + 2 * margin)


def nice_length(target: float) -> float:
    """Round a length to 1, 2 or 5 × 10ⁿ (for the scale bar).

    Args:
        target: Approximate length.

    Returns:
        float: The largest "nice" value ≤ target.
    """
    if target <= 0:
        return 0.0
    exp = np.floor(np.log10(target))
    base = target / 10 ** exp
    nice = 5 if base >= 5 else 2 if base >= 2 else 1
    return float(nice * 10 ** exp)


def format_length(um: float) -> str:
    """Format a µm length with a sensible unit ("500 µm", "2 mm")."""
    if um >= 1000:
        return f"{um / 1000:g} mm"
    if um < 1:
        return f"{um * 1000:g} nm"
    return f"{um:g} µm"
