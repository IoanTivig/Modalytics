"""
Micrometre world geometry: the single source of truth for where images sit.

Conventions used everywhere in Modalytics:

* **World** coordinates are micrometres (µm); x points right, y points down
  (the usual image convention), so screen and data agree without flips.
* Every image has a **local frame** in µm: its pixel grid scaled by its
  calibrated pixel size. Pixel ``(row, col)`` covers the local square
  ``[col·sx, (col+1)·sx) × [row·sy, (row+1)·sy)``; its centre is at
  ``((col+0.5)·sx, (row+0.5)·sy)``.
* A ``Placement`` maps the local frame to the world:
  ``world = T(x, y) · R(rotation) · S(scale) · F(flip) · local``.
  Masks are stored in their parent image's local frame, so moving an image
  moves its masks for free and never touches their coordinates.

All matrices are 3×3 homogeneous affines acting on column vectors ``(x, y, 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


## Affine helpers ##

def translation(tx: float, ty: float) -> np.ndarray:
    """Return a 3×3 translation matrix."""
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def scaling(sx: float, sy: float | None = None) -> np.ndarray:
    """Return a 3×3 scaling matrix (``sy`` defaults to ``sx``)."""
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def rotation(degrees: float) -> np.ndarray:
    """Return a 3×3 rotation, counter-clockwise **as seen on screen**.

    With y pointing down, a counter-clockwise screen rotation by θ is
    ``x' = x·cosθ + y·sinθ``, ``y' = −x·sinθ + y·cosθ``.

    Args:
        degrees: Rotation angle in degrees (positive = counter-clockwise on screen).

    Returns:
        np.ndarray: 3×3 rotation matrix.
    """
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def apply(matrix: np.ndarray, points) -> np.ndarray:
    """Apply a 3×3 affine to points.

    Args:
        matrix: 3×3 affine.
        points: (N, 2) array-like of (x, y), or a single (x, y).

    Returns:
        np.ndarray: Transformed points, same shape as the input.
    """
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts2 = np.atleast_2d(pts)
    out = pts2 @ matrix[:2, :2].T + matrix[:2, 2]
    return out[0] if single else out


def to_4x4(matrix: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Embed a 2D 3×3 affine in a 4×4 matrix (for the GPU scene graph).

    Args:
        matrix: 3×3 affine.
        z: Translation along z (used only for draw ordering).

    Returns:
        np.ndarray: 4×4 float matrix.
    """
    m = np.eye(4)
    m[:2, :2] = matrix[:2, :2]
    m[:2, 3] = matrix[:2, 2]
    m[2, 3] = z
    return m


## Placement ##

@dataclass(frozen=True)
class Placement:
    """Where an image's local µm frame sits in the world.

    Frozen (immutable) on purpose: a change is always a *new* Placement set
    through the project, which emits a change event and can be undone.

    Attributes:
        x_um: World x of the local origin (image top-left corner when unrotated).
        y_um: World y of the local origin.
        rotation_deg: Counter-clockwise screen rotation about the local origin.
        scale: Dimensionless correction on top of the calibrated pixel size
            (e.g. 1.02 found by registration). Calibration itself lives on the
            image (``pixel_size_um``), so µm stay correct.
        flip_x: Mirror the local x axis (before rotation).
    """

    x_um: float = 0.0
    y_um: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    flip_x: bool = False

    def local_to_world(self) -> np.ndarray:
        """Return the 3×3 affine mapping local µm to world µm."""
        flip = scaling(-1.0, 1.0) if self.flip_x else np.eye(3)
        return (translation(self.x_um, self.y_um)
                @ rotation(self.rotation_deg)
                @ scaling(self.scale)
                @ flip)

    def moved_by(self, dx_um: float, dy_um: float) -> "Placement":
        """Return a copy translated by (dx, dy) world µm."""
        return replace(self, x_um=self.x_um + dx_um, y_um=self.y_um + dy_um)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {"x_um": self.x_um, "y_um": self.y_um, "rotation_deg": self.rotation_deg,
                "scale": self.scale, "flip_x": self.flip_x}

    @classmethod
    def from_dict(cls, d: dict) -> "Placement":
        """Build from a dict written by ``to_dict``."""
        return cls(**{k: d[k] for k in ("x_um", "y_um", "rotation_deg", "scale", "flip_x") if k in d})


def rect_corners(width: float, height: float) -> np.ndarray:
    """Return the 4 corners (x, y) of the rectangle [0, w] × [0, h], clockwise from top-left."""
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])


def bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of an (N, 2) point array."""
    p = np.asarray(points, dtype=float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())
