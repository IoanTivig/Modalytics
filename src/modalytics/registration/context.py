"""
What a registration method script works with (Qt-free).

``RegistrationContext`` holds the two images and the shared search (see
``registration.registry``) and offers the pieces methods are built from:

* **pixels**: ``read`` (a region at a resolution), ``grey`` (one channel), ``search_rect``
  (the part of the reference the search can reach);
* **feature maps**: ``feature_map`` (a segmentation model's probability, e.g. glands);
* **matching**: ``match`` — the exhaustive normalised cross-correlation search over the
  poses the tool asks for (coarse), each candidate refined (fine) and polished with ECC
  (``services.registration.search`` / ``polish``) — for any pair of maps;
* **results**: ``candidate`` (from a local → world matrix or a Placement), ``refine_ecc``
  (one placement polished on two maps), ``pose_grid`` (the rotations, scales and mirrors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from modalytics.core import geometry as geo
from modalytics.core.geometry import Placement


@dataclass
class Search:
    """The search every method gets from the Register tool.

    Attributes:
        centre_world: Rough position of the moving image's centre (world µm).
        radius_um: How far from it the centre may end up.
        angle_range: Rotations tried: ±angle_range around the current rotation (180 = any, 0 = keep).
        scale_range: Scale corrections tried: ±scale_range (fraction, e.g. 0.02).
        mirror: Also try the mirrored image.
        top_k: How many candidates to return.
    """

    centre_world: tuple
    radius_um: float
    angle_range: float = 180.0
    scale_range: float = 0.0
    mirror: bool = True
    top_k: int = 5


def _base_cancelled():
    from modalytics.services.segmentation import Cancelled as SegmentationCancelled
    return SegmentationCancelled


class Cancelled(_base_cancelled()):
    """The user cancelled the registration (a kind of the background tasks' Cancelled)."""


class RegistrationContext:
    """The moving image, the reference, the search, and helpers (see the module docstring).

    Attributes:
        moving: The image being placed (ImageNode).
        reference: The image it is placed on (ImageNode).
        search: Search.
        device: Where models run ("auto", "cpu" or a device key).
        profile: The project's profile (which segmentation models), or None.
    """

    def __init__(self, moving, reference, search: Search, device: str = "auto", profile=None,
                 progress=None, cancelled=None) -> None:
        self.moving, self.reference, self.search = moving, reference, search
        self.device, self.profile = device, profile
        self._progress, self._cancelled = progress, cancelled

    # ------------------------------------------------------------ progress
    def progress(self, fraction: float) -> None:
        if self._progress is not None:
            self._progress(max(0.0, min(1.0, float(fraction))))

    def check(self) -> None:
        """Stop here if the user cancelled (raises Cancelled)."""
        if self._cancelled is not None and self._cancelled():
            raise Cancelled()

    def stage(self, a: float, b: float):
        """A progress callback for the part a…b of the whole."""
        return lambda f: self.progress(a + (b - a) * f)

    # ------------------------------------------------------------ pixels
    def search_rect(self, margin_um: float = 0.0) -> tuple[float, float, float, float]:
        """The reference's local rectangle the search can reach: the circle plus the moving image's half diagonal."""
        mw, mh = self.moving.size_um
        s = self.search
        half_diag = 0.5 * math.hypot(mw, mh) * self.moving.placement.scale * (1 + s.scale_range)
        reach = s.radius_um + half_diag + margin_um
        cx, cy = s.centre_world
        corners = np.array([[cx - reach, cy - reach], [cx + reach, cy - reach], [cx + reach, cy + reach],
                            [cx - reach, cy + reach]])
        local = geo.apply(np.linalg.inv(self.reference.local_to_world()), corners)
        return float(local[:, 0].min()), float(local[:, 1].min()), float(local[:, 0].max()), float(local[:, 1].max())

    def read(self, image, um_per_px: float, rect=None) -> tuple[np.ndarray, tuple]:
        """Pixels of an image's local rectangle at ``um_per_px`` (default: the moving image whole, the reference's
        search area). Returns (pixels float32, frame (x0, y0, px, py) in the image's local µm)."""
        from modalytics.services.segmentation import read_region
        if rect is None:
            rect = (0.0, 0.0, *image.size_um) if image is self.moving else self.search_rect(4 * um_per_px)
        pixels, frame = read_region(image, rect, um_per_px)
        return np.asarray(pixels, dtype=np.float32), frame

    @staticmethod
    def grey(pixels: np.ndarray) -> np.ndarray:
        """One channel: colour → luminance (0.299 R + 0.587 G + 0.114 B); single-channel as it is."""
        p = np.asarray(pixels, dtype=np.float32)
        if p.ndim == 3:
            p = p[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.ascontiguousarray(p)

    def feature_map(self, image, um_per_px: float, rect=None, group: str = "Gland") -> tuple[np.ndarray, tuple]:
        """A segmentation model's probability of ``group`` (0–1) over a region, at ``um_per_px``.

        Raises:
            ValueError: If no installed model segments this image type.
        """
        from modalytics.services.segmentation import find_models, pick_model, probability_map
        pkg = pick_model(find_models(), image.modality, self.profile, group)
        if pkg is None:
            raise ValueError(f"this method uses a segmentation model: none for {image.modality} images")
        if rect is None:
            rect = (0.0, 0.0, *image.size_um) if image is self.moving else self.search_rect(4 * um_per_px)
        return probability_map(image, rect, pkg, um_per_px, self.device, None, self._cancelled)

    # ------------------------------------------------------------ poses
    def pose_grid(self, angle_step: float = 3.0) -> tuple[list, list, tuple]:
        """The rotations (degrees), scale corrections and mirror states the search asks for."""
        s, base = self.search, self.moving.placement.rotation_deg
        if s.angle_range >= 180:
            angles = list(np.arange(-180.0, 180.0, angle_step))
        elif s.angle_range <= 0:
            angles = [base]
        else:
            angles = list(base + np.arange(-s.angle_range, s.angle_range + 1e-9, angle_step))
        scales = [1.0] if s.scale_range <= 0 else list(np.linspace(1 - s.scale_range, 1 + s.scale_range, 3))
        mirrors = (False, True) if s.mirror else (self.moving.placement.flip_x,)
        return angles, scales, mirrors

    # ------------------------------------------------------------ matching
    def match(self, fixed_map: np.ndarray, fixed_frame: tuple, moving_map: np.ndarray, coarse_factor: float = 2.0,
              angle_step: float = 3.0, refine: bool = True) -> list:
        """The shared search on two maps: exhaustive masked normalised cross-correlation over the poses (at the
        maps' resolution divided by ``coarse_factor``), then each candidate refined at full resolution (0.5°,
        0.5 % steps) and polished with ECC. ``moving_map`` covers the moving image whole.

        Args:
            fixed_map, fixed_frame: The reference's map and its frame (from ``read`` / ``feature_map``).
            moving_map: The moving image's map (same µm per pixel as the reference's).
            coarse_factor: How much coarser the first search is.
            angle_step: Rotation step of the first search (degrees).
            refine: Refine and polish the candidates.

        Returns:
            list[services.registration.Candidate]: Best first.
        """
        import cv2
        from modalytics.services import registration as reg
        mw, mh = self.moving.size_um
        l2w = self.reference.local_to_world()
        fixed_map = np.ascontiguousarray(fixed_map, dtype=np.float32)
        moving_map = np.ascontiguousarray(moving_map, dtype=np.float32)
        k = max(1.0, float(coarse_factor))

        def shrink(a):
            return cv2.resize(a, (max(1, int(round(a.shape[1] / k))), max(1, int(round(a.shape[0] / k)))),
                              interpolation=cv2.INTER_AREA)
        fixed_c, moving_c = (shrink(fixed_map), shrink(moving_map)) if k > 1 else (fixed_map, moving_map)
        frame_c = (fixed_frame[0], fixed_frame[1], fixed_frame[2] * fixed_map.shape[1] / fixed_c.shape[1],
                   fixed_frame[3] * fixed_map.shape[0] / fixed_c.shape[0])
        angles, scales, mirrors = self.pose_grid(angle_step)
        s = self.search
        coarse = reg.search(fixed_c, frame_c, l2w, moving_c, mw / moving_c.shape[1], s.centre_world, s.radius_um,
                            angles, scales, mirrors, top_k=s.top_k, progress=self.stage(0.0, 0.8 if refine else 1.0),
                            cancelled=self._cancelled)
        if not refine:
            return coarse
        refined = []
        fine_px = mw / moving_map.shape[1]
        for n, cand in enumerate(coarse):
            self.check()
            p = cand.placement
            centre = geo.apply(p.local_to_world(), (mw / 2, mh / 2))
            fine_angles = [p.rotation_deg] if s.angle_range <= 0 else \
                list(p.rotation_deg + np.arange(-angle_step, angle_step + 1e-9, 0.5))
            fine_scales = [p.scale] if s.scale_range <= 0 else list(p.scale + np.arange(-0.01, 0.0101, 0.005))
            best = reg.search(fixed_map, fixed_frame, l2w, moving_map, fine_px, centre, 3 * frame_c[2], fine_angles,
                              fine_scales, (p.flip_x,), top_k=1, cancelled=self._cancelled)
            c = best[0] if best else cand
            refined.append(reg.polish(c, fixed_map, fixed_frame, l2w, moving_map, fine_px))
            self.progress(0.8 + 0.2 * (n + 1) / max(len(coarse), 1))
        refined.sort(key=lambda c: -c.score)
        return refined

    def refine_ecc(self, placement: Placement, fixed_map: np.ndarray, fixed_frame: tuple, moving_map: np.ndarray):
        """One placement polished with ECC on two maps (a Candidate; unchanged if ECC does not converge)."""
        from modalytics.services import registration as reg
        mw = self.moving.size_um[0]
        cand = reg.Candidate(placement, 0.0)
        return reg.polish(cand, np.ascontiguousarray(fixed_map, dtype=np.float32), fixed_frame,
                          self.reference.local_to_world(), np.ascontiguousarray(moving_map, dtype=np.float32),
                          mw / moving_map.shape[1])

    # ------------------------------------------------------------ results
    @staticmethod
    def candidate(matrix=None, score: float = 0.0, placement: Placement | None = None):
        """A result: from a 3×3 moving-local-µm → world-µm matrix (rotation, scale, mirror allowed) or a Placement."""
        from modalytics.services import registration as reg
        if placement is None:
            placement = reg.placement_from_matrix(np.asarray(matrix, float))
        return reg.Candidate(placement, float(score))

    def to_world(self, frame: tuple) -> np.ndarray:
        """3×3 mapping a reference map's pixel (col, row) to world µm (pixel centres)."""
        x0, y0, px, py = frame
        return self.reference.local_to_world() @ geo.translation(x0 + 0.5 * px, y0 + 0.5 * py) @ geo.scaling(px, py)

    def moving_grid(self, moving_shape: tuple) -> np.ndarray:
        """3×3 mapping a whole-moving-image map's pixel (col, row) to the moving image's local µm."""
        h, w = moving_shape[:2]
        mw, mh = self.moving.size_um
        return geo.translation(0.5 * mw / w, 0.5 * mh / h) @ geo.scaling(mw / w, mh / h)
