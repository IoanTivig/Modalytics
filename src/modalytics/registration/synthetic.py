"""
Testing a registration method against a known answer (Qt-free).

A patch is cut from a reference image at a known position, rotation, scale and
mirroring (the "moving" image, of the reference's own type); it is then put back
roughly — its centre moved by an error inside the search radius, rotation and scale
reset — and the method must find the true placement. The errors (µm, degrees, %,
mirroring) and the time are reported per trial. A method that fails here, on its own
image type, cannot be trusted on two different ones.

``synthetic_tissue`` makes a reference without any file: crypt-like rings on a noisy
stroma texture (not tissue — fine for the methods working on intensities or edges,
not for those needing a segmentation model).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from modalytics.core import geometry as geo
from modalytics.core.geometry import Placement
from modalytics.registration.context import Search
from modalytics.registration.run import run_method


@dataclass
class Trial:
    """One synthetic trial: the truth, what was found, the errors."""

    truth: Placement
    found: Placement | None
    score: float
    position_um: float
    rotation_deg: float
    scale_pct: float
    mirror_ok: bool
    seconds: float
    error: str = ""

    def success(self, position_um: float = 5.0, rotation_deg: float = 2.0) -> bool:
        return (not self.error and self.found is not None and self.position_um <= position_um
                and self.rotation_deg <= rotation_deg and self.mirror_ok)


@dataclass
class Report:
    """All trials of a method on one reference."""

    method: str
    trials: list = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.trials)
        ok = sum(t.success() for t in self.trials)
        done = [t for t in self.trials if t.found is not None]
        if not done:
            return f"{self.method}: 0/{n} found" + (f" ({self.trials[0].error})" if self.trials else "")
        med = lambda xs: float(np.median(xs))
        return (f"{self.method}: {ok}/{n} correct (≤ 5 µm, ≤ 2°) · median error "
                f"{med([t.position_um for t in done]):.1f} µm, {med([t.rotation_deg for t in done]):.1f}°, "
                f"{med([t.scale_pct for t in done]):.1f} % · {med([t.seconds for t in self.trials]):.1f} s each")


def synthetic_tissue(size_um: float = 1500.0, um_per_px: float = 1.0, seed: int = 0):
    """A reference image without a file: crypt-like rings (dark lumen, bright epithelium) on a noisy stroma."""
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    rng = np.random.default_rng(seed)
    n = int(size_um / um_per_px)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    img = rng.normal(0.35, 0.05, (n, n)).astype(np.float32)
    for _ in range(int((size_um / 70.0) ** 2)):
        cx, cy = rng.uniform(0, n, 2)
        a, b = rng.uniform(18, 35, 2) / um_per_px
        th = rng.uniform(0, math.pi)
        dx, dy = xx - cx, yy - cy
        u = (dx * math.cos(th) + dy * math.sin(th)) / a
        v = (-dx * math.sin(th) + dy * math.cos(th)) / b
        r = np.sqrt(u * u + v * v)
        img[(r > 0.55) & (r < 1.0)] += 0.45                  # epithelium ring
        img[r <= 0.55] -= 0.15                               # lumen
    from scipy import ndimage
    img = ndimage.gaussian_filter(img, 1.0 / um_per_px).astype(np.float32)
    return ImageNode("Synthetic tissue", ArraySource(img), (um_per_px, um_per_px), "brightfield")


def _cut(reference, truth: Placement, size_um: tuple, um_per_px: float):
    """The moving image: the reference's pixels under ``truth`` (moving local µm → world µm), at ``um_per_px``."""
    import cv2
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.services.segmentation import read_region
    w, h = int(round(size_um[0] / um_per_px)), int(round(size_um[1] / um_per_px))
    corners = geo.apply(truth.local_to_world(), geo.rect_corners(*size_um))
    local = geo.apply(np.linalg.inv(reference.local_to_world()), corners)
    pad = 4 * um_per_px
    rect = (local[:, 0].min() - pad, local[:, 1].min() - pad, local[:, 0].max() + pad, local[:, 1].max() + pad)
    crop, (x0, y0, px, py) = read_region(reference, rect, um_per_px)
    crop = np.asarray(crop, dtype=np.float32)
    # moving pixel centre → moving local µm → world → reference local → crop pixel
    grid_m = geo.translation(0.5 * um_per_px, 0.5 * um_per_px) @ geo.scaling(um_per_px)
    to_crop = np.linalg.inv(geo.translation(x0 + 0.5 * px, y0 + 0.5 * py) @ geo.scaling(px, py)) @ \
        np.linalg.inv(reference.local_to_world()) @ truth.local_to_world() @ grid_m
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if crop.ndim == 3:
        chans = [cv2.warpAffine(np.ascontiguousarray(crop[..., c]), to_crop[:2].astype(np.float32), (w, h), flags=flags,
                                borderMode=cv2.BORDER_REFLECT) for c in range(crop.shape[2])]
        pixels = np.stack(chans, axis=2)
    else:
        pixels = cv2.warpAffine(crop, to_crop[:2].astype(np.float32), (w, h), flags=flags,
                                borderMode=cv2.BORDER_REFLECT)
    return ImageNode("Synthetic patch", ArraySource(pixels), (um_per_px, um_per_px), reference.modality)


def _textured_centre(reference, size_um: tuple, rng) -> np.ndarray:
    """A world point where the reference has structure (the most varied of a few random spots), so the patch
    is not plain background or glass."""
    from modalytics.services.segmentation import read_region
    rw, rh = reference.size_um
    margin = 0.75 * math.hypot(*size_um) + 50
    best, best_v = None, -1.0
    for _ in range(12):
        lx, ly = rng.uniform(margin, max(margin + 1, rw - margin)), rng.uniform(margin, max(margin + 1, rh - margin))
        px = max(size_um) / 32
        sample, _f = read_region(reference, (lx - size_um[0] / 2, ly - size_um[1] / 2, lx + size_um[0] / 2,
                                             ly + size_um[1] / 2), px)
        sample = np.asarray(sample, dtype=np.float32)
        v = float(np.std(sample.mean(axis=2) if sample.ndim == 3 else sample))
        if v > best_v:
            best, best_v = (lx, ly), v
    return geo.apply(reference.local_to_world(), best)


def evaluate_method(spec, reference, trials: int = 5, size_um: tuple = (235.0, 235.0), um_per_px: float | None = None,
                radius_um: float = 150.0, rotation: bool = True, scale_range: float = 0.0, mirror: bool = False,
                seed: int = 1, profile=None, device: str = "auto", progress=None, cancelled=None) -> Report:
    """Run ``trials`` synthetic registrations of patches cut from ``reference`` (see the module docstring).

    Args:
        spec: The method (registration.registry.MethodSpec).
        reference: The image patches are cut from (ImageNode).
        trials: How many.
        size_um: The patch size (µm; default a DHM phase map).
        um_per_px: The patch's resolution (default: the reference's own pixel size).
        radius_um: The search radius; the rough position is off by up to 60 % of it.
        rotation: Turn the patch by a random angle (and search all angles); else keep it upright.
        scale_range: Scale the patch by up to ± this (and search it).
        mirror: Mirror half the patches (and search mirroring).
        seed: Random seed (the same seed gives the same trials).

    Returns:
        Report
    """
    rng = np.random.default_rng(seed)
    report = Report(spec.name)
    um_per_px = um_per_px or float(reference.pixel_size_um[0])
    rotation = rotation and "rotation" in spec.finds          # what the method cannot find is not asked of it
    scale_range = scale_range if "scale" in spec.finds else 0.0
    mirror = mirror and "mirror" in spec.finds
    for n in range(trials):
        centre = _textured_centre(reference, size_um, rng)
        theta = (float(rng.uniform(-3, 3)) if spec.refines_only else float(rng.uniform(-180, 180))) if rotation else 0.0
        scale = float(1 + rng.uniform(-scale_range, scale_range)) if scale_range > 0 else 1.0
        flip = bool(mirror and n % 2 == 1)
        # The true placement: the patch centred on ``centre`` at that pose
        lin = geo.rotation(theta) @ geo.scaling(scale) @ (geo.scaling(-1.0, 1.0) if flip else np.eye(3))
        off = geo.apply(lin, (size_um[0] / 2, size_um[1] / 2)) - geo.apply(lin, (0.0, 0.0))
        truth = Placement(float(centre[0] - off[0]), float(centre[1] - off[1]), theta, scale, flip)
        moving = _cut(reference, truth, size_um, um_per_px)
        # Put back roughly: centre off by up to 60 % of the radius, upright, unscaled, not mirrored
        # (a method that only refines starts close: within 8 µm and at the true rotation ± 3°)
        reach = 8.0 if spec.refines_only else 0.6 * radius_um
        err = rng.uniform(0, reach) * np.array([math.cos(a := rng.uniform(0, 2 * math.pi)), math.sin(a)])
        rough = centre + err
        if spec.refines_only:
            moving.placement = Placement(float(truth.x_um + err[0]), float(truth.y_um + err[1]),
                                         truth.rotation_deg + float(rng.uniform(-3, 3)), scale, flip)
        else:
            moving.placement = Placement(float(rough[0] - size_um[0] / 2), float(rough[1] - size_um[1] / 2))
        search = Search(tuple(rough), radius_um, 180.0 if rotation else 0.0, scale_range, mirror, top_k=3)
        t0 = time.perf_counter()
        sub = None if progress is None else (lambda f, n=n: progress((n + f) / trials))
        try:
            found = run_method(spec, moving, reference, search, None, device, profile, sub, cancelled)
            error = "" if found else "no candidate"
        except ValueError as exc:
            found, error = [], str(exc)
        seconds = time.perf_counter() - t0
        if not found:
            report.trials.append(Trial(truth, None, 0.0, math.inf, math.inf, math.inf, False, seconds, error))
            continue
        best = found[0].placement
        got = geo.apply(best.local_to_world(), (size_um[0] / 2, size_um[1] / 2))
        d_rot = abs((best.rotation_deg - theta + 180) % 360 - 180)
        report.trials.append(Trial(truth, best, found[0].score, float(np.hypot(*(got - centre))), d_rot,
                                   100 * abs(best.scale - scale) / scale, best.flip_x == flip, seconds))
        if progress is not None:
            progress((n + 1) / trials)
    return report
