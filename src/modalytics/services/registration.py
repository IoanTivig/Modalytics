"""
Registration: computing where an image belongs relative to another (Qt-free).

Placement helpers (rotate about a point, mirror in place) and the matching engine
the registration methods share: masked normalised cross-correlation over poses
(``search``) and ECC polishing (``polish``). The methods themselves are scripts
(``modalytics.registration``, ``resources/registration``).

Conventions (see ``core.geometry``): world µm, y down; a ``Placement`` maps the
image's local µm to world µm as T(x, y) · R(θ, counter-clockwise on screen) ·
S(scale) · F(mirror x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from modalytics.core import geometry as geo
from modalytics.core.geometry import Placement


def rotated_about(placement: Placement, pivot_world, delta_deg: float) -> Placement:
    """The placement rotated by ``delta_deg`` (counter-clockwise on screen) about a world point."""
    r = geo.rotation(delta_deg)[:2, :2]
    p = np.asarray(pivot_world, dtype=float)
    origin = r @ np.array([placement.x_um, placement.y_um]) + p - r @ p
    return Placement(float(origin[0]), float(origin[1]), placement.rotation_deg + delta_deg,
                     placement.scale, placement.flip_x)


def mirrored_in_place(placement: Placement, width_um: float) -> Placement:
    """The placement mirrored left↔right about the image's own vertical centre line.

    The image stays where it is on screen; only its content is mirrored (masks too).

    Args:
        placement: Current placement.
        width_um: Width of the image's local frame (µm).
    """
    # New mapping L' = L · M with M: x → w − x. Its linear part is R S F·diag(−1, 1), i.e. the
    # flip flag toggles; its origin is L applied to the local point (w, 0).
    ox, oy = geo.apply(placement.local_to_world(), (width_um, 0.0))
    return Placement(float(ox), float(oy), placement.rotation_deg, placement.scale, not placement.flip_x)


## Automatic registration: feature-map search ##

def placement_from_matrix(m: np.ndarray) -> Placement:
    """Decompose a 3×3 similarity (possibly mirrored) into a Placement.

    Args:
        m: local µm → world µm affine whose linear part is s·R(θ)·[mirror].

    Returns:
        Placement: Same mapping (to numerical precision).
    """
    lin = np.asarray(m, float)[:2, :2]
    mirror = np.linalg.det(lin) < 0
    if mirror:
        lin = lin @ np.diag([-1.0, 1.0])
    s = math.sqrt(abs(np.linalg.det(lin)))
    theta = math.degrees(math.atan2(lin[0, 1], lin[0, 0]))
    return Placement(float(m[0, 2]), float(m[1, 2]), theta, s, bool(mirror))


@dataclass
class Candidate:
    """A possible placement of the moving image.

    Attributes:
        placement: Where the moving image would go.
        score: Normalised cross-correlation of the feature maps (−1…1).
        rank: 1 = best.
    """

    placement: Placement
    score: float
    rank: int = 0


def _grid(x0: float, y0: float, px: float) -> np.ndarray:
    """Array pixel (col, row) → local µm, pixel centres."""
    return geo.translation(x0 + 0.5 * px, y0 + 0.5 * px) @ geo.scaling(px)


def _warp_template(moving_map: np.ndarray, a0: np.ndarray):
    """Warp the moving map by a 3×3 (moving px → fixed px, no translation yet) into a tight canvas.

    Returns:
        tuple: (template, mask, shifted 3×3 such that template = warp(moving, shifted)).
    """
    import cv2
    h, w = moving_map.shape
    corners = geo.apply(a0, np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]]))
    lo, hi = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
    shifted = geo.translation(-lo[0], -lo[1]) @ a0
    size = (int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1)
    tpl = cv2.warpAffine(moving_map, shifted[:2].astype(np.float32), size, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.ones_like(moving_map), shifted[:2].astype(np.float32), size, flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT)
    return tpl, mask, shifted


def search(fixed_map: np.ndarray, fixed_frame: tuple, fixed_l2w: np.ndarray,
           moving_map: np.ndarray, moving_px: float, center_world, radius_um: float,
           angles, scales=(1.0,), mirrors=(False,), top_k: int = 5,
           min_separation_um: float | None = None, progress=None, cancelled=None) -> list[Candidate]:
    """Exhaustive masked-NCC search of the moving feature map in the fixed one.

    For every (mirror, scale, angle) the moving map is warped into the fixed
    map's pixel grid and matched with masked, mean-normalised cross-correlation
    (OpenCV ``TM_CCOEFF_NORMED``); only positions whose centre lies within
    ``radius_um`` of ``center_world`` count. The best candidates at clearly
    different positions are returned, best first.

    Args:
        fixed_map: (H, W) float32 feature map of the reference.
        fixed_frame: (x0, y0, px, py) of ``fixed_map`` in the reference's local µm.
        fixed_l2w: Reference image local → world (3×3).
        moving_map: (h, w) float32 feature map of the moving image, in its own local frame.
        moving_px: µm per pixel of ``moving_map`` (its origin is the image's local origin).
        center_world: Search centre (world µm), usually the current position of the moving image.
        radius_um: Search radius (µm).
        angles: Rotations to try (degrees, counter-clockwise on screen).
        scales: Scale corrections to try.
        mirrors: Mirror states to try.
        top_k: Number of distinct candidates returned.
        min_separation_um: Candidates closer than this are the same match (default: half
            the moving image's size).
        progress: Called with 0–1.
        cancelled: Returns True to abort.

    Returns:
        list[Candidate]: Best first.
    """
    import cv2
    fx0, fy0, fpx, _ = fixed_frame
    grid_f = _grid(fx0, fy0, fpx)
    grid_m = _grid(0.0, 0.0, moving_px)
    to_fixed_px = np.linalg.inv(grid_f) @ np.linalg.inv(fixed_l2w)
    center_px = geo.apply(to_fixed_px, center_world)
    radius_px = radius_um / fpx
    h, w = moving_map.shape
    m_center = np.array([(w - 1) / 2, (h - 1) / 2])
    if min_separation_um is None:
        min_separation_um = 0.5 * max(h, w) * moving_px
    found = []                                        # (score, dx, dy, shifted, mirror, scale, angle)
    combos = [(mi, sc, an) for mi in mirrors for sc in scales for an in angles]
    for n, (mirror, scale, angle) in enumerate(combos):
        if cancelled is not None and cancelled():
            from modalytics.services.segmentation import Cancelled
            raise Cancelled()
        lin = geo.rotation(angle) @ geo.scaling(scale) @ (geo.scaling(-1.0, 1.0) if mirror else np.eye(3))
        # Linear part of moving px → fixed px for this pose (the match supplies the translation)
        lin_px = np.eye(3)
        lin_px[:2, :2] = (to_fixed_px @ lin @ grid_m)[:2, :2]
        tpl, mask, shifted = _warp_template(moving_map, lin_px)
        th, tw = tpl.shape
        if th >= fixed_map.shape[0] or tw >= fixed_map.shape[1]:
            continue
        res = cv2.matchTemplate(fixed_map, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        res[~np.isfinite(res)] = -1.0
        # Restrict to positions whose moving centre lies in the search circle
        c_off = geo.apply(shifted, m_center)
        ys, xs = np.ogrid[:res.shape[0], :res.shape[1]]
        res[(xs + c_off[0] - center_px[0]) ** 2 + (ys + c_off[1] - center_px[1]) ** 2 > radius_px ** 2] = -1.0
        # Keep a few local maxima per pose (distinct positions)
        for _ in range(3):
            dy, dx = np.unravel_index(int(np.argmax(res)), res.shape)
            score = float(res[dy, dx])
            if score <= -1.0:
                break
            found.append((score, dx, dy, shifted, mirror, scale, angle))
            r = int(min_separation_um / fpx)
            res[max(0, dy - r):dy + r + 1, max(0, dx - r):dx + r + 1] = -1.0
        if progress is not None:
            progress((n + 1) / len(combos))
    # Best first, one candidate per distinct position
    found.sort(key=lambda f: -f[0])
    out: list[Candidate] = []
    centres = []
    for score, dx, dy, shifted, mirror, scale, angle in found:
        m = fixed_l2w @ grid_f @ geo.translation(dx, dy) @ shifted @ np.linalg.inv(grid_m)
        c = geo.apply(m, (w * moving_px / 2, h * moving_px / 2))
        if any(np.hypot(*(c - o)) < min_separation_um for o in centres):
            continue
        centres.append(c)
        out.append(Candidate(placement_from_matrix(m), score, len(out) + 1))
        if len(out) >= top_k:
            break
    return out


def polish(cand: Candidate, fixed_map: np.ndarray, fixed_frame: tuple, fixed_l2w: np.ndarray,
           moving_map: np.ndarray, moving_px: float) -> Candidate:
    """Sub-pixel, sub-degree refinement of a candidate with ECC (rigid, on the feature maps).

    The moving map is warped into a crop of the fixed map at the candidate pose;
    OpenCV's ECC maximisation (Evangelidis & Psarakis 2008) then finds the small
    rigid correction. If ECC does not converge the candidate is returned unchanged.

    Args:
        cand: Candidate to refine.
        fixed_map, fixed_frame, fixed_l2w: Reference feature map, its frame and placement.
        moving_map, moving_px: Moving feature map and its pixel size.

    Returns:
        Candidate: Refined (same score field).
    """
    import cv2
    fx0, fy0, fpx, _ = fixed_frame
    grid_f, grid_m = _grid(fx0, fy0, fpx), _grid(0.0, 0.0, moving_px)
    h, w = moving_map.shape
    # moving px → fixed px at the candidate pose
    t = np.linalg.inv(grid_f) @ np.linalg.inv(fixed_l2w) @ cand.placement.local_to_world() @ grid_m
    c = geo.apply(t, np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]]))
    pad = 8
    x0, y0 = int(max(0, np.floor(c[:, 0].min()) - pad)), int(max(0, np.floor(c[:, 1].min()) - pad))
    x1 = int(min(fixed_map.shape[1], np.ceil(c[:, 0].max()) + pad))
    y1 = int(min(fixed_map.shape[0], np.ceil(c[:, 1].max()) + pad))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return cand
    crop = np.ascontiguousarray(fixed_map[y0:y1, x0:x1], dtype=np.float32)
    to_crop = geo.translation(-x0, -y0) @ t
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(moving_map, to_crop[:2].astype(np.float32), size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(moving_map, dtype=np.uint8), to_crop[:2].astype(np.float32), size,
                           flags=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        # template = fixed crop, input = warped moving: warp maps crop px → warped-moving px
        _, warp = cv2.findTransformECC(crop, warped, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), valid, 3)
    except cv2.error:
        return cand
    w3 = np.vstack([warp, [0, 0, 1]]).astype(float)
    if np.hypot(w3[0, 2], w3[1, 2]) > pad:                     # implausible jump: keep the search result
        return cand
    m = fixed_l2w @ grid_f @ geo.translation(x0, y0) @ np.linalg.inv(w3) @ to_crop @ np.linalg.inv(grid_m)
    return Candidate(placement_from_matrix(m), cand.score, cand.rank)
