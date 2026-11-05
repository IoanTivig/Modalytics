"""Registration maths: rotation about a pivot, mirroring in place."""
import numpy as np
import pytest

from modalytics.core import geometry as geo
from modalytics.core.geometry import Placement
from modalytics.services import registration as reg

PTS = np.array([[10, 20], [200, 35], [120, 210], [30, 180], [220, 230]], float)


def test_rotate_about_pivot_keeps_the_pivot_fixed():
    p = Placement(100, 50, 10, 1.2, False)
    pivot = geo.apply(p.local_to_world(), (40, 30))
    q = reg.rotated_about(p, pivot, 33)
    assert np.isclose(q.rotation_deg, 43)
    assert np.allclose(geo.apply(q.local_to_world(), (40, 30)), pivot)


def test_mirror_in_place_keeps_the_footprint_and_swaps_sides():
    p = Placement(100, 50, 25, 1.0, False)
    w, h = 235.5, 235.5
    q = reg.mirrored_in_place(p, w)
    corners = lambda pl: geo.apply(pl.local_to_world(), geo.rect_corners(w, h))
    assert np.allclose(sorted(map(tuple, np.round(corners(p), 6))), sorted(map(tuple, np.round(corners(q), 6))))
    # The local top-left now lands where the top-right was
    assert np.allclose(geo.apply(q.local_to_world(), (0, 0)), geo.apply(p.local_to_world(), (w, 0)))
    # Mirroring twice gives the original mapping back
    twice = reg.mirrored_in_place(q, w)
    assert twice.flip_x == p.flip_x
    assert np.allclose(geo.apply(twice.local_to_world(), PTS), geo.apply(p.local_to_world(), PTS))


def _blobs(shape, seed, sigma=4.0):
    import cv2
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random(shape) > 0.985).astype(np.float32), (0, 0), sigma) * 40


@pytest.mark.parametrize("truth", [
    Placement(400.0, 300.0, 0.0, 1.0, False),
    Placement(520.0, 210.0, 37.0, 1.0, False),
    Placement(300.0, 450.0, -120.0, 1.0, True),
])
def test_search_recovers_a_known_pose(truth):
    """Moving map = a crop of the fixed map seen through a known placement."""
    import cv2
    fpx, mpx = 4.0, 4.0
    fixed = _blobs((300, 300), 3)                                   # 1200 × 1200 µm at 4 µm/px
    h = w = 60                                                      # moving image 240 µm
    # Sample the fixed map at the moving image's pixels (moving px → world → fixed px)
    m = np.linalg.inv(reg._grid(0, 0, fpx)) @ truth.local_to_world() @ reg._grid(0, 0, mpx)
    moving = cv2.warpAffine(fixed, m[:2].astype(np.float32), (w, h),            # dst px → src px
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    start = geo.apply(truth.local_to_world(), (120, 120)) + (60, -40)          # rough guess, 72 µm off
    cands = reg.search(fixed, (0, 0, fpx, fpx), np.eye(3), moving, mpx, start, 150,
                       angles=np.arange(-180, 180, 3.0), mirrors=(False, True))
    best = cands[0].placement
    got = geo.apply(best.local_to_world(), [[0, 0], [240, 240]])
    want = geo.apply(truth.local_to_world(), [[0, 0], [240, 240]])
    assert best.flip_x == truth.flip_x
    assert np.abs(got - want).max() < 2 * fpx, (got, want)
    assert cands[0].score > 0.8 and (len(cands) < 2 or cands[1].score < cands[0].score)


def test_placement_from_matrix_round_trip():
    for p in (Placement(1, 2, 33, 1.1, False), Placement(-5, 7, -150, 0.9, True)):
        q = reg.placement_from_matrix(p.local_to_world())
        assert np.allclose(q.local_to_world(), p.local_to_world())
