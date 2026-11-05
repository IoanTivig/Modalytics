"""µm geometry: placements, affines, and their round trips."""
import numpy as np
import pytest

from modalytics.core import geometry as geo
from modalytics.core.geometry import Placement


def test_identity_placement_maps_local_to_world_unchanged():
    pts = np.array([[0, 0], [10, 5]])
    assert np.allclose(geo.apply(Placement().local_to_world(), pts), pts)


def test_translation_and_scale():
    m = Placement(x_um=100, y_um=-20, scale=2).local_to_world()
    assert np.allclose(geo.apply(m, (3, 4)), (106, -12))


def test_rotation_is_counter_clockwise_on_screen():
    # With y down, the local x axis rotated 90° CCW on screen points "up" = −y
    m = geo.rotation(90)
    assert np.allclose(geo.apply(m, (1, 0)), (0, -1), atol=1e-12)


def test_flip_mirrors_x_before_rotation():
    m = Placement(flip_x=True).local_to_world()
    assert np.allclose(geo.apply(m, (5, 2)), (-5, 2))


@pytest.mark.parametrize("p", [Placement(12.5, -3, 37, 1.3, False), Placement(-1e4, 2e4, -120, 0.7, True)])
def test_inverse_round_trip(p):
    m = p.local_to_world()
    pts = np.random.default_rng(0).uniform(-500, 500, (20, 2))
    back = geo.apply(np.linalg.inv(m), geo.apply(m, pts))
    assert np.allclose(back, pts)


def test_placement_dict_round_trip():
    p = Placement(1, 2, 3, 4, True)
    assert Placement.from_dict(p.to_dict()) == p


def test_to_4x4_keeps_2d_part_and_z():
    m4 = geo.to_4x4(geo.translation(3, 4), z=7)
    assert m4[0, 3] == 3 and m4[1, 3] == 4 and m4[2, 3] == 7 and m4[3, 3] == 1
