"""Tile choice and view maths (pure functions)."""
import numpy as np

from modalytics.io.sources import Level
from modalytics.render import tiling
from modalytics.render.view import View, format_length, nice_length

LEVEL0 = Level(shape=(60373, 147407), tile=(1024, 1024), downsample=(1.0, 1.0))
LEVEL2 = Level(shape=(3773, 9212), tile=(1024, 1024), downsample=(60373 / 3773, 147407 / 9212))
SIZES = [0.2512, 1.005, 4.02, 16.08]


def test_choose_level():
    assert tiling.choose_level(0.25, SIZES) == 0
    assert tiling.choose_level(0.7, SIZES) == 1          # 1.005 ≤ 0.7 × 1.5
    assert tiling.choose_level(0.6, SIZES) == 0          # 1.005 > 0.9 → too blurry
    assert tiling.choose_level(100, SIZES) == 3


def test_visible_tiles_centre_first_and_clamped():
    # View [-500, 3500]² µm-px: clamped at 0, covers tiles 0–3, centre (1500, 1500) is in tile (1, 1)
    view = np.array([[-500, -500], [3500, -500], [3500, 3500], [-500, 3500]], float)
    tiles = tiling.visible_tiles(LEVEL0, view, margin_tiles=0)
    assert set(tiles) == {(ty, tx) for ty in range(4) for tx in range(4)}
    assert tiles[0] == (1, 1)


def test_visible_tiles_outside_is_empty():
    view = np.array([[-5000, -5000], [-4000, -5000], [-4000, -4000], [-5000, -4000]], float)
    assert tiling.visible_tiles(LEVEL0, view) == []


def test_edge_tile_rect_is_cropped():
    gy, gx = LEVEL2.grid
    x0, y0, w, h = tiling.tile_rect_px(LEVEL2, gy - 1, gx - 1)
    assert np.isclose(x0 + w, 147407) and np.isclose(y0 + h, 60373)


def test_view_round_trip_and_zoom_anchor():
    v = View(cx=100, cy=50, um_per_px=2, width=800, height=600)
    assert np.allclose(v.world_to_screen(*v.screen_to_world(123, 456)), (123, 456))
    anchor = v.screen_to_world(200, 100)
    v.zoom_at(3.0, 200, 100)
    assert np.allclose(v.screen_to_world(200, 100), anchor)
    assert np.isclose(v.um_per_px, 2 / 3)


def test_fit_shows_whole_rect():
    v = View(width=1000, height=500)
    v.fit((0, 0, 4000, 1000), margin=0)
    c = v.corners()
    assert c[:, 0].min() <= 0 and c[:, 0].max() >= 4000 and c[:, 1].min() <= 0 and c[:, 1].max() >= 1000


def test_scale_bar_lengths():
    assert nice_length(130) == 100 and nice_length(270) == 200 and nice_length(0.6) == 0.5
    assert format_length(2000) == "2 mm" and format_length(50) == "50 µm" and format_length(0.5) == "500 nm"


def test_reframe_keeps_content_still_on_the_monitor():
    # A dock on the left grows by 120 px: canvas moves right by 120 and shrinks by 120
    v = View(cx=500, cy=300, um_per_px=2, width=1000, height=800)
    left_global = 280
    world_at_monitor_px = v.screen_to_world(700 - left_global, 400)
    v.reframe(120, 0, -120, 0)
    v.width = 880
    assert np.allclose(v.screen_to_world(700 - (left_global + 120), 400), world_at_monitor_px)
