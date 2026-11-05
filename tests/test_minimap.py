"""Minimap thumbnails: colormapped phase data and the mask highlight."""
import numpy as np

from modalytics.core.annotations import as_multipolygon
from modalytics.core.model import ImageNode
from modalytics.io.sources import ArraySource
from modalytics.render import minimap
from modalytics.render.visuals import COLORMAPS


def test_phase_thumbnail_uses_the_image_contrast():
    data = np.tile(np.linspace(0, 2, 100, dtype=np.float32), (50, 1))
    img = ImageNode("p", ArraySource(data, robust_range=False), (0.5, 0.5), "phase")
    rgba, px, py = minimap.thumbnail(img, COLORMAPS)
    assert rgba.shape == (50, 100, 4) and px == py == 0.5
    assert rgba[0, 0, 0] < 10 and rgba[0, -1, 0] > 245          # gray map: black → white across the range


def test_large_images_are_downsampled_with_correct_pixel_size():
    data = np.zeros((1000, 2000, 3), np.uint8)
    img = ImageNode("w", ArraySource(data), (0.25, 0.25), "wsi")
    rgba, px, py = minimap.thumbnail(img, COLORMAPS)
    assert max(rgba.shape[:2]) <= minimap.MAX_PX and np.isclose(px * rgba.shape[1], 0.25 * 2000)


def test_highlight_dims_outside_the_mask_only():
    rgba = np.full((20, 40, 4), 200, np.uint8)
    square = as_multipolygon([np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])   # local µm
    out = minimap.highlight(rgba, square, 1.0, 1.0)
    assert out[5, 5, 0] == 200                                   # inside: untouched
    assert out[15, 30, 0] == int(200 * minimap.DIM)               # outside: dimmed
    empty = minimap.highlight(rgba, None, 1.0, 1.0)
    assert (empty[..., 0] == int(200 * minimap.DIM)).all()
