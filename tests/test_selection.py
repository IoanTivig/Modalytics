"""
Selecting in the viewer (a click, Ctrl+click, a rectangle, Shift) and the image a drawing goes on.
"""

import numpy as np

from modalytics.core.model import ImageNode, Placement, Project
from modalytics.io.sources import ArraySource
from modalytics.tools.annotate import drawing_target
from modalytics.tools.select import RectSelect, images_in_rect


def image(name, x, y):
    return ImageNode(name, ArraySource(np.zeros((100, 100), np.float32)), (1.0, 1.0), "phase", Placement(x, y))


def at(img, fx=0.5):
    """A world point fx of the way across the image (at mid height)."""
    lo, hi = img.world_corners().min(axis=0), img.world_corners().max(axis=0)
    return lo[0] + fx * (hi[0] - lo[0]), (lo[1] + hi[1]) / 2


def centre(img):
    """A point only this image covers (a: its left part, b: its right part, as they overlap)."""
    return at(img, {"a": 0.1, "b": 0.9}.get(img.name, 0.5))


class Ctx:
    """The little of a tool context RectSelect needs."""

    class Preview:
        def set_box(self, a=None, b=None):
            self.box = a

    def __init__(self, project):
        self.project, self.preview = project, self.Preview()

    def request_draw(self):
        pass


def three():
    """a and b overlap (b on top); c apart."""
    p = Project()
    a, b, c = image("a", 0, 0), image("b", 50, 0), image("c", 400, 0)
    p.reset([a, b, c])
    return p, a, b, c


def test_drawing_goes_on_the_top_image_under_the_start_unless_the_selected_one_is_there():
    p, a, b, c = three()
    overlap = at(a, 0.75)                                          # a and b both cover it
    assert drawing_target(p, *overlap) is b                       # nothing selected: the top one
    p.select(a)
    assert drawing_target(p, *overlap) is a                       # the selected image, though lower, is there
    assert drawing_target(p, *centre(c)) is c                     # elsewhere: the image under the start, not a
    assert drawing_target(p, 1e6, 1e6) is None                    # outside every image: nothing
    b.visible = False
    p.select(None)
    assert drawing_target(p, *overlap) is a                       # a hidden image is not drawn on


def test_a_rectangle_selects_the_images_it_touches():
    p, a, b, c = three()
    (ax, ay), (cx, cy) = centre(a), centre(c)
    assert images_in_rect(p, (ax, ay), (ax + 1, ay + 1)) == [a]
    assert images_in_rect(p, (ax, ay), (cx, cy)) == [a, b, c]    # touching is enough
    assert images_in_rect(p, (1e6, 1e6), (2e6, 2e6)) == []


def press(ctx, wx, wy, add=False, toggle=False, to=None):
    """A click at (wx, wy), or a drag from there to ``to``."""
    pick = RectSelect(ctx, {"x": 0, "y": 0}, wx, wy, add=add, toggle=toggle)
    if to is not None:
        pick.move(ctx, {"x": 50, "y": 50}, *to)
    pick.finish(ctx, *(to or (wx, wy)))


def test_clicks_and_rectangles_with_shift_and_ctrl():
    p, a, b, c = three()
    ctx = Ctx(p)
    press(ctx, *centre(c))
    assert p.selection == [c]
    press(ctx, *centre(a), add=True)                              # Shift + click adds
    assert p.selection == [c, a] and p.selected is a
    press(ctx, *centre(c), toggle=True)                           # Ctrl+click on a selected one removes it
    assert p.selection == [a]
    press(ctx, *centre(c), toggle=True)                           # …and on another adds it
    assert p.selection == [a, c]
    press(ctx, *centre(b), to=(centre(b)[0] + 1, centre(b)[1] + 1))
    assert p.selection == [b]                                     # a plain rectangle replaces the selection
    assert ctx.preview.box is None                                # the rubber band is gone
    press(ctx, *centre(c), add=True, to=(centre(c)[0] + 1, centre(c)[1] + 1))
    assert p.selection == [b, c]                                  # Shift + drag adds
    press(ctx, *at(a, 0.1), add=True, to=at(c, 0.5))
    assert p.selection == [b, c, a]                               # one over all three adds the one missing
    press(ctx, 1e6, 1e6, add=True)
    assert p.selection == [b, c, a]                               # Shift + click on nothing keeps the selection
    press(ctx, 1e6, 1e6)
    assert p.selection == []                                      # a plain click on nothing clears it
