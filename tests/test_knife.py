"""
Touching objects drawn by hand stay separate, and the Knife cuts objects (and suggestions) in two.
"""

from types import SimpleNamespace

import numpy as np
import shapely

from modalytics.core import annotations as ann
from modalytics.core.commands import UndoStack
from modalytics.core.model import ImageNode, MaskNode, Project, Suggestion
from modalytics.io.sources import ArraySource
from modalytics.tools.annotate import apply_shape, cut_along

GLAND = ("Gland", "")


def setup(*objects):
    """An image with a Gland mask of ``objects``; the tool context the functions need."""
    img = ImageNode("i", ArraySource(np.zeros((300, 300), np.float32)), (1.0, 1.0), "phase")
    p = Project()
    p.reset([img])
    if objects:
        m = MaskNode(geometry=ann.as_multipolygon(list(objects)), group="Gland")
        p.add_mask(img, m)
    ctx = SimpleNamespace(project=p, undo=UndoStack(), tasks=None)
    return ctx, img


def objects(img):
    return [len(shapely.get_parts(m.geometry)) for m in img.masks]


def test_a_shape_that_only_touches_stays_separate_one_mostly_over_extends():
    ctx, img = setup(shapely.box(0, 0, 50, 50))
    # a neighbour drawn overlapping the gland a little (2 of its 50 µm): two glands, a hair apart
    apply_shape(ctx, img, ann.as_multipolygon([shapely.box(48, 0, 100, 50)]), GLAND, False, separate=True)
    assert objects(img) == [2]
    a, b = sorted(shapely.get_parts(img.masks[0].geometry), key=lambda g: g.bounds[0])
    assert a.area == 2500 and 0 < a.distance(b) < 0.1                         # the first one is not changed
    # a bulge drawn mostly over the first gland: it extends it
    apply_shape(ctx, img, ann.as_multipolygon([shapely.box(10, 40, 30, 60)]), GLAND, False, separate=True)
    assert objects(img) == [2] and max(g.bounds[3] for g in shapely.get_parts(img.masks[0].geometry)) == 60
    # the brush (no separating) joins what it touches
    apply_shape(ctx, img, ann.as_multipolygon([shapely.box(98, 0, 120, 50)]), GLAND, False)
    assert objects(img) == [2]


def test_the_knife_cuts_an_object_crossed_from_side_to_side_in_two():
    ctx, img = setup(shapely.box(0, 0, 100, 50), shapely.box(200, 0, 250, 50))
    wiggle = [(50, -10), (45, 10), (55, 25), (48, 40), (50, 60)]               # freeform, across the first
    assert cut_along(ctx, img, wiggle, GLAND, all_classes=True)
    pieces = sorted(shapely.get_parts(img.masks[0].geometry), key=lambda g: (g.bounds[0], g.area))
    assert objects(img) == [3] and abs(sum(g.area for g in pieces) - 7500) < 10  # only a hair lost
    assert pieces[-1].bounds == (200, 0, 250, 50)                                 # the other one untouched
    ctx.undo.undo()
    assert objects(img) == [2]
    # a line that ends inside the object cuts nothing (no slit)
    assert not cut_along(ctx, img, [(50, -10), (50, 25)], GLAND, all_classes=True)
    assert objects(img) == [2] and img.masks[0].geometry.equals(ann.as_multipolygon(
        [shapely.box(0, 0, 100, 50), shapely.box(200, 0, 250, 50)]))


def test_the_knife_cuts_suggestions_too_keeping_their_class():
    ctx, img = setup()
    p = ctx.project
    node = p.suggestions_for(img)
    p.set_suggestions(node, [Suggestion(shapely.box(0, 0, 100, 50), "Gland", "Low grade dysplasia", 0.3, "why")])
    assert cut_along(ctx, img, [(50, -10), (50, 60)], GLAND, all_classes=True)
    assert len(node.items) == 2 and all(s.subgroup == "Low grade dysplasia" and s.reason == "why" for s in node.items)
    ctx.undo.undo()
    assert len(node.items) == 1
