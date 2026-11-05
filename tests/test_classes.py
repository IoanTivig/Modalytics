"""
Selecting single objects (glands) and giving them, or a whole class layer, another class.
"""

from types import SimpleNamespace

import numpy as np
import shapely

from modalytics.core import annotations as ann
from modalytics.core.commands import UndoStack
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io.sources import ArraySource
from modalytics.services.classes import change_class, change_layer_class, delete_objects
from modalytics.tools.select import RectSelect, object_at

LOW, HIGH = "Low grade dysplasia", "High grade dysplasia"


def setup():
    """An image with three plain "Gland" objects (not graded) and one High grade one."""
    img = ImageNode("i", ArraySource(np.zeros((300, 300), np.float32)), (1.0, 1.0), "phase")
    p = Project()
    p.reset([img])
    p.labels.ensure("Gland", LOW)
    p.labels.ensure("Gland", HIGH)
    p.add_mask(img, MaskNode(geometry=[[(0, 0), (40, 0), (40, 40), (0, 40)], [(100, 0), (140, 0), (140, 40), (100, 40)],
                                       [(200, 0), (240, 0), (240, 40), (200, 40)]], group="Gland"))
    p.add_mask(img, MaskNode(geometry=[[(0, 100), (40, 100), (40, 140), (0, 140)]], group="Gland", subgroup=HIGH))
    return p, img


def count(img, sub):
    m = next((m for m in img.masks if m.group == "Gland" and m.subgroup == sub), None)
    return len(m.geometry.geoms) if m is not None else 0


def test_a_click_selects_one_object_shift_and_ctrl_add_and_remove():
    p, img = setup()
    ctx = SimpleNamespace(project=p, preview=SimpleNamespace(set_box=lambda *a: None), request_draw=lambda: None)

    def click(x, y, **kw):
        RectSelect(ctx, {"x": 0, "y": 0}, x, y, **kw).finish(ctx, x, y)
    click(20, 20)
    assert len(p.objects) == 1 and p.objects[0][1].bounds == (0, 0, 40, 40) and p.selected is img.masks[0]
    click(120, 20, add=True)
    click(20, 120, add=True)
    assert len(p.objects) == 3 and set(p.selection) == set(img.masks)       # objects of two layers
    click(120, 20, toggle=True)
    assert len(p.objects) == 2
    click(280, 280)                                                       # the image, not an object
    assert p.objects == [] and p.selected is img


def test_change_class_moves_objects_unchanged_and_undo_restores():
    p, img = setup()
    undo = UndoStack()
    two = [object_at(p, 20, 20), object_at(p, 120, 20)]
    cmd, moved = change_class(p, two, "Gland", LOW)                        # the class layer is made
    undo.push(cmd)
    assert count(img, "") == 1 and count(img, LOW) == 2 and len(moved) == 2
    assert {q.bounds for _m, q in moved} == {(0, 0, 40, 40), (100, 0, 140, 40)}   # outlines exactly kept
    cmd, _ = change_class(p, [object_at(p, 20, 120)], "Gland", LOW)          # High → Low: joins, kept separate
    undo.push(cmd)
    assert count(img, LOW) == 3 and count(img, HIGH) == 0
    undo.undo()
    undo.undo()
    assert count(img, "") == 3 and count(img, HIGH) == 1 and count(img, LOW) == 0
    assert all(m.subgroup != LOW for m in img.masks)                        # the layer made for them is gone


def test_selected_objects_follow_edits_and_delete_only_the_objects():
    p, img = setup()
    p.select_objects([object_at(p, 20, 20)])
    cmd = delete_objects(p, list(p.objects))
    cmd.do()
    assert count(img, "") == 2 and len(img.masks) == 2                      # the layer stays
    assert p.objects == []                                                   # the deleted object left the selection


def test_a_whole_layer_relabelled_or_merged():
    p, img = setup()
    plain = img.masks[0]
    cmd = change_layer_class(p, plain, "Gland", LOW)                        # no Low layer yet: relabelled
    cmd.do()
    assert (plain.group, plain.subgroup, plain.name) == ("Gland", LOW, f"Gland › {LOW}") and count(img, LOW) == 3
    cmd.undo()
    assert (plain.subgroup, plain.name) == ("", "Gland")
    high = img.masks[1]
    cmd = change_layer_class(p, high, "Gland", "")                          # a Gland layer exists: merged into it
    cmd.do()
    assert count(img, "") == 4 and high not in img.masks
    cmd.undo()
    assert high in img.masks and count(img, HIGH) == 1 and count(img, "") == 3
    assert shapely.equals(ann.as_multipolygon(high.geometry), high.geometry)
