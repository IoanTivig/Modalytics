"""Domain model: hierarchy, events, µm-preserving edits, undo."""
import numpy as np

from modalytics.core import geometry as geo
from modalytics.core.commands import UndoStack, set_placement_command
from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io.sources import ArraySource


def image(name="img", ps=0.5, shape=(100, 200), placement=None):
    return ImageNode(name, ArraySource(np.zeros(shape, np.float32)), (ps, ps), "phase", placement)


def test_size_and_contains():
    img = image(ps=0.5, shape=(100, 200), placement=Placement(10, 20))
    assert img.size_um == (100.0, 50.0)
    assert img.contains_world(10.1, 20.1) and img.contains_world(109.9, 69.9)
    assert not img.contains_world(111, 30)


def test_image_at_returns_top_most_visible():
    p = Project()
    low, high = p.add_image(image("low")), p.add_image(image("high"))
    assert p.image_at(5, 5) is high
    p.set_visible(high, False)
    assert p.image_at(5, 5) is low


def test_masks_follow_image_moves_without_changing_coordinates():
    p = Project()
    img = p.add_image(image())
    mask = p.add_mask(img, MaskNode("m", [np.array([[1, 1], [5, 1], [5, 5]])]))
    before = mask.polygons[0].copy()
    p.set_placement(img, Placement(1000, 2000, 30))
    assert np.array_equal(mask.polygons[0], before)          # local coordinates untouched
    world = geo.apply(img.local_to_world(), before)
    assert not np.allclose(world, before)                     # but the world position moved


def test_transfer_mask_keeps_world_position():
    p = Project()
    a = p.add_image(image("a", placement=Placement(0, 0)))
    b = p.add_image(image("b", placement=Placement(37, -12, 25, 1.1)))
    poly = np.array([[10, 10], [30, 10], [30, 25], [10, 25]], float)
    mask = p.add_mask(a, MaskNode("m", [poly]))
    world_before = geo.apply(a.local_to_world(), poly)
    copy = p.transfer_mask(mask, b, duplicate=True)
    assert mask.parent is a and copy.parent is b and copy.id != mask.id
    got = geo.apply(b.local_to_world(), copy.polygons[0])
    assert sorted(map(tuple, np.round(got, 6))) == sorted(map(tuple, np.round(world_before, 6)))
    p.transfer_mask(mask, b)
    assert mask.parent is b and mask not in a.children


def test_pixel_size_change_rescales_masks_with_the_image():
    p = Project()
    img = p.add_image(image(ps=0.5))
    mask = p.add_mask(img, MaskNode("m", [np.array([[50, 50], [60, 50], [60, 60]], float)]))
    p.set_pixel_size(img, (1.0, 1.0))
    assert img.size_um == (200.0, 100.0)
    assert np.isclose(mask.area_um2, 4 * 50.0)                 # 10×10/2 µm² triangle → ×4 in area
    assert (100, 100) in set(map(tuple, np.round(mask.polygons[0], 6)))   # same pixel, new µm


def test_events_are_emitted():
    p = Project()
    seen = []
    p.structure_changed.connect(lambda: seen.append("structure"))
    p.node_changed.connect(lambda n, f: seen.append(f))
    img = p.add_image(image())
    p.set_opacity(img, 1.7)
    assert img.opacity == 1.0 and seen == ["structure", "opacity"]


def test_undo_redo_placement():
    p, undo = Project(), UndoStack()
    img = p.add_image(image())
    old, new = img.placement, Placement(5, 6)
    undo.push(set_placement_command(p, img, old, new))
    assert img.placement == new
    undo.undo()
    assert img.placement == old
    undo.redo()
    assert img.placement == new


def test_mask_for_creates_one_mask_per_class_with_schema_colour():
    p = Project()
    img = p.add_image(image())
    m1 = p.mask_for(img, "Gland", "Normal")
    m2 = p.mask_for(img, "Gland", "Normal")
    m3 = p.mask_for(img, "Gland", "Dysplasia high grade")
    assert m1 is m2 and m1 is not m3 and len(img.masks) == 2
    assert m3.color == p.labels.color("Gland", "Dysplasia high grade")
    assert m1.name == "Gland › Normal" and m1.geometry.is_empty


def test_unknown_classes_are_added_to_the_schema():
    p = Project()
    img = p.add_image(image())
    p.add_mask(img, MaskNode(geometry=None, group="Crypt", subgroup="Branched"))
    assert ("Crypt", "Branched") in p.labels.choices()
