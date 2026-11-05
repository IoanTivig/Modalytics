"""Vector annotation geometry: validity, holes, brush strokes, transforms, schema."""
import numpy as np

from modalytics.core import annotations as ann
from modalytics.core import geometry as geo


def square(x, y, s):
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], float)


def test_self_intersection_is_repaired():
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float)
    mp = ann.as_multipolygon([bowtie])
    assert mp.is_valid and len(mp.geoms) == 2 and np.isclose(mp.area, 50)


def test_brush_stroke_area_is_a_capsule():
    r, length = 5.0, 40.0
    mp = ann.stroke_polygon(np.array([[0, 0], [length, 0]]), r)
    assert np.isclose(mp.area, 2 * r * length + np.pi * r * r, rtol=1e-3)


def test_erasing_inside_makes_a_hole():
    mp = ann.as_multipolygon([square(0, 0, 100)])
    holed = ann.as_multipolygon(mp.difference(ann.stroke_polygon(np.array([[50, 50]]), 10)))
    (ext, holes), = ann.rings(holed)
    assert len(holes) == 1 and np.isclose(holed.area, 100 * 100 - np.pi * 100, rtol=1e-3)


def test_transform_matches_point_transform():
    mp = ann.as_multipolygon([square(10, 20, 5)])
    m = geo.translation(3, -4) @ geo.rotation(30) @ geo.scaling(2)
    out = ann.transform(mp, m)
    assert np.isclose(out.area, mp.area * 4)
    c = np.asarray(out.centroid.coords)[0]
    assert np.allclose(c, geo.apply(m, np.asarray(mp.centroid.coords)[0]))


def test_remove_specks():
    mp = ann.as_multipolygon([square(0, 0, 10), square(50, 50, 0.5)])
    assert len(ann.remove_specks(mp, 1.0).geoms) == 1


def test_schema_round_trip_and_aliases():
    s = ann.default_schema()
    assert ann.LabelSchema.from_dict(s.to_dict()).choices() == s.choices()
    assert ann.canonical("Glande") == "Gland" and ann.canonical("Displasia High Grade") == "Dysplasia high grade"


def test_local_edit_equals_full_overlay():
    rng = np.random.default_rng(4)
    import shapely
    blobs = [shapely.Point(x, y).buffer(r) for x, y, r in zip(rng.uniform(0, 1000, 300), rng.uniform(0, 1000, 300),
                                                                  rng.uniform(5, 20, 300))]
    mp = ann.as_multipolygon(shapely.union_all(blobs))
    stroke = ann.stroke_polygon(np.array([[100, 500], [900, 520]]), 15)
    added, changed = ann.edit(mp, stroke, "add")
    assert changed and added.is_valid and added.equals(ann.as_multipolygon(mp.union(stroke)))
    cut, changed = ann.edit(mp, stroke, "cut")
    assert changed and cut.is_valid and np.isclose(cut.area, mp.difference(stroke).area)
    far = ann.stroke_polygon(np.array([[5000, 5000]]), 5)
    same, changed = ann.edit(mp, far, "cut")
    assert not changed and same is mp


def test_delete_vertices_keeps_outlines_and_drops_small_holes():
    from modalytics.tools.edit_points import delete_vertices
    ext = square(0, 0, 100)
    pent = np.vstack([ext[:2], [[100, 50]], ext[2:]])            # 5 points
    hole = square(40, 40, 10)
    assert len(delete_vertices([pent, hole], {(0, 2)})[0]) == 4
    assert delete_vertices([ext, hole], {(0, 0), (0, 1)}) is None          # outline would drop below 3
    out = delete_vertices([ext, hole], {(1, 0), (1, 1)})                    # hole drops below 3 → removed
    assert out is not None and len(out) == 1
    out = delete_vertices([pent, hole], {(0, 2), (1, 0), (1, 1)})           # both at once
    assert len(out) == 1 and len(out[0]) == 4
