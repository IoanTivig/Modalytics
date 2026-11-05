"""Judging segmentation: overlap, object matching, merges and splits, the evaluated area, projects."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from modalytics.training import segmentation as se


def sq(x, y, s=10.0):
    return box(x, y, x + s, y + s)


def test_overlap_and_object_counts():
    hand = [sq(0, 0), sq(20, 0), sq(40, 0), sq(60, 0)]
    found = [sq(1, 0),                      # matches hand 0 (IoU 90/110)
             sq(20, 0),                     # matches hand 1 exactly
             sq(80, 0)]                     # invented; hands 2 and 3 missed
    c, items = se.compare(hand, found, "img")
    assert (c.hand, c.found, c.tp, c.fp, c.fn) == (4, 3, 2, 1, 2)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(0.5)
    assert c.f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 2))
    inter = 90 + 100
    assert c.dice == pytest.approx(2 * inter / (400 + 300))
    assert c.iou == pytest.approx(inter / (400 + 300 - inter))
    assert c.matched_iou == pytest.approx((90 / 110 + 1) / 2)
    assert c.pq == pytest.approx(c.matched_iou * c.f1)                     # PQ = SQ · F1
    kinds = sorted(i.kind for i in items)
    assert kinds == ["invented", "missed", "missed"]
    inv = next(i for i in items if i.kind == "invented")
    assert inv.outline and inv.area == pytest.approx(100) and Polygon(inv.outline).contains(
        Polygon(inv.outline).representative_point())


def test_matching_threshold_poor_merge_split():
    # IoU 0.5 is the default threshold: a half-shifted square (IoU 1/3) does not match
    c, _ = se.compare([sq(0, 0)], [sq(5, 0)])
    assert c.tp == 0 and c.fp == 1 and c.fn == 1
    c, _ = se.compare([sq(0, 0)], [sq(5, 0)], iou_min=0.3)
    assert c.tp == 1
    # a match outlined poorly (IoU 0.6) is listed for a look
    c, items = se.compare([sq(0, 0)], [box(0, 0, 10, 6)])
    assert c.tp == 1 and [i.kind for i in items] == ["poor"]
    # one model object over two hand objects: a merge (and both hands missed, the big object invented)
    c, items = se.compare([sq(0, 0), sq(12, 0)], [box(0, 0, 22, 10)])
    assert c.merges == 1 and c.tp == 0 and "merge" in [i.kind for i in items]
    # two model objects inside one hand object: a split
    c, items = se.compare([box(0, 0, 22, 10)], [sq(0, 0), sq(12, 0)])
    assert c.splits == 1 and "split" in [i.kind for i in items]
    # nothing found / nothing drawn
    c, _ = se.compare([sq(0, 0)], [])
    assert c.fn == 1 and c.dice == 0 and math.isnan(c.precision)
    c, _ = se.compare([], [sq(0, 0)])
    assert c.fp == 1 and math.isnan(c.recall)


def test_evaluated_area_and_limit():
    hand = [sq(10, 10), sq(30, 10)]
    whole = se.evaluated_area(hand, 200, 100, "image", 0)
    assert whole.area == pytest.approx(200 * 100)
    hull = se.evaluated_area(hand, 200, 100, "hull", 5)
    assert hull.contains(sq(10, 10)) and not hull.contains(sq(150, 50))
    assert se.evaluated_area([], 200, 100, "hull", 5) is None
    found = [sq(10, 10), sq(150, 50), box(43, 10, 60, 20)]          # inside, far away, 2/17 inside the margin
    kept = se.limit(found, hull)
    assert kept == [found[0]]


def test_counts_add_up_over_projects(monkeypatch, tmp_path):
    """evaluate(): per project and total, skipped images, items tagged with their project (model stubbed)."""
    from synthetic_projects import graded_project
    from modalytics.parameters.engine import objects_of
    from modalytics.training import dataset as ds
    paths = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(2)]

    def fake_found(image, rect, pkg, path, cache_dir, progress, cancelled, device):
        hand = [p for m in image.masks if m.group == "Gland" for p in objects_of(m.geometry)]
        return hand[1:] + [sq(rect[0] + 1, rect[1] + 1, 5)]               # one missed, one invented

    monkeypatch.setattr(se, "_found", fake_found)
    monkeypatch.setattr(ds, "library_folder", lambda: tmp_path / "lib")

    class Pkg:
        id, path = "stub", ""

    ev = se.evaluate(paths, "Gland", {"phase": Pkg()})
    assert [c.hand for c in ev.per_project] == [5, 5]
    assert ev.total.tp == 8 and ev.total.fn == 2 and ev.total.fp == 2 and ev.total.images == 2
    assert {i.project for i in ev.items} == {0, 1} and ev.models == {"phase": "stub"}
    none = se.evaluate(paths, "Gland", {"wsi": Pkg()})
    assert none.total.images == 0 and len(none.skipped) == 2 and "no model for phase" in none.skipped[0]


def test_marked_area_and_unreviewed_suggestions_left_out():
    import numpy as np
    from shapely.geometry import box
    from modalytics.core.model import ImageNode, Project, Suggestion
    from modalytics.io.sources import ArraySource
    from modalytics.training.segmentation import evaluated_area, without_pending
    hand = [box(40, 40, 60, 60)]
    assert evaluated_area(hand, 200, 200, "marked", 10, complete=True).area == 200 * 200     # marked: whole image
    hull = evaluated_area(hand, 200, 200, "marked", 10, complete=False)
    assert hull.area < 50 * 50                                                             # not marked: around them
    img = ImageNode("i", ArraySource(np.zeros((200, 200), np.float32)), (1.0, 1.0), "phase")
    p = Project()
    p.reset([img])
    p.set_suggestions(p.suggestions_for(img), [Suggestion(box(100, 100, 120, 120), "Gland"),
                                               Suggestion(box(40, 40, 60, 60), "Gland", regrade=True),
                                               Suggestion(box(0, 0, 10, 10), "Lumen")])
    region, n = without_pending(box(0, 0, 200, 200), img, "Gland")
    assert n == 1 and not region.contains(box(101, 101, 119, 119).centroid)             # the new object: left out
    assert region.contains(box(40, 40, 60, 60).centroid) and region.contains(box(0, 0, 10, 10).centroid)


def test_glas_object_measures_and_boundaries_on_known_shapes():
    from shapely.geometry import box
    from modalytics.training.segmentation import boundary_measures, compare
    a = box(0, 0, 10, 10)
    same, _ = compare([a], [box(0, 0, 10, 10)])
    assert same.object_dice == 1.0 and same.object_hausdorff == 0.0 and same.hd95 == 0.0 and same.nsd == 1.0
    shifted, _ = compare([a], [box(1, 0, 11, 10)])                  # 1 µm to the right
    assert abs(shifted.object_dice - 0.9) < 1e-9 and abs(shifted.object_hausdorff - 1.0) < 1e-9
    assert abs(shifted.hd95 - 1.0) < 0.05 and shifted.nsd == 1.0     # within τ = 2 µm everywhere
    assert shifted.sq == shifted.matched_iou and shifted.rq == shifted.f1
    hd95, nsd = boundary_measures(a, box(5, 0, 15, 10), tolerance=2.0)
    assert abs(hd95 - 5.0) < 0.05 and 0 < nsd < 1
    # A hand object nothing overlaps: Dice 0 for it, Hausdorff to the nearest model object
    apart, _ = compare([a, box(100, 0, 110, 10)], [box(0, 0, 10, 10)])
    assert abs(apart.object_dice - 0.75) < 1e-9                      # ½ (½ · 1 + ½ · 0) + ½ · 1
    assert abs(apart.object_hausdorff - 25.0) < 1e-9 and apart.fn == 1     # ½ (½ · 0 + ½ · 100) + ½ · 0


def test_glas_f1_counts_an_outline_covering_half_its_gland_even_when_too_large():
    hand = [sq(0, 0), sq(20, 0), sq(40, 0)]
    found = [box(0, 0, 25, 10),               # covers all of hand 0 but is 2.5× its size: IoU 0.4 (no match)…
             sq(45, 0),                       # half of hand 2: IoU 1/3 (no match), covers 50 % (GlaS: found)
             sq(80, 0)]                       # invented
    c, _ = se.compare(hand, found)
    assert c.tp == 0 and c.f1 == 0.0                                        # IoU ≥ 0.5: nothing found
    assert c.glas_tp == 2 and c.glas_f1 == pytest.approx(2 * 2 / (3 + 3))   # GlaS: 0 and 2 found
    # Two model objects each covering half of the same gland: it is found once
    two, _ = se.compare([sq(0, 0)], [box(0, 0, 5, 10), box(5, 0, 10, 10)])
    assert two.glas_tp == 1
    # Results saved before the GlaS F1 existed: unknown, not 0
    old = se.Counts(hand=3, found=3, images=1)
    assert math.isnan(old.glas_f1) and math.isnan(se.Counts().add(c).add(old).glas_f1)
