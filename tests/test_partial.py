"""Partial objects (shape incomplete): cut by the image edge (found by itself) or marked by hand (undoable); measured
with what needs the whole object left empty, graded only by a classifier that reads none of that, left out of
classifier training unless the set includes them where the parameter allows it, and of the report's shape
statistics. And the Training tables' sort order."""

from __future__ import annotations

import os
import sys

from shapely.geometry import Polygon, box

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_projects import PX, graded_project  # noqa: E402

LOW, HIGH = "Low grade dysplasia", "High grade dysplasia"


def _project(tmp_path):
    """A graded project with one more High-grade gland cut by the image's left edge, and one gland marked partial."""
    from modalytics.core.model import MaskNode, Project
    from modalytics.io.project_file import load_project
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.services.partial import set_partial
    path = graded_project(str(tmp_path / "case"), seed=3, n_low=4, n_high=3)
    data = load_project(path)
    img = data.images[0]
    cut = MaskNode(geometry=[Polygon([(0, 90), (14, 88), (16, 102), (0, 104)])], group="Gland", subgroup=HIGH)
    cut.parent = img
    img.children.append(cut)
    prof = ProfileLibrary().get("colonic_tissue")
    prof.parameters = {u: {} for u in ("object:any:area", "object:any:circularity", "object:any:in_image",
                                       "object:phase:mean")}
    prof.report = {"parameters": ["object:any:area", "object:any:circularity"]}
    project = Project()
    project.reset(data.images, profile=prof)
    project.path = path
    low = next(m for m in img.masks if m.subgroup == LOW)
    marked = list(low.geometry.geoms)[0]
    cmd = set_partial(project, [(low, marked)], True)
    return project, img, cut, low, marked, cmd


def test_edge_cut_and_marked_by_hand_with_undo(tmp_path):
    from modalytics.core.model import PARTIAL_EDGE, PARTIAL_MARKED
    project, img, cut, low, marked, cmd = _project(tmp_path)
    assert img.edge_cut(list(cut.geometry.geoms)[0]) and img.partial_reason(list(cut.geometry.geoms)[0]) == PARTIAL_EDGE
    others = [p for p in low.geometry.geoms if not p.equals(marked)]
    assert all(img.partial_reason(p) == "" for p in others)              # inside the image: complete
    assert img.partial_reason(marked) == ""                              # not marked until the command runs
    cmd.do()
    assert img.partial_reason(marked) == PARTIAL_MARKED and len(img.partial_points) == 1
    cmd.undo()
    assert img.partial_reason(marked) == "" and "partial" not in img.metadata
    cmd.do()
    # Saved with the project
    from modalytics.io.project_file import load_project, save_project
    save_project(project, str(tmp_path / "again"))
    again = load_project(str(tmp_path / "again.modalytics")).images[0]
    assert again.partial_reason(marked) == PARTIAL_MARKED
    assert not img.edge_cut(box(5, 5, 20, 20)) and img.edge_cut(box(5, 5, 20, img.size_um[1]))


def test_measured_classified_and_reported_with_their_shape_incomplete(tmp_path):
    from modalytics.parameters.results import classify, fixed_columns, fixed_values, measure_project
    from modalytics.report.html import build
    from modalytics.training.classifiers import Classifier, Condition, Rule
    project, img, cut, low, marked, cmd = _project(tmp_path)
    cmd.do()
    res = measure_project(project)
    rows = res.of_scope("object")
    assert len(rows) == 8                                               # every object is measured
    reasons = sorted(r.partial for r in rows if r.partial)
    assert reasons == ["cut by the image edge", "marked partial"]
    assert "Partial" in fixed_columns(res, "object")
    edge_row = next(r for r in rows if r.partial == "cut by the image edge")
    assert edge_row.values["object:any:in_image"] == 0.0                # the existing edge flag agrees
    import math
    assert math.isnan(edge_row.values["object:any:circularity"]) and "needs the whole object" in \
        edge_row.problems["object:any:circularity"]                     # shape: left empty, said why
    assert math.isfinite(edge_row.values["object:phase:mean"])           # pixel statistics: the visible part
    whole = next(r for r in rows if not r.partial)
    assert math.isfinite(whole.values["object:any:circularity"])
    # A classifier reading a shape parameter does not grade it; one reading pixel statistics does
    texture = Classifier("ranges", "Gland", [LOW, HIGH], rules=[
        Rule(HIGH, [Condition("object:phase:mean", ">=", -1e9, "Phase mean")])], note="test")   # any value: High
    graded, _p = classify(res, texture)
    edge_row = next(r for r in graded.of_scope("object") if r.partial == "cut by the image edge")
    assert edge_row.prediction["grade"] == HIGH and "none of its parameters needs the whole object" in \
        edge_row.prediction["reason"]
    clf = Classifier("ranges", "Gland", [LOW, HIGH], rules=[
        Rule(HIGH, [Condition("object:any:circularity", "<=", 0.8, "Circularity")])], note="test")
    res, _p = classify(res, clf)
    project.profile.classification["Gland"] = clf.to_dict()                # the report shows the profile's
    edge_row = next(r for r in res.of_scope("object") if r.partial == "cut by the image edge")
    assert edge_row.prediction["grade"] == "" and "not graded" in edge_row.prediction["reason"] \
        and "Circularity" in edge_row.prediction["reason"]
    assert fixed_values(edge_row, True)[-1] == ""                       # not judged against the annotation
    project.set_results(res)
    page = build(project)
    assert "2 object(s) with an incomplete shape" in page and "· partial</span>" in page
    assert "not compared with the annotation" in page                     # nor counted in the agreement


def test_classifier_training_leaves_partial_objects_out(tmp_path):
    from modalytics.io.project_file import save_project
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.training import dataset as ds
    project, img, cut, low, marked, cmd = _project(tmp_path)
    cmd.do()
    save_project(project, project.path)
    prof = ProfileLibrary().get("colonic_tissue")
    prof.parameters = {"object:any:area": {}}
    tset = ds.TrainingSet("t", prof, "colonic_tissue", "Gland", [ds.ProjectEntry(project.path)])
    data = ds.gather(tset, use_cache=False)
    assert len(data.grade) == 6 and data.partial == {"case": 2}         # 8 glands, 2 with an incomplete shape
    # Included where the parameter allows it: area needs the whole object (empty), the phase mean does not
    import numpy as np
    from modalytics.training import classifiers as cl
    prof.parameters = {"object:any:area": {}, "object:phase:mean": {}}
    tset.include_partial = True
    data = ds.gather(tset)
    area, mean = (next(i for i, c in enumerate(data.columns) if c[0] == u)
                  for u in ("object:any:area", "object:phase:mean"))
    assert len(data.grade) == 8 and int(data.partial_rows.sum()) == 2 and not data.partial
    assert np.all(np.isnan(data.values[data.partial_rows, area])) and np.all(np.isfinite(data.values[:, mean]))
    every = np.ones(8, bool)
    assert cl.usable(data, every, ["object:phase:mean"]).sum() == 8      # its classifier may use them
    assert cl.usable(data, every, ["object:any:area", "object:phase:mean"]).sum() == 6   # not with area
    assert ds.TrainingSet.from_dict(tset.to_dict()).include_partial
    ov = ds.project_overview(project.path, prof, "Gland")
    assert ov.objects == 8 and ov.partial == 2


def test_sort_keys_numbers_text_and_empty_last():
    from modalytics.ui.training.sorting import sort_key
    cells = ["12 %", "—", "abc", "0.95 (0.90–0.98)", "1,234 µm²", "", "★ 3", "Zeta", "−2"]
    ordered = sorted(cells, key=sort_key)
    assert ordered[:5] == ["−2", "0.95 (0.90–0.98)", "★ 3", "12 %", "1,234 µm²"]
    assert ordered[5:7] == ["abc", "Zeta"] and set(ordered[7:]) == {"—", ""}


def test_sort_item_keeps_the_pinned_row_last():
    from PySide6 import QtCore, QtWidgets
    from modalytics.ui.training.sorting import SortItem, sortable
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tree = sortable(QtWidgets.QTreeWidget())
    tree.setColumnCount(2)
    for name, v in (("b", "0.5"), ("a", "0.9"), ("c", "—")):
        SortItem(tree, [name, v])
    total = SortItem(tree, ["All", "0.7"])
    total.pinned = True
    assert [tree.topLevelItem(i).text(0) for i in range(4)] == ["b", "a", "c", "All"]   # unsorted until clicked
    for order, expected in ((QtCore.Qt.SortOrder.AscendingOrder, ["b", "a", "c", "All"]),
                            (QtCore.Qt.SortOrder.DescendingOrder, ["a", "b", "c", "All"])):   # "—", All last
        tree.sortByColumn(1, order)
        assert [tree.topLevelItem(i).text(0) for i in range(4)] == expected


def test_a_gland_grazing_the_edge_counts_as_whole_by_the_profiles_share(tmp_path):
    from modalytics.core.model import EDGE_SHARE, PARTIAL_EDGE
    from modalytics.profiles.model import Profile
    project, img, cut, low, marked, cmd = _project(tmp_path)
    w, h = img.size_um
    graze = Polygon([(0, 50), (20, 40), (40, 50), (20, 60)])            # touches the left edge at one point
    side = box(0, 120, 30, 140)                                         # a fifth of its outline on the edge
    assert img.edge_share(graze) < 0.02 and abs(img.edge_share(side) - 0.2) < 0.02     # (the 1-px band)
    assert img.partial_reason(graze) == "" and img.partial_reason(graze, 0.0) == PARTIAL_EDGE   # 0 %: any touch
    assert img.partial_reason(side) == PARTIAL_EDGE and img.partial_reason(side, 0.25) == ""
    assert img.edge_share(box(10, 10, 20, 20)) == 0.0
    prof = Profile("p")
    assert prof.edge_share == EDGE_SHARE == 0.05
    prof.measuring = {"edge_percent": 12.5}
    again = Profile.from_dict(prof.to_dict())
    assert again.edge_share == 0.125 and again.to_dict()["measuring"] == {"edge_percent": 12.5}


def test_the_share_changes_what_classifier_training_leaves_out(tmp_path):
    from modalytics.io.project_file import save_project
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.training import dataset as ds
    project, img, cut, low, marked, cmd = _project(tmp_path)
    save_project(project, project.path)
    prof = ProfileLibrary().get("colonic_tissue")
    prof.parameters = {"object:any:area": {}}
    tset = ds.TrainingSet("t", prof, "colonic_tissue", "Gland", [ds.ProjectEntry(project.path)])
    assert ds.gather(tset).partial == {"case": 1}                       # the edge gland (24 % of its outline)
    prof.measuring = {"edge_percent": 30}                                # now it counts as whole
    assert ds.gather(tset).partial == {} and ds.project_overview(project.path, prof, "Gland").partial == 0
