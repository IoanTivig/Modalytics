"""Measuring a whole project: the three kinds of rows, reuse of object values by class summaries, out-of-date
detection, measuring one image again, saving with the project, CSV export, cancelling."""
import csv
import math

import numpy as np
import pytest
from shapely.geometry import Point, box

from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io.sources import ArraySource
from modalytics.parameters import engine, results as res
from modalytics.parameters.registry import ParameterLibrary
from modalytics.profiles.model import Profile

PARAMS = {"object:any:area": {}, "object:any:best_alignment_metric": {}, "object:phase:mean": {},
          "object:od:mean": {}, "class:any:count": {}, "class:any:mean_bam": {"exclude_edge": False},
          "image:any:width_um": {}, "image:phase:phase_mean": {}}


def phase_image(name, x0=0.0):
    rng = np.random.default_rng(len(name))
    img = ImageNode(name, ArraySource(rng.normal(1.0, 0.2, (200, 200)).astype(np.float32)), (0.5, 0.5), "phase",
                    Placement(x0, 0))
    for group, sub, geoms in (("Gland", "Normal", [Point(30, 30).buffer(10, 128), Point(70, 70).buffer(8, 128)]),
                              ("Gland", "Dysplasia high grade", [box(10, 60, 30, 90)]),
                              ("Lamina propria", "Normal", [box(50, 5, 95, 40)])):
        m = MaskNode(geometry=geoms, group=group, subgroup=sub)
        m.parent = img
        img.children.append(m)
    return img


@pytest.fixture
def project():
    p = Project()
    p.reset([phase_image("A"), phase_image("Bee", 200)], profile=Profile("Test", images=["phase"],
                                                                          parameters=dict(PARAMS)))
    return p


def test_three_kinds_of_rows(project):
    r = res.measure_project(project)
    assert r.counts() == {"object": 8, "class": 2 * 5, "image": 2}
    labels = [(row.group, row.subgroup) for row in r.of_scope("class") if row.image_name == "A"]
    assert labels == [("Gland", ""), ("Gland", "Dysplasia high grade"), ("Gland", "Normal"),
                      ("Lamina propria", ""), ("Lamina propria", "Normal")]
    assert {c.uid for c in r.columns["object"]} == {"object:any:area", "object:any:best_alignment_metric",
                                                    "object:phase:mean", "object:od:mean"}
    assert all(set(row.values) == {c.uid for c in r.columns[row.scope]} for row in r.rows)   # every column
    first = r.of_scope("object")[0]
    lib = ParameterLibrary()
    direct, _ = engine.measure_object(Point(30, 30).buffer(10, 128), project.images[0],
                                      engine.select(lib, PARAMS, "object")[0], project.images)
    assert first.values["object:any:area"] == pytest.approx(direct["object:any:area"])
    assert math.isnan(first.values["object:od:mean"]) and "no od view" in first.problems["object:od:mean"]
    glands = [row for row in r.of_scope("class") if row.image_name == "A" and row.class_name == "Gland"][0]
    bams = [row.values["object:any:best_alignment_metric"] for row in r.of_scope("object")
            if row.image_name == "A" and row.group == "Gland"]
    assert glands.values["class:any:count"] == 3
    assert glands.values["class:any:mean_bam"] == pytest.approx(np.mean(bams))     # the object rows' values
    assert r.of_scope("image")[1].values["image:any:width_um"] == pytest.approx(100.0)


def test_out_of_date_and_measuring_one_image_again(project):
    r = res.measure_project(project)
    project.set_results(r)
    assert r.outdated(project) == []
    a, b = project.images
    project.set_mask_geometry(a.masks[0], [Point(30, 30).buffer(12, 128)])
    assert r.outdated(project) == ["1 mask(s) were edited, added or removed"]
    b.masks[0].geometry = b.masks[0].geometry.buffer(1)                     # B changes too, not measured again
    again = res.measure_project(project, images=[a])
    merged = r.merged(again, [a.id], [i.id for i in project.images])
    assert merged.counts() == dict(r.counts(), object=7)                    # A's two Normal glands are one now
    assert [row.image_name for row in merged.of_scope("image")] == ["A", "Bee"]
    new_area = [row for row in merged.of_scope("object") if row.image_name == "A"][0].values["object:any:area"]
    assert new_area == pytest.approx(math.pi * 144, rel=1e-3)
    assert merged.outdated(project) == ["1 mask(s) were edited, added or removed"]   # B is still out of date
    project.profile.parameters["object:any:perimeter"] = {}
    assert "the profile's parameters changed" in merged.outdated(project)


def test_saved_with_the_project_and_exported(project, tmp_path):
    from modalytics.io.project_file import load_project, save_project
    project.set_results(res.measure_project(project))
    path = str(tmp_path / "p.modalytics")
    save_project(project, path)
    data = load_project(path)
    back = data.results
    assert back is not None and back.counts() == project.results.counts()
    assert back.columns == project.results.columns
    row, old = back.of_scope("object")[0], project.results.of_scope("object")[0]
    assert row.values["object:any:area"] == old.values["object:any:area"] and math.isnan(row.values["object:od:mean"])
    assert row.problems == old.problems
    out = tmp_path / "objects.csv"
    assert res.export_csv(back, "object", str(out)) == 8
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                                    # Excel reads µm² right
    rows = list(csv.reader(out.open(encoding="utf-8-sig")))
    assert rows[0][:5] == ["Image", "Class", "Group", "Subgroup", "Object"] and "Area [µm²]" in rows[0]
    od = rows[0].index(next(c.title for c in back.columns["object"] if c.uid == "object:od:mean"))
    assert rows[1][:5] == ["A", "Gland › Normal", "Gland", "Normal", "1"] and rows[1][od] == ""     # no OD: blank


def test_cancel(project):
    calls = []
    with pytest.raises(res.Cancelled):
        res.measure_project(project, progress=calls.append, cancelled=lambda: len(calls) >= 2)
    assert len(calls) == 2


def test_a_second_included_version_adds_labelled_columns(project):
    """A filtered view ticked for parameters on one image: its own columns after the plain ones; the outdated
    check sees the tick; the other image leaves them empty (said)."""
    from modalytics.processing import run
    from modalytics.processing.registry import ProcessingLibrary
    a = project.images[0]
    view = run.make_view(a, ProcessingLibrary().get("filters:gaussian_blur"), {"sigma_um": 1.0}, measure=True)
    view.name = "Blurred"
    project.add_view(a, view)
    r = res.measure_project(project)
    cols = r.columns["object"]
    uid = engine.version_uid("object:phase:mean", "Blurred")
    at = [c.uid for c in cols].index("object:phase:mean")
    assert cols[at + 1].uid == uid and cols[at + 1].label == cols[at].label + " · Blurred"   # right after its own
    on_a = [row for row in r.of_scope("object") if row.image_name == "A"]
    on_b = [row for row in r.of_scope("object") if row.image_name == "Bee"]
    assert all(not math.isnan(row.values[uid]) for row in on_a)
    assert all(math.isnan(row.values[uid]) and "no “Blurred” version" in row.problems[uid] for row in on_b)
    assert not r.outdated(project)
    project.set_measured(a, view, False)                                  # untick it in Layers
    assert any("views changed" in why for why in r.outdated(project))
    again = res.measure_project(project)
    assert uid not in {c.uid for c in again.columns["object"]}
