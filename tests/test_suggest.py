"""Model-assisted annotation: suggestions from a model's outlines, classified; accepting and deleting them."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, box

from modalytics.core.commands import UndoStack
from modalytics.core.model import MaskNode, Project, Suggestion, SuggestionNode
from modalytics.services import suggest as sg

LOW, HIGH = "Low grade dysplasia", "High grade dysplasia"


def _project(classify: bool = True):
    """One phase map with two graded glands and one gland without a grade; the profile grades by area."""
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.profiles.library import ProfileLibrary
    import numpy as np
    prof = ProfileLibrary().get("colonic_tissue")
    prof.classification.pop("Gland", None)          # the built-in profile's own rule is replaced (or removed)
    if classify:
        from modalytics.training.classifiers import Classifier, Condition, Rule
        prof.classification["Gland"] = Classifier("ranges", "Gland", [LOW, HIGH], rules=[
            Rule(HIGH, [Condition("object:any:area", ">=", 1000.0, "Area")])]).to_dict()
    image = ImageNode("map", ArraySource(np.zeros((400, 400), np.float32)), (0.5, 0.5), "phase")
    project = Project()
    project.reset([image], profile=prof)
    project.add_mask(image, MaskNode(geometry=[Point(30, 30).buffer(10).exterior.coords], group="Gland", subgroup=LOW))
    project.add_mask(image, MaskNode(geometry=[Point(150, 30).buffer(10).exterior.coords], group="Gland",
                                     subgroup=""))
    return project, image


def test_make_skips_drawn_objects_suggests_new_ones_and_grades():
    project, image = _project()
    found = [Point(30, 30).buffer(10.5),          # the graded drawn gland: nothing suggested
             Point(150, 30).buffer(10),           # drawn but ungraded: its grade is suggested
             Point(100, 120).buffer(12),          # new, small → low grade
             Point(60, 160).buffer(25)]           # new, large (≈ 1963 µm²) → high grade
    items = sg.make(found, image, project.images, project.profile, "Gland")
    assert len(items) == 3
    regrade = [s for s in items if s.regrade]
    assert len(regrade) == 1 and regrade[0].geometry.equals(next(
        p for m in image.masks if m.subgroup == "" for p in m.geometry.geoms))
    new = {round(s.geometry.area / 100): s for s in items if not s.regrade}
    assert {s.subgroup for s in new.values()} == {LOW, HIGH}
    big = max(new.values(), key=lambda s: s.geometry.area)
    assert big.subgroup == HIGH and "Area" in big.reason and big.score == 1
    ys = [s.geometry.representative_point().y for s in items]
    assert ys == sorted(ys) or len({round(y / 50) for y in ys}) < len(ys)    # reading order
    # without a classifier: outlines only, no grade, and no regrade suggestion is possible
    project2, image2 = _project(classify=False)
    plain = sg.make(found, image2, project2.images, project2.profile, "Gland")
    assert len(plain) == 2 and all(s.subgroup == "" and math.isnan(s.score) for s in plain)


def test_accept_reject_undo():
    project, image = _project()
    undo = UndoStack()
    node = project.suggestions_for(image)
    near = Suggestion(box(35, 20, 60, 40), "Gland", HIGH, 1.0, "big")          # overlaps the drawn low-grade gland
    other = Suggestion(Point(250, 250).buffer(10), "Gland", LOW, 0.0, "small")
    regrade = Suggestion(next(p for m in image.masks if m.subgroup == "" for p in m.geometry.geoms), "Gland", HIGH,
                         regrade=True)
    project.set_suggestions(node, [near, other, regrade])
    low_before = project.mask_for(image, "Gland", LOW, create=False).geometry
    # 1. Accept as suggested: a new High grade mask; the drawing wins where they overlap (a separate object)
    undo.push(sg.accept_command(project, node, [near]))
    high = project.mask_for(image, "Gland", HIGH, create=False)
    low = project.mask_for(image, "Gland", LOW, create=False)
    assert high is not None and len(high.geometry.geoms) == 1 and low.geometry.equals(low_before)
    assert high.geometry.intersection(low.geometry).area == 0 and high.geometry.area < near.geometry.area
    assert [s.id for s in node.items] == [other.id, regrade.id]
    # 2. Accept with another grade
    undo.push(sg.accept_command(project, node, [other], grade=HIGH))
    assert len(high.geometry.geoms) == 2 and len(low.geometry.geoms) == 1
    # 3. A regrade: the ungraded drawn gland moves to High grade unchanged
    ungraded = project.mask_for(image, "Gland", "", create=False)
    undo.push(sg.accept_command(project, node, [regrade]))
    assert ungraded.geometry.is_empty and len(high.geometry.geoms) == 3 and node.items == []
    # 4. Undo all three: as before; a delete is undoable too
    for _ in range(3):
        undo.undo()
    assert project.mask_for(image, "Gland", HIGH, create=False) is None
    assert len(node.items) == 3 and len(ungraded.geometry.geoms) == 1
    undo.push(sg.reject_command(project, node, [other]))
    assert [s.id for s in node.items] == [near.id, regrade.id]
    undo.undo()
    assert len(node.items) == 3


def test_neighbour_and_counts():
    node = SuggestionNode([Suggestion(box(i, 0, i + 1, 1), "Gland", HIGH if i else LOW) for i in range(3)])
    assert sg.neighbour(node).id == node.items[0].id
    node.current = node.items[2].id
    assert sg.neighbour(node).id == node.items[0].id and sg.neighbour(node, -1).id == node.items[1].id
    assert sg.counts(node) == {LOW: 1, HIGH: 2}


def test_suggestions_are_not_masks_and_are_saved(tmp_path):
    from modalytics.io.project_file import load_project, save_project
    project, image = _project()
    node = project.suggestions_for(image)
    project.set_suggestions(node, [Suggestion(box(200, 200, 230, 230), "Gland", HIGH, 0.93, "why")],
                            {"model": "m1", "classifier": "ranges"})
    assert node not in image.masks and image.suggestions is node                  # never measured as annotation
    project.set_suggestions_shown(node, "Gland", False)                           # its group hidden in Layers
    project.set_measured(image, None, False)                                       # the original left out
    assert node.groups() == [("Gland", 1)] and node.hidden_groups == ["Gland"]
    save_project(project, str(tmp_path / "p"))
    data = load_project(str(tmp_path / "p.modalytics"))
    again = data.images[0].suggestions
    assert again.hidden_groups == ["Gland"] and not data.images[0].original_measured
    assert again is not None and len(again.items) == 1 and again.items[0].subgroup == HIGH
    assert again.items[0].score == pytest.approx(0.93) and again.source["model"] == "m1"
    assert again.items[0].geometry.equals(box(200, 200, 230, 230))


def test_measure_annotation_only_unless_suggestions_are_included():
    from modalytics.parameters.results import measure_project
    project, image = _project()
    project.profile.parameters = {"object:any:area": {}}
    node = project.suggestions_for(image)
    project.set_suggestions(node, [Suggestion(Point(160, 160).buffer(20), "Gland", HIGH, 1.0, "big"),
                                   Suggestion(next(p for m in image.masks if m.subgroup == "" for p in m.geometry.geoms),
                                              "Gland", HIGH, regrade=True)])
    plain = measure_project(project)
    assert len(plain.of_scope("object")) == 2 and plain.suggested == 0       # the two drawn glands only
    both = measure_project(project, include_suggestions=True)
    objs = both.of_scope("object")
    assert len(objs) == 3 and both.suggested == 1                            # + the new one (not the regrade)
    sug = next(r for r in objs if r.suggested)
    assert sug.subgroup == HIGH and sug.values["object:any:area"] == pytest.approx(Point(0, 0).buffer(20).area)
    assert image.masks == [m for m in image.children if isinstance(m, MaskNode)] and len(image.masks) == 2
    from modalytics.parameters.results import Results, fixed_values
    assert "(suggested)" in fixed_values(sug)[1] and Results.from_dict(both.to_dict()).suggested == 1


def _logistic_on_area(project, threshold=None):
    """The profile's classifier: a logistic regression on area (P(high) = ½ at 1000 µm², steep)."""
    from modalytics.training.classifiers import Classifier
    project.profile.classification["Gland"] = Classifier(
        "logistic", "Gland", [LOW, HIGH], features=[("object:any:area", "Area")], medians=[1000.0], importance=[1.0],
        weights=[[3.0]], intercepts=[0.0], means=[1000.0], scales=[200.0]).to_dict()
    if threshold is not None:
        project.profile.min_confidence["Gland"] = threshold


def test_a_grade_is_suggested_from_the_minimum_confidence_else_the_plain_group():
    project, image = _project()
    _logistic_on_area(project)
    found = [Point(100, 120).buffer(12),          # ≈ 452 µm²: surely low
             Point(60, 160).buffer(25),           # ≈ 1963 µm²: surely high
             Point(250, 250).buffer(17.84)]       # ≈ 1000 µm²: a coin toss
    items = {round(s.geometry.area): s for s in sg.make(found, image, project.images, project.profile, "Gland")}
    small, middle, big = (items[k] for k in sorted(items))      # by area
    assert small.subgroup == LOW and big.subgroup == HIGH and small.score >= 0.8 and big.score >= 0.8
    assert middle.subgroup == "" and "unsure" in middle.reason and 0.5 <= middle.score < 0.8
    # stricter: only the surest keep their grade; off: none graded, and no drawn object regraded
    project.profile.min_confidence["Gland"] = 0.999
    strict = sg.make(found, image, project.images, project.profile, "Gland")
    assert sum(1 for s in strict if s.subgroup) < 3
    off = sg.make(found + [Point(150, 30).buffer(10)], image, project.images, project.profile, "Gland",
                  classify_on=False)
    assert all(s.subgroup == "" and not s.regrade for s in off) and len(off) == 3


def test_classify_selected_suggests_a_sure_grade_that_differs_and_undoes():
    project, image = _project()
    _logistic_on_area(project)
    big_low = MaskNode(geometry=[Point(300, 300).buffer(25).exterior.coords], group="Gland", subgroup="")
    project.add_mask(image, big_low)
    middle = MaskNode(geometry=[Point(300, 100).buffer(17.84).exterior.coords], group="Gland", subgroup="")
    project.add_mask(image, middle)
    selected = [(m, p) for m in image.masks for p in m.geometry.geoms]
    made, said = sg.classify_objects(project, selected)
    grades = sorted((now, g) for _grp, now, g, _s, _w in said)
    assert (LOW, LOW) in grades                    # the small graded gland: agrees, nothing suggested
    assert ("", HIGH) in grades and ("", "") in grades
    items = made[image]
    assert len(items) == 2 and all(s.regrade for s in items)           # the ungraded small one and the big one
    assert {s.subgroup for s in items} == {LOW, HIGH}
    undo = UndoStack()
    undo.push(sg.add_command(project, image, items))
    assert image.suggestions is not None and len(image.suggestions.items) == 2
    undo.undo()
    assert image.suggestions is None or not image.suggestions.items


def test_the_minimum_confidence_is_kept_in_the_profile():
    from modalytics.profiles.model import MIN_CONFIDENCE, Profile
    project, _image = _project()
    prof = project.profile
    assert prof.confidence_for("Gland") == MIN_CONFIDENCE
    prof.min_confidence["Gland"] = 0.9
    back = Profile.from_dict(prof.to_dict())
    assert back.confidence_for("Gland") == 0.9 and back.confidence_for("Other") == MIN_CONFIDENCE
