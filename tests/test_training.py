"""Training sets: graded objects gathered from projects (older class names included), the evaluation split by
whole projects, and the range finder's AUC and cut-offs, with known answers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_projects import graded_project  # noqa: E402

from modalytics.profiles.library import ProfileLibrary  # noqa: E402
from modalytics.training import dataset as ds  # noqa: E402
from modalytics.training import ranges  # noqa: E402

PARAMS = {"object:any:area": {}, "object:any:best_alignment_metric": {}, "object:any:solidity": {},
          "object:phase:std": {}}


@pytest.fixture
def tset(tmp_path):
    profile = ProfileLibrary().get("colonic_tissue")
    profile.parameters = dict(PARAMS)
    projects = [ds.ProjectEntry(graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=6, n_high=4,
                                               old_names=(i % 2 == 1)))
                for i in range(6)]
    projects[5].role = "evaluation"
    return ds.TrainingSet("Test set", profile, "colonic_tissue", "Gland", projects)


def test_gather_reads_grades_and_older_names(tset):
    assert tset.grades == ["Low grade dysplasia", "High grade dysplasia"]
    data = ds.gather(tset)
    assert data.values.shape == (60, 4) and data.counts() == {"Low grade dysplasia": 36, "High grade dysplasia": 24}
    assert data.per_project()["p1"] == {"Low grade dysplasia": 6, "High grade dysplasia": 4}   # "Normal" → Low
    assert not data.problems and not data.ungraded
    again = ds.gather(tset)                                     # from the cache: the same table
    assert np.array_equal(np.nan_to_num(again.values), np.nan_to_num(data.values))


def test_evaluation_split_is_by_whole_projects(tset):
    data = ds.gather(tset)
    held = ds.split(tset, data)
    assert set(data.project[held]) == {5} and held.sum() == 10            # the project marked Evaluation
    tset.mode, tset.share = "share", 0.34
    chosen = ds.evaluation_projects(tset, data)
    assert len(chosen) == 2 and chosen == ds.evaluation_projects(tset, data)   # 2 of 6 projects, repeatable
    held = ds.split(tset, data)
    for g in range(2):                                          # every grade on both sides
        assert np.any(held & (data.grade == g)) and np.any(~held & (data.grade == g))
    assert all(np.all(held[data.project == p]) or not np.any(held[data.project == p]) for p in range(6))


def test_auc_and_youden_known_answers():
    assert ranges.auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert ranges.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0
    assert ranges.auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5          # ties count half
    assert ranges.youden(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), "higher") == (3.5, 1.0, 1.0)
    cut, sens, spec = ranges.youden(np.array([5.0, 6, 7]), np.array([1.0, 2, 8]), "lower")
    assert cut == 3.5 and sens == pytest.approx(2 / 3) and spec == 1.0


def test_range_finder_on_known_differences(tset):
    data = ds.gather(tset)
    seps = ranges.separations(data, ~ds.split(tset, data))
    by = {s.uid: s for s in seps}
    bam, grain = by["object:any:best_alignment_metric"], by["object:phase:std"]
    assert bam.auc > 0.9 and bam.direction == "higher" and bam.n_lower + bam.n_higher == 50   # evaluation left out
    assert grain.auc > 0.9 and grain.direction == "higher"
    assert by["object:any:solidity"].direction == "lower"                # irregular glands are less solid
    assert seps[0].auc >= seps[-1].auc and "≥" in bam.rule and bam.rule.endswith("High grade dysplasia")
    assert by["object:any:area"].auc < bam.auc


def test_training_set_file_round_trip(tset, tmp_path):
    path = tmp_path / "set.mtrain"
    tset.save(path)
    back = ds.TrainingSet.load(path)
    assert back.name == tset.name and back.group == "Gland" and back.profile.parameters == tset.profile.parameters
    assert [(p.path, p.role) for p in back.projects] == [(p.path, p.role) for p in tset.projects]
    assert back.profile.label_for("Gland", "Normal") == ("Gland", "Low grade dysplasia")


def test_test_projects_are_never_trained_on_nor_drawn_for_evaluation(tset):
    tset.projects[0].role = tset.projects[1].role = "test"
    data = ds.gather(tset)
    assert 0 not in set(data.project) and 1 not in set(data.project)       # classifiers never see them
    assert data.values.shape[0] == 40
    assert ds.test_projects(tset) == {0, 1}
    assert ds.training_projects(tset, data) == {2, 3, 4}                   # not Evaluation (5), not Test
    tset.mode, tset.share = "share", 0.9                                    # a share too large: one must train
    chosen = ds.evaluation_projects(tset, data)
    assert chosen and not chosen & {0, 1} and len(chosen) <= 3 and ds.training_projects(tset, data)


def test_named_test_groups_are_measured_apart_and_kept(tset, tmp_path):
    from modalytics.training import classifiers as cl
    from modalytics.training import metrics as mt
    tset.projects[0].role = tset.projects[1].role = tset.projects[2].role = "test"
    tset.projects[1].group = tset.projects[2].group = "Hospital B"            # project 0: the plain "Test" group
    tset.choose_by = "evaluation"
    assert ds.test_groups(tset) == {"Test": {0}, "Hospital B": {1, 2}}
    # 1. Kept in the file
    tset.save(tmp_path / "set.mtrain")
    back = ds.TrainingSet.load(tmp_path / "set.mtrain")
    assert [(p.role, p.group) for p in back.projects[:3]] == [("test", ""), ("test", "Hospital B"),
                                                              ("test", "Hospital B")]
    assert ds.test_groups(back) == ds.test_groups(tset) and back.choose_by == "evaluation"
    # 2. Measured apart: the test projects only, never with the others
    data, tdata = ds.gather(tset), ds.gather(tset, test=True)
    assert set(data.project.tolist()) == {3, 4, 5} and set(tdata.project.tolist()) == {0, 1, 2}
    # 3. Each group a final check of its own; chosen on the evaluation project, the evaluation is not one
    train, held = ds.training_rows(tset, data), ds.split(tset, data)
    built = {"logistic": cl.fit_logistic(data, train, [c[0] for c in data.columns], "Gland")}
    finals = {g: (tdata, np.isin(tdata.project, sorted(idx))) for g, idx in ds.test_groups(tset).items()}
    a = mt.assess(data, built, train, held, choose_by="evaluation", finals=finals, n_boot=50)
    assert a["judged_on"] == "evaluation" and list(a["finals"]) == ["Test", "Hospital B"]
    assert a["finals"]["Test"]["logistic"]["n"] == 10 and a["finals"]["Hospital B"]["logistic"]["n"] == 20
    # by cross-validation (not run here: it falls back, and says so)
    b = mt.assess(data, built, train, held, finals=finals, n_boot=50)
    assert b["judged_on"] == "evaluation" and "not cross-validated" in b["fallback"]


def test_acquisition_copies_are_the_objects_measured_on_changed_pixels(tset):
    from modalytics.training import acquisition as aq
    data = ds.gather(tset)
    j = {c[0]: i for i, c in enumerate(data.columns)}
    # 1. A copy that changes nothing is the measurement itself
    same = aq.measure(tset, {0, 1}, aq.Acquisition(copies=1, pixel_size=0, gain=0, offset=0, noise=0))
    mine = np.isin(data.project, [0, 1])
    assert np.allclose(np.nan_to_num(same.values), np.nan_to_num(data.values[mine]))
    assert same.variant.tolist() == [1] * int(mine.sum())
    # 2. The calibration off: every area × s², shapes the same; the pixels changed: the phase spread too
    acq = aq.Acquisition(copies=2, pixel_size=0.1, gain=0.2, offset=0.3, noise=0.1)
    got = aq.measure(tset, {0}, acq)
    first = (got.variant == 1)
    s = acq.draw(tset.projects[0].path, 0)["scale"]
    ratio = got.values[first, j["object:any:area"]] / data.values[data.project == 0, j["object:any:area"]]
    assert np.allclose(ratio, s * s, rtol=1e-6)
    assert np.allclose(got.values[first, j["object:any:solidity"]],
                       data.values[data.project == 0, j["object:any:solidity"]], atol=1e-6)
    assert not np.allclose(got.values[first, j["object:phase:std"]], data.values[data.project == 0,
                                                                                 j["object:phase:std"]])
    assert set(got.variant.tolist()) == {1, 2} and set(got.project.tolist()) == {0}


def test_project_overview_reads_the_masks_without_measuring(tset, tmp_path):
    from modalytics.io.project_file import load_project, save_project
    ov = ds.project_overview(tset.projects[1].path, tset.profile, "Gland")    # older names: read as the grades
    assert ov.images == {"phase": 1} and ov.outlined == 1 and ov.fully == 0 and ov.objects == 10
    assert ov.grades == {"Low grade dysplasia": 6, "High grade dysplasia": 4} and not ov.problem
    path = tset.projects[2].path
    project = load_project(path)
    project.images[0].pixel_size_um = (5.0, 5.0)                             # a wrong pixel size, 10× the others
    save_project(project, path)
    overviews = {p.path: ds.project_overview(p.path, tset.profile, "Gland") for p in tset.projects}
    assert overviews[path].um_per_px == {"phase": [5.0]}                      # read again: the file changed
    assert list(ds.resolution_outliers(overviews)) == [path]
    assert ds.project_overview(str(tmp_path / "gone.modalytics"), tset.profile, "Gland").problem.startswith("not found")
