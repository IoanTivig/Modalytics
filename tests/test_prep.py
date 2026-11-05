"""Preparing objects for a classifier: per-project normalisation (the same when it is made and when it is
applied), augmentation from the training projects only, and the acquisition copies."""
import numpy as np
import pytest

from modalytics.training import classifiers as cl
from modalytics.training import prep as pp
from modalytics.training.dataset import Dataset


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    project = np.repeat(np.arange(8), n // 8)
    grade = rng.integers(0, 2, n)
    shift = rng.normal(0, 3, 8)[project]                      # what a whole project shares (acquisition)
    X = np.column_stack([grade * 1.0 + shift + rng.normal(0, 0.5, n), 5 + shift + rng.normal(0, 1, n)])
    return Dataset([("a", "A", "", ""), ("b", "B", "", "")], X, grade, ["L", "H"], project,
                   [f"P{i}" for i in range(8)], [("", "", i, (0.0, 0.0)) for i in range(n)])


def test_normalisation_takes_out_what_a_project_shares():
    data = _data()
    rows = np.ones(len(data.grade), bool)
    raw = cl.fit_logistic(data, rows, ["a"], "G")
    norm = pp.normalized(data, "difference")
    made = cl.fit_logistic(norm, rows, ["a"], "G")
    acc = lambda c, d: cl.evaluate(c, d, rows, intervals=False).accuracy        # noqa: E731
    assert acc(made, norm) > acc(raw, data) + 0.1                 # the projects' shift no longer hides the grade
    for p in range(8):                                            # each project's median is 0 now
        assert abs(np.median(norm.values[norm.project == p, 1])) < 1e-9
    r = pp.normalized(data, "ratio").values
    z = pp.normalized(data, "robust_z").values
    assert np.isfinite(r).all() and np.isfinite(z).all()


def test_the_same_normalisation_when_a_project_is_classified():
    data = _data()
    norm = pp.normalized(data, "robust_z")
    clf = cl.fit_logistic(norm, np.ones(len(data.grade), bool), ["a", "b"], "G")
    clf.prep = pp.Prep(normalize="robust_z").to_dict()
    back = cl.Classifier.from_dict(clf.to_dict())
    assert back.normalize == "robust_z" and "normalised per project" in cl.describe(back)
    mine = data.project == 3
    objects = [{"a": v[0], "b": v[1]} for v in data.values[mine]]
    prepared, why = cl.prepare_objects(back, objects)
    assert not why and np.allclose([[o["a"], o["b"]] for o in prepared], norm.values[mine])
    few, why = cl.prepare_objects(back, objects[:pp.MIN_REFERENCE - 1])
    assert "too few" in why
    # the reference: complete objects only (an incomplete one is normalised, but does not move it)
    complete = [True] * len(objects)
    odd = [dict(objects[0], a=1e6)] + objects[1:]
    complete[0] = False
    a, _w = cl.prepare_objects(back, odd, complete)
    b, _w = cl.prepare_objects(back, objects[1:])
    assert np.allclose([o["a"] for o in a[1:]], [o["a"] for o in b])


def test_augmentation_comes_from_the_training_rows_only():
    data = _data()
    train = data.project < 6
    prep = pp.Prep(noise_copies=2, noise_sd=0.1, smote=True)
    aug, rows = pp.augmented(data, train, prep)
    added = rows.copy()
    added[:len(data.grade)] = False
    assert rows[:len(data.grade)].tolist() == train.tolist()      # the real training rows, and only those
    assert set(aug.project[added].tolist()) <= set(range(6))       # never an evaluation project's
    n0, n1 = np.bincount(data.grade[train])
    assert int(added.sum()) == 2 * int(train.sum()) + abs(int(n0) - int(n1))   # two noisy copies + SMOTE's top-up
    # acquisition copies: only those of the projects trained on
    acquired = pp.normalized(_data(seed=1), "none")
    aug, rows = pp.augmented(data, data.project != 2, pp.Prep(acquisition={"copies": 1}), acquired)
    assert 2 not in set(aug.project[len(data.grade):].tolist())


def test_smote_balances_the_classes():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (60, 3))
    grade = np.array([0] * 45 + [1] * 15)
    v, g, s = pp.smote(values, grade, rng)
    assert len(g) == 30 and (g == 1).all() and (grade[s] == 1).all()
    lo, hi = values[grade == 1].min(axis=0), values[grade == 1].max(axis=0)
    assert ((v >= lo - 1e-9) & (v <= hi + 1e-9)).all()            # between real objects of the class


def test_cross_validation_never_augments_from_the_project_it_judges():
    from modalytics.training import validation as va
    data = _data()
    train = np.ones(len(data.grade), bool)
    acquired = _data(seed=2)
    acquired.grade = 1 - acquired.grade                          # copies that lie: they would spoil a fold they reach
    clf = cl.fit_logistic(data, train, ["a"], "G")
    clf.prep = pp.Prep(acquisition={"copies": 1}).to_dict()
    seen = []
    real = pp.augmented

    def spy(d, rows, prep, acq=None):
        out = real(d, rows, prep, acq)
        seen.append((set(d.project[rows].tolist()), set(out[0].project[len(d.grade):].tolist())))
        return out
    pp.augmented = spy
    try:
        va.cross_validate(clf, data, train, acquired=acquired)
    finally:
        pp.augmented = real
    assert len(seen) == 8 and all(added <= trained for trained, added in seen)


def test_a_project_with_no_complete_object_is_left_empty_not_a_crash():
    data = _data()
    data.partial_rows = data.project == 0                       # every object of project 0 incomplete
    norm = pp.normalized(data, "ratio")
    assert np.isnan(norm.values[data.project == 0]).all() and np.isfinite(norm.values[data.project == 1]).all()


def test_projects_can_weigh_equally_or_by_the_root_of_their_objects():
    data = _data()
    rows = (data.project != 0) | (np.arange(len(data.grade)) % 30 == 0)    # project 0: one object only
    for how, total in (("equal", lambda n: 1.0), ("sqrt", lambda n: np.sqrt(n))):
        w = pp.project_weights(data, rows, how)
        assert abs(w.mean() - 1) < 1e-9
        per = {p: (w[data.project[rows] == p].sum(), int((data.project[rows] == p).sum()))
               for p in np.unique(data.project[rows])}
        ratios = [s / total(n) for s, n in per.values()]
        assert np.allclose(ratios, ratios[0])                    # each project's total as asked
    assert pp.project_weights(data, rows, "none") is None
    # applied when fitted, and made again the same way in cross-validation
    from modalytics.training import validation as va
    plain = cl.fit_logistic(data, rows, ["a", "b"], "G")
    equal = cl.fit_logistic(data, rows, ["a", "b"], "G", weigh="equal")
    assert not np.allclose(plain.weights, equal.weights)
    equal.prep = pp.Prep(project_weights="equal").to_dict()
    again = va.refit(equal, data, rows)
    assert np.allclose(again.weights, equal.weights) and "projects weighted" in cl.describe(again)
