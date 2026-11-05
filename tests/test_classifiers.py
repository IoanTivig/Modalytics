"""Classifiers from parameters: ranges, the fitted points score, agreement and applying them to results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modalytics.training import classifiers as cl
from modalytics.training.dataset import Dataset

GRADES = ["Low grade dysplasia", "High grade dysplasia"]


def _data(n=60, seed=0) -> Dataset:
    """Two grades: "irregularity" separates them well, "graininess" a little, "noise" not at all."""
    rng = np.random.default_rng(seed)
    grade = np.repeat([0, 1], n // 2)
    irregular = np.where(grade == 1, rng.normal(3.0, 0.6, n), rng.normal(1.0, 0.6, n))
    grainy = np.where(grade == 1, rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n))
    round_ = np.where(grade == 1, rng.normal(0.6, 0.1, n), rng.normal(0.9, 0.1, n))     # lower in high grade
    noise = rng.normal(0, 1, n)
    cols = [("irr", "Irregularity", "", ""), ("grain", "Graininess", "", ""), ("round", "Roundness", "", ""),
            ("noise", "Noise", "", "")]
    return Dataset(cols, np.column_stack([irregular, grainy, round_, noise]), grade, list(GRADES),
                   np.arange(n) % 6, [f"p{i}" for i in range(6)])


def test_condition_and_ranges():
    c = cl.Condition("irr", ">=", 2.0, "Irregularity")
    assert c.holds(2.0) and not c.holds(1.9) and c.holds(math.nan) is None
    clf = cl.Classifier("ranges", "Gland", GRADES, rules=[cl.Rule(GRADES[1], [
        c, cl.Condition("round", "<=", 0.75, "Roundness")], "all")])
    assert clf.classify({"irr": 3, "round": 0.6})[0] == GRADES[1]
    assert clf.classify({"irr": 3, "round": 0.9})[0] == GRADES[0]            # AND: one is not enough
    clf.rules[0].combine = "any"
    grade, score, why = clf.classify({"irr": 3, "round": 0.9})
    assert grade == GRADES[1] and score == 1 and "Irregularity ≥ 2" in why
    grade, _s, why = clf.classify({"round": 0.9})
    assert grade == GRADES[0] and "missing: Irregularity" in why             # a missing value never counts
    again = cl.Classifier.from_dict(clf.to_dict())
    assert again.to_dict() == clf.to_dict() and again.parameters == ["irr", "round"]


def test_fit_score_points_and_thresholds():
    data = _data()
    rows = np.ones(len(data.grade), bool)
    clf = cl.fit_score(data, rows, ["irr", "round", "noise"], "Gland")
    points = {c.condition.uid: c.points for c in clf.criteria}
    # 1. the strong parameters get points, in the right direction; whole points, at least 1
    assert points["irr"] >= 1 and points["round"] >= 1
    assert all(isinstance(p, int) and 1 <= p <= 5 for p in points.values())
    assert next(c for c in clf.criteria if c.condition.uid == "round").condition.op == "<="
    # 2. one threshold for two grades; the score separates the training objects well
    assert len(clf.thresholds) == 1
    a = cl.evaluate(clf, data, rows)
    assert a.accuracy > 0.9 and a.kappa > 0.8
    grade, total, why = clf.classify({"irr": 4.0, "round": 0.5, "noise": 0.0})
    assert grade == GRADES[1] and "points" in why and total >= clf.thresholds[0]
    assert cl.Classifier.from_dict(clf.to_dict()).to_dict() == clf.to_dict()
    with pytest.raises(ValueError):
        cl.fit_score(data, rows, [], "Gland")


def test_agreement_numbers():
    true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    a = cl.agreement(true, pred, GRADES)
    assert a.confusion.tolist() == [[3, 1], [1, 3]]
    assert a.accuracy == pytest.approx(0.75)
    assert a.sensitivity[1] == pytest.approx(0.75) and a.specificity[1] == pytest.approx(0.75)
    assert a.kappa == pytest.approx(0.5)                     # po 0.75, pe 0.5 → (0.75 − 0.5) / 0.5
    perfect = cl.agreement(true, true, GRADES)
    assert perfect.kappa == pytest.approx(1.0)
    # three grades: quadratic weights — one grade off costs less than two
    t3 = np.array([0, 1, 2, 0, 1, 2])
    near = cl.agreement(t3, np.array([1, 1, 2, 0, 1, 1]), ["a", "b", "c"])
    far = cl.agreement(t3, np.array([2, 1, 2, 0, 1, 0]), ["a", "b", "c"])
    assert near.kappa > far.kappa


def test_classify_results_keeps_annotation():
    from modalytics.parameters.results import Column, Results, Row, classify
    cols = {"object": [Column("irr", "Irregularity", "", "")], "class": [], "image": []}
    rows = [Row("object", "i1", "img", "Gland", GRADES[0], "m1", 0, {"irr": 3.0}),
            Row("object", "i1", "img", "Gland", GRADES[0], "m1", 1, {"irr": 1.0}),
            Row("object", "i1", "img", "Lamina propria", "", "m2", 0, {"irr": 3.0})]
    res = Results(cols, rows, 0.0, "Colonic tissue")
    clf = cl.Classifier("ranges", "Gland", GRADES, rules=[cl.Rule(GRADES[1], [cl.Condition("irr", ">=", 2.0)])],
                        note="pilot")
    out, problems = classify(res, clf)
    pred = [r.prediction.get("grade") for r in out.rows]
    assert pred == [GRADES[1], GRADES[0], None] and not problems
    assert [r.subgroup for r in out.rows] == [GRADES[0], GRADES[0], ""]      # the annotation is untouched
    assert res.rows[0].prediction == {}                                       # the input is not changed
    again = Results.from_dict(out.to_dict())
    assert again.rows[0].prediction["grade"] == GRADES[1] and again.classified == {"Gland": "ranges: pilot"}
    _out, problems = classify(res, cl.Classifier("ranges", "Gland", GRADES, rules=[
        cl.Rule(GRADES[1], [cl.Condition("absent", ">=", 1.0)])]))
    assert problems and "absent" in problems[0]


def test_forest_matches_scikit_learn_and_round_trips():
    from sklearn.ensemble import RandomForestClassifier
    data = _data(n=80, seed=3)
    data.values[5, 1] = np.nan                                   # a missing value
    rows = np.arange(len(data.grade)) % 4 != 0                   # a quarter held out
    clf = cl.fit_forest(data, rows, ["irr", "grain", "round", "noise"], "Gland", trees=50, seed=7)
    # 1. our tree walk gives scikit-learn's probabilities exactly (same seed, same median filling)
    X = data.values[rows].copy()
    med = np.nanmedian(X, axis=0)
    X[~np.isfinite(X)] = np.broadcast_to(med, X.shape)[~np.isfinite(X)]
    ref = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, oob_score=True, class_weight="balanced",
                                 random_state=7, n_jobs=-1).fit(X, data.grade[rows])
    held = data.values[~rows]
    np.testing.assert_allclose(clf.probabilities(held), ref.predict_proba(np.where(np.isfinite(held), held, med)),
                               atol=1e-6)
    # 2. it separates the grades; the strong parameter matters most, noise least
    a = cl.evaluate(clf, data, ~rows)
    assert a.accuracy > 0.85 and 0.5 < clf.info["oob_accuracy"] <= 1
    imp = dict(zip([u for u, _l in clf.features], clf.importance))
    assert imp["irr"] > imp["noise"] and abs(sum(clf.importance) - 1) < 1e-6
    # 3. one object: grade, confidence, the reason; a missing value uses the median
    grade, conf, why = clf.classify({"irr": 4.0, "grain": 1.0, "round": 0.5})
    assert grade == GRADES[1] and 0.5 < conf <= 1 and "P(High grade dysplasia)" in why and "missing" in why
    # 4. stored as numbers in the profile and read back identically
    again = cl.Classifier.from_dict(clf.to_dict())
    np.testing.assert_allclose(again.probabilities(held), clf.probabilities(held))
    assert again.parameters == ["irr", "grain", "round", "noise"]
    with pytest.raises(ValueError):
        cl.fit_forest(data, rows, [], "Gland")


def test_wilson_interval_known_values():
    # Newcombe (1998), example 81 / 263: 0.2553–0.3662; 0 / 10 → 0–0.2775
    assert cl.wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert cl.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert all(math.isnan(x) for x in cl.wilson(0, 0))
    a = cl.agreement(np.array([0] * 10 + [1] * 10), np.array([0] * 10 + [1] * 10), GRADES)
    assert a.accuracy_ci == pytest.approx((0.8389, 1.0), abs=1e-4)          # 100 % of 20 is not certainty
    assert a.sensitivity_ci[1] == pytest.approx(cl.wilson(10, 10))


def test_cross_validation_by_project_and_mistakes():
    from modalytics.training import validation as va
    data = _data(n=120, seed=5)                                   # 6 projects
    train = data.project != 5                                     # project 5 kept for evaluation
    for clf in (cl.fit_score(data, train, ["irr", "round"], "Gland"),
                cl.fit_forest(data, train, ["irr", "grain", "round"], "Gland", trees=40),
                cl.Classifier("ranges", "Gland", GRADES, rules=[cl.Rule(GRADES[1], [
                    cl.Condition("irr", ">=", 99.0, "Irregularity")])])):        # a cut-off typed by hand
        cv = va.cross_validate(clf, data, train)
        # every training object graded once, by a classifier that did not see its project; none of project 5
        assert cv.rows.sum() == train.sum() and not cv.rows[data.project == 5].any()
        assert len(cv.folds) == 5 and cv.agreement.n == train.sum() and cv.agreement.accuracy > 0.85
        lo, mean, hi = cv.spread()
        assert lo <= mean <= hi
    # the hand-typed cut-off is found again in each fold: cross-validation judges the recipe
    assert cl.evaluate(clf, data, train).accuracy < 0.6
    found = va.details(clf, data, train)
    wrong = va.mistakes(data, train, found)
    assert len(wrong) == sum(found[r][0] != GRADES[data.grade[r]] for r in found)
    assert all(hand == GRADES[1] and got[0] == GRADES[0] for _r, hand, got in wrong)
    with pytest.raises(ValueError):
        va.cross_validate(clf, data, data.project == 0)


def test_three_grades():
    from modalytics.training import ranges as rg
    from modalytics.training import validation as va
    rng = np.random.default_rng(2)
    n = 90
    grade = np.repeat([0, 1, 2], n // 3)
    x = grade * 2.0 + rng.normal(0, 0.5, n)                      # rises with the grade
    y = rng.normal(0, 1, n)
    three = ["Normal", "Low grade dysplasia", "High grade dysplasia"]
    data = Dataset([("x", "X", "", ""), ("y", "Y", "", "")], np.column_stack([x, y]), grade, three,
                   np.arange(n) % 5, [f"p{i}" for i in range(5)])
    rows = data.project != 4
    seps = [s for s in rg.separations(data, rows) if s.uid == "x"]
    assert [(s.lower, s.higher) for s in seps] == [("Normal", "Low grade dysplasia"),
                                                   ("Low grade dysplasia", "High grade dysplasia")]
    rules = [cl.Rule(s.higher, [cl.Condition("x", ">=", s.cutoff, "X")]) for s in seps]
    ranges_clf = cl.Classifier("ranges", "Gland", three, rules=rules)
    score = cl.fit_score(data, rows, ["x"], "Gland")
    forest = cl.fit_forest(data, rows, ["x", "y"], "Gland", trees=60)
    assert len(score.thresholds) == 2 and score.thresholds[0] <= score.thresholds[1]
    for clf in (ranges_clf, forest):
        a = cl.evaluate(clf, data, ~rows)
        assert a.confusion.shape == (3, 3) and a.accuracy > 0.8 and a.kappa > 0.8   # quadratic-weighted
        assert va.cross_validate(clf, data, rows).agreement.accuracy > 0.8
    assert ranges_clf.classify({"x": 5.0})[0] == "High grade dysplasia"
    assert ranges_clf.classify({"x": 2.0})[0] == "Low grade dysplasia"
    assert forest.probabilities(np.array([[5.0, 0.0]])).shape == (1, 3)
