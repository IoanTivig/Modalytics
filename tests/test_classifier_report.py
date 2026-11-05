"""Logistic regression and gradient boosting (stored as numbers, as scikit-learn predicts), the standard measures
with known answers, two classifiers compared, the recommendation (readable when as good) and the report."""
import math

import numpy as np
import pytest

from modalytics.training import classifiers as cl
from modalytics.training import metrics as mt
from modalytics.training.dataset import Dataset


def _data(k=2, n=300, seed=0, ordered=True):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, k, n)
    X = np.column_stack([g * 1.2 + rng.normal(0, 1, n), rng.normal(0, 1, n), g * 0.6 + rng.normal(0, 1, n)])
    proj = rng.integers(0, 10, n)
    return Dataset([(u, u.upper(), "", "") for u in "abc"], X, g, [f"G{i}" for i in range(k)], proj,
                   [f"P{i}" for i in range(10)], ordered=ordered)


@pytest.mark.parametrize("k", [2, 3])
def test_logistic_and_boosting_predict_as_scikit_learn_and_survive_saving(k):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    data = _data(k)
    rows = np.ones(len(data.grade), bool)
    X = data.values
    for clf in (cl.fit_logistic(data, rows, list("abc"), "G"), cl.fit_boosting(data, rows, list("abc"), "G")):
        again = cl.Classifier.from_dict(clf.to_dict())                     # as kept in the profile
        if clf.kind == "logistic":
            z = (X - X.mean(0)) / X.std(0)
            ref = LogisticRegression(class_weight="balanced", max_iter=5000).fit(z, data.grade).predict_proba(z)
        else:
            ref = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                                             init="zero", random_state=1).fit(X, data.grade).predict_proba(X)
        assert np.abs(again.probabilities(X) - ref).max() < 1e-6, clf.kind
        grade, score, why = again.classify({"a": 3.0, "b": 0.0, "c": 2.0})
        assert grade == f"G{k - 1}" and 0 < score <= 1 and "P(G0)" in why
        assert again.learned and sorted(again.parameters) == ["a", "b", "c"] and cl.describe(again)


def test_measures_with_known_answers():
    true = np.array([0] * 50 + [1] * 30)
    pred = true.copy()
    pred[:5], pred[50:58] = 1, 0                             # 45 / 50 low and 22 / 30 high right
    m = mt.measures(true, pred, ["L", "H"], True, n_boot=200)
    assert m["accuracy"] == pytest.approx(67 / 80) and m["balanced_accuracy"] == pytest.approx((0.9 + 22 / 30) / 2)
    low, high = m["per_class"]
    assert low["sensitivity"] == pytest.approx(0.9) and high["ppv"] == pytest.approx(22 / 27)
    assert high["npv"] == pytest.approx(45 / 53) and high["f1"] == pytest.approx(2 * 22 / (2 * 22 + 5 + 8))
    assert m["mcc"] == pytest.approx((45 * 22 - 5 * 8) / math.sqrt(50 * 30 * 53 * 27))
    assert m["balanced_accuracy_ci"][0] < m["balanced_accuracy"] < m["balanced_accuracy_ci"][1]
    perfect = mt.measures(true, true, ["L", "H"], True, np.column_stack([1 - true, true]).astype(float), n_boot=0)
    assert perfect["kappa"] == 1.0 and perfect["mcc"] == 1.0 and perfect["auc"] == 1.0


def test_compare_mcnemar_and_the_difference():
    true = np.array([0, 1] * 50)
    a = true.copy()
    b = true.copy()
    b[:20] = 1 - b[:20]                                       # b wrong on 20 that a grades right
    c = mt.compare(true, a, b, 2, n_boot=300)
    assert c["only_a"] == 20 and c["only_b"] == 0 and c["p"] < 0.001 and c["difference_ci"][0] > 0
    same = mt.compare(true, a, a, 2, n_boot=100)
    assert same["p"] == 1.0 and same["difference"] == 0.0


def test_a_readable_classifier_as_good_as_the_best_is_recommended(tmp_path):
    from modalytics.report import classifiers as rep
    data = _data(2, n=400, seed=3)
    train = data.project < 7
    held = ~train
    built = {"logistic": cl.fit_logistic(data, train, ["a", "c"], "G"),
             "forest": cl.fit_forest(data, train, ["a", "b", "c"], "G", trees=100),
             "boosting": cl.fit_boosting(data, train, ["a", "b", "c"], "G", stages=60)}
    a = mt.assess(data, built, train, held, n_boot=200)
    assert a["judged_on"] == "evaluation" and set(a["evaluation"]) == set(built)
    assert a["recommended"] == "logistic" or a["best"] == "logistic", a["reason"]      # the simple truth here
    assert "balanced accuracy" in a["reason"]
    for kind, m in a["evaluation"].items():
        assert 0.6 < m["balanced_accuracy"] <= 1 and len(m["confusion"]) == 2 and m["n"] == int(held.sum())

    class Set:                                                 # what the report reads of a training set
        name, group = "Set", "Gland"

        class profile:
            name = "Colonic"
    page = rep.build(a, built, Set, data, train, held)
    for part in ("classifiers compared", "Recommended:", "Balanced accuracy", "MCC", "ROC AUC", "Per class",
                 "Confusion matrices", "The classifiers", "Logistic regression", "Gradient boosting", "Methods"):
        assert part in page, part


def test_categories_unweighted_kappa_and_mean_auc():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 2, 1, 1, 2, 0])
    graded = mt.measures(true, pred, ["N", "L", "H"], True, n_boot=0)["kappa"]
    plain = mt.measures(true, pred, ["A", "B", "C"], False, n_boot=0)["kappa"]
    assert plain == pytest.approx(0.5) and graded < plain
    scores = np.eye(3)[pred]
    assert 0.5 < mt.measures(true, pred, ["A", "B", "C"], False, scores, n_boot=0)["auc"] < 1


def test_intervals_resample_whole_projects_when_there_are_enough():
    # 1. 12 projects of 20 objects each; a project's objects are all right or all wrong together (8 of 12 right)
    true = np.tile([0, 1], 120)
    projects = np.repeat(np.arange(12), 20)
    pred = np.where(projects < 8, true, 1 - true)
    by_projects = mt.measures(true, pred, ["L", "H"], True, n_boot=500, projects=projects)
    by_objects = mt.measures(true, pred, ["L", "H"], True, n_boot=500)
    assert by_projects["resampled"] == "projects" and by_projects["projects"] == 12
    assert by_objects["resampled"] == "objects"
    width = lambda ci: ci[1] - ci[0]                                     # noqa: E731
    # the objects are not independent: the interval by projects is far wider than Wilson's
    assert width(by_projects["accuracy_ci"]) > 2.5 * width(by_objects["accuracy_ci"])
    assert width(by_projects["balanced_accuracy_ci"]) > 2.5 * width(by_objects["balanced_accuracy_ci"])
    assert width(by_projects["per_class"][1]["sensitivity_ci"]) > 2 * width(by_objects["per_class"][1]["sensitivity_ci"])
    # 2. The Build table's agreement does the same; with fewer than MIN_PROJECTS it falls back to Wilson
    a = cl.agreement(true, pred, ["L", "H"], projects=projects)
    assert a.resampled == "projects" and width(a.accuracy_ci) > 2.5 * width(cl.wilson(int(np.sum(true == pred)), 240))
    few = cl.agreement(true, pred, ["L", "H"], projects=projects % 3)
    assert few.resampled == "objects" and few.projects == 3 and few.accuracy_ci == cl.wilson(160, 240)
    # 3. Two classifiers compared by projects: the verdict rests on the interval alone
    c = mt.compare(true, true, pred, 2, n_boot=500, projects=projects)
    assert c["resampled"] == "projects" and c["p"] < 0.001                # McNemar alone would call it clear
    assert mt.alike(c) == (c["difference_ci"][0] <= 0)


def test_cross_validation_keeps_scores_for_the_auc():
    from modalytics.training import validation as va
    data = _data(2, n=300, seed=5)
    rows = np.ones(len(data.grade), bool)
    clf = cl.fit_logistic(data, rows, ["a", "c"], "G")
    cv = va.cross_validate(clf, data, rows)
    assert cv.scores is not None and np.isfinite(cv.scores[cv.rows]).all()
    assert cv.agreement.resampled == "projects" and cv.agreement.projects == 10
    a = mt.assess(data, {"logistic": clf}, rows, ~rows, {"logistic": cv}, n_boot=200)
    m = a["cv"]["logistic"]
    assert a["judged_on"] == "cross-validation" and 0.6 < m["auc"] <= 1 and np.isfinite(m["auc_ci"]).all()
    assert m["resampled"] == "projects"


def test_chosen_by_cross_validation_the_evaluation_projects_become_a_final_check():
    from modalytics.report import classifiers as rep
    from modalytics.training import validation as va
    data = _data(2, n=400, seed=7)
    train, held = data.project < 8, data.project >= 8
    built = {"logistic": cl.fit_logistic(data, train, ["a", "c"], "G"),
             "forest": cl.fit_forest(data, train, ["a", "b", "c"], "G", trees=60)}
    cvs = {k: va.cross_validate(c, data, train) for k, c in built.items()}
    finals = {"Setup 20×": (data, data.project == 9)}                       # a test group (here: data reused)
    a = mt.assess(data, built, train, held, cvs, n_boot=100, finals=finals)
    assert a["judged_on"] == "cross-validation" and not a["fallback"]
    assert list(a["finals"]) == [mt.EVALUATION, "Setup 20×"]
    assert a["finals"][mt.EVALUATION]["logistic"]["n"] == int(held.sum())
    assert a["against"] and all(c["resampled"] == "projects" for c in a["against"].values())   # 8 projects

    class Set:
        name, group = "Set", "Gland"

        class profile:
            name = "Colonic"
    page = rep.build(a, built, Set, data, train, held, finals)
    for part in ("Final checks", "Cross-validation (the choice)", mt.EVALUATION, "Setup 20×", "test group Setup 20×",
                 "Chosen by cross-validation"):
        assert part in page, part


def test_and_cut_offs_found_together_balance_the_rule():
    from modalytics.training import ranges as rg
    rng = np.random.default_rng(4)
    lo = rng.normal(0, 1, (300, 3))
    hi = rng.normal(0.9, 1, (300, 3))                       # three conditions, each separating moderately
    alone = [rg.youden(lo[:, j], hi[:, j], "higher")[0] for j in range(3)]
    ops = [">="] * 3
    j0 = lambda cuts: (np.all(hi >= cuts, axis=1).mean() + 1 - np.all(lo >= cuts, axis=1).mean() - 1)  # noqa: E731
    cuts, sens, spec = rg.joint_youden(lo, hi, ops, alone, "all")
    assert j0(np.array(cuts)) >= j0(np.array(alone)) + 0.05          # clearly better than each one's own best
    assert abs(sens + spec - 1 - j0(np.array(cuts))) < 1e-9
    assert all(c <= a + 1e-9 for c, a in zip(cuts, alone))            # with AND, the cut-offs loosen


def test_ranges_made_again_in_cross_validation_find_their_cut_offs_together():
    data = _data(2, n=400, seed=6)
    rows = np.ones(len(data.grade), bool)
    clf = cl.Classifier("ranges", "G", list(data.grades),
                        rules=[cl.Rule("G1", [cl.Condition("a", ">=", 0.6, "A"), cl.Condition("c", ">=", 0.3, "C")])])
    joint = cl.fit_ranges_jointly(clf, data, rows)
    from modalytics.training import validation as va
    again = va.refit(clf, data, rows)
    assert [c.value for c in again.rules[0].conditions] == [c.value for c in joint.rules[0].conditions]
    acc = lambda c: cl.evaluate(c, data, rows, intervals=False)          # noqa: E731
    bal = lambda a: (a.sensitivity[0] + a.sensitivity[1]) / 2            # noqa: E731
    from modalytics.training import ranges as rg
    alone = cl.Classifier.from_dict(clf.to_dict())
    col = {c[0]: i for i, c in enumerate(data.columns)}
    for c in alone.rules[0].conditions:
        v = data.values[:, col[c.uid]]
        c.value = rg.youden(v[data.grade == 0], v[data.grade == 1], "higher")[0]
    assert bal(acc(joint)) >= bal(acc(alone))


def test_the_report_warns_of_objects_left_out_and_of_many_selections_tried():
    from modalytics.report import classifiers as rep
    data = _data(2, n=300, seed=8)
    data.partial_rows = np.arange(300) % 3 == 0                    # a third incomplete …
    data.values[data.partial_rows, 2] = np.nan                     # … without "c" (a parameter of the whole object)
    train, held = data.project < 7, data.project >= 7
    built = {"logistic": cl.fit_logistic(data, train, ["a", "c"], "G")}
    a = mt.assess(data, built, train, held, n_boot=50)

    class Set:
        name, group = "Set", "Gland"

        class profile:
            name = "Colonic"
    page = rep.build(a, built, Set, data, train, held, None, tried=6)
    assert "objects: the others have an incomplete shape" in page and "6 different parameter selections" in page
    assert "different parameter selections" not in rep.build(a, built, Set, data, train, held, None, tried=1)


def test_score_cut_offs_found_together_with_the_points():
    rng = np.random.default_rng(5)
    n = 400
    g = rng.integers(0, 2, n)
    X = np.column_stack([g * 0.9 + rng.normal(0, 1, n), g * 0.9 + rng.normal(0, 1, n), g * 0.6 + rng.normal(0, 1, n)])
    data = Dataset([(u, u.upper(), "", "") for u in "abc"], X, g, ["L", "H"], rng.integers(0, 8, n),
                   [f"P{i}" for i in range(8)])
    rows = np.ones(n, bool)
    alone = cl.fit_score(data, rows, ["a", "b", "c"], "G", joint=False)
    together = cl.fit_score(data, rows, ["a", "b", "c"], "G")
    bal = lambda c: np.mean(cl.evaluate(c, data, rows, intervals=False).sensitivity)        # noqa: E731
    assert bal(together) >= bal(alone) and together.kind == "score" and together.thresholds
    assert any(abs(x.condition.value - y.condition.value) > 1e-12
               for x in together.criteria for y in alone.criteria if x.condition.uid == y.condition.uid) or \
        bal(together) == bal(alone)
