"""
The standard measures of a classifier against the grades given by hand, and two classifiers compared (Qt-free).

``measures(true, predicted, grades, ordered, scores)`` gives, with 95 % intervals:

* **accuracy** — the share graded right (Wilson interval);
* **balanced accuracy** — the mean of the classes' sensitivities: each class counts the same
  however many objects it has, so a classifier that always says the commoner grade does not
  look good (bootstrap interval);
* **Cohen's κ** — agreement beyond chance: κ = (p_o − p_e) / (1 − p_e); quadratic-weighted for
  more than two ordered grades (a mistake two grades off counts more), plain for categories;
* **MCC** — the Matthews correlation coefficient, for any number of classes:
  (c·s − Σ p_k t_k) / √((s² − Σ p_k²)(s² − Σ t_k²)), c the objects graded right, s all, p_k
  and t_k the objects predicted as and truly of class k; −1 … 1, 0 = chance;
* per class (one against the others): **sensitivity** (recall), **specificity**, **PPV**
  (precision), **NPV**, **F1** = 2·PPV·sensitivity / (PPV + sensitivity) — proportions with
  Wilson intervals — and **macro-F1**, the mean of the classes' F1;
* **ROC AUC** — from the classifier's score (a learned one's probability; with two grades a
  Score's points; Ranges give a grade only, so their AUC is that of the yes/no decision): with
  two grades the AUC of the higher one; with more, the mean of each class's one-against-the-
  others AUC.

``compare(a, b)`` judges two classifiers on the **same** objects: **McNemar's test** on the
objects one grades right and the other wrong (exact, two-sided), and the difference in
balanced accuracy with its 95 % interval (paired bootstrap: the same objects drawn for both).

**Intervals and projects.** Objects of one project resemble each other, so they are not
independent, and intervals that treat them as independent come out too narrow. Given each
object's project, every interval (proportions included) resamples **whole projects** when there
are ``classifiers.MIN_PROJECTS`` or more (a cluster bootstrap), and the verdict of ``compare``
rests on that interval alone — McNemar's test treats the objects as independent, so it is shown
but not used. With fewer projects, proportions get Wilson's interval and the rest a bootstrap over
objects, and ``resampled`` says "objects": likely too narrow.
"""

from __future__ import annotations

import math

import numpy as np

from modalytics.training import classifiers as cl
from modalytics.training.classifiers import wilson
from modalytics.training.ranges import auc as _auc

BOOTSTRAP = 2000


def _confusion(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), int)
    np.add.at(cm, (np.asarray(true, int), np.asarray(pred, int)), 1)
    return cm


def balanced_accuracy(cm: np.ndarray) -> float:
    """The mean of the classes' sensitivities (classes with objects)."""
    rows = cm.sum(axis=1)
    known = rows > 0
    return float(np.mean(np.diag(cm)[known] / rows[known])) if known.any() else math.nan


def kappa(cm: np.ndarray, ordered: bool = True) -> float:
    """Cohen's κ (quadratic weights for more than two ordered grades)."""
    k, n = len(cm), cm.sum()
    if not n:
        return math.nan
    idx = np.arange(k)
    weights = ((idx[:, None] - idx[None, :]) ** 2) / max((k - 1) ** 2, 1) if k > 2 and ordered else \
        (idx[:, None] != idx[None, :]).astype(float)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (weights * expected).sum()
    return float(1 - (weights * cm).sum() / denom) if denom > 0 else math.nan


def mcc(cm: np.ndarray) -> float:
    """Matthews correlation coefficient, for any number of classes."""
    s, c = cm.sum(), np.trace(cm)
    p, t = cm.sum(axis=0), cm.sum(axis=1)
    den = math.sqrt(float(s * s - (p * p).sum()) * float(s * s - (t * t).sum()))
    return float((c * s - (p * t).sum()) / den) if den > 0 else math.nan


def roc_auc(true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Two grades: the AUC of the higher grade's score; more: the mean one-against-the-others AUC."""
    true = np.asarray(true, int)
    if scores is None or not len(true):
        return math.nan
    if k == 2:
        s = scores[:, 1] if scores.ndim == 2 else scores
        return _auc(s[true == 0], s[true == 1])
    out = [_auc(scores[true != c, c], scores[true == c, c]) for c in range(k) if np.any(true == c) and
           np.any(true != c)]
    out = [a for a in out if math.isfinite(a)]
    return float(np.mean(out)) if out else math.nan


def measures(true: np.ndarray, pred: np.ndarray, grades: list, ordered: bool = True, scores: np.ndarray | None = None,
             n_boot: int = BOOTSTRAP, seed: int = 0, projects: np.ndarray | None = None) -> dict:
    """Every measure in the module docstring, as plain numbers ({name: value}, intervals as (low, high)).

    Args:
        projects: (objects,) each object's project: the intervals resample whole projects when there are enough
            (see the module docstring).

    Returns:
        The measures, with "resampled" ("projects" | "objects") and "projects" (how many; 0 when not told).
    """
    true, pred = np.asarray(true, int), np.asarray(pred, int)
    k, n = len(grades), len(true)
    cm = _confusion(true, pred, k)
    whole = cl.by_projects(projects)
    out = {"n": int(n), "confusion": cm.tolist(), "resampled": "projects" if whole else "objects",
           "projects": len(np.unique(projects)) if projects is not None else 0}
    if not n:
        return out
    right = int(np.trace(cm))
    out["accuracy"], out["accuracy_ci"] = right / n, wilson(right, n)
    out["balanced_accuracy"] = balanced_accuracy(cm)
    out["kappa"], out["mcc"] = kappa(cm, ordered), mcc(cm)
    per = []
    for c in range(k):
        tp, fn = int(cm[c, c]), int(cm[c].sum() - cm[c, c])
        fp, tn = int(cm[:, c].sum() - cm[c, c]), int(n - cm[c].sum() - cm[:, c].sum() + cm[c, c])
        sens = tp / (tp + fn) if tp + fn else math.nan
        ppv = tp / (tp + fp) if tp + fp else math.nan
        per.append({"class": grades[c], "n": tp + fn, "sensitivity": sens, "sensitivity_ci": wilson(tp, tp + fn),
                    "specificity": tn / (tn + fp) if tn + fp else math.nan, "specificity_ci": wilson(tn, tn + fp),
                    "ppv": ppv, "ppv_ci": wilson(tp, tp + fp), "npv": tn / (tn + fn) if tn + fn else math.nan,
                    "npv_ci": wilson(tn, tn + fn),
                    "f1": 2 * ppv * sens / (ppv + sens) if math.isfinite(ppv) and math.isfinite(sens) and ppv + sens
                    else math.nan})
    out["per_class"] = per
    f1s = [p["f1"] for p in per if math.isfinite(p["f1"])]
    out["macro_f1"] = float(np.mean(f1s)) if f1s else math.nan
    out["auc"] = roc_auc(true, scores, k)
    # Bootstrap intervals: of every measure by whole projects; else of those that are not plain proportions
    draws = {"balanced_accuracy": [], "kappa": [], "mcc": [], "macro_f1": [], "auc": [], "accuracy": []}
    per_draws = {key: [[] for _c in range(k)] for key in ("sensitivity", "specificity", "ppv", "npv")}
    for i in cl.resamples(n, projects, n_boot, seed):
        cmb = _confusion(true[i], pred[i], k)
        draws["balanced_accuracy"].append(balanced_accuracy(cmb))
        draws["kappa"].append(kappa(cmb, ordered))
        draws["mcc"].append(mcc(cmb))
        draws["accuracy"].append(np.trace(cmb) / len(i))
        f = []
        for c in range(k):
            tp, fn, fp = cmb[c, c], cmb[c].sum() - cmb[c, c], cmb[:, c].sum() - cmb[c, c]
            tn = len(i) - tp - fn - fp
            if 2 * tp + fp + fn:
                f.append(2 * tp / (2 * tp + fp + fn))
            for key, num, den in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp), ("ppv", tp, tp + fp),
                                  ("npv", tn, tn + fn)):
                per_draws[key][c].append(num / den if den else math.nan)
        draws["macro_f1"].append(float(np.mean(f)) if f else math.nan)
        if scores is not None and (k == 2 or len(set(true[i].tolist())) > 1):
            draws["auc"].append(roc_auc(true[i], scores[i], k))
    for name, values in draws.items():
        if name != "accuracy" or whole:
            out[name + "_ci"] = cl.interval(values)
    if whole:
        for c, pc in enumerate(per):
            for key, values in per_draws.items():
                pc[key + "_ci"] = cl.interval(values[c])
    return out


def compare(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, k: int, n_boot: int = BOOTSTRAP,
            seed: int = 1, projects: np.ndarray | None = None) -> dict:
    """Two classifiers on the same objects: McNemar's exact test and the balanced accuracy difference (a − b).

    Args:
        projects: (objects,) each object's project: the paired bootstrap draws whole projects when there are enough.

    Returns:
        {"only_a": objects a grades right and b wrong, "only_b": the reverse, "p": McNemar's two-sided p,
         "difference": balanced accuracy a − b, "difference_ci": its 95 % interval (paired bootstrap),
         "resampled": "projects" | "objects"}.
    """
    from scipy.stats import binom
    true, a, b = (np.asarray(x, int) for x in (true, pred_a, pred_b))
    ra, rb = a == true, b == true
    only_a, only_b = int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))
    m = only_a + only_b
    p = float(min(1.0, 2 * binom.cdf(min(only_a, only_b), m, 0.5))) if m else 1.0
    diff = balanced_accuracy(_confusion(true, a, k)) - balanced_accuracy(_confusion(true, b, k))
    boots = [balanced_accuracy(_confusion(true[i], a[i], k)) - balanced_accuracy(_confusion(true[i], b[i], k))
             for i in cl.resamples(len(true), projects, n_boot, seed)]
    return {"only_a": only_a, "only_b": only_b, "p": p, "difference": float(diff), "difference_ci": cl.interval(boots),
            "resampled": "projects" if cl.by_projects(projects) else "objects"}


def alike(c: dict) -> bool:
    """True if a classifier is not distinguishable from the best (``compare(best, it)``): the interval of the
    difference contains 0 — and, when it resampled objects only, McNemar's p ≥ 0.05 too (with whole projects
    resampled McNemar is not used: it treats the objects as independent)."""
    lo = (c.get("difference_ci") or (math.nan, math.nan))[0]
    if not c or not math.isfinite(lo) or lo > 0:
        return False
    return c.get("resampled") == "projects" or c.get("p", 0) >= 0.05


def scores_of(classifier, data, rows: np.ndarray) -> np.ndarray:
    """(objects, grades) scores for ROC: a learned classifier's probabilities; with two grades a Score's points
    total (column 1, the higher grade's); otherwise the grade given, one-hot (a yes/no decision: Ranges, and a
    Score with more grades, whose one total does not rank each class against the others)."""
    rows = np.asarray(rows, bool)
    k = len(data.grades)
    col = {c[0]: i for i, c in enumerate(data.columns)}
    if classifier.learned:
        X = data.values[np.ix_(np.flatnonzero(rows), [col[u] for u, _l in classifier.features])]
        return classifier.probabilities(X) if len(X) else np.zeros((0, k))
    out = np.zeros((int(rows.sum()), k))
    for i, r in enumerate(np.flatnonzero(rows)):
        grade, score, _why = classifier.classify({uid: data.values[r, j] for uid, j in col.items()})
        if classifier.kind == "score" and k == 2:
            out[i] = (-score, score)
        else:
            out[i, data.grades.index(grade) if grade in data.grades else 0] = 1.0
    return out


# ---------------------------------------------------------------- the classifiers of a training set, assessed
READABLE = ("ranges", "score", "logistic")          # a rule, points or weights a person can read and write down
NAMES = {"ranges": "Ranges", "score": "Score", "logistic": "Logistic regression", "forest": "Random forest",
         "boosting": "Gradient boosting"}


EVALUATION = "Evaluation projects"         # the final check on them (chosen by cross-validation)
CHOOSE_BY = {"cross-validation": "Cross-validation (training projects)", "evaluation": "Evaluation projects"}


def assess(data, built: dict, train: np.ndarray, held: np.ndarray, cvs: dict | None = None,
           n_boot: int = BOOTSTRAP, choose_by: str = "cross-validation", finals: dict | None = None) -> dict:
    """Every classifier built, measured where it is chosen, compared with the best by balanced accuracy, the one to
    recommend, and the final checks on the projects that took no part in the choice.

    1. **The choice** — on ``choose_by``: cross-validation on the training projects (each project graded by the
       classifier made again without it), or the evaluation projects. When that is not possible (no
       cross-validation, or no evaluation objects) the other one is used, and ``fallback`` says so.
    2. **The recommendation**: the best by balanced accuracy — unless it is not readable (a random forest,
       gradient boosting) and a readable one (Ranges, Score, logistic regression) is not distinguishable from it
       on the same objects (``alike``): then the best of those readable ones, since one can see how its
       parameters decide.
    3. **Final checks** — each group of projects not used to choose, apart: the evaluation projects (when the
       choice was by cross-validation) and each test group. They are measured after the choice and never change
       it.

    Args:
        data: The measured objects (``dataset.Dataset``; the test projects not among them).
        built: {kind: ``classifiers.Classifier``}.
        train, held: (objects,) the training and the evaluation objects of ``data``.
        cvs: {kind: ``validation.CrossValidation``} (as checked now), optional.
        choose_by: "cross-validation" or "evaluation".
        finals: {test group: (its measured objects (``dataset.Dataset``), (objects,) its rows)}.

    Returns:
        {"evaluation" | "cv": {kind: measures}, "training": {kind: measures (no intervals)},
         "judged_on": "cross-validation" | "evaluation" | "", "fallback": why the other one ("" if none),
         "against": {kind: compare(best, kind)}, "best": kind, "recommended": kind, "reason": text,
         "finals": {group: {kind: measures}} (in order: the evaluation projects first, then the test groups),
         "objects": {kind: evaluation objects}}.
    """
    k, grades, ordered = len(data.grades), list(data.grades), data.ordered
    out = {"evaluation": {}, "training": {}, "cv": {}, "against": {}, "objects": {}, "finals": {}, "fallback": ""}
    preds = {}

    def measured(clf, d, rows) -> tuple[dict, np.ndarray, np.ndarray]:
        rows = cl.usable(d, rows, clf.parameters)            # partial objects only with every value it reads
        pred = cl.predict(clf, d, rows)
        m = measures(d.grade[rows], pred, grades, ordered, scores_of(clf, d, rows), n_boot=n_boot,
                     projects=d.project[rows]) if rows.any() else {"n": 0}
        return m, rows, pred

    for kind, clf in built.items():
        m, rows, pred = measured(clf, data, held)
        out["evaluation"][kind], out["objects"][kind], preds[kind] = m, int(rows.sum()), (rows, pred)
        tr = cl.usable(data, train, clf.parameters)
        out["training"][kind] = measures(data.grade[tr], cl.predict(clf, data, tr), grades, ordered, n_boot=0)
        cv = (cvs or {}).get(kind)
        if cv is not None and cv.rows.any():
            scores = cv.scores[cv.rows] if getattr(cv, "scores", None) is not None else None
            out["cv"][kind] = measures(data.grade[cv.rows], cv.predicted[cv.rows], grades, ordered, scores,
                                       n_boot=n_boot, projects=data.project[cv.rows])
    # 1. Where it is chosen: as asked, else the other one
    on_eval = {kind: m for kind, m in out["evaluation"].items() if m.get("n")}
    if choose_by == "evaluation":
        out["judged_on"] = "evaluation" if on_eval else ("cross-validation" if out["cv"] else "")
        if not on_eval and out["cv"]:
            out["fallback"] = "no evaluation objects: chosen by cross-validation instead"
    else:
        out["judged_on"] = "cross-validation" if out["cv"] else ("evaluation" if on_eval else "")
        if not out["cv"] and on_eval:
            out["fallback"] = "not cross-validated: chosen on the evaluation projects instead"
    pool = out["cv"] if out["judged_on"] == "cross-validation" else on_eval
    # 2. Final checks: the groups that took no part in the choice
    if out["judged_on"] == "cross-validation" and on_eval:
        out["finals"][EVALUATION] = out["evaluation"]
    for group, (d, rows) in (finals or {}).items():
        out["finals"][group] = {kind: measured(clf, d, rows)[0] for kind, clf in built.items()}
    if not pool:
        out.update(best="", recommended="", reason="no evaluation objects and no cross-validation: mark evaluation "
                                                    "projects (Data) or cross-validate")
        return out
    ba = lambda kind: pool[kind].get("balanced_accuracy", math.nan)       # noqa: E731
    best = max(pool, key=lambda kind: (ba(kind) if math.isfinite(ba(kind)) else -1, kind in READABLE))
    # 3. Each against the best, on the objects both graded
    for kind in pool:
        if kind == best:
            continue
        if out["judged_on"] == "evaluation":
            ra, pa = preds[best]
            rb, pb = preds[kind]
            both = ra & rb
            a = pa[np.isin(np.flatnonzero(ra), np.flatnonzero(both))]
            b = pb[np.isin(np.flatnonzero(rb), np.flatnonzero(both))]
            truth = data.grade[both]
        else:
            ca, cb = cvs[best], cvs[kind]
            both = ca.rows & cb.rows
            a, b, truth = ca.predicted[both], cb.predicted[both], data.grade[both]
        out["against"][kind] = compare(truth, a, b, k, n_boot=n_boot, projects=data.project[both])             if len(truth) else {}
    # 4. The recommendation: a readable one when it is not distinguishable from the best
    if best in READABLE:
        rec, reason = best, f"{NAMES[best]} has the best balanced accuracy, and it is readable."
    else:
        close = [kind for kind in pool if kind in READABLE and alike(out["against"].get(kind) or {})]
        if close:
            rec = max(close, key=ba)
            c = out["against"][rec]
            why = ("the interval of the difference, resampling whole projects, contains 0" if c["resampled"] ==
                   "projects" else f"the interval of the difference contains 0, McNemar p {c['p']:.2f}")
            reason = (f"{NAMES[best]} has the best balanced accuracy, but {NAMES[rec]} is not distinguishable from it "
                      f"on these objects ({why}) and one can read how its parameters decide: {NAMES[rec]} is "
                      f"recommended.")
        else:
            rec = best
            reason = (f"{NAMES[best]} has the best balanced accuracy, and no readable classifier comes close enough "
                      f"(each is clearly below it, or none was built).")
    out.update(best=best, recommended=rec, reason=reason)
    return out
