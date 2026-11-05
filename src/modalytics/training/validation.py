"""
Checking a classifier beyond one split: cross-validation by project, and the objects it gets wrong (Qt-free).

**Cross-validation by project.** With few projects, one training / evaluation
split gives an unsteady figure: a different choice of evaluation projects could
give quite another. Cross-validation uses every training project once for
checking:

1. The training projects are dealt into k folds (k = the number of projects, at
   most 10: with up to 10 projects, each project is its own fold —
   leave-one-project-out). Whole projects only, as in the evaluation split: objects
   of one project (one sample) resemble each other.
2. For each fold, the classifier is made again **the same way** on the other folds
   and grades the fold's objects: Ranges — the same parameters, directions and
   AND / OR, with cut-offs found again together (``ranges.joint_youden``); Score — the points fitted
   again to the same parameters; Trained classifier — grown again with the same
   parameters and settings. (Values typed by hand are not kept: they were chosen
   looking at all the training projects.) Its data augmentation (``training.prep``)
   is made again from the other folds' projects only — never from the fold judged.
3. Every training object is so graded once by a classifier that did not see its
   project; these grades are compared with the hand grades (``classifiers.agreement``,
   with 95 % intervals resampling whole projects), and each fold's accuracy shows how
   much it varies. Each object's out-of-fold score (``metrics.scores_of``: a learned
   classifier's probabilities) is kept too, for the ROC AUC.

The evaluation projects are not used, so they stay an independent last check.

**Mistakes.** ``mistakes`` lists the objects a classifier grades differently from
the hand grade, with the predicted grade, its score or confidence and the reason —
to see whether the errors are borderline objects, poor outlines or annotations to
look at again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from modalytics.training import classifiers as cl
from modalytics.training.metrics import scores_of

MAX_FOLDS = 10


class Cancelled(Exception):
    """Raised when a cross-validation is cancelled."""


@dataclass
class CrossValidation:
    """The outcome of a cross-validation (see the module docstring).

    Attributes:
        kind: The classifier kind it checked.
        signature: The classifier it checked (its dict), to tell when it is out of date.
        folds: [(project names, objects, accuracy)] per fold.
        rows: (objects,) True for the objects graded (the training objects of the folds that could be made).
        predicted: (objects,) the out-of-fold grade index (−1 where not graded).
        details: {row: (grade, score, reason)} for the graded objects.
        agreement: Out-of-fold grades against the hand grades.
        skipped: Folds that could not be made, and why.
        scores: (objects, grades) the out-of-fold scores (NaN where not graded), for the ROC AUC.
    """

    kind: str
    signature: dict
    folds: list
    rows: np.ndarray
    predicted: np.ndarray
    details: dict
    agreement: cl.Agreement
    skipped: list = field(default_factory=list)
    scores: np.ndarray | None = None

    def spread(self) -> tuple[float, float, float]:
        """(lowest, mean, highest) accuracy over the folds."""
        acc = [a for _p, n, a in self.folds if n and math.isfinite(a)]
        return (min(acc), float(np.mean(acc)), max(acc)) if acc else (math.nan, math.nan, math.nan)


def refit(clf: cl.Classifier, data, rows: np.ndarray, uids: list[str] | None = None,
          seed: int = 1, acquired=None) -> cl.Classifier:
    """The classifier made again the same way on other ``rows`` (see the module docstring, step 2).

    Args:
        clf: The classifier.
        data: The ``training.dataset.Dataset`` (normalised as the classifier's ``prep`` says).
        rows: The objects to make it on.
        uids: Score: the parameters ticked (default: those with points).
        seed: Trained classifier: the seed.
        acquired: The training projects measured again (``training.acquisition``, normalised the same way), for
            its acquisition augmentation: only ``rows``' projects are used.

    Raises:
        ValueError: If a grade has no objects in ``rows``.
    """
    from modalytics.training import prep as pp
    prep = pp.Prep.from_dict(clf.prep)
    made = _refit(clf, *pp.augmented(data, rows, prep, acquired), uids, seed, prep.project_weights)
    made.prep = dict(clf.prep)
    return made


def _refit(clf: cl.Classifier, data, rows: np.ndarray, uids: list[str] | None, seed: int,
           weigh: str = "none") -> cl.Classifier:
    g = data.grade[rows]
    missing = [name for k, name in enumerate(data.grades) if not np.any(g == k)]
    if missing:
        raise ValueError(f"no objects of {', '.join(missing)}")
    if clf.kind == "score":
        return cl.fit_score(data, rows, uids or clf.parameters, clf.group, weigh=weigh)
    info = clf.info
    if clf.kind == "logistic":
        return cl.fit_logistic(data, rows, [u for u, _l in clf.features], clf.group,
                               strength=info.get("strength", 1.0), balanced=info.get("balanced", True), weigh=weigh)
    if clf.kind == "boosting":
        return cl.fit_boosting(data, rows, [u for u, _l in clf.features], clf.group,
                               stages=info.get("stages", 200), depth=info.get("depth", 3),
                               learning_rate=info.get("learning_rate", 0.1), seed=seed, weigh=weigh)
    if clf.kind == "forest":
        return cl.fit_forest(data, rows, [u for u, _l in clf.features], clf.group, trees=info.get("trees", 300),
                             min_leaf=info.get("min_leaf", 2), balanced=info.get("balanced", True), seed=seed,
                             weigh=weigh)
    # Ranges: the same conditions, their cut-offs found again — together, for each rule — between the grade and
    # the one below
    return cl.fit_ranges_jointly(clf, data, rows)


def details(clf: cl.Classifier, data, rows: np.ndarray) -> dict:
    """{row: (grade, score, reason)} of a classifier for ``rows`` of a dataset."""
    col = {c[0]: i for i, c in enumerate(data.columns)}
    out = {}
    for r in np.flatnonzero(rows):
        out[int(r)] = clf.classify({uid: data.values[r, j] for uid, j in col.items()})
    return out


def cross_validate(clf: cl.Classifier, data, train: np.ndarray, uids: list[str] | None = None, seed: int = 1,
                   progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] | None = None, acquired=None) -> CrossValidation:
    """Cross-validate a classifier by project on the training objects (see the module docstring).

    Args:
        clf: The classifier (made on all of ``train``).
        data: The ``training.dataset.Dataset``.
        train: The training objects (the evaluation projects left out).
        uids: Score: the parameters ticked.
        seed: Deals the projects into folds (when there are more than 10) and seeds the forests.
        acquired: For the acquisition augmentation (``refit``).

    Raises:
        ValueError: With fewer than two training projects.
        Cancelled: If ``cancelled`` returned True.
    """
    # 1. Projects into folds
    projects = sorted(set(data.project[train].tolist()))
    if len(projects) < 2:
        raise ValueError("cross-validation needs at least two training projects")
    k = min(MAX_FOLDS, len(projects))
    order = list(np.random.default_rng(seed).permutation(projects)) if len(projects) > k else projects
    folds = [order[i::k] for i in range(k)]
    # 2. Each fold graded by the classifier made on the others
    predicted = np.full(len(data.grade), -1, int)
    scores = np.full((len(data.grade), len(data.grades)), np.nan)
    found, per_fold, skipped = {}, [], []
    names = data.projects
    for f, fold in enumerate(folds):
        if cancelled is not None and cancelled():
            raise Cancelled()
        test = cl.usable(data, train & np.isin(data.project, fold), uids or clf.parameters)
        label = ", ".join(names[p] for p in fold)
        try:
            made = refit(clf, data, train & ~np.isin(data.project, fold), uids, seed, acquired)   # the whole fold out
        except ValueError as exc:
            skipped.append(f"{label}: the other projects have {exc}")
            continue
        got = details(made, data, test)
        for r, (grade, score, why) in got.items():
            predicted[r] = data.grades.index(grade)
            found[r] = (grade, score, why)
        if test.any():
            scores[test] = scores_of(made, data, test)
        acc = float(np.mean(predicted[test] == data.grade[test])) if test.any() else math.nan
        per_fold.append((label, int(test.sum()), acc))
        if progress is not None:
            progress((f + 1) / len(folds))
    # 3. Out-of-fold grades against the hand grades
    rows = predicted >= 0
    return CrossValidation(clf.kind, clf.to_dict(), per_fold, rows, predicted, found,
                           cl.agreement(data.grade[rows], predicted[rows], data.grades, data.ordered,
                                        data.project[rows]), skipped, scores)


def mistakes(data, rows: np.ndarray, found: dict) -> list[tuple]:
    """[(row, hand grade, (grade, score, reason))] of the objects in ``rows`` graded differently by hand."""
    out = []
    for r in np.flatnonzero(rows):
        got = found.get(int(r))
        if got is not None and got[0] != data.grades[data.grade[r]]:
            out.append((int(r), data.grades[data.grade[r]], got))
    return out
