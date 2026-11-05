"""
The range finder: how well each parameter separates two grades, and where to cut (Qt-free).

For each parameter and each pair of neighbouring grades (with two grades, the one
pair), on the **training** objects only:

* **AUC** — the area under the ROC curve: the probability that an object of the
  higher grade has a value beyond that of an object of the lower grade (Mann–Whitney
  U / (n₁·n₂)). 0.5 = the parameter does not separate them; 1 = it separates them
  perfectly. It is given as ≥ 0.5 with the direction: "higher" when the higher
  grade has the larger values, "lower" when it has the smaller ones.
* **Cut-off** — the value that best separates the two grades by Youden's index
  J = sensitivity + specificity − 1 (maximised over all cut-offs), with the
  **sensitivity** (share of the higher grade called higher) and **specificity**
  (share of the lower grade called lower) at that cut-off.

The evaluation projects are left out, so they can judge the ranges you choose
independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass
class Separation:
    """How one parameter separates two grades (see the module docstring)."""

    uid: str
    label: str
    unit: str
    help: str
    lower: str                 # the lower grade's name
    higher: str                # the higher grade's name
    auc: float                 # ≥ 0.5 (NaN if a grade has no values)
    direction: str             # "higher" or "lower": the higher grade's values are larger / smaller
    cutoff: float
    sensitivity: float
    specificity: float
    n_lower: int
    n_higher: int

    @property
    def rule(self) -> str:
        """The cut-off as a rule, e.g. "≥ 0.031 → High grade dysplasia"."""
        if not math.isfinite(self.cutoff):
            return "—"
        sign = "≥" if self.direction == "higher" else "≤"
        return f"{sign} {self.cutoff:.4g} → {self.higher}"


def auc(lower: np.ndarray, higher: np.ndarray) -> float:
    """P(a value of ``higher`` > a value of ``lower``), ties counting half (Mann–Whitney U / (n₁·n₂))."""
    n1, n2 = len(lower), len(higher)
    if n1 == 0 or n2 == 0:
        return math.nan
    ranks = rankdata(np.concatenate([lower, higher]))
    u = ranks[n1:].sum() - n2 * (n2 + 1) / 2.0
    return float(u / (n1 * n2))


def youden(lower: np.ndarray, higher: np.ndarray, direction: str) -> tuple[float, float, float]:
    """(cut-off, sensitivity, specificity) maximising sensitivity + specificity − 1.

    "higher": values ≥ cut-off are called the higher grade; "lower": values ≤ cut-off are.
    The cut-off is placed halfway between neighbouring distinct values.
    """
    if len(lower) == 0 or len(higher) == 0:
        return math.nan, math.nan, math.nan
    sign = 1.0 if direction == "higher" else -1.0
    lo, hi = sign * np.asarray(lower, float), sign * np.asarray(higher, float)
    values = np.unique(np.concatenate([lo, hi]))
    if len(values) == 1:
        return float(sign * values[0]), 1.0, 0.0
    cuts = np.concatenate([[values[0]], (values[:-1] + values[1:]) / 2.0])
    lo_sorted, hi_sorted = np.sort(lo), np.sort(hi)
    # Called higher when value ≥ cut: sensitivity = share of higher ≥ cut, specificity = share of lower < cut
    sens = 1.0 - np.searchsorted(hi_sorted, cuts, side="left") / len(hi)
    spec = np.searchsorted(lo_sorted, cuts, side="left") / len(lo)
    j = sens + spec - 1.0
    best = int(np.argmax(j))
    return float(sign * cuts[best]), float(sens[best]), float(spec[best])


def joint_youden(lower: np.ndarray, higher: np.ndarray, ops: list[str], start: list[float], combine: str = "all",
                 rounds: int = 10, candidates: int = 200) -> tuple[list[float], float, float]:
    """Cut-offs of conditions combined into one rule, found **together**: the rule's sensitivity + specificity − 1
    made as large as can be, one cut-off at a time (each over its parameter's values, the others kept), from
    ``start`` until nothing improves (coordinate ascent).

    Each cut-off found alone is the best for its parameter alone; joined with AND every condition must hold, so
    the rule calls far fewer objects the higher grade than any of them (with OR far more) — found together, they
    loosen (or tighten) to balance the rule.

    Args:
        lower, higher: (objects, conditions) the values of each condition's parameter for the lower and the higher
            grade (NaN: missing — the condition does not hold, as when classifying).
        ops: Per condition ">=" or "<=".
        start: Per condition its cut-off to start from (e.g. each one's own best).
        combine: "all" (AND) or "any" (OR).

    Returns:
        (cut-offs, the rule's sensitivity, its specificity).
    """
    lower, higher = np.atleast_2d(np.asarray(lower, float)), np.atleast_2d(np.asarray(higher, float))
    m = len(ops)
    cuts = [float(c) for c in start]
    if not len(lower) or not len(higher) or m == 0:
        return cuts, math.nan, math.nan

    def holds(values, j, c):
        with np.errstate(invalid="ignore"):
            return (values[:, j] >= c) if ops[j] == ">=" else (values[:, j] <= c)

    def score(hl, hh):
        ok_lo = hl.all(axis=1) if combine == "all" else hl.any(axis=1)
        ok_hi = hh.all(axis=1) if combine == "all" else hh.any(axis=1)
        return ok_hi.mean() + (1.0 - ok_lo.mean()) - 1.0, ok_hi.mean(), 1.0 - ok_lo.mean()

    hl = np.column_stack([holds(lower, j, cuts[j]) for j in range(m)])
    hh = np.column_stack([holds(higher, j, cuts[j]) for j in range(m)])
    best, sens, spec = score(hl, hh)
    for _r in range(rounds):
        moved = False
        for j in range(m):
            v = np.unique(np.concatenate([lower[:, j], higher[:, j]]))
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            options = (v[:-1] + v[1:]) / 2.0
            if len(options) > candidates:
                options = np.quantile(options, np.linspace(0, 1, candidates))
            for c in options:
                hl2, hh2 = hl.copy(), hh.copy()
                hl2[:, j], hh2[:, j] = holds(lower, j, c), holds(higher, j, c)
                s, se, sp = score(hl2, hh2)
                if s > best + 1e-12:
                    best, sens, spec, cuts[j], hl, hh, moved = s, se, sp, float(c), hl2, hh2, True
        if not moved:
            break
    return cuts, float(sens), float(spec)


def separations(data, train: np.ndarray | None = None) -> list[Separation]:
    """Every parameter × neighbouring-grade pair, best-separating first.

    Args:
        data: A ``training.dataset.Dataset``.
        train: Rows to use (default: all); pass ``~split(...)`` to leave the evaluation projects out.
    """
    out = []
    for k in range(len(data.grades) - 1):
        out += between(data, train, [k], [k + 1], data.grades[k], data.grades[k + 1])
    out.sort(key=lambda s: (-(s.auc if math.isfinite(s.auc) else 0.0), s.label))
    return out


def between(data, train: np.ndarray | None, lower: list[int], higher: list[int], lower_name: str,
            higher_name: str) -> list[Separation]:
    """Every parameter separating two sets of classes (two classes, or one against the rest), best first.

    Args:
        data: A ``training.dataset.Dataset``.
        train: Rows to use (default: all).
        lower, higher: The classes (indices) on each side; ``higher`` is the "positive" one.
        lower_name, higher_name: How the two sides are named (e.g. "Low grade", or "the rest").
    """
    rows = np.ones(len(data.grade), bool) if train is None else np.asarray(train, bool)
    lo_rows = rows & np.isin(data.grade, lower)
    hi_rows = rows & np.isin(data.grade, higher)
    out = []
    for j, (uid, label, unit, help_) in enumerate(data.columns):
        lo = data.values[lo_rows, j]
        hi = data.values[hi_rows, j]
        lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
        a = auc(lo, hi)
        direction = "higher" if not math.isfinite(a) or a >= 0.5 else "lower"
        a = a if not math.isfinite(a) or a >= 0.5 else 1.0 - a
        cut, sens, spec = youden(lo, hi, direction)
        out.append(Separation(uid, label, unit, help_, lower_name, higher_name, a, direction, cut, sens, spec,
                              len(lo), len(hi)))
    out.sort(key=lambda s: (-(s.auc if math.isfinite(s.auc) else 0.0), s.label))
    return out
