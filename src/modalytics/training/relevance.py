"""
Which parameters carry information about the classes, and which repeat each other (Qt-free).

For each object parameter of a training set's measured objects (``dataset.Dataset``), on
the training projects (or the training and evaluation ones, for a descriptive table). The
classes are the group's subgroups: **ordered grades** (a scale, e.g. low < high grade
dysplasia) or **categories** with no order (e.g. four pathologies), as the profile says.

**What is compared** (``UNITS``, the user's choice):

* **objects** (the default) — each object (gland, cell) is a case: "do the classes differ?"
  in this material. Objects of one sample may resemble each other (one patient or source,
  one slide, one acquisition), so the p-values describe this material and may be
  optimistic about new samples.
* **images** — each image's class is one case, its value the median of its objects (for
  projects that hold images of several samples).
* **projects** — each project's class is one case, the median of its objects: "will it
  hold for new samples?" — the conservative choice; it needs many projects per class.

For each parameter, on the cases:

* **The main effect** — *grades*: the **AUC across the grades**, the probability that, of
  two cases of different grades, the higher-grade one has the larger value (ties count
  half), over every pair of grades: the Jonckheere–Terpstra trend statistic as a share
  (with two grades, the Mann–Whitney AUC: U / n₁n₂); given as ≥ 0.5 with its direction.
  *Categories*: each class against all the others (**one-vs-rest AUC**); the main AUC is
  the class that stands apart most (``best``).
* **95 % interval** — bootstrap of the cases (``BOOTSTRAP`` draws with replacement), the
  2.5th and 97.5th percentiles.
* **p-value by permutation** — the classes shuffled among the cases (``BOOTSTRAP`` times).
  *Grades*: the trend test (two-sided, a normal curve with the shuffles' spread: p-values
  below 1 / draws can be told apart too, as many parameters need). *Categories*: the
  omnibus test — does any class differ? (the Kruskal–Wallis idea: the pairs of classes'
  squared departures of their AUC from 0.5, a gamma curve fitted to the shuffles). **q**
  corrects the p-values for the many parameters (Benjamini–Hochberg false discovery rate).
* **Every pair** of classes: its AUC (the probability that the second class's case has the
  larger value) and its p-value from the same shuffles, corrected for the number of pairs
  (Holm, as a post-hoc test).
* **Per class**: n, median (interquartile range), mean ± SD of the objects, and the
  Shapiro–Wilk test of normality (for information: the tests here compare ranks).
* **Consistency** — in how many projects (with two classes of ≥ ``MIN_PER_PROJECT``
  objects) the parameter goes the same way; and whether it varies more **between
  projects** within a class than **between classes** (Kruskal–Wallis ε² of each): then it
  may follow the acquisition or the sample rather than the class.
* **Redundancy** — parameters whose values follow each other (Spearman |ρ| ≥
  ``REDUNDANT_RHO``, average linkage) form a group: they measure one thing; one of each
  group is enough.
* **Verdict** — *separates* (q < ``Q_MAX`` and AUC ≥ ``SEPARATES_AUC``), *weak* (real but
  small: q < ``Q_MAX``), *promising* (AUC ≥ ``SEPARATES_AUC`` and p < ``Q_MAX`` before the
  correction, but not after it: too few cases to confirm), *chance* (not distinguishable
  from chance), or *too few* (fewer than ``MIN_OBJECTS`` objects — ``MIN_CASES`` images
  or projects — of a class). The **proposal**: the best parameter of each group that
  separates.

Choosing parameters on the objects a classifier is then judged on flatters it, so the
analysis meant for choosing uses the training projects only; "all" (the training and
evaluation projects: test projects are never measured for classifiers) describes the data.

References:
    Benjamini Y, Hochberg Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple
        Testing. J R Stat Soc B. 1995;57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

BOOTSTRAP = 2000
SEPARATES_AUC = 0.7            # an AUC from which a parameter is useful on its own
Q_MAX = 0.05                   # false discovery rate accepted
REDUNDANT_RHO = 0.8            # |Spearman ρ| from which two parameters measure the same thing
MIN_OBJECTS = 10               # objects per class (comparing objects), below: too few to tell
MIN_CASES = 5                  # images or projects per class (comparing them), below: too few to tell
MIN_PER_PROJECT = 3            # objects of a class in a project for its own direction to count
PROJECT_EFFECT = 0.1           # ε² between projects from which a parameter may follow them rather than the class
VERDICTS = {"separates": "Separates", "weak": "Weak", "promising": "Promising, not confirmed",
            "chance": "Not distinguishable from chance", "too few": "Too few cases"}
UNITS = {"objects": "objects", "images": "images", "projects": "projects",
         "within": "objects within projects"}          # (the last: analyses made before the choice existed)


class Cancelled(Exception):
    """The analysis was cancelled."""


@dataclass
class Relevance:
    """One parameter's relevance for the classes (see the module docstring).

    Attributes:
        auc: The main AUC, ≥ 0.5: grades: across them; categories: the class that stands apart most against the
            others (``best``).
        direction: "higher": larger values with the higher grade (categories: in ``best``); "lower": smaller.
        best: Categories: the class that stands apart most.
        pairs: Every pair of classes: (first, second, P(second's case has the larger value), p corrected for the
            pairs (Holm)).
        vs_rest: Categories: (class, P(its case has a larger value than another class's)).
        stats: {class: [n, median, Q1, Q3, mean, SD, Shapiro–Wilk p]} of the objects.
        n: {class: cases with a value} (objects, images or projects, as compared).
    """

    uid: str
    label: str
    unit: str = ""
    help: str = ""
    auc: float = math.nan
    direction: str = "higher"
    ci: tuple = (math.nan, math.nan)
    p: float = math.nan
    q: float = math.nan
    best: str = ""
    pairs: list = field(default_factory=list)
    vs_rest: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)
    projects: int = 0                                # projects with values
    agree: tuple = (0, 0)                            # (projects going the same way, projects with two classes)
    grade_effect: float = math.nan                   # Kruskal–Wallis ε² between classes
    project_effect: float = math.nan                 # mean ε² between projects within a class
    group: int = 0                                   # its redundancy group (1, 2…; 0: alone)
    verdict: str = "chance"
    proposed: bool = False

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label

    @property
    def follows_projects(self) -> bool:
        """It varies more between projects within a class than between classes (acquisition, sample?)."""
        return (math.isfinite(self.project_effect) and math.isfinite(self.grade_effect)
                and self.project_effect >= PROJECT_EFFECT and self.project_effect > self.grade_effect)

    def notes(self) -> list[str]:
        """What to look at before trusting it."""
        out = []
        if self.follows_projects:
            out.append(f"varies more between projects (ε² {self.project_effect:.2f}) than between classes "
                       f"(ε² {self.grade_effect:.2f})")
        a, n = self.agree
        if n >= 2 and a < n:
            out.append(f"goes the same way in {a} of {n} projects")
        return out

    def normal(self) -> str:
        """The Shapiro–Wilk result per class in a few words ("normal in every class", "not normal in High"…)."""
        bad = [c for c, s in self.stats.items() if len(s) > 6 and math.isfinite(s[6]) and s[6] < 0.05]
        known = [c for c, s in self.stats.items() if len(s) > 6 and math.isfinite(s[6])]
        if not known:
            return ""
        return "normal in every class" if not bad else f"not normal in {', '.join(bad)}"


@dataclass
class Analysis:
    """A relevance analysis of a training set's parameters (see the module docstring).

    Attributes:
        name: Its name.
        on: "training" (the training projects) or "all".
        group: The objects (the set's group).
        grades: The classes, in order.
        projects: The projects' names used.
        counts: {class: objects}.
        rows: One ``Relevance`` per parameter, the proposal first, then by verdict and AUC.
        groups: The redundancy groups: [[parameter ids]] (two or more each), the group numbers' order.
        settings: The thresholds and draws used.
        made: When (ISO time).
        training_set, profile: Their names (for the report).
        ordered: The classes are ordered grades (else categories).
        unit: What was compared (``UNITS``): "objects", "images" or "projects".
    """

    name: str
    on: str
    group: str
    grades: list
    projects: list
    counts: dict
    rows: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    made: str = ""
    training_set: str = ""
    profile: str = ""
    ordered: bool = True
    unit: str = "objects"

    def proposed(self) -> list[Relevance]:
        return [r for r in self.rows if r.proposed]

    def row(self, uid: str) -> Relevance | None:
        return next((r for r in self.rows if r.uid == uid), None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rows"] = [dict(asdict(r), ci=list(r.ci), agree=list(r.agree), pairs=[list(p) for p in r.pairs],
                          vs_rest=[list(v) for v in r.vs_rest]) for r in self.rows]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        keys = set(Relevance.__dataclass_fields__)
        rows = [Relevance(**{k: v for k, v in r.items() if k in keys}) for r in d.get("rows", [])]
        for r in rows:
            r.ci, r.agree = tuple(r.ci), tuple(r.agree)
            r.pairs, r.vs_rest = [tuple(p) for p in r.pairs], [tuple(v) for v in r.vs_rest]
        own = set(cls.__dataclass_fields__) - {"rows"}
        a = cls(**{k: v for k, v in d.items() if k in own}, rows=rows)
        if "unit" not in d and "design" in d:                # made before the choice: what it did by itself
            a.unit = "projects" if d["design"] == "between" else "within"
        return a

    def save(self, path: str | os.PathLike) -> None:
        from modalytics import fileio
        fileio.write_text(path, json.dumps(self.to_dict(), indent=1, default=float))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Analysis":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------- the measures
class _Sorted:
    """One parameter's values sorted once (ties as blocks), for many draws that only reweight cases (bootstrap) or
    relabel them (permutation)."""

    def __init__(self, values: np.ndarray, n_grades: int) -> None:
        self.order = np.argsort(values, kind="mergesort")
        _u, self.block = np.unique(values[self.order], return_inverse=True)
        self.n_blocks = int(self.block.max()) + 1 if len(self.block) else 0
        self.n_grades = n_grades

    def counts(self, grades: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(C[a, b], D[a, b]) for classes a < b: the concordant pairs (b's value the larger; ties one half) and all
        the pairs, each pair counting w_i·w_j (a case drawn twice by the bootstrap counts twice)."""
        g = grades[self.order]
        w = np.ones(len(g)) if weights is None else weights[self.order]
        k = self.n_grades
        per = np.zeros((k, self.n_blocks))
        for c in range(k):
            m = g == c
            per[c] = np.bincount(self.block[m], weights=w[m], minlength=self.n_blocks)
        below = np.cumsum(per, axis=1) - per                 # weight strictly below each block
        total = per.sum(axis=1)
        conc, den = np.zeros((k, k)), np.zeros((k, k))
        for a in range(k):
            for b in range(a + 1, k):
                conc[a, b] = float(np.sum(per[b] * (below[a] + 0.5 * per[a])))
                den[a, b] = float(total[a] * total[b])
        return conc, den


def _trend(conc: np.ndarray, den: np.ndarray) -> float:
    """The concordance over every pair of classes a < b (the Jonckheere–Terpstra statistic as a share)."""
    return float(conc.sum() / den.sum()) if den.sum() else math.nan


def _omnibus(conc: np.ndarray, den: np.ndarray) -> float:
    """Does any class differ: the pairs' squared departures of their concordance from 0.5, weighted by their
    numbers of pairs (the Kruskal–Wallis idea)."""
    m = den > 0
    if not m.any():
        return math.nan
    return float(np.sum(den[m] * (conc[m] / den[m] - 0.5) ** 2) / den[m].sum())


def concordance(values: np.ndarray, grades: np.ndarray, weights: np.ndarray, n_grades: int) -> tuple[float, list]:
    """(The share of concordant pairs over every pair of grades, [each neighbouring pair's]), weighted.

    For cases i, j of grades a < b: concordant when x_j > x_i (a tie: one half); each pair counts w_i·w_j.

    Args:
        values, grades, weights: Per case (finite values only).
        n_grades: The number of grades.
    """
    conc, den = _Sorted(values, n_grades).counts(grades, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        pairs = [float(conc[k, k + 1] / den[k, k + 1]) if den[k, k + 1] else math.nan for k in range(n_grades - 1)]
    return _trend(conc, den), pairs


def epsilon_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Kruskal–Wallis ε² = H / (n − 1): the share of the ranks' variance between the labels' groups."""
    from scipy.stats import rankdata
    keep = [k for k in np.unique(labels) if np.sum(labels == k) >= 2]
    mask = np.isin(labels, keep)
    v, lab = values[mask], labels[mask]
    n = len(v)
    if len(keep) < 2 or n < 3:
        return math.nan
    r = rankdata(v)
    h = 12.0 / (n * (n + 1)) * sum(np.sum(lab == k) * (r[lab == k].mean() - (n + 1) / 2) ** 2 for k in keep)
    ties = np.unique(v, return_counts=True)[1]
    h /= 1 - np.sum(ties ** 3 - ties) / (n ** 3 - n) if n > 1 and np.any(ties > 1) else 1.0
    return float(h / (n - 1))


def benjamini_hochberg(p: list[float]) -> list[float]:
    """q-values (Benjamini–Hochberg) of p-values (NaN kept out and returned as NaN)."""
    idx = [i for i, x in enumerate(p) if math.isfinite(x)]
    m = len(idx)
    q = [math.nan] * len(p)
    if not m:
        return q
    order = sorted(idx, key=lambda i: p[i])
    running = 1.0
    for rank in range(m, 0, -1):                             # from the largest p down: the running minimum
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def holm(p: list[float]) -> list[float]:
    """p-values corrected for several comparisons of one parameter (Holm's step-down; NaN kept as NaN)."""
    idx = sorted((i for i, x in enumerate(p) if math.isfinite(x)), key=lambda i: p[i])
    m = len(idx)
    out = [math.nan] * len(p)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, p[i] * (m - rank)))
        out[i] = running
    return out


def class_stats(values: np.ndarray) -> list[float]:
    """[n, median, Q1, Q3, mean, SD, Shapiro–Wilk p] of one class's values (NaN where it cannot be said)."""
    n = len(values)
    if not n:
        return [0] + [math.nan] * 6
    q1, med, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
    shapiro = math.nan
    if 3 <= n <= 5000 and np.ptp(values) > 0:
        from scipy.stats import shapiro as _shapiro
        shapiro = float(_shapiro(values).pvalue)
    return [n, med, q1, q3, float(np.mean(values)), sd, shapiro]


def spearman_groups(values: np.ndarray, uids: list[str], rho_min: float = REDUNDANT_RHO) -> list[list[str]]:
    """Groups of parameters that follow each other (|Spearman ρ| ≥ ``rho_min``, average linkage), two or more."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import rankdata
    k = values.shape[1]
    if k < 2:
        return []
    ranks = np.full(values.shape, np.nan)
    for j in range(k):
        ok = np.isfinite(values[:, j])
        if ok.sum() > 2:
            ranks[ok, j] = rankdata(values[ok, j])
    rho = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            ok = np.isfinite(ranks[:, a]) & np.isfinite(ranks[:, b])
            if ok.sum() > 2 and np.std(ranks[ok, a]) > 0 and np.std(ranks[ok, b]) > 0:
                rho[a, b] = rho[b, a] = abs(float(np.corrcoef(ranks[ok, a], ranks[ok, b])[0, 1]))
    dist = np.clip(1.0 - rho, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), t=1.0 - rho_min, criterion="distance")
    groups: dict[int, list[str]] = {}
    for uid, lab in zip(uids, labels):
        groups.setdefault(int(lab), []).append(uid)
    return [g for g in groups.values() if len(g) > 1]


def _normal_p(observed: float, null: np.ndarray) -> float:
    """Two-sided p of a departure from 0.5 against the shuffles' spread, as a normal curve (the statistic is close
    to normal under shuffling: a p-value below 1 / draws can be told apart too, as many parameters need)."""
    from scipy.stats import norm
    null = null[np.isfinite(null)]
    if not math.isfinite(observed) or len(null) < 2 or np.std(null) <= 0:
        return math.nan
    return float(2.0 * norm.sf(abs(observed - 0.5) / float(np.std(null))))


def _gamma_p(observed: float, null: np.ndarray) -> float:
    """One-sided p of a non-negative statistic against the shuffles, as a gamma curve fitted to their mean and
    variance (a squared departure behaves like a χ²)."""
    from scipy.stats import gamma
    null = null[np.isfinite(null)]
    if not math.isfinite(observed) or len(null) < 2:
        return math.nan
    mean, var = float(np.mean(null)), float(np.var(null))
    if mean <= 0 or var <= 0:
        return math.nan
    return float(gamma.sf(observed, a=mean * mean / var, scale=var / mean))


def case_keys(data, unit: str) -> np.ndarray:
    """(objects,) the case each object belongs to: itself ("objects"), its image ("images") or its project."""
    if unit == "objects":
        return np.arange(len(data.grade))
    if unit == "projects":
        return np.asarray(data.project, int)
    images: dict = {}
    where = data.where or [("", "", 0, (0, 0))] * len(data.grade)
    return np.array([images.setdefault((int(p), str(w[0])), len(images)) for p, w in zip(data.project, where)], int)


# ---------------------------------------------------------------- the analysis
def analyse(data, rows: np.ndarray, on: str, group: str, name: str = "", n_boot: int = BOOTSTRAP, seed: int = 0,
            progress: Callable[[float], None] | None = None, cancelled: Callable[[], bool] | None = None,
            training_set: str = "", profile: str = "", ordered: bool | None = None,
            unit: str = "objects") -> Analysis:
    """Each parameter's relevance for the classes on the chosen objects (see the module docstring).

    Args:
        data: The measured objects (``dataset.Dataset``).
        rows: (objects,) True for the objects used (the training projects', or all).
        on: "training" or "all" (said in the report).
        group: The objects' group (for the report).
        name: The analysis' name.
        n_boot: Bootstrap draws (and shuffles).
        seed: For the draws.
        progress: Called with 0–1.
        cancelled: True stops (raises ``Cancelled``).
        ordered: The classes are ordered grades; False: categories (default: as the data says).
        unit: What is compared: "objects" (each a case), "images" or "projects" (each one's class a case: the
            median of its objects).

    Returns:
        The ``Analysis``.
    """
    rows = np.asarray(rows, bool)
    ordered = getattr(data, "ordered", True) if ordered is None else bool(ordered)
    names = list(data.grades)
    k = len(names)
    keys = case_keys(data, unit)
    rng = np.random.default_rng(seed)
    out = []
    n_par = len(data.columns)
    for j, (uid, label, unit_, help_) in enumerate(data.columns):
        if cancelled is not None and cancelled():
            raise Cancelled()
        ok = rows & np.isfinite(data.values[:, j])
        x, g, proj = data.values[ok, j], data.grade[ok], data.project[ok]
        r = Relevance(uid, label, unit_, help_)
        r.stats = {names[c]: class_stats(x[g == c]) for c in range(k)}           # of the objects, whatever the unit
        r.projects = len(set(proj.tolist()))
        # 1. The cases: the objects themselves, or each image's / project's class (the median of its objects)
        if unit == "objects":
            xc, gc, kc = x, g, keys[ok]
        else:
            kk = keys[ok]
            cases = [(float(np.median(x[(kk == key) & (g == c)])), c, key) for key in sorted(set(kk.tolist()))
                     for c in range(k) if np.any((kk == key) & (g == c))]
            xc = np.array([v for v, _c, _k in cases])
            gc = np.array([c for _v, c, _k in cases], int)
            kc = np.array([key for _v, _c, key in cases], int)
        r.n = {names[c]: int(np.sum(gc == c)) for c in range(k)}
        if r.projects == 0 or sum(1 for v in r.n.values() if v) < 2:
            r.verdict = "too few"
            out.append(r)
            continue
        # 2. The main effect: the trend across the grades, or the class that stands apart most
        ranked = _Sorted(xc, k)
        conc, den = ranked.counts(gc)
        binary = None
        if ordered:
            trend = _trend(conc, den)
            r.direction = "higher" if not math.isfinite(trend) or trend >= 0.5 else "lower"
            r.auc = trend if r.direction == "higher" else 1.0 - trend
        else:
            two = _Sorted(xc, 2)
            r.vs_rest = []
            for c in range(k):
                if r.n[names[c]] and r.n[names[c]] < len(xc):
                    r.vs_rest.append((names[c], _trend(*two.counts((gc == c).astype(int)))))
            known = [(c, a) for c, a in r.vs_rest if math.isfinite(a)]
            target = max(known, key=lambda ca: abs(ca[1] - 0.5))[0] if known else names[0]
            a = dict(r.vs_rest).get(target, math.nan)
            r.best, r.direction = target, "higher" if not math.isfinite(a) or a >= 0.5 else "lower"
            r.auc = a if r.direction == "higher" else 1.0 - a
            binary = (gc == names.index(target)).astype(int)
        sign = 1.0 if r.direction == "higher" else -1.0
        # 3. Bootstrap of the cases (the main AUC's interval); shuffles of the classes among them (the p-values)
        boot, null_main, null_pairs = np.empty(n_boot), np.empty(n_boot), np.empty((n_boot, k, k))
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, len(xc), len(xc)), minlength=len(xc)).astype(float)
            cb = _trend(*ranked.counts(gc, w)) if binary is None else _trend(*two.counts(binary, w))
            boot[b] = (cb if sign > 0 else 1.0 - cb) if math.isfinite(cb) else math.nan
            cn, dn = ranked.counts(gc[rng.permutation(len(gc))])
            null_main[b] = _trend(cn, dn) if ordered else _omnibus(cn, dn)
            with np.errstate(invalid="ignore", divide="ignore"):
                null_pairs[b] = cn / dn
        boot = boot[np.isfinite(boot)]
        if len(boot):
            lo, hi = np.percentile(boot, [2.5, 97.5])
            r.ci = (float(lo), float(hi))
        r.p = _normal_p(_trend(conc, den), null_main) if ordered else _gamma_p(_omnibus(conc, den), null_main)
        # 4. Every pair: its AUC and its p (the same shuffles), corrected for the pairs (Holm)
        pair_list, raw = [], []
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                if not den[a_, b_]:
                    continue
                c_ = float(conc[a_, b_] / den[a_, b_])
                pair_list.append((names[a_], names[b_], c_))
                raw.append(_normal_p(c_, null_pairs[:, a_, b_]))
        r.pairs = [(a_, b_, auc, p_) for (a_, b_, auc), p_ in zip(pair_list, holm(raw))]
        # 5. Consistency over projects, and projects against classes (of the objects)
        if binary is not None:
            target_index = names.index(r.best)
        agree = total = 0
        for p in sorted(set(proj.tolist())):
            m = proj == p
            if ordered:
                present = [c for c in range(k) if np.sum(g[m] == c) >= MIN_PER_PROJECT]
                if len(present) < 2:
                    continue
                keep = m & np.isin(g, present)
                cp = concordance(x[keep], g[keep], np.ones(int(keep.sum())), k)[0]
            else:
                inside = (g[m] == target_index).astype(int)
                if inside.sum() < MIN_PER_PROJECT or (len(inside) - inside.sum()) < MIN_PER_PROJECT:
                    continue
                cp = concordance(x[m], inside, np.ones(int(m.sum())), 2)[0]
            if math.isfinite(cp) and cp != 0.5:
                total += 1
                agree += int((cp > 0.5) == (sign > 0))
        r.agree = (agree, total)
        r.grade_effect = epsilon_squared(x, g)
        within = [epsilon_squared(x[g == c], proj[g == c]) for c in range(k)]
        within = [e for e in within if math.isfinite(e)]
        r.project_effect = float(np.mean(within)) if within else math.nan
        out.append(r)
        if progress is not None:
            progress(0.9 * (j + 1) / max(n_par, 1))
    # 6. Many parameters at once: q-values; the verdicts
    tested = [r for r in out if r.verdict != "too few"]
    for r, q in zip(tested, benjamini_hochberg([r.p for r in tested])):
        r.q = q
    least = MIN_OBJECTS if unit == "objects" else MIN_CASES
    for r in tested:
        if min(r.n.values()) < least:
            r.verdict = "too few"
        elif math.isfinite(r.q) and r.q < Q_MAX:
            r.verdict = "separates" if r.auc >= SEPARATES_AUC else "weak"
        elif math.isfinite(r.p) and r.p < Q_MAX and r.auc >= SEPARATES_AUC:
            r.verdict = "promising"                      # strong, but too few cases to pass the correction
        else:
            r.verdict = "chance"
    # 7. Redundancy groups; the proposal: the best of each group that separates
    uids = [c[0] for c in data.columns]
    groups = spearman_groups(data.values[rows], uids)
    number = {uid: i + 1 for i, members in enumerate(groups) for uid in members}
    for r in out:
        r.group = number.get(r.uid, 0)
    separating = [r for r in out if r.verdict == "separates"]
    for r in separating:
        rivals = [s for s in separating if r.group and s.group == r.group]
        r.proposed = not rivals or r is max(rivals, key=lambda s: (s.auc, -len(s.notes())))
    rank = {"separates": 0, "promising": 1, "weak": 2, "chance": 3, "too few": 4}
    out.sort(key=lambda r: (not r.proposed, rank[r.verdict], -(r.auc if math.isfinite(r.auc) else 0.0), r.label))
    if progress is not None:
        progress(1.0)
    used = sorted(set(data.project[rows].tolist()))
    counts = {names[c]: int(np.sum(rows & (data.grade == c))) for c in range(k)}
    return Analysis(name, on, group, names, [data.projects[p] for p in used], counts, out, groups,
                    {"bootstrap": n_boot, "separates_auc": SEPARATES_AUC, "q_max": Q_MAX,
                     "redundant_rho": REDUNDANT_RHO, "min_objects": MIN_OBJECTS, "min_cases": MIN_CASES},
                    dt.datetime.now().isoformat(timespec="seconds"), training_set, profile, ordered, unit)
