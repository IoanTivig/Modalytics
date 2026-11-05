"""
Classifiers from parameters, and how well they agree with the grades given by hand (Qt-free).

A classifier gives each object a **grade** (one of a group's subgroups, e.g. Low /
High grade dysplasia) from its parameter values, with a **reason** a person can
read. Two kinds are plain data, stored in the profile (``Profile.classification``):

* **Ranges** — conditions on parameters, e.g. "Irregularity (BAM) ≥ 0.03 AND Phase
  Std Dev ≥ 0.1": for each grade above the lowest, a list of conditions combined
  with all (AND) or any (OR); the highest grade whose conditions hold is given,
  else the lowest. A missing value never satisfies a condition.
* **Score** — a points score, the form clinical grading scores take: each criterion
  (a parameter beyond a cut-off) adds its points; the total, compared with the
  bands' thresholds, gives the grade. ``fit_score`` chooses the criteria's cut-offs
  (Youden's index) and fits the points by logistic regression on the training
  objects, scaled so the smallest weight is 1 point and rounded to whole points; the
  thresholds between grades are the totals that separate them best.

* **Logistic regression** — each grade's log-odds a weighted sum of the parameters, standardised
  (each parameter's training mean and SD, so a weight is per SD; a missing value counts as the
  training median); fitted with scikit-learn (L2-penalised, grades weighing equally;
  multinomial with more than two classes) and stored as its weights, so classifying needs
  only numpy. Readable: a weight and, with two grades, an odds ratio per SD for each parameter.
* **Gradient boosting** — shallow regression trees added one after another, each correcting
  the sum of the ones before (``init="zero"``: the sum starts at 0); the probabilities are the
  logistic (two grades) or softmax of the learning rate × that sum. Grown with scikit-learn and
  stored as the trees' split arrays and leaf values, like the forest.

* **Random forest** (the trained classifier) — Breiman 2001: many decision trees,
  each grown on a bootstrap sample of the training objects with a random subset of
  the parameters tried at every split; the forest's probability of each grade is
  the mean of the trees' leaf proportions, and the grade given is the most probable
  one (its probability is the object's confidence). Grown with scikit-learn
  (``fit_forest``) and stored in the profile as the trees' split arrays (compressed
  numbers, no pickled code), so classifying needs only numpy. A missing value is
  replaced by the parameter's median on the training objects.

``evaluate`` compares a classifier's grades with the hand-given ones: the confusion
matrix, accuracy, each grade's sensitivity and specificity, and Cohen's kappa
(agreement beyond chance; quadratic-weighted when there are more than two grades,
which then counts how far apart the grades are), each proportion with its 95 %
**Wilson score interval** — with few objects a proportion is uncertain, and this
interval stays sensible near 0 % and 100 % (where p ± 1.96·√(p(1−p)/n) does not):

    centre = (p + z²/2n) / (1 + z²/n),  half-width = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n),  z = 1.96

References:
    Breiman L. Random Forests. Machine Learning. 2001;45(1):5-32. doi:10.1023/A:1010933404324
    Wilson EB. Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American
        Statistical Association. 1927;22(158):209-212. doi:10.1080/01621459.1927.10502953
    Cohen J. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement.
        1960;20(1):37-46. doi:10.1177/001316446002000104
"""

from __future__ import annotations

import base64
import io
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

OPS = {">=": "≥", "<=": "≤"}
LEARNED = ("logistic", "forest", "boosting")          # learned from the objects: they give probabilities


# ---------------------------------------------------------------- the classifier kinds
@dataclass
class Condition:
    """One parameter compared with a value: ``uid`` ``op`` ``value`` (op ">=" or "<=")."""

    uid: str
    op: str
    value: float
    label: str = ""

    def holds(self, v: float) -> bool | None:
        """True / False, or None when the value is missing."""
        if v is None or not math.isfinite(v):
            return None
        return v >= self.value if self.op == ">=" else v <= self.value

    def text(self) -> str:
        return f"{self.label or self.uid} {OPS.get(self.op, self.op)} {self.value:.4g}"

    def to_dict(self) -> dict:
        return {"parameter": self.uid, "op": self.op, "value": float(self.value), "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(str(d["parameter"]), str(d.get("op", ">=")), float(d["value"]), str(d.get("label", "")))


@dataclass
class Rule:
    """The conditions that give a grade, combined with "all" (AND) or "any" (OR)."""

    grade: str
    conditions: list = field(default_factory=list)
    combine: str = "all"

    def to_dict(self) -> dict:
        return {"grade": self.grade, "combine": self.combine, "conditions": [c.to_dict() for c in self.conditions]}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(str(d["grade"]), [Condition.from_dict(c) for c in d.get("conditions", [])],
                   str(d.get("combine", "all")))


@dataclass
class Criterion:
    """A score criterion: ``points`` when the parameter is beyond its cut-off."""

    condition: Condition
    points: int

    def to_dict(self) -> dict:
        return dict(self.condition.to_dict(), points=int(self.points))

    @classmethod
    def from_dict(cls, d: dict) -> "Criterion":
        return cls(Condition.from_dict(d), int(d.get("points", 1)))


@dataclass
class Classifier:
    """A classifier of one group's objects (see the module docstring).

    Attributes:
        kind: "ranges", "score", "logistic", "forest" (random forest) or "boosting" (gradient boosting).
        group: The class group whose objects it grades (e.g. "Gland").
        grades: The grades, lowest first.
        rules: Ranges: one rule per grade above the lowest (checked from the highest).
        criteria: Score: the criteria and their points.
        thresholds: Score: the total from which each grade above the lowest is given (one per grade above the
            lowest, increasing).
        note: Where it comes from (training set, when), shown with it.
        features: Learned kinds: [(parameter id, label)] it reads, in order.
        medians: Learned kinds: each feature's median on the training objects (stands in for a missing value).
        importance: Forest, boosting: each feature's importance (mean decrease in impurity, sums to 1); logistic:
            each feature's |weight| as a share of their sum.
        forest: Forest, boosting: the trees, encoded (``_encode_trees``).
        info: How it was grown (trees, depth, learning rate, out-of-bag accuracy, …).
        weights: Logistic: one row of weights per SD of each feature (one row with two grades: the higher grade's
            log-odds; else one per grade).
        intercepts: Logistic: one per row of ``weights``.
        means, scales: Logistic: each feature's training mean and SD (the features are standardised with them).
        prep: How the objects were prepared before it was made (``training.prep.Prep`` as a dict: per-project
            normalisation — applied the same way to every project it classifies, ``prepare_objects`` — and data
            augmentation); {}: neither.
    """

    kind: str
    group: str
    grades: list
    rules: list = field(default_factory=list)
    criteria: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
    note: str = ""
    features: list = field(default_factory=list)
    medians: list = field(default_factory=list)
    importance: list = field(default_factory=list)
    forest: str = ""
    info: dict = field(default_factory=dict)
    weights: list = field(default_factory=list)
    intercepts: list = field(default_factory=list)
    means: list = field(default_factory=list)
    scales: list = field(default_factory=list)
    prep: dict = field(default_factory=dict)
    _trees: dict | None = field(default=None, repr=False, compare=False)

    @property
    def parameters(self) -> list[str]:
        """The parameter ids it reads."""
        conds = [c for r in self.rules for c in r.conditions] + [c.condition for c in self.criteria]
        return list(dict.fromkeys([c.uid for c in conds] + [u for u, _l in self.features]))

    @property
    def learned(self) -> bool:
        """A learned classifier (logistic regression, random forest, gradient boosting): it gives probabilities."""
        return self.kind in LEARNED

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Learned kinds: (objects, grades) probabilities for a (objects, features) array (NaN = missing)."""
        X = np.array(X, float, copy=True)
        med = np.asarray(self.medians, float)
        gaps = ~np.isfinite(X)
        X[gaps] = np.broadcast_to(med, X.shape)[gaps]
        if self.kind == "logistic":
            Z = (X - np.asarray(self.means, float)) / np.asarray(self.scales, float)
            raw = Z @ np.asarray(self.weights, float).T + np.asarray(self.intercepts, float)
            return _to_probabilities(raw)
        if self._trees is None:
            self._trees = _decode_trees(self.forest)
        if self.kind == "boosting":
            k = int(self.info.get("outputs", 1))
            leaves = _leaf_values(self._trees, X)[:, :, 0]                      # (objects, stages × outputs)
            raw = float(self.info.get("learning_rate", 0.1)) * leaves.reshape(len(X), -1, k).sum(axis=1)
            return _to_probabilities(raw)
        return _forest_proba(self._trees, X, len(self.grades))

    def classify(self, values: dict) -> tuple[str, float, str]:
        """(grade, score, reason) for one object's {parameter id: value}.

        The score is the points total (Score) or the number of conditions met (Ranges).
        """
        if self.learned:
            x = np.array([[values.get(u, math.nan) for u, _l in self.features]], float)
            missing = [lab for (u, lab), v in zip(self.features, x[0]) if not math.isfinite(v)]
            p = self.probabilities(x)[0]
            k = int(np.argmax(p))
            reason = " · ".join(f"P({g}) {q:.2f}" for g, q in zip(self.grades, p))
            if self.kind == "logistic":                      # what weighed most for this object
                z = (np.where(np.isfinite(x[0]), x[0], self.medians) - np.asarray(self.means)) / np.asarray(self.scales)
                row = np.asarray(self.weights)[0 if len(self.weights) == 1 else k]
                push = sorted(zip(self.features, row * z), key=lambda t: -abs(t[1]))[:2]
                reason += " · most: " + ", ".join(f"{lab} {v:+.2f}" for (_u, lab), v in push)
            reason += f" · missing (training median used): {', '.join(missing)}" if missing else ""
            return self.grades[k], float(p[k]), reason
        if self.kind == "score":
            total, parts, missing = 0, [], []
            for c in self.criteria:
                ok = c.condition.holds(values.get(c.condition.uid, math.nan))
                if ok is None:
                    missing.append(c.condition.label or c.condition.uid)
                elif ok:
                    total += c.points
                    parts.append(f"{c.condition.text()} (+{c.points})")
            grade = self.grades[0]
            for g, t in zip(self.grades[1:], self.thresholds):
                if total >= t:
                    grade = g
            reason = f"{total} points" + (": " + ", ".join(parts) if parts else "") + \
                (f" · missing: {', '.join(missing)}" if missing else "")
            return grade, float(total), reason
        for rule in sorted(self.rules, key=lambda r: -self.grades.index(r.grade) if r.grade in self.grades else 0):
            states = [c.holds(values.get(c.uid, math.nan)) for c in rule.conditions]
            met = [c for c, s in zip(rule.conditions, states) if s]
            if rule.conditions and ((rule.combine == "all" and all(states)) or (rule.combine == "any" and any(states))):
                joined = " AND " if rule.combine == "all" else " OR "
                return rule.grade, float(len(met)), joined.join(c.text() for c in met)
        missing = [c.label or c.uid for r in self.rules for c in r.conditions
                   if c.holds(values.get(c.uid, math.nan)) is None]
        return self.grades[0], 0.0, "no rule met" + (f" · missing: {', '.join(missing)}" if missing else "")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "group": self.group, "grades": list(self.grades), "note": self.note}
        if self.kind == "score":
            d["criteria"] = [c.to_dict() for c in self.criteria]
            d["thresholds"] = [int(t) for t in self.thresholds]
        elif self.learned:
            d.update(features=[{"parameter": u, "label": lab} for u, lab in self.features],
                     medians=[float(m) for m in self.medians], importance=[round(float(i), 6) for i in self.importance],
                     info=dict(self.info))
            if self.kind == "logistic":
                d.update(weights=[[float(w) for w in row] for row in self.weights],
                         intercepts=[float(b) for b in self.intercepts], means=[float(m) for m in self.means],
                         scales=[float(s) for s in self.scales])
            else:
                d["forest"] = self.forest
        else:
            d["rules"] = [r.to_dict() for r in self.rules]
        if self.prep:
            d["prep"] = dict(self.prep)
        return d

    @property
    def normalize(self) -> str:
        """Its per-project normalisation (``training.prep.NORMALIZE``; "none")."""
        return str((self.prep or {}).get("normalize", "none"))

    @classmethod
    def from_dict(cls, d: dict) -> "Classifier":
        return cls(str(d.get("kind", "ranges")), str(d.get("group", "")), [str(g) for g in d.get("grades", [])],
                   [Rule.from_dict(r) for r in d.get("rules", [])],
                   [Criterion.from_dict(c) for c in d.get("criteria", [])],
                   [int(t) for t in d.get("thresholds", [])], str(d.get("note", "")),
                   [(str(f["parameter"]), str(f.get("label", ""))) for f in d.get("features", [])],
                   [float(m) for m in d.get("medians", [])], [float(i) for i in d.get("importance", [])],
                   str(d.get("forest", "")), dict(d.get("info") or {}),
                   [[float(w) for w in row] for row in d.get("weights", [])],
                   [float(b) for b in d.get("intercepts", [])], [float(m) for m in d.get("means", [])],
                   [float(s) for s in d.get("scales", [])], dict(d.get("prep") or {}))


# ---------------------------------------------------------------- fitting a score
def _logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, rounds: int = 100,
              sample: np.ndarray | None = None) -> np.ndarray:
    """Weights (intercept first) of an L2-penalised logistic regression (Newton's method; ``sample``: each
    object's weight)."""
    A = np.column_stack([np.ones(len(X)), X])
    s = np.ones(len(X)) if sample is None else np.asarray(sample, float)
    w = np.zeros(A.shape[1])
    penalty = np.full(A.shape[1], l2)
    penalty[0] = 0.0                                 # the intercept is not shrunk
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        grad = A.T @ (s * (p - y)) + penalty * w
        hess = (A * (s * p * (1 - p))[:, None]).T @ A + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fit_ranges_jointly(clf: Classifier, data, rows: np.ndarray) -> Classifier:
    """Ranges with each rule's cut-offs found together on ``rows`` (``ranges.joint_youden``): a rule's grade
    against the grade below, its conditions combined as it says; each starts from its own best cut-off."""
    from modalytics.training import ranges as rg
    col = {c[0]: i for i, c in enumerate(data.columns)}
    out = Classifier.from_dict(clf.to_dict())
    for rule in out.rules:
        if rule.grade not in data.grades or not rule.conditions:
            continue
        k = data.grades.index(rule.grade)
        idx = [col[c.uid] for c in rule.conditions]
        lo = data.values[np.ix_(np.flatnonzero(rows & (data.grade == k - 1)), idx)]
        hi = data.values[np.ix_(np.flatnonzero(rows & (data.grade == k)), idx)]
        start = []
        for j, c in enumerate(rule.conditions):
            a, b = lo[:, j][np.isfinite(lo[:, j])], hi[:, j][np.isfinite(hi[:, j])]
            cut, _s, _p = rg.youden(a, b, "higher" if c.op == ">=" else "lower")
            start.append(cut if math.isfinite(cut) else c.value)
        cuts, _sens, _spec = rg.joint_youden(lo, hi, [c.op for c in rule.conditions], start, rule.combine)
        for c, v in zip(rule.conditions, cuts):
            if math.isfinite(v):
                c.value = v
    return out


def usable(data, rows: np.ndarray, uids: list[str]) -> np.ndarray:
    """``rows`` without the partial objects (``Dataset.partial_rows``) that lack a value of these parameters — their
    shape is incomplete and the parameter needs the whole object: a classifier reading it cannot use them."""
    rows = np.asarray(rows, bool)
    partial = getattr(data, "partial_rows", None)
    if partial is None or not len(partial) or not partial.any():
        return rows
    col = {c[0]: i for i, c in enumerate(data.columns)}
    idx = [col[u] for u in uids if u in col]
    lacking = partial & ~np.all(np.isfinite(data.values[:, idx]), axis=1) if idx else np.zeros(len(rows), bool)
    return rows & ~lacking


SCORE_CANDIDATES = 30          # cut-offs tried per criterion and round when they are found together
SCORE_ROUNDS = 4


def fit_score(data, rows: np.ndarray, uids: list[str], group: str, max_points: int = 5,
              weigh: str = "none", joint: bool = True) -> Classifier:
    """A points score from these parameters, fitted on ``rows`` of a ``training.dataset.Dataset``.

    1. Each parameter becomes a criterion at its Youden cut-off (direction from its AUC), between the lowest
       and the highest grade.
    2. A logistic regression of "higher than the lowest grade" on the criteria (met = 1) gives each a weight;
       weights are scaled so the smallest positive one is 1 point (at most ``max_points``) and rounded.
    3. Each threshold is the points total that best separates the grades below from those above (Youden).
    4. ``joint``: the cut-offs are then found **together** — each criterion's cut-off moved in turn over its
       parameter's values (``SCORE_CANDIDATES`` of them), steps 2–3 made again, and kept when the training
       objects' balanced accuracy rises; until nothing improves (at most ``SCORE_ROUNDS`` rounds). Found alone,
       each cut-off is the best for its parameter alone; with points added up, they work together (as Ranges'
       conditions, ``fit_ranges_jointly``).

    ``weigh``: the projects weighed in step 2 (``training.prep.project_weights``).

    Raises:
        ValueError: If no parameter is given or a grade has no objects.
    """
    from modalytics.training import ranges as rg
    from modalytics.training.prep import project_weights
    if not uids:
        raise ValueError("choose at least one parameter for the score")
    rows = usable(data, rows, uids)                          # partial objects only with every value it reads
    grades = data.grades
    g = data.grade[rows]
    if len(set(g.tolist())) < 2:
        raise ValueError("the training objects need at least two grades")
    labels = {c[0]: c[1] for c in data.columns}
    col = {c[0]: i for i, c in enumerate(data.columns)}
    V = data.values[np.ix_(np.flatnonzero(rows), [col[u] for u in uids])].astype(float)
    sample = project_weights(data, rows, weigh)
    # 1. Each parameter's direction and its own best cut-off
    ops, cuts = [], []
    for j, _uid in enumerate(uids):
        v = V[:, j]
        lo, hi = v[(g == 0) & np.isfinite(v)], v[(g == len(grades) - 1) & np.isfinite(v)]
        a = rg.auc(lo, hi)
        direction = "higher" if not math.isfinite(a) or a >= 0.5 else "lower"
        cut, _s, _p = rg.youden(lo, hi, direction)
        ops.append(">=" if direction == "higher" else "<=")
        cuts.append(cut)

    def made(cuts: list) -> tuple[Classifier, float]:
        """Steps 2–3 for these cut-offs: (the score, its balanced accuracy on the training objects)."""
        with np.errstate(invalid="ignore"):                  # a missing value never meets a criterion
            X = np.column_stack([(V[:, j] >= c) if op == ">=" else (V[:, j] <= c)
                                 for j, (op, c) in enumerate(zip(ops, cuts))]).astype(float)
        w = _logistic(X, (g > 0).astype(float), sample=sample)[1:]
        positive = w[w > 0]
        unit = positive.min() if positive.size else 1.0
        points = np.array([int(max(0, min(max_points, round(wi / unit)))) if wi > 0 else 0 for wi in w])
        if not points.any():
            points[0] = 1
        totals = X @ points
        thresholds = []
        for k in range(1, len(grades)):
            cut, _s, _p = rg.youden(totals[g < k], totals[g >= k], "higher")
            t = int(math.ceil(cut)) if math.isfinite(cut) else 1
            thresholds.append(max(t, thresholds[-1] if thresholds else 0))
        given = np.sum(totals[:, None] >= np.asarray(thresholds)[None, :], axis=1)
        ba = float(np.mean([np.mean(given[g == k] == k) for k in range(len(grades)) if np.any(g == k)]))
        chosen = [Criterion(Condition(u, op, c, labels.get(u, u)), int(pt))
                  for u, op, c, pt in zip(uids, ops, cuts, points) if pt > 0]
        return Classifier("score", group, list(grades), criteria=chosen, thresholds=thresholds), ba

    best, best_ba = made(cuts)
    # 4. The cut-offs found together
    for _round in range(SCORE_ROUNDS if joint else 0):
        moved = False
        for j in range(len(uids)):
            v = np.unique(V[:, j][np.isfinite(V[:, j])])
            if len(v) < 2:
                continue
            options = (v[:-1] + v[1:]) / 2.0
            if len(options) > SCORE_CANDIDATES:
                options = np.quantile(options, np.linspace(0, 1, SCORE_CANDIDATES))
            for c in options:
                trial = list(cuts)
                trial[j] = float(c)
                clf, ba = made(trial)
                if ba > best_ba + 1e-9:
                    best, best_ba, cuts, moved = clf, ba, trial, True
        if not moved:
            break
    return best


# ---------------------------------------------------------------- the trained classifier (random forest)
def fit_forest(data, rows: np.ndarray, uids: list[str], group: str, trees: int = 300, min_leaf: int = 2,
               balanced: bool = True, seed: int = 1, weigh: str = "none") -> Classifier:
    """A random forest grown on ``rows`` of a ``training.dataset.Dataset`` (needs scikit-learn).

    1. The features are the chosen parameters; a missing value is replaced by the parameter's training median.
    2. scikit-learn grows the forest (``balanced``: the grades weigh equally however many objects each has;
       ``seed`` makes it repeatable); its out-of-bag accuracy is kept as a first, internal estimate.
    3. The trees are stored as numbers (``_encode_trees``), with each parameter's importance.

    Raises:
        ValueError: If no parameter is given or a grade has no training objects.
        ImportError: If scikit-learn is not installed.
    """
    from sklearn.ensemble import RandomForestClassifier
    if not uids:
        raise ValueError("choose at least one parameter for the trained classifier")
    rows = usable(data, rows, uids)                          # partial objects only with every value it reads
    col = {c[0]: i for i, c in enumerate(data.columns)}
    labels = {c[0]: c[1] for c in data.columns}
    X = data.values[np.ix_(np.flatnonzero(rows), [col[u] for u in uids])].astype(float)
    y = data.grade[rows]
    missing = [g for k, g in enumerate(data.grades) if not np.any(y == k)]
    if missing:
        raise ValueError(f"the training projects have no objects of: {', '.join(missing)}")
    # 1. Missing values → training medians (a parameter never measured counts as 0)
    med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in X.T])
    gaps = ~np.isfinite(X)
    X[gaps] = np.broadcast_to(med, X.shape)[gaps]
    # 2. Grow
    forest = RandomForestClassifier(n_estimators=int(trees), min_samples_leaf=int(min_leaf), oob_score=True,
                                    class_weight="balanced" if balanced else None, random_state=int(seed),
                                    n_jobs=-1)
    from modalytics.training.prep import project_weights
    forest.fit(X, y, sample_weight=project_weights(data, rows, weigh))
    # 3. Store
    info = {"trees": int(trees), "min_leaf": int(min_leaf), "balanced": bool(balanced), "seed": int(seed),
            "objects": int(len(y)), "oob_accuracy": round(float(forest.oob_score_), 4)}
    return Classifier("forest", group, list(data.grades), features=[(u, labels.get(u, u)) for u in uids],
                      medians=med.tolist(), importance=forest.feature_importances_.tolist(),
                      forest=_encode_trees(forest.estimators_, len(data.grades), classes=forest.classes_), info=info)


def _prepared(data, rows: np.ndarray, uids: list[str], what: str) -> tuple:
    """(X with missing values as the training medians, y, medians, labels, the rows used) of the usable training
    rows."""
    if not uids:
        raise ValueError(f"choose at least one parameter for the {what}")
    rows = usable(data, rows, uids)
    col = {c[0]: i for i, c in enumerate(data.columns)}
    labels = {c[0]: c[1] for c in data.columns}
    X = data.values[np.ix_(np.flatnonzero(rows), [col[u] for u in uids])].astype(float)
    y = data.grade[rows]
    missing = [g for k, g in enumerate(data.grades) if not np.any(y == k)]
    if missing:
        raise ValueError(f"the training projects have no objects of: {', '.join(missing)}")
    med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in X.T])
    gaps = ~np.isfinite(X)
    X[gaps] = np.broadcast_to(med, X.shape)[gaps]
    return X, y, med, labels, rows


def fit_logistic(data, rows: np.ndarray, uids: list[str], group: str, strength: float = 1.0,
                 balanced: bool = True, weigh: str = "none") -> Classifier:
    """A logistic regression fitted on ``rows`` of a ``training.dataset.Dataset`` (needs scikit-learn).

    1. Missing values → training medians; each parameter standardised (its training mean and SD), so a weight is
       per SD and weights can be compared.
    2. scikit-learn fits it, L2-penalised (``strength`` = 1/C: larger, weights pulled more towards 0, steadier
       with few objects); ``balanced``: the grades weigh equally.

    Raises:
        ValueError: If no parameter is given or a grade has no training objects.
    """
    from sklearn.linear_model import LogisticRegression
    from modalytics.training.prep import project_weights
    X, y, med, labels, used = _prepared(data, rows, uids, "logistic regression")
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    model = LogisticRegression(C=1.0 / max(strength, 1e-6), class_weight="balanced" if balanced else None,
                               max_iter=5000)
    model.fit((X - mean) / sd, y, sample_weight=project_weights(data, used, weigh))
    w = np.abs(model.coef_).sum(axis=0)
    info = {"strength": float(strength), "balanced": bool(balanced), "objects": int(len(y))}
    return Classifier("logistic", group, list(data.grades), features=[(u, labels.get(u, u)) for u in uids],
                      medians=med.tolist(), importance=(w / max(w.sum(), 1e-12)).tolist(), info=info,
                      weights=model.coef_.tolist(), intercepts=model.intercept_.tolist(), means=mean.tolist(),
                      scales=sd.tolist())


def fit_boosting(data, rows: np.ndarray, uids: list[str], group: str, stages: int = 200, depth: int = 3,
                 learning_rate: float = 0.1, seed: int = 1, weigh: str = "none") -> Classifier:
    """Gradient boosting grown on ``rows`` of a ``training.dataset.Dataset`` (needs scikit-learn).

    Shallow regression trees (``depth``) added ``stages`` times, each scaled by ``learning_rate``, the sum starting
    at 0 (``init="zero"``); with more than two classes one tree per class and stage. 80 % of the objects drawn for
    each tree (stochastic gradient boosting: steadier).

    Raises:
        ValueError: If no parameter is given or a grade has no training objects.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from modalytics.training.prep import project_weights
    X, y, med, labels, used = _prepared(data, rows, uids, "gradient boosting")
    model = GradientBoostingClassifier(n_estimators=int(stages), max_depth=int(depth),
                                       learning_rate=float(learning_rate), subsample=0.8, init="zero",
                                       random_state=int(seed))
    model.fit(X, y, sample_weight=project_weights(data, used, weigh))
    outputs = model.estimators_.shape[1]
    info = {"stages": int(stages), "depth": int(depth), "learning_rate": float(learning_rate), "seed": int(seed),
            "objects": int(len(y)), "outputs": int(outputs)}
    trees = [est for stage in model.estimators_ for est in stage]          # stage by stage, one per output
    return Classifier("boosting", group, list(data.grades), features=[(u, labels.get(u, u)) for u in uids],
                      medians=med.tolist(), importance=model.feature_importances_.tolist(), info=info,
                      forest=_encode_trees(trees, 1, leaf=lambda t: t.value[:, 0, :1]))


def _to_probabilities(raw: np.ndarray) -> np.ndarray:
    """(objects, grades) probabilities from raw scores: one column (two grades) → logistic; else softmax."""
    if raw.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 0], -40, 40)))
        return np.column_stack([1.0 - p, p])
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _encode_trees(estimators, n_values: int, leaf=None, classes=None) -> str:
    """Trees as one compressed text: their nodes' split feature, threshold, children and the values at the leaves
    (``numpy.savez`` arrays, zlib, base64 — numbers only, no code).

    Args:
        estimators: The fitted trees (a forest's ``estimators_``, or a list of a boosting's trees).
        n_values: Values per node: the grades (a forest: their proportions), or 1 (a boosting tree's output).
        leaf: ``leaf(tree_) -> (nodes, n_values)``; default: the classes' proportions (a forest).
        classes: A forest's ``classes_`` (which grade each column of its trees is).
    """
    left, right, feat, thr, val, starts = [], [], [], [], [], []
    offset = 0
    for est in estimators:
        t = est.tree_
        n = t.node_count
        starts.append(offset)
        lc, rc = t.children_left.astype(np.int64), t.children_right.astype(np.int64)
        left.append(np.where(lc < 0, -1, lc + offset))
        right.append(np.where(rc < 0, -1, rc + offset))
        feat.append(t.feature.astype(np.int32))
        thr.append(t.threshold.astype(np.float64))
        if leaf is None:
            v = np.zeros((n, n_values))
            v[:, np.asarray(classes).astype(int)] = t.value[:, 0, :]
            v /= np.maximum(v.sum(axis=1, keepdims=True), 1e-12)
        else:
            v = np.asarray(leaf(t), float).reshape(n, n_values)
        val.append(v.astype(np.float32))
        offset += n
    buf = io.BytesIO()
    np.savez(buf, left=np.concatenate(left), right=np.concatenate(right), feature=np.concatenate(feat),
             threshold=np.concatenate(thr), value=np.concatenate(val), starts=np.asarray(starts, np.int64))
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode("ascii")


def _decode_trees(text: str) -> dict:
    with np.load(io.BytesIO(zlib.decompress(base64.b64decode(text))), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _forest_proba(trees: dict, X: np.ndarray, n_grades: int) -> np.ndarray:
    """Mean leaf proportions over the trees."""
    return _leaf_values(trees, X).mean(axis=1).astype(float).reshape(len(X), n_grades)


def _leaf_values(trees: dict, X: np.ndarray) -> np.ndarray:
    """(objects, trees, values) at the leaf each object reaches in each tree, every tree walked at once (as
    scikit-learn: value ≤ threshold → left; values compared in single precision, as scikit-learn does)."""
    X = X.astype(np.float32).astype(np.float64)
    node = np.repeat(trees["starts"][None, :], len(X), axis=0)             # (objects, trees)
    left, right, feat, thr = trees["left"], trees["right"], trees["feature"], trees["threshold"]
    rows = np.arange(len(X))[:, None]
    while True:
        inner = left[node] >= 0
        if not inner.any():
            break
        go_left = X[rows, np.where(inner, feat[node], 0)] <= thr[node]
        node = np.where(inner, np.where(go_left, left[node], right[node]), node)
    return trees["value"][node].astype(float)


def ranges_from(separation, group: str, grades: list) -> Classifier:
    """A one-condition Ranges classifier from a range-finder row (``ranges.Separation``)."""
    cond = Condition(separation.uid, ">=" if separation.direction == "higher" else "<=", separation.cutoff,
                     separation.label)
    return Classifier("ranges", group, list(grades), rules=[Rule(separation.higher, [cond], "all")])


# ---------------------------------------------------------------- agreement with the hand-given grades
@dataclass
class Agreement:
    """How a classifier's grades compare with the hand-given ones.

    Attributes:
        grades: The grade names.
        confusion: [true grade][predicted grade] counts.
        accuracy: Share classified right.
        sensitivity, specificity: Per grade (one against the others).
        kappa: Cohen's kappa (quadratic-weighted for more than two ordered grades: a mistake two grades off counts
            more; plain for categories); 1 = perfect, 0 = chance.
        n: Objects compared.
        accuracy_ci, sensitivity_ci, specificity_ci: 95 % intervals (low, high) of those proportions: by
            resampling whole projects when the objects come from ``MIN_PROJECTS`` or more, else Wilson's.
        resampled: "projects" or "objects" — how the intervals were made.
        projects: The projects the objects come from (0: not told).
    """

    grades: list
    confusion: np.ndarray
    accuracy: float
    sensitivity: list
    specificity: list
    kappa: float
    n: int
    accuracy_ci: tuple = (math.nan, math.nan)
    sensitivity_ci: list = field(default_factory=list)
    specificity_ci: list = field(default_factory=list)
    resampled: str = "objects"
    projects: int = 0


def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval of the proportion k / n (see the module docstring); (NaN, NaN) if n = 0."""
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# Objects of one project (one sample) resemble each other, so they are not independent: an interval that treats
# them as independent (Wilson's, or a bootstrap over objects) comes out too narrow. With enough projects, the
# intervals resample whole projects instead (a cluster bootstrap); with fewer, they fall back to the objects and
# say so.
MIN_PROJECTS = 5
BOOT_PROJECTS = 1000


def by_projects(projects) -> bool:
    """True if objects of these projects (one id per object) are enough to resample whole projects."""
    return projects is not None and len(np.unique(projects)) >= MIN_PROJECTS


def resamples(n: int, projects=None, n_boot: int = BOOT_PROJECTS, seed: int = 0):
    """Bootstrap draws of object indices: whole projects drawn with replacement when ``by_projects``, else objects.

    Args:
        n: The objects.
        projects: (n,) each object's project, or None.
        n_boot: The draws.
        seed: The random seed.

    Yields:
        (m,) object indices of one draw (m = n for objects; about n for projects).
    """
    rng = np.random.default_rng(seed)
    if by_projects(projects):
        _ids, which = np.unique(np.asarray(projects), return_inverse=True)
        members = [np.flatnonzero(which == c) for c in range(len(_ids))]
        for _b in range(n_boot):
            yield np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
    else:
        for _b in range(n_boot):
            yield rng.integers(0, n, n)


def interval(values) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles of the finite bootstrap values ((NaN, NaN) with 10 or fewer)."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    return tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if len(v) > 10 else (math.nan, math.nan)


def agreement(true: np.ndarray, predicted: np.ndarray, grades: list, ordered: bool = True,
              projects=None) -> Agreement:
    """Compare predicted grade indices with true ones (``ordered``: the grades are a scale; else categories).

    Args:
        projects: (objects,) each object's project: with ``MIN_PROJECTS`` or more, the intervals resample whole
            projects (see ``resamples``); without, Wilson's.
    """
    k = len(grades)
    true, predicted = np.asarray(true, int), np.asarray(predicted, int)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (true, predicted), 1)
    n = int(cm.sum())
    n_projects = len(np.unique(projects)) if projects is not None else 0
    if n == 0:
        return Agreement(grades, cm, math.nan, [math.nan] * k, [math.nan] * k, math.nan, 0,
                         (math.nan, math.nan), [(math.nan, math.nan)] * k, [(math.nan, math.nan)] * k,
                         projects=n_projects)
    acc = float(np.trace(cm) / n)
    sens, spec, sens_ci, spec_ci = [], [], [], []
    for i in range(k):
        tp, fn = cm[i, i], cm[i].sum() - cm[i, i]
        fp, tn = cm[:, i].sum() - cm[i, i], n - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        sens.append(float(tp / (tp + fn)) if tp + fn else math.nan)
        spec.append(float(tn / (tn + fp)) if tn + fp else math.nan)
        sens_ci.append(wilson(tp, tp + fn))
        spec_ci.append(wilson(tn, tn + fp))
    # Cohen's kappa; quadratic weights (i − j)² / (k − 1)² for more than two ordered grades (categories: plain)
    idx = np.arange(k)
    weights = ((idx[:, None] - idx[None, :]) ** 2) / max((k - 1) ** 2, 1) if k > 2 and ordered else \
        (idx[:, None] != idx[None, :]).astype(float)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (weights * expected).sum()
    kappa = float(1 - (weights * cm).sum() / denom) if denom > 0 else math.nan
    acc_ci = wilson(np.trace(cm), n)
    if by_projects(projects):                                      # whole projects resampled
        code = true * k + predicted
        draws = np.array([np.bincount(code[i], minlength=k * k).reshape(k, k) for i in resamples(n, projects)])
        rows, cols, hits = draws.sum(axis=2), draws.sum(axis=1), np.diagonal(draws, axis1=1, axis2=2)
        totals = draws.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            acc_ci = interval(hits.sum(axis=1) / totals)
            sens_ci = [interval(hits[:, i] / rows[:, i]) for i in range(k)]
            spec_ci = [interval((totals - rows[:, i] - cols[:, i] + hits[:, i]) / (totals - rows[:, i]))
                       for i in range(k)]
    return Agreement(grades, cm, acc, sens, spec, kappa, n, acc_ci, sens_ci, spec_ci,
                     "projects" if by_projects(projects) else "objects", n_projects)


def predict(classifier: Classifier, data, rows: np.ndarray) -> np.ndarray:
    """Predicted grade indices for ``rows`` of a dataset."""
    col = {c[0]: i for i, c in enumerate(data.columns)}
    if classifier.learned:                                           # all objects at once
        X = data.values[np.ix_(np.flatnonzero(rows), [col[u] for u, _l in classifier.features])]
        return np.argmax(classifier.probabilities(X), axis=1).astype(int) if len(X) else np.zeros(0, int)
    out = []
    for r in np.flatnonzero(rows):
        values = {uid: data.values[r, j] for uid, j in col.items()}
        grade, _score, _why = classifier.classify(values)
        out.append(classifier.grades.index(grade) if grade in classifier.grades else 0)
    return np.asarray(out, int)


def evaluate(classifier: Classifier, data, rows: np.ndarray, intervals: bool = True) -> Agreement:
    """Agreement of the classifier with the hand-given grades on ``rows`` of a dataset (a partial object only when
    it has every value the classifier reads); ``intervals``: by resampling whole projects when there are enough
    (else Wilson's only)."""
    rows = usable(data, rows, classifier.parameters)
    return agreement(data.grade[rows], predict(classifier, data, rows), data.grades, data.ordered,
                     data.project[rows] if intervals else None)


def prepare_objects(clf: Classifier, rows: list[dict], complete: list[bool] | None = None
                    ) -> tuple[list[dict], str]:
    """One project's objects of the group ({parameter id: value}, all of them) ready for the classifier: read
    relative to the project as when it was made (``Classifier.prep``; unchanged without normalisation).

    Args:
        complete: Per object: its shape is complete (the reference is made of those; default: all).

    Returns:
        (the objects' values; why they cannot be graded, "" if they can).
    """
    from modalytics.training.prep import normalize_objects
    return normalize_objects(rows, clf.parameters, clf.normalize, complete)


def describe(clf: Classifier) -> str:
    """The classifier in one line a person can read (profile editor, report)."""
    from modalytics.training.prep import Prep
    prep = Prep.from_dict(clf.prep).describe() if clf.prep else "none"
    text = _describe(clf)
    return text if prep == "none" else f"{text} · {prep}"


def _describe(clf: Classifier) -> str:
    if clf.kind == "score":
        return ("Points score: " + ", ".join(f"{c.condition.text()} (+{c.points})" for c in clf.criteria) + "; "
                + ", ".join(f"{g} from {t} points" for g, t in zip(clf.grades[1:], clf.thresholds)))
    if clf.kind == "forest":
        top = sorted(zip(clf.features, clf.importance), key=lambda x: -x[1])[:5]
        return (f"Random forest of {clf.info.get('trees')} trees on {len(clf.features)} "
                "parameters; most important: " + ", ".join(f"{lab} ({100 * imp:.0f} %)" for (_u, lab), imp in top))
    if clf.kind == "boosting":
        top = sorted(zip(clf.features, clf.importance), key=lambda x: -x[1])[:5]
        return (f"Gradient boosting of {clf.info.get('stages')} trees (depth {clf.info.get('depth')}) on "
                f"{len(clf.features)} parameters; most important: "
                + ", ".join(f"{lab} ({100 * imp:.0f} %)" for (_u, lab), imp in top))
    if clf.kind == "logistic":
        W = np.asarray(clf.weights)
        if len(W) == 1:
            parts = [f"{lab} {w:+.2f} (OR {math.exp(w):.2f})" for (_u, lab), w in zip(clf.features, W[0])]
            return f"Logistic regression, log-odds of {clf.grades[-1]} per SD: " + ", ".join(parts)
        return (f"Logistic regression on {len(clf.features)} parameters, one row of weights per class "
                f"(multinomial)")
    return "Ranges: " + "; ".join(f"{r.grade} when " + (" AND " if r.combine == "all" else " OR ").join(
        c.text() for c in r.conditions) for r in clf.rules) + f"; otherwise {clf.grades[0]}"
