"""
Preparing the measured objects for a classifier: per-project normalisation and data augmentation (Qt-free).

A classifier is made from the objects of the training projects. Two things can be done to them first, both set
in Training › Build, recorded with the classifier (``Classifier.prep``) and named in the report:

**Normalisation per project** (``normalize``). Objects of one project resemble each other more than objects of
the same grade from different projects — partly because of how that project was acquired (illumination,
calibration, sample thickness). Each parameter can be read relative to the project's own objects of the group:

* ``difference``: v − the project's median;
* ``ratio``: v / the project's median;
* ``robust_z``: (v − the project's median) / the project's interquartile range.

The reference is the median (and IQR) over the project's complete objects of the group (not those whose shape
is incomplete), whatever their grade, so it is the same when a new project is classified (all its complete
objects of the group). It needs ``MIN_REFERENCE`` objects with a value; with fewer, the value is left empty. **Caution**: a project whose objects are all of one grade
loses that grade's difference — normalisation suits projects that hold several grades, or a reference tissue.

**Data augmentation** — more training rows, made from the training projects' objects only (never from the
evaluation or test projects; in cross-validation, from the projects of the fold's training part only):

* **acquisition** (``training.acquisition``): each training project measured again as if acquired a little
  differently (pixel size, gain, offset, noise, defocus) — copies that keep their project, so a fold leaves them
  out with it;
* **noise copies**: each training object copied with Gaussian noise added to every parameter (SD ×
  ``noise_sd`` of that parameter over the training objects);
* **SMOTE** (synthetic minority oversampling): each class with fewer objects than the largest gets new objects
  on the line between one of its objects and one of its 5 nearest neighbours of the same class (standardised
  parameters), until it has as many.

Augmented rows are used to **make** a classifier only; it is always **judged** on real objects.

**Weighing projects** (``project_weights``). A project with many objects counts more than one with few. With
``equal`` each project's objects together weigh the same (one object of a one-object project then weighs as much
as all the objects of another — use it with care); with ``sqrt`` a project's total grows with the square root of
its objects (a compromise). The weights are applied when a logistic regression, a forest, gradient boosting and a
Score's points are fitted (not Ranges' or a Score's cut-offs); with the grades weighing equally too, both apply.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

NORMALIZE = {"none": "None", "difference": "Difference from the project's median",
             "ratio": "Ratio to the project's median", "robust_z": "Robust z within the project"}
MIN_REFERENCE = 5           # objects with a value a project needs to be a reference
SMOTE_NEIGHBOURS = 5
WEIGHTS = {"none": "None", "sqrt": "√ objects", "equal": "Equal"}


@dataclass
class Prep:
    """How the objects are prepared before a classifier is made (see the module docstring).

    Attributes:
        normalize: A key of ``NORMALIZE``.
        noise_copies: Noisy copies of each training object (0: none).
        noise_sd: Their noise: SD × each parameter's SD over the training objects.
        smote: Classes with fewer objects topped up by SMOTE.
        acquisition: ``training.acquisition.Acquisition`` as a dict ({}: off).
        seed: Makes the augmentation repeatable.
        project_weights: A key of ``WEIGHTS``: how much each project's objects weigh together.
    """

    normalize: str = "none"
    noise_copies: int = 0
    noise_sd: float = 0.1
    smote: bool = False
    acquisition: dict = field(default_factory=dict)
    seed: int = 1
    project_weights: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Prep":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def augments(self) -> bool:
        return bool(self.noise_copies or self.smote or self.acquisition.get("copies"))

    def describe(self) -> str:
        """In words, e.g. "normalised: ratio to the project's median; augmented: 2 noisy copies (SD 0.1), SMOTE"."""
        parts = []
        if self.normalize != "none":
            parts.append(f"normalised per project: {NORMALIZE.get(self.normalize, self.normalize).lower()}")
        aug = []
        if self.acquisition.get("copies"):
            from modalytics.training.acquisition import Acquisition
            aug.append(Acquisition.from_dict(self.acquisition).describe())
        if self.noise_copies:
            aug.append(f"{self.noise_copies} noisy cop{'y' if self.noise_copies == 1 else 'ies'} of each object "
                       f"(SD {self.noise_sd:g} × each parameter's)")
        if self.smote:
            aug.append("SMOTE (smaller classes topped up)")
        if aug:
            parts.append("augmented: " + "; ".join(aug))
        if self.project_weights != "none":
            parts.append("projects weighted: " + ("each the same" if self.project_weights == "equal" else
                                                  "each ∝ √ its objects"))
        return "; ".join(parts) if parts else "none"


# ---------------------------------------------------------------- normalisation
def reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median, IQR) per column of one project's objects; NaN where fewer than ``MIN_REFERENCE`` have a value."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    enough = np.sum(np.isfinite(values), axis=0) >= MIN_REFERENCE   # none with no complete object at all
    med, iqr = np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    if enough.any():
        with np.errstate(all="ignore"):
            q1, q2, q3 = np.nanpercentile(values[:, enough], [25, 50, 75], axis=0)
        med[enough], iqr[enough] = q2, q3 - q1
    return med, iqr


def normalize_block(values: np.ndarray, method: str, complete: np.ndarray | None = None) -> np.ndarray:
    """One project's objects (objects, parameters) read relative to themselves (see the module docstring).

    Args:
        complete: (objects,) the objects the reference is made of (default: all).
    """
    values = np.asarray(values, float)
    if method == "none" or not len(values):
        return values.copy()
    med, iqr = reference(values if complete is None else values[np.asarray(complete, bool)])
    with np.errstate(all="ignore"):
        if method == "difference":
            out = values - med
        elif method == "ratio":
            out = values / np.where(med != 0, med, np.nan)
        elif method == "robust_z":
            out = (values - med) / np.where(iqr > 0, iqr, np.nan)
        else:
            raise ValueError(f"unknown normalisation: {method}")
    return out


def _blocks(data) -> np.ndarray:
    """Each object's reference block: its project — and, for acquisition copies, its copy (a project acquired
    again)."""
    variant = getattr(data, "variant", None)
    return data.project.astype(np.int64) * 1000 + (variant if variant is not None else 0)


def normalized(data, method: str):
    """A copy of a ``Dataset`` with every value normalised within its project (``normalize_block``)."""
    if method == "none":
        return data
    values = np.array(data.values, float)
    blocks = _blocks(data)
    partial = data.partial_rows if data.partial_rows is not None and len(data.partial_rows) == len(blocks) else         np.zeros(len(blocks), bool)
    for b in np.unique(blocks):
        rows = blocks == b
        values[rows] = normalize_block(data.values[rows], method, ~partial[rows])
    return replace(data, values=values)


def normalize_objects(rows: list[dict], uids: list[str], method: str,
                      complete: list[bool] | None = None) -> tuple[list[dict], str]:
    """One project's objects ({parameter id: value}) normalised for a classifier — when it is applied.

    Args:
        complete: Per object: its shape is complete (the reference; default: all).

    Returns:
        (the objects' values, normalised for ``uids``; why not, "" when they could be).
    """
    if method == "none" or not rows:
        return rows, ""
    values = np.array([[_num(r.get(u)) for u in uids] for r in rows], float)
    complete = np.ones(len(rows), bool) if complete is None else np.asarray(complete, bool)
    if complete.sum() < MIN_REFERENCE:
        return rows, (f"too few complete objects to normalise per project ({int(complete.sum())}; the classifier "
                      f"needs {MIN_REFERENCE} or more of the group)")
    out = normalize_block(values, method, complete)
    return [dict(r, **{u: float(out[i, j]) for j, u in enumerate(uids)}) for i, r in enumerate(rows)], ""


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


# ---------------------------------------------------------------- weighing projects
def project_weights(data, rows: np.ndarray, how: str) -> np.ndarray | None:
    """Each of ``rows``' objects' weight (their mean 1; None for "none"): a project's objects share its total —
    1 (``equal``) or √ its objects (``sqrt``)."""
    if how == "none":
        return None
    rows = np.asarray(rows, bool)
    w = np.zeros(len(rows))
    for p in np.unique(data.project[rows]):
        mine = rows & (data.project == p)
        n = int(mine.sum())
        w[mine] = (1.0 if how == "equal" else math.sqrt(n)) / n
    w[rows] *= rows.sum() / max(w[rows].sum(), 1e-12)
    return w[rows]


# ---------------------------------------------------------------- augmentation
def _concat(data, parts: list) -> tuple:
    """``data`` with rows added: parts = [(values, grade, project, source rows)]; (dataset, added-rows mask)."""
    if not parts:
        return data, np.zeros(len(data.grade), bool)
    values = np.vstack([data.values] + [p[0] for p in parts])
    grade = np.concatenate([data.grade] + [p[1] for p in parts])
    project = np.concatenate([data.project] + [p[2] for p in parts])
    src = np.concatenate([p[3] for p in parts])
    where = list(data.where) + [data.where[i] if i < len(data.where) else ("", "", 0, (math.nan, math.nan))
                                for i in src] if data.where else []
    partial = np.concatenate([data.partial_rows, data.partial_rows[src]]) \
        if data.partial_rows is not None and len(data.partial_rows) == len(data.grade) else None
    added = np.zeros(len(grade), bool)
    added[len(data.grade):] = True
    return replace(data, values=values, grade=grade, project=project, where=where, partial_rows=partial,
                   variant=None), added


def noise_copies(values: np.ndarray, copies: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """``copies`` noisy copies of each row (SD × ``sd`` of each column; NaN stays NaN), stacked copy by copy."""
    with np.errstate(all="ignore"):
        col_sd = np.nanstd(values, axis=0)
    col_sd = np.where(np.isfinite(col_sd), col_sd, 0.0)
    return np.vstack([values + rng.normal(0.0, 1.0, values.shape) * sd * col_sd for _c in range(copies)])


def smote(values: np.ndarray, grade: np.ndarray, rng: np.random.Generator,
          k: int = SMOTE_NEIGHBOURS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New objects for the smaller classes, until each has as many as the largest (see the module docstring).

    Returns:
        (values, grade, the source row of each new object).
    """
    counts = np.bincount(grade)
    top = counts.max() if len(counts) else 0
    with np.errstate(all="ignore"):
        med, sd = np.nanmedian(values, axis=0), np.nanstd(values, axis=0)
    filled = np.where(np.isfinite(values), values, med)                 # for the distances only
    z = (filled - med) / np.where(sd > 0, sd, 1.0)
    out_v, out_g, out_s = [], [], []
    for c, n in enumerate(counts):
        if n < 2 or n >= top:
            continue
        idx = np.flatnonzero(grade == c)
        d = ((z[idx, None, :] - z[None, idx, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d, np.inf)
        near = np.argsort(d, axis=1)[:, :min(k, n - 1)]
        for _m in range(top - n):
            i = int(rng.integers(n))
            j = int(near[i, int(rng.integers(near.shape[1]))])
            a, b = values[idx[i]], values[idx[j]]
            out_v.append(a + rng.random() * (b - a))                      # NaN where either is NaN
            out_g.append(c)
            out_s.append(idx[i])
    if not out_v:
        return np.zeros((0, values.shape[1])), np.zeros(0, int), np.zeros(0, int)
    return np.asarray(out_v), np.asarray(out_g, int), np.asarray(out_s, int)


def augmented(data, rows: np.ndarray, prep: Prep, acquired=None) -> tuple:
    """The rows to make a classifier from: ``rows`` of ``data``, and the augmented ones made from them.

    Args:
        data: The (normalised) ``Dataset``.
        rows: (objects,) the training objects.
        prep: The preparation (its augmentation; the normalisation is ``data``'s already).
        acquired: The training projects measured again (``acquisition.measure``, normalised the same way):
            its rows of ``rows``' projects are added.

    Returns:
        (dataset, rows): ``data`` with the augmented rows appended, and the rows to make the classifier from
        (``rows`` and every one added).
    """
    rows = np.asarray(rows, bool)
    if not prep.augments or not rows.any():
        return data, rows
    rng = np.random.default_rng(prep.seed)
    idx = np.flatnonzero(rows)
    parts = []
    # 1. Acquisition copies of the training projects (their own rows, of these projects only)
    if acquired is not None and prep.acquisition.get("copies") and len(acquired.grade):
        keep = np.isin(acquired.project, np.unique(data.project[idx]))
        if keep.any():
            parts.append((acquired.values[keep], acquired.grade[keep], acquired.project[keep],
                          np.full(int(keep.sum()), idx[0])))
    # 2. Noisy copies of the training objects
    if prep.noise_copies:
        v = noise_copies(data.values[idx], prep.noise_copies, prep.noise_sd, rng)
        parts.append((v, np.tile(data.grade[idx], prep.noise_copies), np.tile(data.project[idx], prep.noise_copies),
                      np.tile(idx, prep.noise_copies)))
    # 3. SMOTE: the smaller classes topped up from the training objects
    if prep.smote:
        v, g, s = smote(data.values[idx], data.grade[idx], rng)
        if len(g):
            parts.append((v, g, data.project[idx][s], idx[s]))
    out, added = _concat(data, parts)
    return out, np.concatenate([rows, np.zeros(len(out.grade) - len(rows), bool)]) | added
