"""
Training sets: graded objects from annotated projects, with an evaluation part (Qt-free).

A **training set** says:

* which **profile** (its parameters are measured, its classes are the grades);
* which **group** is classified — the segmentation layer, e.g. "Gland"; its
  **subgroups**, in the profile's order, are the grades (the classification layer),
  e.g. Low grade dysplasia < High grade dysplasia. Older names are read through the
  profile's ``aliases`` (a HoloPath "Normal" gland counts as Low grade dysplasia);
* which **projects** (``.modalytics`` or HoloPath ``.hpro``) hold the graded objects;
* how the **evaluation** part is chosen: the projects you mark "Evaluation", or an
  automatic share of the projects. It is always whole projects (one sample each), never
  objects of one project on both sides: objects of one sample resemble each other,
  so splitting them would make the evaluation look better than it is;
* the **test groups**: projects marked "Test", each in a named group (e.g. "Naive",
  "External – Hospital B", "Setup 20×"). They are never trained on and never used to choose:
  each group is a final check of its own — most telling when it differs on purpose (another
  microscope setup, another lab), to see how far the classifier or model carries over.

``gather`` measures every graded object of the projects with the profile's object
parameters (one ``Dataset`` table) — the test projects apart (``test=True``), so that
nothing that builds or chooses a classifier can see them; measurements are cached per
project file and profile, so opening the set again is quick. Objects of the group without a grade
(e.g. outlined but not yet classified) are counted, not used.

A training set is a small JSON file (``.mtrain``) in your library's ``training`` folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from modalytics import fileio, paths
from modalytics.profiles.model import Profile

SUFFIX = ".mtrain"
# test: never trained on and never used to choose (an epoch, an architecture): only for the final, reported numbers
ROLES = ("training", "evaluation", "test")
TEST = "Test"                          # the name of a test group not named
MODES = ("projects", "share")          # evaluation: the projects marked so, or an automatic share of them


class Cancelled(Exception):
    """Gathering was cancelled."""


@dataclass
class ProjectEntry:
    """One project of a training set: its file, its role ("training", "evaluation" or "test") and, for a test
    project, its test group ("" = ``TEST``)."""

    path: str
    role: str = "training"
    group: str = ""

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def test_group(self) -> str:
        """The test group's name ("" for a project that is not a test project)."""
        return (self.group or TEST) if self.role == "test" else ""


@dataclass
class TrainingSet:
    """What to learn from (see the module docstring).

    Attributes:
        name: Shown in the Training section.
        profile: The profile (a copy, kept with the set, so it measures the same way later).
        profile_id: Its id in the library (for showing where it came from).
        group: The class group whose objects are classified (e.g. "Gland").
        projects: The annotated projects and their roles.
        mode: "projects" (the projects marked Evaluation) or "share" (chosen automatically).
        share: The share of projects held out in "share" mode (0–0.9).
        seed: Makes the automatic choice repeatable.
        built: {"ranges" | "score": classifier dict} — the classifiers being built (Training › Build).
        segmentation: How segmentation models are judged on these projects: {"group", "model" ("" = the
            profile's), "projects" ("evaluation" | "all"), "area" ("image" | "hull"), "margin", "iou"}.
        runs_folder: Its folder of training runs (``training.runs``; claimed at the first run).
        include_partial: Objects whose shape is incomplete (cut by the image edge, or marked partial) count too, for
            the parameters that do not need the whole object (``ParameterSpec.whole_object``: their values for the
            others are left empty); off: left out altogether.
        choose_by: How Build's report chooses the classifier: "cross-validation" (on the training projects; the
            evaluation projects then a final check too) or "evaluation" (on the evaluation projects).
    """

    name: str
    profile: Profile
    profile_id: str = ""
    group: str = ""
    projects: list[ProjectEntry] = field(default_factory=list)
    mode: str = "projects"
    share: float = 0.2
    seed: int = 1
    built: dict = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)
    runs_folder: str = ""
    include_partial: bool = False
    choose_by: str = "cross-validation"

    @property
    def grades(self) -> list[str]:
        """The grades: the group's subgroups in the profile's order."""
        g = next((g for g in self.profile.classes.groups if g.name == self.group), None)
        return [s.name for s in g.subgroups] if g else []

    def to_dict(self) -> dict:
        return {"format": "modalytics-training-set", "format_version": 1, "name": self.name,
                "profile_id": self.profile_id, "profile": self.profile.to_dict(), "group": self.group,
                "projects": [{"path": p.path, "role": p.role, **({"group": p.group} if p.test_group and p.group
                                                                   else {})} for p in self.projects],
                "evaluation": {"mode": self.mode, "share": self.share, "seed": self.seed},
                "built": self.built, "segmentation": self.segmentation,
                **({"runs_folder": self.runs_folder} if self.runs_folder else {}),
                **({"include_partial": True} if self.include_partial else {}),
                **({"choose_by": self.choose_by} if self.choose_by != "cross-validation" else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSet":
        if d.get("format") != "modalytics-training-set":
            raise ValueError("not a Modalytics training set")
        ev = d.get("evaluation") or {}
        return cls(str(d.get("name", "")), Profile.from_dict(d["profile"]), str(d.get("profile_id", "")),
                   str(d.get("group", "")),
                   [ProjectEntry(str(p["path"]), str(p.get("role", "training")), str(p.get("group", "")))
                    for p in d.get("projects", [])],
                   str(ev.get("mode", "projects")), float(ev.get("share", 0.2)), int(ev.get("seed", 1)),
                   dict(d.get("built") or {}), dict(d.get("segmentation") or {}), str(d.get("runs_folder", "")),
                   bool(d.get("include_partial", False)), str(d.get("choose_by", "cross-validation")))

    def save(self, path: str | os.PathLike) -> None:
        fileio.write_text(path, json.dumps(self.to_dict(), indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TrainingSet":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def library_folder():
    """Where your training sets are kept."""
    return paths.user_dir("training")


# ---------------------------------------------------------------- the table of graded objects
@dataclass
class Dataset:
    """Graded objects of a training set, one row each.

    Attributes:
        columns: [(parameter id, label, unit, help)] of the profile's object parameters.
        values: (objects, parameters) float array (NaN where not computed).
        grade: (objects,) index into ``grades``.
        grades: The grade names.
        project: (objects,) index into ``projects``.
        projects: The projects' names.
        where: Per object: (image name, mask class, object number, a point inside it (x, y) in its image's µm
            frame) — to find it again (the point still finds it after older class names were merged).
        ungraded: {project name: objects of the group without a grade}.
        problems: {project name: why it could not be read}.
        partial: {project name: graded objects left out because their shape is incomplete (cut by the image edge,
            or marked partial) — their shape parameters would mislead the classifier}.
        ordered: The grades are a scale (in their order); False: categories with no order (the profile says).
        partial_rows: (objects,) True for an object whose shape is incomplete, included (``include_partial``): its
            values for the parameters that need the whole object are empty.
        variant: (objects,) for objects measured again as if acquired differently (``training.acquisition``):
            which copy (each copy of a project is normalised on its own, like another project); None otherwise.
    """

    columns: list
    values: np.ndarray
    grade: np.ndarray
    grades: list
    project: np.ndarray
    projects: list
    where: list = field(default_factory=list)
    ungraded: dict = field(default_factory=dict)
    problems: dict = field(default_factory=dict)
    partial: dict = field(default_factory=dict)
    ordered: bool = True
    partial_rows: np.ndarray | None = None
    variant: np.ndarray | None = None

    def counts(self, mask: np.ndarray | None = None) -> dict:
        """{grade: objects} (of the rows in ``mask``)."""
        g = self.grade if mask is None else self.grade[mask]
        return {name: int(np.sum(g == i)) for i, name in enumerate(self.grades)}

    def per_project(self) -> dict:
        """{project name: {grade: objects}}."""
        out = {name: {g: 0 for g in self.grades} for name in self.projects}
        for p, g in zip(self.project, self.grade):
            out[self.projects[p]][self.grades[g]] += 1
        return out


def _open(path: str):
    """The images (with their masks) of a project file."""
    if path.lower().endswith(".hpro"):
        from modalytics.io.importers import import_hpro
        return import_hpro(path)
    from modalytics.io.project_file import load_project
    return load_project(path).images


CACHE_VERSION = 4          # 2: each object's point is kept; 3: partial objects left out; 4: measured too, marked


def _cache_key(path: str, profile: Profile, group: str) -> str:
    st = os.stat(path)
    what = json.dumps([CACHE_VERSION, os.path.abspath(path), st.st_size, st.st_mtime_ns, profile.parameters, group,
                       profile.aliases, [(g.name, [s.name for s in g.subgroups]) for g in profile.classes.groups],
                       profile.edge_share],
                      sort_keys=True, default=list)
    return hashlib.sha1(what.encode()).hexdigest()[:24]


def gather(tset: TrainingSet, progress: Callable[[float], None] | None = None,
           cancelled: Callable[[], bool] | None = None, library=None, use_cache: bool = True,
           test: bool = False) -> Dataset:
    """Measure every graded object of the training set's projects (see the module docstring).

    Args:
        test: Measure the test projects only (for the final checks); default: every project but them. Either way
            ``project`` indexes ``tset.projects``.

    Raises:
        Cancelled: If ``cancelled`` returned True.
        ValueError: If the group has no grades in the profile.
    """
    from modalytics.parameters import engine
    from modalytics.parameters.classes import ClassIndex
    from modalytics.parameters.registry import ParameterLibrary
    library = library or ParameterLibrary()
    grades = tset.grades
    if len(grades) < 2:
        raise ValueError(f"'{tset.group}' needs at least two subgroups (grades) in the profile")
    from modalytics.parameters.registry import distinct_labels
    choices, _missing = engine.select(library, tset.profile.parameters, "object")
    labels = distinct_labels(s for s, _c in choices)                       # same name, other image type
    columns = [(s.uid, labels[s.uid], s.unit, s.help_text()) for s, _c in choices]
    uids = [c[0] for c in columns]
    cache_dir = library_folder() / "cache"
    rows, grade, project, where, partial_rows = [], [], [], [], []
    whole = [s.whole_object for s, _c in choices]          # left empty on a partial object (``include_partial``)
    ungraded, problems, partial = {}, {}, {}
    for pi, entry in enumerate(tset.projects):
        if cancelled is not None and cancelled():
            raise Cancelled()
        if (entry.role == "test") != test:          # test projects apart: final checks only
            continue
        name = entry.name
        try:
            key = _cache_key(entry.path, tset.profile, tset.group)
        except OSError as exc:
            problems[name] = f"not found ({exc.strerror or exc})"
            continue
        cached = cache_dir / f"{key}.json"
        if use_cache and cached.exists():
            with open(cached, encoding="utf-8") as fh:
                got = json.load(fh)
        else:
            # 1. The project's graded objects of the group, measured with the training profile
            try:
                images = _open(entry.path)
            except Exception as exc:                        # unreadable file: reported, the others go on
                problems[name] = str(exc)
                continue
            got = {"rows": [], "ungraded": 0, "partial": 0}
            objects = [(img, m, i, poly) for img in images for m in img.masks if m.group == tset.group
                       for i, poly in enumerate(engine.objects_of(m.geometry))]
            for k, (img, m, i, poly) in enumerate(objects):
                if cancelled is not None and cancelled():
                    raise Cancelled()
                _g, sub = tset.profile.label_for(m.group, m.subgroup)
                if sub not in grades:
                    got["ungraded"] += 1
                    continue
                cut = img.partial_reason(poly, tset.profile.edge_share)   # shape incomplete: measured, marked
                values, _problems = engine.measure_object(poly, img, choices, images, ClassIndex(img, images),
                                                          extras=False)
                got["rows"].append({"grade": sub, "image": img.name, "class": f"{m.group} › {m.subgroup}",
                                    "index": i, "point": list(poly.representative_point().coords[0]),
                                    "values": [_num(values.get(u)) for u in uids],
                                    **({"partial": cut} if cut else {})})
                if progress is not None:
                    progress((pi + (k + 1) / max(len(objects), 1)) / max(len(tset.projects), 1))
            cache_dir.mkdir(parents=True, exist_ok=True)
            fileio.write_text(cached, json.dumps(got))
        left_out = 0
        for r in got["rows"]:
            if r["grade"] not in grades:
                continue
            values = [math.nan if v is None else v for v in r["values"]]
            if r.get("partial"):                         # shape incomplete: left out, or only where it may count
                if not tset.include_partial:
                    left_out += 1
                    continue
                values = [math.nan if w else v for v, w in zip(values, whole)]
            partial_rows.append(bool(r.get("partial")))
            rows.append(values)
            grade.append(grades.index(r["grade"]))
            project.append(pi)
            where.append((r["image"], r["class"], r["index"] + 1, tuple(r.get("point") or (math.nan, math.nan))))
        if got["ungraded"]:
            ungraded[name] = got["ungraded"]
        if left_out:
            partial[name] = left_out
        if progress is not None:
            progress((pi + 1) / max(len(tset.projects), 1))
    values = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(uids))
    return Dataset(columns, values, np.asarray(grade, int), grades, np.asarray(project, int),
                   [p.name for p in tset.projects], where, ungraded, problems, partial,
                   tset.profile.classes.ordered(tset.group), np.asarray(partial_rows, bool))


# ---------------------------------------------------------------- what the projects hold (no measuring)
@dataclass
class ProjectOverview:
    """What a project holds for the set's object group, read from its masks (nothing measured).

    Attributes:
        images: {image type: number of images}.
        um_per_px: {image type: the distinct pixel sizes (µm/px), sorted}.
        outlined: Images with at least one object of the group drawn.
        fully: Images whose annotation is marked complete for the group (everything not drawn is not the group).
        objects: Objects of the group.
        grades: {grade: objects} (older subgroup names read as the profile says).
        ungraded: Objects of the group without one of the grades.
        partial: Objects of the group whose shape is incomplete (cut by the image edge, or marked partial).
        unreviewed: Unreviewed suggestions of the group that may be objects nobody outlined yet.
        problem: Why the project could not be read ("" if it could).
    """

    images: dict = field(default_factory=dict)
    um_per_px: dict = field(default_factory=dict)
    outlined: int = 0
    fully: int = 0
    objects: int = 0
    grades: dict = field(default_factory=dict)
    ungraded: int = 0
    partial: int = 0
    unreviewed: int = 0
    problem: str = ""


_OVERVIEWS: dict = {}                     # {(path, modified, group, grades, aliases): ProjectOverview}


def project_overview(path: str, profile: Profile, group: str) -> ProjectOverview:
    """What one project holds for ``group`` (see ``ProjectOverview``); remembered per file and modification time."""
    from modalytics.parameters.engine import objects_of
    from modalytics.training.segmentation import pending
    g = next((g for g in profile.classes.groups if g.name == group), None)
    grades = [s.name for s in g.subgroups] if g else []
    try:
        key = (os.path.abspath(path), os.path.getmtime(path), group, tuple(grades),
               json.dumps(profile.aliases.get(group, {}), sort_keys=True), profile.edge_share)
    except OSError as exc:
        return ProjectOverview(problem=f"not found ({exc.strerror or exc})")
    if key in _OVERVIEWS:
        return _OVERVIEWS[key]
    try:
        images = _open(path)
    except Exception as exc:                                        # unreadable: said, the others go on
        return ProjectOverview(problem=str(exc))
    out = ProjectOverview(grades={gr: 0 for gr in grades})
    sizes: dict[str, set] = {}
    for img in images:
        out.images[img.modality] = out.images.get(img.modality, 0) + 1
        sizes.setdefault(img.modality, set()).add(float(f"{float(np.mean(img.pixel_size_um)):.4g}"))
        n = 0
        for m in img.masks:
            if m.group != group:
                continue
            _g, sub = profile.label_for(m.group, m.subgroup)
            polys = objects_of(m.geometry)
            k = len(polys)
            n += k
            out.partial += sum(1 for poly in polys if img.partial_reason(poly, profile.edge_share))
            if sub in out.grades:
                out.grades[sub] += k
            else:
                out.ungraded += k
        out.objects += n
        out.outlined += n > 0
        out.fully += bool(img.fully_outlined(group))
        out.unreviewed += len(pending(img, group))
    out.um_per_px = {t: sorted(v) for t, v in sizes.items()}
    _OVERVIEWS[key] = out
    return out


def forget_overviews(paths: list[str]) -> None:
    """Read these projects again next time (``project_overview``), even if their files look unchanged."""
    gone = {os.path.abspath(p) for p in paths}
    for key in [k for k in _OVERVIEWS if k[0] in gone]:
        del _OVERVIEWS[key]


SLIP_TOLERANCE = 1.25        # a pixel size within ×1.25 of 10×, 100×, 1000×… another one: a slip, not a magnification


def size_key(um_per_px: float) -> float:
    """A pixel size rounded to 4 significant digits (sizes of one system read alike)."""
    return float(f"{um_per_px:.4g}")


def unit_slips(sizes: dict) -> dict:
    """The pixel sizes that look like a decimal-point or unit slip of a more common one.

    Different pixel sizes are normal (several systems and magnifications: 5× and 20× differ 4×; each image is
    resampled to the model's resolution). Only a size about a power of ten (10×, 100×, 1000×… within
    ``SLIP_TOLERANCE``) away from a size more images have looks wrongly stored: 3.14 for 0.314 µm/px, nm read as
    µm. (A real 10× step, e.g. 4× and 40× objectives, is asked about too: one look confirms it.)

    Args:
        sizes: {pixel size (µm/px): number of images}.

    Returns:
        {the doubtful size: the more common size it is a power of ten away from}.
    """
    out = {}
    for u, n in sizes.items():
        for other, m in sorted(sizes.items(), key=lambda kv: -kv[1]):
            if m <= n or not (u > 0 and other > 0):
                continue
            decades = abs(math.log10(u / other))
            if round(decades) >= 1 and abs(decades - round(decades)) <= math.log10(SLIP_TOLERANCE):
                out[u] = other
                break
    return out


def resolution_outliers(overviews: dict) -> dict:
    """{project key: why} for projects with a pixel size that looks wrongly stored (``unit_slips``: a power of ten
    from a size more images of that type have, e.g. 3.14 stored where 0.314 µm/px was meant)."""
    by_type: dict[str, dict[float, int]] = {}
    for ov in overviews.values():
        for t, values in ov.um_per_px.items():
            for v in values:
                counts = by_type.setdefault(t, {})
                counts[size_key(v)] = counts.get(size_key(v), 0) + 1
    slips = {t: unit_slips(c) for t, c in by_type.items()}
    out = {}
    for k, ov in overviews.items():
        for t, values in ov.um_per_px.items():
            far = [v for v in values if size_key(v) in slips[t]]
            if far:
                base = slips[t][size_key(far[0])]
                out[k] = (f"{far[0]:g} µm/px for {t}, {far[0] / base:.3g}× the {base:g} µm/px of "
                          f"{by_type[t][base]} other image(s) — a power of ten: is it stored right (a decimal point "
                          f"or unit slip)?")
    return out


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


# ---------------------------------------------------------------- training / evaluation
def evaluation_projects(tset: TrainingSet, data: Dataset | None = None) -> set[int]:
    """The indices of the projects held out for evaluation.

    "projects" mode: those marked Evaluation. "share" mode: projects drawn at random (repeatably, from
    ``seed``) until about ``share`` of the graded objects are held out, preferring a choice that leaves
    every grade on both sides.
    """
    if tset.mode == "projects":
        return {i for i, p in enumerate(tset.projects) if p.role == "evaluation"}
    n = len(tset.projects)
    tests = test_projects(tset)
    if n - len(tests) < 2 or tset.share <= 0:
        return set()
    if data is not None and len(data.grade):
        per = np.zeros((n, len(data.grades)), int)
        for p, g in zip(data.project, data.grade):
            per[p, g] += 1
    else:
        per = np.ones((n, 1), int)
    per[sorted(tests)] = 0                                  # test projects are never drawn
    total = per.sum()
    target = tset.share * total
    rng = np.random.default_rng(tset.seed)
    best, best_score = set(), math.inf
    for _ in range(200):                                    # many random orders: keep the best
        order = [p for p in rng.permutation(n) if p not in tests]
        chosen, held = [], 0
        for p in order:
            more = per[p].sum()
            if len(chosen) >= n - len(tests) - 1 or (chosen and abs(held + more - target) >= abs(held - target)):
                break                                       # one more project would move away from the share
            chosen.append(int(p))
            held += more
        ev = per[chosen].sum(axis=0)
        tr = per.sum(axis=0) - ev
        missing = int(np.sum((ev == 0) & (per.sum(axis=0) > 0))) + int(np.sum((tr == 0) & (per.sum(axis=0) > 0)))
        score = missing * total + abs(held - target)        # every grade on both sides first, then the share
        if score < best_score:
            best, best_score = set(chosen), score
    return best


def test_projects(tset: TrainingSet) -> set[int]:
    """The indices of the test projects: never trained on, never used to choose; only for the final numbers."""
    return {i for i, p in enumerate(tset.projects) if p.role == "test"}


def test_groups(tset: TrainingSet) -> dict[str, set[int]]:
    """{test group: the indices of its projects}, in the order the groups first appear."""
    out: dict[str, set[int]] = {}
    for i, p in enumerate(tset.projects):
        if p.test_group:
            out.setdefault(p.test_group, set()).add(i)
    return out


def training_projects(tset: TrainingSet, data: Dataset | None = None) -> set[int]:
    """The indices of the projects trained on: neither evaluation nor test."""
    held = evaluation_projects(tset, data) | test_projects(tset)
    return {i for i in range(len(tset.projects)) if i not in held}


def split(tset: TrainingSet, data: Dataset) -> np.ndarray:
    """(objects,) True for the objects held out for evaluation."""
    held = evaluation_projects(tset, data)
    return np.isin(data.project, sorted(held))


def training_rows(tset: TrainingSet, data: Dataset) -> np.ndarray:
    """(objects,) True for the objects trained on: the training projects' (not the evaluation nor the test
    projects' — test projects are never used to build or choose)."""
    return np.isin(data.project, sorted(training_projects(tset, data)))
