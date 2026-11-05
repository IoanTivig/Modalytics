"""
Measuring a whole project with its profile, and the results (Qt-free).

``measure_project(project)`` measures every image of the project with the
parameters its profile chose, and gives three kinds of rows:

* **object** rows: one per object, i.e. per polygon of each mask (each separate
  outline of a class is one object), with the profile's object parameters — on an
  object whose shape is incomplete (cut by the image edge, or marked partial) those
  that need the whole object (``ParameterSpec.whole_object``: shape, totals) are left
  empty, with the reason;
* **class** rows: one per class present on each image — each group as a whole
  (e.g. "Gland": all glands) and each of its subgroups (e.g. "Gland › Normal") —
  with the profile's per-class parameters (counts, densities, summaries);
* **image** rows: one per image, with the profile's whole-image parameters.

When several versions of an image type are included in the parameters (Layers ›
Processed), each other version adds a column after its parameter's, labelled with
its name ("Mean phase · Denoised"); images without that version leave it empty.

A value that could not be computed is NaN, with the reason in the row's
``problems`` (shown when hovering the empty cell).

The results remember what they were measured on (``signature``: each mask's
outline, each image's pixel size and views, the profile's parameter choices), so
``outdated`` can tell when masks were edited, images added or removed, or the
profile changed since: the values are then those of an earlier state.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import shapely

from modalytics.core.annotations import label_text
from modalytics.parameters import engine
from modalytics.parameters.classes import ClassIndex
from modalytics.parameters.registry import ParameterLibrary

SCOPE_ORDER = ("object", "class", "image")


class Cancelled(Exception):
    """Measuring was cancelled (nothing is kept)."""


@dataclass
class Row:
    """One row of results.

    Attributes:
        scope: "object", "class" or "image".
        image_id, image_name: The image measured.
        group, subgroup: The class (object and class rows; subgroup "" = the whole group for class rows).
        mask_id: The mask holding the object (object rows).
        index: The object's position among the mask's polygons (object rows; 0-based).
        values: {parameter id: value} (NaN = not computed).
        problems: {parameter id: why it was not computed}.
        prediction: Object rows, once classified: {"grade", "score", "reason"} — the classifier's grade,
            kept apart from the class the object was annotated with.
        suggested: An object (or class row) measured from unreviewed suggestions (Measure › Include suggestions).
        partial: Object rows: why its shape is incomplete ("" if it is not; ``ImageNode.partial_reason``).
    """

    scope: str
    image_id: str
    image_name: str
    group: str = ""
    subgroup: str = ""
    mask_id: str = ""
    index: int = -1
    values: dict = field(default_factory=dict)
    problems: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)
    suggested: bool = False
    partial: str = ""

    @property
    def class_name(self) -> str:
        """"Group › Subgroup", or the group alone."""
        return label_text(self.group, self.subgroup)

    def to_dict(self) -> dict:
        return {"scope": self.scope, "image_id": self.image_id, "image_name": self.image_name, "group": self.group,
                "subgroup": self.subgroup, "mask_id": self.mask_id, "index": self.index,
                "values": {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
                           for k, v in self.values.items()},
                "problems": self.problems, **({"prediction": self.prediction} if self.prediction else {}),
                **({"suggested": True} if self.suggested else {}), **({"partial": self.partial} if self.partial else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "Row":
        values = {k: (math.nan if v is None else float(v)) for k, v in d.get("values", {}).items()}
        return cls(d["scope"], d["image_id"], d.get("image_name", ""), d.get("group", ""), d.get("subgroup", ""),
                   d.get("mask_id", ""), int(d.get("index", -1)), values, dict(d.get("problems", {})),
                   dict(d.get("prediction") or {}), bool(d.get("suggested", False)), str(d.get("partial", "")))


@dataclass
class Column:
    """A measured parameter as the results show it (kept, so results read even if the library changes)."""

    uid: str
    label: str
    unit: str = ""
    help: str = ""

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


@dataclass
class Results:
    """The results of measuring a project (see the module docstring).

    Attributes:
        columns: {scope: [Column]} in the library's order.
        rows: All rows (object, class and image).
        measured: When (ISO time).
        profile_name: The profile used.
        signature: What was measured (for ``outdated``).
        seconds: How long it took.
        classified: {group: what classified its objects (the classifier's kind and note)}.
    """

    columns: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    measured: str = ""
    profile_name: str = ""
    signature: dict = field(default_factory=dict)
    seconds: float = 0.0
    classified: dict = field(default_factory=dict)

    def of_scope(self, scope: str) -> list[Row]:
        return [r for r in self.rows if r.scope == scope]

    def counts(self) -> dict:
        return {s: len(self.of_scope(s)) for s in SCOPE_ORDER}

    @property
    def suggested(self) -> int:
        """Objects measured from unreviewed suggestions (0 unless Measure › Include suggestions was on)."""
        return sum(1 for r in self.of_scope("object") if r.suggested)

    def outdated(self, project) -> list[str]:
        """Why the results no longer match the project (empty: they do)."""
        now = signature(project)
        before = self.signature or {}
        reasons = []
        old_images, new_images = before.get("images", {}), now["images"]
        names = {i.id: i.name for i in project.images}
        for iid in new_images.keys() - old_images.keys():
            reasons.append(f"{names.get(iid, iid)} was added")
        if old_images.keys() - new_images.keys():
            reasons.append(f"{len(old_images.keys() - new_images.keys())} image(s) were removed")
        for iid in new_images.keys() & old_images.keys():
            if new_images[iid] != old_images[iid]:
                reasons.append(f"{names.get(iid, iid)}: pixel size or views changed")
        flat = lambda masks: {m: d for per_image in masks.values() for m, d in per_image.items()}  # noqa: E731
        old_masks, new_masks = flat(before.get("masks", {})), flat(now["masks"])
        changed = {m for m in old_masks.keys() | new_masks.keys() if old_masks.get(m) != new_masks.get(m)}
        if changed:
            reasons.append(f"{len(changed)} mask(s) were edited, added or removed")
        if before.get("profile") != now["profile"]:
            reasons.append("the profile's parameters changed")
        return reasons

    def merged(self, newer: "Results", image_ids, order=None) -> "Results":
        """These results with some images measured again (``newer``, from ``measure_project(images=…)``).

        The other images keep their rows and what they were measured on, so they still show as out of date
        if they changed since. If the columns differ (the profile changed), only ``newer`` is kept.

        Args:
            newer: The new measurement of ``image_ids``.
            image_ids: The images measured again.
            order: Image ids in the project's order (rows are sorted by it).
        """
        ids = set(image_ids)
        if newer.columns != self.columns:
            return newer
        rows = [r for r in self.rows if r.image_id not in ids] + list(newer.rows)
        if order:
            pos = {iid: i for i, iid in enumerate(order)}
            rows.sort(key=lambda r: (pos.get(r.image_id, len(pos)), SCOPE_ORDER.index(r.scope)))
        sig = {"images": dict(self.signature.get("images", {})), "masks": dict(self.signature.get("masks", {})),
               "profile": newer.signature.get("profile")}
        for iid in ids:
            for part in ("images", "masks"):
                if iid in newer.signature.get(part, {}):
                    sig[part][iid] = newer.signature[part][iid]
                else:
                    sig[part].pop(iid, None)
        return Results(newer.columns, rows, newer.measured, newer.profile_name, sig, newer.seconds)
        # (predictions are made again after measuring: Classify)

    def to_dict(self) -> dict:
        return {"columns": {s: [vars(c) for c in cols] for s, cols in self.columns.items()},
                "rows": [r.to_dict() for r in self.rows], "measured": self.measured,
                "profile_name": self.profile_name, "signature": self.signature, "seconds": self.seconds,
                "classified": self.classified}

    @classmethod
    def from_dict(cls, d: dict) -> "Results":
        return cls({s: [Column(**c) for c in cols] for s, cols in d.get("columns", {}).items()},
                   [Row.from_dict(r) for r in d.get("rows", [])], d.get("measured", ""),
                   d.get("profile_name", ""), d.get("signature", {}), float(d.get("seconds", 0.0)),
                   dict(d.get("classified") or {}))


# ---------------------------------------------------------------- what was measured
def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def signature(project) -> dict:
    """What a measurement depends on: each mask's outline and class (per image), each image's pixel size and
    measured views, the profile's parameter choices."""
    images, masks = {}, {}
    for img in project.images:
        views = [json.dumps(v.chain, sort_keys=True) for v in img.views if v.measure]
        images[img.id] = _digest(json.dumps([list(img.pixel_size_um), views, img.original_measured]).encode())
        masks[img.id] = {m.id: _digest(shapely.to_wkb(m.geometry) + m.group.encode() + b"|" + m.subgroup.encode())
                         for m in img.masks}
    params = project.profile.parameters if project.profile is not None else {}
    return {"images": images, "masks": masks,
            "profile": _digest(json.dumps(params, sort_keys=True, default=list).encode())}


# ---------------------------------------------------------------- measuring
def class_labels(image) -> list[tuple[str, str]]:
    """The classes of an image's class rows: each group with objects, then each of its subgroups with objects."""
    out: list[tuple[str, str]] = []
    for m in image.masks:
        if m.geometry.is_empty:
            continue
        if (m.group, "") not in out:
            out.append((m.group, ""))
    for group, _ in list(out):
        subs = []
        for m in image.masks:
            if m.group == group and m.subgroup and not m.geometry.is_empty and m.subgroup not in subs:
                subs.append(m.subgroup)
        out += [(group, s) for s in subs]
    return sorted(out, key=lambda lab: (out.index((lab[0], "")), lab[1]))


def measure_project(project, library: ParameterLibrary | None = None,
                    progress: Callable[[float], None] | None = None,
                    cancelled: Callable[[], bool] | None = None, images=None,
                    include_suggestions: bool = False) -> Results:
    """Measure the project's images with its profile's parameters.

    Args:
        project: The Project (its ``profile.parameters`` say what to measure).
        library: The parameter library (default: all here).
        progress: Called with 0–1 as it goes.
        cancelled: Polled between objects; True stops (raises ``Cancelled``).
        images: Only these images (default: all).
        include_suggestions: Measure the images' unreviewed suggestions too, as if accepted with their suggested
            grade (their rows are marked ``suggested``); by default only the annotation is measured.

    Returns:
        Results: Object, class and image rows (see the module docstring).

    Raises:
        Cancelled: If ``cancelled`` returned True.
    """
    import time
    t0 = time.perf_counter()
    library = library or ParameterLibrary()
    chosen = project.profile.parameters if project.profile is not None else library.defaults()
    choices = {s: engine.select(library, chosen, s)[0] for s in SCOPE_ORDER}
    whole = {spec.uid for spec, _c in choices["object"] if spec.whole_object}     # left empty on partial objects
    from modalytics.parameters.registry import distinct_labels
    columns = {}
    for s in SCOPE_ORDER:
        labels = distinct_labels(spec for spec, _c in choices[s])           # same name, other image type
        columns[s] = [Column(spec.uid, labels[spec.uid], spec.unit, spec.help_text()) for spec, _c in choices[s]]
    images = list(images) if images is not None else list(project.images)
    every = list(project.images)
    from modalytics.core.model import EDGE_SHARE
    share = project.profile.edge_share if project.profile is not None else EDGE_SHARE   # cut by the image edge
    temporary: set = set()
    if include_suggestions:                                  # stand-ins whose masks also hold the suggestions
        twins = {img.id: with_suggestions(img, temporary) for img in every}
        every = [twins[img.id] for img in every]
        images = [twins.get(img.id, img) for img in images]
    # 1. How much work: one unit per object, per class row and per image
    work = sum(len(engine.objects_of(m.geometry)) for img in images for m in img.masks) \
        + sum(len(class_labels(img)) + 1 for img in images)
    done = 0

    def step():
        nonlocal done
        done += 1
        if cancelled is not None and cancelled():
            raise Cancelled()
        if progress is not None:
            progress(min(1.0, done / max(work, 1)))

    rows: list[Row] = []
    for img in images:
        classes = ClassIndex(img, every)                 # gathered once per image, shared by its rows
        cache: dict = {}                                  # object values, reused by the class summaries
        # 2. Objects: each polygon of each mask (a partial one: what needs the whole object left empty)
        for m in img.masks:
            for i, poly in enumerate(engine.objects_of(m.geometry)):
                values, problems = engine.measure_object(poly, img, choices["object"], every, classes) \
                    if choices["object"] else ({}, {})
                partial = img.partial_reason(poly, share)
                shown, said = values, problems
                if partial and whole:
                    shown, said = dict(values), dict(problems)
                    for uid in values:
                        if engine.split_uid(uid)[0] in whole:
                            shown[uid] = math.nan
                            said[uid] = f"needs the whole object: this one is incomplete ({partial})"
                rows.append(Row("object", img.id, img.name, m.group, m.subgroup, m.id, i, shown, said,
                                suggested=m.id in temporary, partial=partial))
                key = shapely.to_wkb(poly)
                for spec, constants in choices["object"]:
                    cache[(spec.uid, _frozen(constants), key)] = values.get(spec.uid, math.nan)
                step()
        # 3. Classes: each group, and each of its subgroups
        for label in class_labels(img):
            values, problems = engine.measure_class(img, label, choices["class"], every, classes, library,
                                                    object_constants=chosen, value_cache=cache) \
                if choices["class"] else ({}, {})
            rows.append(Row("class", img.id, img.name, label[0], label[1], values=values, problems=problems))
            step()
        # 4. The whole image
        values, problems = engine.measure_image(img, choices["image"], every, classes) \
            if choices["image"] else ({}, {})
        rows.append(Row("image", img.id, img.name, values=values, problems=problems))
        step()
    # 5. The other versions' results: a labelled column after their parameter's
    for s in SCOPE_ORDER:
        extra = sorted({u for r in rows if r.scope == s for u in r.values if engine.VERSION in u})
        if not extra:
            continue
        cols = []
        for col in columns[s]:
            cols.append(col)
            for uid in extra:
                base, version = engine.split_uid(uid)
                if base == col.uid:
                    cols.append(Column(uid, f"{col.label} · {version}", col.unit, col.help))
        columns[s] = cols
    # 6. Every row has every column of its kind (NaN with the reason where it was not computed)
    for row in rows:
        for col in columns[row.scope]:
            if col.uid not in row.values:
                row.values[col.uid] = math.nan
                version = engine.split_uid(col.uid)[1]
                if version:
                    row.problems.setdefault(col.uid, f"{row.image_name} has no “{version}” version included in "
                                                     f"the parameters")
    return Results(columns, rows, _dt.datetime.now().isoformat(timespec="seconds"),
                   project.profile.name if project.profile is not None else "", signature(project),
                   time.perf_counter() - t0)


def with_suggestions(image, temporary: set):
    """A stand-in of an image whose masks also hold its suggestions, as if accepted (for measuring them).

    The image itself is not changed (it may be drawn meanwhile). Grade-only suggestions are left out: their object
    is already one of the masks. The stand-in masks' ids are added to ``temporary``.
    """
    import copy
    from modalytics.core.model import MaskNode
    node = image.suggestions
    items = [it for it in (node.items if node is not None else []) if not it.regrade]
    if not items:
        return image
    twin = copy.copy(image)
    twin.children = []
    for m in image.masks:
        clone = copy.copy(m)
        clone.parent = twin
        twin.children.append(clone)
    by_class: dict = {}
    for it in items:
        by_class.setdefault((it.group, it.subgroup), []).append(it.geometry)
    for (group, sub), polys in by_class.items():
        temp = MaskNode(geometry=polys, group=group, subgroup=sub)
        temp.parent = twin
        twin.children.append(temp)
        temporary.add(temp.id)
    return twin


def _frozen(constants: dict) -> tuple:
    """Constants as a hashable key, the same as ``ClassContext.object_values`` makes."""
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in constants.items()))


# ---------------------------------------------------------------- export
FIXED = {"object": ["Image", "Class", "Group", "Subgroup", "Object", "Partial"],
         "class": ["Image", "Class", "Group", "Subgroup"], "image": ["Image"]}


PREDICTED = ["Predicted", "Score", "Matches annotation"]       # object rows, once classified


def fixed_columns(results: "Results | None", scope: str) -> list[str]:
    """The identifying columns of a table: with the prediction's once the objects were classified."""
    return FIXED[scope] + (PREDICTED if scope == "object" and results is not None and results.classified else [])


def fixed_values(row: Row, predicted: bool = False) -> list:
    """The identifying cells of a row (before its parameter values); ``predicted``: with the prediction's."""
    if row.scope == "object":
        cells = [row.image_name, row.class_name + (" (suggested)" if row.suggested else ""), row.group,
                 row.subgroup, row.index + 1, row.partial]
        if predicted:
            p = row.prediction
            grade = p.get("grade", "")
            matches = "" if not (grade and row.subgroup) or row.partial else ("yes" if grade == row.subgroup else "no")
            cells += [grade, p.get("score", "") if grade else "", matches]
        return cells
    if row.scope == "class":
        return [row.image_name, row.class_name, row.group, row.subgroup]
    return [row.image_name]


def export_csv(results: Results, scope: str, path: str) -> int:
    """Write one kind of rows as CSV (UTF-8 with a byte-order mark, so spreadsheet programs read µm² right).

    Returns:
        int: Rows written.
    """
    cols = results.columns.get(scope, [])
    rows = results.of_scope(scope)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        fixed = fixed_columns(results, scope)
        w.writerow(fixed + [c.title for c in cols])
        for r in rows:
            vals = [r.values.get(c.uid, math.nan) for c in cols]
            w.writerow(fixed_values(r, len(fixed) > len(FIXED[scope]))
                       + ["" if (v is None or math.isnan(v)) else f"{float(v):.10g}" for v in vals])
    return len(rows)


# ---------------------------------------------------------------- classifying
def classify(results: Results, classifier, library: ParameterLibrary | None = None) -> tuple["Results", list[str]]:
    """The results with each object of the classifier's group given a predicted grade (its ``prediction``).

    An object whose shape is incomplete is graded only when none of the classifier's parameters needs the whole
    object; otherwise it is left ungraded (grade ""), with the reason.

    Args:
        results: Measured results.
        classifier: A ``training.classifiers.Classifier``.

    Returns:
        tuple: (new Results, problems — e.g. parameters the classifier reads that were not measured).
    """
    import copy
    out = copy.deepcopy(results)
    measured = {c.uid for c in out.columns.get("object", [])}
    missing = [u for u in classifier.parameters if u not in measured]
    problems = [f"the classifier reads {len(missing)} parameter(s) the profile does not measure: "
                f"{', '.join(missing)} (tick them in the profile, then Measure)"] if missing else []
    library = library or ParameterLibrary()
    labels = {c.uid: c.label for c in out.columns.get("object", [])}
    whole = [u for u in classifier.parameters if (s := library.specs.get(u)) is not None and s.whole_object]
    n = 0
    # The group's objects of the project, read as the classifier was made (per-project normalisation, if any)
    mine = [row for row in out.of_scope("object") if row.group == classifier.group]
    from modalytics.training.classifiers import prepare_objects
    prepared, why = prepare_objects(classifier, [row.values for row in mine], [not row.partial for row in mine])
    if why:
        problems.append(f"{classifier.group}: not graded — {why}")
        for row in mine:
            row.prediction = {"grade": "", "score": math.nan, "reason": f"not graded: {why}"}
        mine = []
    for row, values in zip(mine, prepared):
        if row.partial and whole:                        # an incomplete shape, read by the classifier: not graded
            row.prediction = {"grade": "", "score": math.nan,
                              "reason": f"not graded: its shape is incomplete ({row.partial}) and the classifier "
                                        f"reads {', '.join(labels.get(u, u) for u in whole)}, which need the whole "
                                        f"object"}
            continue
        grade, score, reason = classifier.classify(values)
        if row.partial:
            reason = f"shape incomplete ({row.partial}), but none of its parameters needs the whole object · {reason}"
        row.prediction = {"grade": grade, "score": score, "reason": reason}
        n += 1
    out.classified[classifier.group] = f"{classifier.kind}: {classifier.note}".strip(": ")
    return out, problems
