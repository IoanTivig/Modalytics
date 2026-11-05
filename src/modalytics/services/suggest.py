"""
Model-assisted annotation: a model's outlines become suggestions, and the expert reviews them (Qt-free).

**Suggesting** (after Segment). Each object the segmentation model found is
compared with the objects of its group already drawn by hand on the image:

* it is (mostly) one of them — IoU ≥ 0.5: no suggestion, the drawing stays; but if
  that drawn object has no grade yet and the profile classifies the group, its
  grade is suggested (a *regrade* suggestion: the outline is the expert's own);
* otherwise it becomes a suggestion, with the grade the profile's classifier gives
  it (only the parameters the classifier reads are measured, on the annotation as
  it is — suggestions never count for one another), its score or confidence and
  the reason.

**Confidence** (``decide``). A classifier giving a probability (logistic regression,
random forest, gradient boosting) has its grade suggested only from the profile's
minimum confidence for the group (``Profile.confidence_for``); below it the object is
suggested as the plain group, its confidence and the reason said. Ranges and Score
give no probability: their grade is always suggested, to be checked. Graded or not,
a suggestion is reviewed before it enters the annotation — machine grades never
become training labels unseen. With classification off (Segment's *Classify when
segmenting*), objects are suggested ungraded.

**Classify selected** (``classify_objects``): drawn objects are graded the same way,
each a regrade suggestion when its confident grade differs from the one it has.

Suggestions are listed top to bottom, left to right, the order Next follows.

**Reviewing.** Each suggestion is judged once, outline and grade together:

* *accept* — it moves into the annotation with its suggested grade, or with the
  grade the expert chooses. Where it overlaps objects of its group already drawn,
  the drawing wins: the overlapping part is cut away (with a hair's breadth gap,
  so the two stay separate objects). A regrade suggestion moves the drawn object
  to the chosen grade unchanged;
* *delete* — it is dropped.

Every review step is one undo command (``core.commands.Command``).
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from modalytics.core import annotations as ann
from modalytics.core.commands import Command
from modalytics.core.model import MaskNode, Suggestion

MATCH_IOU = 0.5
GAP_UM = 0.05                 # between an accepted object and a drawn one it touched: stays a separate object


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if inter > 0 else 0.0


def _polys(geom) -> list:
    return [p for p in shapely.get_parts(geom) if isinstance(p, Polygon) and not p.is_empty] if geom is not None \
        else []


def classify(polys: list, image, images: list, profile, group: str, library=None,
             reference: list | tuple = ()) -> list[tuple[str, float, str]]:
    """The profile's grade for each outline: [(grade, score, reason)] — ("", NaN, "") without a classifier.

    A classifier made on values normalised per project reads each outline relative to the outlines given here and
    ``reference`` (other objects of the group on the image), its stand-in for the project's objects.
    """
    clf = profile.classifier_for(group) if profile is not None else None
    if clf is None or not polys:
        return [("", math.nan, "")] * len(polys)
    from modalytics.parameters import engine
    from modalytics.parameters.classes import ClassIndex
    from modalytics.parameters.registry import ParameterLibrary
    library = library or ParameterLibrary()
    wanted = {u: profile.parameters.get(u, {}) for u in clf.parameters}
    choices, _missing = engine.select(library, wanted, "object")
    classes = ClassIndex(image, images)
    shapes = list(polys) + (list(reference) if clf.normalize != "none" else [])
    measured = [engine.measure_object(poly, image, choices, images, classes, extras=False)[0] for poly in shapes]
    # Read as the classifier was made: relative to these objects (per-project normalisation, if any)
    from modalytics.training.classifiers import prepare_objects
    prepared, why = prepare_objects(clf, measured, [not image.partial_reason(p, profile.edge_share) for p in shapes])
    if why:
        return [("", math.nan, f"not graded: {why}")] * len(polys)
    return [clf.classify(values) for values in prepared[:len(polys)]]


def decide(profile, group: str, grade: str, score: float, why: str) -> tuple[str, str]:
    """The grade to suggest from a classifier's (grade, score, reason): its grade when it gives a probability at or
    above the profile's minimum confidence, else "" (the plain group); Ranges' and Score's always (see the module
    docstring). Returns (grade, reason)."""
    clf = profile.classifier_for(group) if profile is not None else None
    if clf is None or not grade:
        return grade, why
    if not clf.learned:
        return grade, f"{why} · no confidence ({clf.kind}): check the grade"
    threshold = profile.confidence_for(group)
    if math.isfinite(score) and score >= threshold:
        return grade, why
    return "", (f"unsure — {grade} with confidence {score:.2f}, below {threshold:.2f}: left as {group} · {why}")


def make(found, image, images: list, profile, group: str, library=None, cancelled=None,
         classify_on: bool = True) -> list[Suggestion]:
    """Suggestions from a model's objects on an image (see the module docstring).

    Args:
        found: The model's objects (geometry in the image's local µm).
        image: The image (its masks: the annotation).
        images: All the project's images (for parameters reading registered images).
        profile: The project's profile (its classifier for ``group``, if any).
        group: The class group the model outlines.
        classify_on: Grade them with the profile's classifier (``decide``); off: ungraded, and drawn objects are not
            regraded.

    Returns:
        list[Suggestion]: In reading order (top to bottom, left to right).
    """
    from modalytics.parameters.engine import objects_of
    drawn = [(m, p) for m in image.masks if m.group == group for p in objects_of(m.geometry)]
    grades = {s.name for g in (profile.classes.groups if profile is not None else []) if g.name == group
              for s in g.subgroups} if classify_on and profile is not None and \
        profile.classifier_for(group) is not None else set()
    tree = shapely.STRtree([p for _m, p in drawn]) if drawn else None
    new, regrade = [], {}
    for f in _polys(found):
        best, best_k = 0.0, -1
        for k in (tree.query(f) if tree is not None else []):
            iou = _iou(f, drawn[int(k)][1])
            if iou > best:
                best, best_k = iou, int(k)
        if best >= MATCH_IOU:
            m, p = drawn[best_k]
            if grades and m.subgroup not in grades:                # drawn, not graded: suggest its grade
                regrade[best_k] = p
            continue
        new.append(f)
    polys = new + list(regrade.values())
    if cancelled is not None and cancelled():
        return []
    graded = classify(polys, image, images, profile, group, library, reference=[p for _m, p in drawn]) \
        if classify_on else [("", math.nan, "")] * len(polys)
    out = []
    for i, (p, (g, s, why)) in enumerate(zip(polys, graded)):
        g, why = decide(profile, group, g, s, why) if classify_on else (g, why)
        if i >= len(new) and not g:
            continue                                            # a drawn object: nothing sure to suggest
        out.append(Suggestion(p, group, g, s, why, regrade=i >= len(new)))
    out.sort(key=lambda s: (round(s.geometry.representative_point().y / 50), s.geometry.representative_point().x))
    return out


def classify_objects(project, objects: list, library=None, cancelled=None) -> tuple[dict, list]:
    """Grade drawn objects with the profile's classifiers (Annotate › Classify selected).

    Args:
        project: The project (its profile).
        objects: [(mask, polygon)] — the objects selected.

    Returns:
        ({image: [regrade Suggestion]} — one where a sure grade differs from the object's, [(group, its grade now,
        the grade given ("" unsure), score, reason)] for every object of a classified group).
    """
    from modalytics.parameters.engine import objects_of
    profile = project.profile
    by_image: dict = {}
    for m, p in objects:
        if profile is not None and profile.classifier_for(m.group) is not None and m.parent is not None:
            by_image.setdefault((m.parent, m.group), []).append((m, p))
    out, said = {}, []
    for (image, group), items in by_image.items():
        if cancelled is not None and cancelled():
            break
        polys = [p for _m, p in items]
        others = [q for mk in image.masks if mk.group == group for q in objects_of(mk.geometry)
                  if not any(q.equals(p) for p in polys)]
        graded = classify(polys, image, list(project.images), profile, group, library, reference=others)
        for (m, p), (g, s, why) in zip(items, graded):
            g, why = decide(profile, group, g, s, why)
            said.append((group, m.subgroup, g, s, why))
            if g and g != m.subgroup:
                out.setdefault(image, []).append(Suggestion(p, group, g, s, why, regrade=True))
    return out, said


def add_command(project, image, items: list, source: dict | None = None) -> Command | None:
    """Add suggestions to an image's review layer (made if it has none), as one undo step; replaces suggestions
    for the same outlines."""
    if not items:
        return None
    from modalytics.core.model import SuggestionNode
    node = project.suggestions_for(image, create=False)
    created = node is None
    if created:
        node = SuggestionNode()
    old_items, old_source = list(node.items), dict(node.source)
    kept = [s for s in old_items if not any(_iou(s.geometry, it.geometry) > 0.99 for it in items)]
    new_items = kept + list(items)
    new_source = dict(old_source, **(source or {}))

    def do():
        if created and node.parent is None:
            node.parent = image
            image.children.append(node)
            project.structure_changed.emit()
        project.set_suggestions(node, new_items, new_source)

    def undo():
        project.set_suggestions(node, old_items, old_source)
        if created and node.parent is not None:
            project.remove(node)

    return Command(f"Suggest {len(items)} grade(s)", do, undo)


# ---------------------------------------------------------------- review
def _target(project, image, group: str, grade: str) -> tuple[MaskNode, bool]:
    """The class mask a suggestion goes into: (mask, created now)."""
    mask = project.mask_for(image, group, grade, create=False)
    if mask is not None:
        return mask, False
    return MaskNode(geometry=None, color=project.labels.color(group, grade), group=group, subgroup=grade), True


def accept_command(project, node, items: list, grade: str | None = None) -> Command:
    """Accept suggestions into the annotation (see the module docstring) — one undo step.

    Args:
        project: The project.
        node: The image's SuggestionNode.
        items: The suggestions to accept.
        grade: The grade to give them (None: each its suggested grade).
    """
    from modalytics.parameters.engine import objects_of
    image = node.parent
    geoms: dict = {}                    # mask → its new geometry
    created: list = []
    for it in items:
        g = it.subgroup if grade is None else grade
        known = project.profile is not None and (it.group, g) in project.profile.classes.choices()
        project.labels.ensure(it.group, g, project.profile.classes.color(it.group, g) if known else None)
        target, new = _target(project, image, it.group, g)
        if new and target not in created:
            # a second item for the same new class reuses the mask made for the first
            same = next((m for m in created if m.group == it.group and m.subgroup == g), None)
            target = same or target
            if same is None:
                created.append(target)
        current = geoms.get(target, target.geometry)
        if it.regrade:
            # the drawn object leaves its mask and joins the chosen grade's, unchanged
            source = next((m for m in image.masks if m.group == it.group
                           and any(_iou(p, it.geometry) > 0.99 for p in objects_of(geoms.get(m, m.geometry)))), None)
            if source is not None and source is not target:
                rest = [p for p in objects_of(geoms.get(source, source.geometry)) if _iou(p, it.geometry) <= 0.99]
                geoms[source] = ann.as_multipolygon(rest)
                current = geoms.get(target, target.geometry)
            geoms[target] = ann.as_multipolygon(objects_of(current) + [it.geometry])
            continue
        # the drawing wins: cut what overlaps objects of the group already there (with a hair's breadth gap)
        poly = it.geometry
        group_geom = [geoms.get(m, m.geometry) for m in image.masks if m.group == it.group] + \
            [geoms[m] for m in created if m in geoms]
        others = shapely.union_all([g_ for g_ in group_geom if not g_.is_empty]) if group_geom else None
        if others is not None and not others.is_empty and others.intersects(poly):
            poly = poly.difference(others.buffer(GAP_UM))
        parts = [p for p in _polys(poly) if p.area > 1.0]
        if parts:
            geoms[target], _ = ann.edit(current, ann.as_multipolygon(parts), "add")
    old = {m: m.geometry for m in geoms}
    ids = {it.id for it in items}
    before = list(node.items)
    after = [s for s in before if s.id not in ids]

    def do():
        for m in created:
            if m.parent is None and m in geoms:
                project.add_mask(image, m)
        for m, g_ in geoms.items():
            project.set_mask_geometry(m, g_)
        project.set_suggestions(node, after)

    def undo():
        for m, g_ in old.items():
            project.set_mask_geometry(m, g_)
        for m in created:
            if m.parent is not None:
                project.remove(m)
        project.set_suggestions(node, before)

    what = f"{len(items)} suggestions" if len(items) != 1 else "suggestion"
    return Command(f"Accept {what}" + (f" as {grade}" if grade else ""), do, undo)


def reject_command(project, node, items: list) -> Command:
    """Delete suggestions — one undo step."""
    ids = {it.id for it in items}
    before = list(node.items)
    after = [s for s in before if s.id not in ids]
    return Command(f"Delete {len(items)} suggestion{'s' if len(items) != 1 else ''}",
                   lambda: project.set_suggestions(node, after), lambda: project.set_suggestions(node, before))


def neighbour(node, step: int = 1):
    """The suggestion after (``step`` = 1) or before (−1) the current one, wrapping round; the first if none."""
    if not node.items:
        return None
    ids = [s.id for s in node.items]
    if node.current not in ids:
        return node.items[0 if step > 0 else -1]
    return node.items[(ids.index(node.current) + step) % len(ids)]


def counts(node) -> dict:
    """{suggested grade ("" = none): suggestions}."""
    out: dict = {}
    for s in node.items:
        out[s.subgroup] = out.get(s.subgroup, 0) + 1
    return out


def bounds_world(node, item) -> np.ndarray:
    """The suggestion's outline in world µm (for drawing it and bringing it into view)."""
    from modalytics.core import geometry as geo
    return geo.apply(node.parent.local_to_world(), np.asarray(item.geometry.exterior.coords))
