"""
Changing the class of objects: single glands, or a whole class layer.

An object is one polygon of a class mask (``MaskNode``: one mask per class and image).
Giving it another class moves it, unchanged, to that class's mask on the same image
(made when the image has none yet). Where it overlaps an object already in that class
it is cut back by a hair's breadth, so the two stay separate objects (as drawing keeps
touching glands apart). One undo step for all the objects, on every image.

``rename_to_profile`` gives an image's masks the profile's current class names (its ``aliases``: e.g. HoloPath's
*Normal* → *Low grade dysplasia*) when a project, a HoloPath project or the demo is opened.
"""

from __future__ import annotations

import shapely

from modalytics.core import annotations as ann
from modalytics.core.commands import Command
from modalytics.core.model import MaskNode

GAP_UM = 0.05


def class_choices(project) -> list[tuple[str, str]]:
    """The classes objects can be given: the profile's (group, then its subgroups), else the project's."""
    labels = project.profile.classes if project.profile is not None else project.labels
    return labels.choices()


def class_color(project, group: str, subgroup: str):
    """A class's colour: the profile's, else the project's."""
    labels = project.profile.classes if project.profile is not None else project.labels
    return labels.color(group, subgroup)


def _parts(geom) -> list:
    return [p for p in shapely.get_parts(geom) if isinstance(p, shapely.Polygon) and not p.is_empty]


def change_class(project, objects: list, group: str, subgroup: str):
    """Move objects [(mask, polygon)] to the class (group, subgroup) on their own image.

    Returns:
        tuple: (Command or None when nothing changes, the moved objects as [(mask, polygon)] once done — to
        select them).
    """
    by_image: dict = {}
    for m, p in objects:
        if m.parent is not None and (m.group, m.subgroup) != (group, subgroup):
            by_image.setdefault(m.parent, []).append((m, p))
    if not by_image:
        return None, []
    geoms: dict = {}                     # mask → its new geometry
    created: list = []                   # (image, new mask)
    moved_after: list = []
    for image, items in by_image.items():
        target = project.mask_for(image, group, subgroup, create=False)
        if target is None:
            target = MaskNode(geometry=None, color=class_color(project, group, subgroup), group=group,
                              subgroup=subgroup)
            created.append((image, target))
        for m, p in items:
            # 1. Out of its mask
            rest = [q for q in _parts(geoms.get(m, m.geometry)) if not q.equals(p)]
            geoms[m] = ann.as_multipolygon(rest)
            # 2. Into the class's mask, a separate object (cut back where it overlaps one there)
            there = _parts(geoms.get(target, target.geometry))
            hit = [q for q in there if q.intersects(p)]
            piece = p.difference(shapely.union_all(hit).buffer(GAP_UM)) if hit else p
            pieces = [q for q in _parts(piece) if q.area > 0.5]
            geoms[target] = ann.as_multipolygon(there + pieces)
            moved_after += [(target, q) for q in pieces]
    old = {m: m.geometry for m in geoms}

    def do():
        for image, m in created:
            if m.parent is None:
                project.add_mask(image, m)
        for m, g in geoms.items():
            project.set_mask_geometry(m, g)

    def undo():
        for m, g in old.items():
            project.set_mask_geometry(m, g)
        for _image, m in created:
            if m.parent is not None:
                project.remove(m)

    n = sum(len(v) for v in by_image.values())
    what = "object" if n == 1 else f"{n} objects"
    return Command(f"Class of {what}: {ann.label_text(group, subgroup)}", do, undo), moved_after


def change_layer_class(project, mask: MaskNode, group: str, subgroup: str) -> Command | None:
    """Give a whole class layer (all its objects) another class. When the image already has that class, its
    objects join that layer (kept separate) and this one goes; otherwise the layer is relabelled."""
    image = mask.parent
    if image is None or (mask.group, mask.subgroup) == (group, subgroup):
        return None
    if project.mask_for(image, group, subgroup, create=False) is not None:
        cmd, _moved = change_class(project, [(mask, p) for p in _parts(mask.geometry)], group, subgroup)
        if cmd is None:
            return None
        index = image.children.index(mask)

        def do():
            cmd.do()
            project.remove(mask)

        def undo():
            mask.parent = image
            image.children.insert(min(index, len(image.children)), mask)
            project.structure_changed.emit()
            cmd.undo()
        return Command(f"Class of {mask.name}: {ann.label_text(group, subgroup)}", do, undo)
    before = (mask.group, mask.subgroup, mask.color, mask.name)
    color = class_color(project, group, subgroup)

    def undo_relabel():
        project.relabel_mask(mask, before[0], before[1], before[2])
        mask.name = before[3]
        project.node_changed.emit(mask, "name")
    return Command(f"Class of {mask.name}: {ann.label_text(group, subgroup)}",
                   lambda: project.relabel_mask(mask, group, subgroup, color), undo_relabel)


def delete_objects(project, objects: list) -> Command | None:
    """Delete single objects [(mask, polygon)] (their masks stay, even if empty) — one undo step."""
    geoms: dict = {}
    for m, p in objects:
        geoms[m] = ann.as_multipolygon([q for q in _parts(geoms.get(m, m.geometry)) if not q.equals(p)])
    if not geoms:
        return None
    old = {m: m.geometry for m in geoms}
    n = len(objects)
    return Command(f"Delete {'object' if n == 1 else f'{n} objects'}",
                   lambda: [project.set_mask_geometry(m, g) for m, g in geoms.items()],
                   lambda: [project.set_mask_geometry(m, g) for m, g in old.items()])


def rename_to_profile(image, profile) -> None:
    """Give an image's masks the profile's current class names (``Profile.aliases``); masks that end up in the
    same class are merged into one (their objects stay separate polygons)."""
    by_class: dict = {}
    for m in list(image.masks):
        m.group, m.subgroup = profile.label_for(m.group, m.subgroup)
        m.name = ann.label_text(m.group, m.subgroup)
        m.color = profile.classes.color(m.group, m.subgroup)
        first = by_class.setdefault((m.group, m.subgroup), m)
        if first is not m:                              # same class as an earlier mask: one mask
            first.geometry = ann.as_multipolygon(shapely.union_all([first.geometry, m.geometry]))
            image.children.remove(m)
