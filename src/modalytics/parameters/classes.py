"""
The annotation classes around an object, for parameters that relate classes (Qt-free).

A class is a ``(group, subgroup)`` pair, as annotation tools and profiles name
them: ``("Gland", "Dysplasia high grade")``; an empty subgroup means the whole
group (``("Gland", "")`` = every gland, classified or not). Each polygon of a
class's masks is one object.

``ClassIndex`` gathers, for one measured image, the polygons of each class in
that image's local µm frame:

* from the image's own masks;
* if the image has none of that class, from the other images of the project
  that overlap it, moved through both images' placements (registration): stroma
  outlined on the slide is found when a gland is measured on the phase map.

Parameters reach it through the measuring contexts (``ctx.other``,
``ctx.others``, ``ctx.inside``, ``ctx.nearest``, ``ctx.area_of``,
``ctx.surroundings(..., only=...)``); one index is shared by every object of an
image, so each class is gathered once.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.strtree import STRtree

from modalytics.core.annotations import label_text
from modalytics.parameters import measures as ms

Label = tuple[str, str]


def as_label(value) -> Label:
    """A class from what a profile or script gives: ("Gland", "Normal"), ["Gland", ""], "Gland › Normal", "Gland"."""
    if isinstance(value, (tuple, list)):
        group = str(value[0]) if value else ""
        sub = str(value[1]) if len(value) > 1 and value[1] is not None else ""
        return group.strip(), sub.strip()
    text = str(value or "")
    for sep in ("›", ">", "/"):
        if sep in text:
            group, sub = text.split(sep, 1)
            return group.strip(), sub.strip()
    return text.strip(), ""


def label_name(label) -> str:
    """"Group › Subgroup" (or "Group") of a class."""
    return label_text(*as_label(label))


def matches(mask, label: Label) -> bool:
    """Whether a mask belongs to a class (a group alone matches all its subgroups)."""
    group, sub = label
    return mask.group == group and (not sub or mask.subgroup == sub)


class ClassIndex:
    """The annotation classes on (or registered onto) one image.

    Args:
        image: The measured ImageNode (its masks are its children).
        images: The project's images, to find classes outlined on another, registered image.
    """

    def __init__(self, image, images=()) -> None:
        self.image = image
        self.images = [i for i in images if i is not image]
        self._polygons: dict[Label, list[Polygon]] = {}
        self._union: dict[Label, object] = {}
        self._tree: dict[Label, STRtree] = {}

    def polygons(self, label) -> list[Polygon]:
        """The class's objects (polygons) in this image's local µm: its own masks, else registered ones."""
        label = as_label(label)
        if label not in self._polygons:
            own = [p for m in self.image.masks if matches(m, label) for p in ms.parts(m.geometry)]
            if not own:
                own = self._from_other_images(label)
            self._polygons[label] = [p for p in own if not p.is_empty]
        return self._polygons[label]

    def _from_other_images(self, label: Label) -> list[Polygon]:
        """The class's polygons on overlapping images, moved into this image's frame (first image that has any)."""
        frame = box(0, 0, *self.image.size_um)
        for other in self.images:
            masks = [m for m in other.masks if matches(m, label)]
            if not masks:
                continue
            m = np.linalg.inv(self.image.local_to_world()) @ other.local_to_world()      # other's µm → this one's
            coeffs = [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]]
            moved = [affinity.affine_transform(p, coeffs) for mask in masks for p in ms.parts(mask.geometry)]
            moved = [p for p in moved if p.intersects(frame)]
            if moved:
                return moved
        return []

    def geometry(self, label):
        """All of the class as one geometry (union of its objects; empty if none)."""
        label = as_label(label)
        if label not in self._union:
            polys = self.polygons(label)
            self._union[label] = shapely.union_all(polys) if polys else MultiPolygon()
        return self._union[label]

    def tree(self, label) -> STRtree:
        """A spatial index of the class's objects (``tree.query`` gives indices into ``polygons``)."""
        label = as_label(label)
        if label not in self._tree:
            self._tree[label] = STRtree(self.polygons(label))
        return self._tree[label]

    def near(self, label, geom, distance: float = 0.0) -> list[Polygon]:
        """The class's objects within ``distance`` µm of a geometry (touching or overlapping when 0)."""
        polys = self.polygons(label)
        if not polys:
            return []
        idx = self.tree(label).query(geom, predicate="dwithin", distance=distance) if distance > 0 \
            else self.tree(label).query(geom, predicate="intersects")
        return [polys[i] for i in idx]


def same_object(a: Polygon, b: Polygon) -> bool:
    """Whether two outlines are the same object (an object is not its own neighbour)."""
    return a.bounds == b.bounds and a.equals(b)
