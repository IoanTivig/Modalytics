"""
Vector annotation geometry and the label schema (groups → subgroups).

Masks are **vector only** (decision 2026-10-16): a mask is a shapely
``MultiPolygon`` (polygons with holes) in its parent image's local µm frame.
Pixel masks are only ever *derived* (rasterised on demand for ML or
statistics), never stored.

Each connected polygon of a mask is one object (e.g. one gland), which is what
measurements and classification will later work on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

EMPTY = MultiPolygon()


## Geometry helpers ##

def as_multipolygon(obj) -> MultiPolygon:
    """Convert anything polygonal to a valid ``MultiPolygon``.

    Args:
        obj: A shapely geometry, a list of (N, 2) outer rings, or None.

    Returns:
        MultiPolygon: Valid, possibly empty. Self-intersections are repaired
        (``make_valid``); lines and points produced by the repair are dropped.
    """
    if obj is None:
        return EMPTY
    if not isinstance(obj, BaseGeometry):
        items = list(obj)
        if items and all(isinstance(i, BaseGeometry) for i in items):
            obj = shapely.GeometryCollection(items)          # a list of shapely polygons
        else:
            rings = [np.asarray(r, dtype=float) for r in items]  # a list of (N, 2) rings
            obj = MultiPolygon([Polygon(r) for r in rings if len(r) >= 3])
    if not obj.is_valid:
        obj = shapely.make_valid(obj)
    polys = _polygons_of(obj)
    return MultiPolygon(polys) if polys else EMPTY


def _polygons_of(geom: BaseGeometry) -> list[Polygon]:
    """All non-empty polygons contained in a geometry (recursing into collections)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_polygons_of(g))
        return out
    return []


def transform(mp: MultiPolygon, matrix: np.ndarray) -> MultiPolygon:
    """Apply a 3×3 affine (column-vector convention) to a MultiPolygon."""
    a, b, xoff = matrix[0]
    d, e, yoff = matrix[1]
    return as_multipolygon(affinity.affine_transform(mp, [a, b, d, e, xoff, yoff]))


def rings(mp: MultiPolygon) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Return (exterior, [holes]) coordinate arrays for every polygon (closing point removed)."""
    out = []
    for poly in mp.geoms:
        ext = np.asarray(poly.exterior.coords)[:-1]
        holes = [np.asarray(h.coords)[:-1] for h in poly.interiors]
        out.append((ext, holes))
    return out


def stroke_polygon(points: np.ndarray, radius: float, radii=None) -> MultiPolygon:
    """Area covered by a round brush of ``radius`` moved along ``points``.

    With ``radii`` (one per point, e.g. from a pen's pressure) the width varies along the stroke: the area is the
    union of the convex hulls of each two consecutive dabs (a tapered capsule between them).

    Args:
        points: (N, 2) path in local µm (one point = a single dab).
        radius: Brush radius in µm.
        radii: Optional radius of each point (µm).

    Returns:
        MultiPolygon: The painted area.
    """
    pts = np.asarray(points, dtype=float)
    if radii is not None and len(pts) > 1 and np.ptp(np.asarray(radii, float)) > 1e-9:
        dabs = [shapely.Point(p).buffer(float(r), quad_segs=16) for p, r in zip(pts, radii)]
        pieces = [shapely.union(a, b).convex_hull for a, b in zip(dabs, dabs[1:])]
        return as_multipolygon(shapely.union_all(pieces))
    if radii is not None and len(radii):
        radius = float(np.asarray(radii, float).max())
    geom = shapely.Point(pts[0]) if len(pts) == 1 else shapely.LineString(pts)
    # 32 segments per quarter circle keeps edges smooth at high zoom
    return as_multipolygon(geom.buffer(radius, quad_segs=32))


def edit(mp: MultiPolygon, shape: MultiPolygon, op: str, min_area: float = 0.0) -> tuple[MultiPolygon, bool]:
    """Add ("add") or cut ("cut") a shape, touching only the polygons it meets.

    A brush stroke usually meets a handful of the mask's objects. A spatial
    index (STRtree) finds them; only those are merged or cut, every other
    polygon is kept as is. On a mask of 8 000 objects this is the difference
    between ~2 s (whole-mask overlay) and a few ms.

    Args:
        mp: Current mask geometry (valid).
        shape: Shape to add or cut (valid).
        op: "add" or "cut".
        min_area: Drop resulting pieces smaller than this (µm²).

    Returns:
        tuple: (new geometry, changed?)
    """
    parts = shapely.get_parts(mp)
    hit = shapely.STRtree(parts).query(shape, predicate="intersects") if len(parts) else np.array([], int)
    if op == "cut" and len(hit) == 0:
        return mp, False
    touched = shapely.union_all(parts[hit]) if len(hit) else EMPTY
    result = touched.union(shape) if op == "add" else touched.difference(shape)
    new_parts = [p for p in _polygons_of(result) if p.area >= min_area]
    kept = np.delete(parts, hit) if len(hit) else parts
    out = list(kept) + new_parts
    return (MultiPolygon(out) if out else EMPTY), True


def remove_specks(mp: MultiPolygon, min_area: float) -> MultiPolygon:
    """Drop polygons smaller than ``min_area`` µm² (tiny leftovers of erasing)."""
    return as_multipolygon([p for p in mp.geoms if p.area >= min_area]) if min_area > 0 else mp


## Label schema ##

@dataclass
class Label:
    """A subgroup (class) inside a group, e.g. Gland › Dysplasia high grade."""

    name: str
    color: tuple[float, float, float, float]


@dataclass
class Group:
    """An annotation group (what is segmented, e.g. Gland) with its subgroups.

    ``ordered``: the subgroups are grades on a scale, in their order (e.g. low < high grade dysplasia); else
    categories with no order (e.g. four pathologies) — statistics and classifiers then compare them as such.
    """

    name: str
    color: tuple[float, float, float, float]
    subgroups: list[Label] = field(default_factory=list)
    ordered: bool = True


@dataclass
class LabelSchema:
    """The groups and subgroups available in a project.

    Will come from the tissue profile (Settings); the default reproduces the
    colonic classes used in HoloPath.
    """

    groups: list[Group] = field(default_factory=list)

    def choices(self) -> list[tuple[str, str]]:
        """Every group (as (group, "") = not yet classified) followed by its subgroups."""
        out = []
        for g in self.groups:
            out += [(g.name, "")] + [(g.name, s.name) for s in g.subgroups]
        return out

    def color(self, group: str, subgroup: str = "") -> tuple[float, float, float, float]:
        """Colour of a subgroup, else of its group, else neutral grey."""
        for g in self.groups:
            if g.name == group:
                for s in g.subgroups:
                    if s.name == subgroup:
                        return s.color
                return g.color
        return (0.7, 0.7, 0.7, 1.0)

    def ordered(self, group: str) -> bool:
        """Whether the group's subgroups are ordered grades (True also for a group not in the schema)."""
        g = next((g for g in self.groups if g.name == group), None)
        return g.ordered if g is not None else True

    def ensure(self, group: str, subgroup: str = "", color=None) -> None:
        """Add a group/subgroup if it is not in the schema yet (e.g. from an imported project)."""
        if not group:
            return
        g = next((g for g in self.groups if g.name == group), None)
        if g is None:
            g = Group(group, tuple(color or (0.7, 0.7, 0.7, 1.0)), [])
            self.groups.append(g)
        if subgroup and not any(s.name == subgroup for s in g.subgroups):
            g.subgroups.append(Label(subgroup, tuple(color or g.color)))

    def to_dict(self) -> dict:
        return {"groups": [dict({"name": g.name, "color": list(g.color),
                                 "subgroups": [{"name": s.name, "color": list(s.color)} for s in g.subgroups]},
                                **({} if g.ordered else {"ordered": False})) for g in self.groups]}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSchema":
        return cls([Group(g["name"], tuple(g["color"]),
                          [Label(s["name"], tuple(s["color"])) for s in g.get("subgroups", [])],
                          bool(g.get("ordered", True)))
                    for g in d.get("groups", [])])


def default_schema() -> LabelSchema:
    """Colonic default (HoloPath classes)."""
    return LabelSchema([
        Group("Gland", (0.20, 0.80, 0.40, 1.0), [
            Label("Normal", (0.20, 0.80, 0.40, 1.0)),
            Label("Dysplasia low grade", (0.98, 0.80, 0.15, 1.0)),
            Label("Dysplasia high grade", (0.92, 0.25, 0.25, 1.0)),
        ]),
        Group("Lamina propria", (0.30, 0.45, 0.95, 1.0), [Label("Normal", (0.30, 0.45, 0.95, 1.0))]),
        Group("Region", (0.75, 0.45, 0.90, 1.0), []),
    ])


_ALIASES = {  # HoloPath (Romanian) class names → schema names
    "glande": "Gland", "glanda": "Gland", "lamina": "Lamina propria",
    "displasia low grade": "Dysplasia low grade", "displasia high grade": "Dysplasia high grade",
    "normal": "Normal",
}


def canonical(name: str) -> str:
    """Map a known alias (HoloPath's Romanian names) to the schema name."""
    return _ALIASES.get(name.strip().lower(), name.strip())


def label_text(group: str, subgroup: str) -> str:
    """Display name "Group › Subgroup" (or just the group)."""
    return f"{group} › {subgroup}" if subgroup else group
