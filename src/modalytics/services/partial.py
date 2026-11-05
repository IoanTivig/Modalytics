"""
Partial objects: glands whose shape is incomplete, so their shape parameters mislead (Qt-free).

An object is partial when the image's border cuts it (found automatically, ``ImageNode.edge_cut``) or when it
was marked so by hand — cut by the section plane, torn, folded, out of focus — stored as a point inside it
(``ImageNode.partial_points``). Partial objects are drawn and segmented like any other; only what depends on their
shape leaves them out: classifier training (``training.dataset.gather``), the report's shape statistics, and a
classified partial object is said to have an incomplete shape.
"""

from __future__ import annotations


def points_after(image, polys: list, on: bool) -> list[tuple[float, float]]:
    """The image's partial points once ``polys`` are marked (``on``) or unmarked."""
    from shapely.geometry import Point
    pts = list(image.partial_points)
    if on:
        for poly in polys:
            if not image.marked_partial(poly):
                p = poly.representative_point()                 # always inside the object
                pts.append((p.x, p.y))
        return pts
    return [p for p in pts if not any(poly.covers(Point(p)) for poly in polys)]


def set_partial(project, objects: list, on: bool):
    """Mark (or unmark) the selected objects partial, as one undoable command.

    Args:
        project: The Project.
        objects: [(mask, polygon)] — the selection.
        on: Mark (True) or unmark.

    Returns:
        A ``core.commands.Command`` to push, or None when nothing changes.
    """
    from modalytics.core.commands import Command
    by_image: dict = {}
    for mask, poly in objects:
        by_image.setdefault(mask.parent, []).append(poly)
    changes = []
    for image, polys in by_image.items():
        before, after = list(image.partial_points), points_after(image, polys, on)
        if before != after:
            changes.append((image, before, after))
    if not changes:
        return None
    n = sum(len(p) for p in by_image.values())

    def apply(which):
        for image, before, after in changes:
            project.set_partial_points(image, after if which else before)
    return Command(f"{'Mark' if on else 'Unmark'} {n} object(s) partial", lambda: apply(True), lambda: apply(False))
