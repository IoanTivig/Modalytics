"""
Edit points tool: reshape one object (polygon) of a mask by its vertices.

    Click an object              activate it: its vertices show as handles
    Click a point                select it (and only it)
    Ctrl/Shift + click a point   add it to / remove it from the selection
    Drag on empty space          rubber-band: select the points inside (Ctrl/Shift adds)
    Drag a selected point        move all selected points together
    Click an edge                insert a point there and drag it
    Delete / right-click         delete the selected points (or the one clicked)
    Arrow keys                   nudge the selection 1 screen px (Shift: 10)
    Ctrl+A                       select every point of the object
    S                            simplify the object (tolerance in Tool Settings)
    Esc                          clear the selection, then leave the object

An outline always keeps at least 3 points; a hole that would drop below 3
disappears. Every change is one undo step and the result is repaired to a
valid polygon (a self-crossing drag splits it cleanly).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from modalytics.core import annotations as ann
from modalytics.core import geometry as geo
from modalytics.core.commands import set_geometry_command
from modalytics.core.model import MaskNode
from modalytics.tools.base import Setting, Tool
from modalytics.tools.select import node_at

LEFT, RIGHT = 1, 2
HIT_VERTEX_PX = 7
HIT_EDGE_PX = 6
DRAG_START_PX = 4           # movement before a press on empty space becomes a rubber band


def _rings_of(poly: Polygon) -> list[np.ndarray]:
    """Exterior then holes, as (N, 2) arrays without the closing point."""
    return [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in poly.interiors]


def _segment_distance(p, a, b) -> tuple[float, np.ndarray]:
    """Distance from point p to segment ab and the closest point."""
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    q = a + t * ab
    return float(np.hypot(*(p - q))), q


def delete_vertices(rings: list[np.ndarray], selected: set[tuple[int, int]]) -> list[np.ndarray] | None:
    """Remove selected vertices from rings.

    The exterior keeps at least 3 points (if the deletion would go below,
    nothing is deleted from it); a hole that would go below 3 is removed.

    Args:
        rings: Exterior then holes.
        selected: {(ring, vertex)} to delete.

    Returns:
        list | None: New rings, or None if nothing could be deleted.
    """
    out, changed = [], False
    for i, r in enumerate(rings):
        drop = sorted(j for (k, j) in selected if k == i)
        if not drop:
            out.append(r)
            continue
        kept = np.delete(r, drop, axis=0)
        if len(kept) >= 3:
            out.append(kept)
            changed = True
        elif i == 0:
            out.append(r)                 # an object keeps its outline; delete objects with the eraser
        else:
            changed = True                # the hole disappears
    return out if changed else None


class EditPointsTool(Tool):
    """Select, move, insert and delete the vertices of one mask object."""

    id = "edit_points"
    uses_shift_click = True        # Shift adds points to the selection
    label = "Edit points"
    icon = "mdi6.vector-polyline-edit"
    shortcut = "A"
    tabs = ("Annotate",)
    group = "Tools"
    hint = ("Click an object, then click or box-select its points (Ctrl/Shift adds) and drag them. Click an "
            "edge to add a point; Delete or right-click removes the selection; arrows nudge; Ctrl+A selects "
            "all; S simplifies; Esc clears, then leaves.")
    settings = [Setting("tolerance", "Simplify tolerance", "float", 0.5, 0.05, 50.0, 0.1, " µm",
                        tooltip="Maximum deviation allowed when pressing S")]

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.mask: MaskNode | None = None
        self.index = -1                          # which polygon of the mask's MultiPolygon
        self.rings: list[np.ndarray] = []        # editable copy, local µm
        self.selected: set[tuple[int, int]] = set()
        self.hot: tuple[int, int] | None = None
        # Gesture in progress: {"kind": "move" | "box", "screen": press position, …}
        self._drag: dict | None = None
        self._geom = None                        # the mask geometry the editable copy was taken from

    # ------------------------------------------------------------ helpers
    def _m(self):
        return self.mask.parent.local_to_world()

    def _world(self, pts):
        return geo.apply(self._m(), pts)

    def _local(self, wx, wy):
        return geo.apply(np.linalg.inv(self._m()), (wx, wy))

    def _show(self, ctx) -> None:
        if self.mask is None:
            ctx.preview.clear()
        else:
            parts = []
            for r in self.rings:
                w = self._world(r)
                parts += [w, w[:1], np.full((1, 2), np.nan)]
            ctx.preview.set_path(np.vstack(parts))
            hot = None if self.hot is None else self._world(self.rings[self.hot[0]][self.hot[1]])
            sel = [self._world(self.rings[i][j]) for i, j in sorted(self.selected)] or None
            ctx.preview.set_handles(np.vstack([self._world(r) for r in self.rings]), hot, sel)
        ctx.request_draw()

    def _leave(self, ctx) -> None:
        self.mask, self.index, self.rings, self.hot, self._drag = None, -1, [], None, None
        self.selected = set()
        ctx.preview.set_box()
        self._show(ctx)

    def _screen(self, ctx, ring: np.ndarray) -> np.ndarray:
        w = self._world(ring)
        return np.array([ctx.view.world_to_screen(*p) for p in w])

    def _pick_vertex(self, ctx, ev):
        for i, r in enumerate(self.rings):
            s = self._screen(ctx, r)
            d = np.hypot(s[:, 0] - ev["x"], s[:, 1] - ev["y"])
            j = int(np.argmin(d))
            if d[j] <= HIT_VERTEX_PX:
                return i, j
        return None

    def _pick_edge(self, ctx, ev):
        p = np.array([ev["x"], ev["y"]], float)
        best = None
        for i, r in enumerate(self.rings):
            s = self._screen(ctx, r)
            for j in range(len(s)):
                d, _q = _segment_distance(p, s[j], s[(j + 1) % len(s)])
                if d <= HIT_EDGE_PX and (best is None or d < best[0]):
                    best = (d, i, j)
        return None if best is None else best[1:]

    def _activate_object(self, ctx, mask: MaskNode, wx, wy) -> bool:
        lx, ly = geo.apply(np.linalg.inv(mask.parent.local_to_world()), (wx, wy))
        pt = shapely.Point(lx, ly)
        for k, poly in enumerate(mask.geometry.geoms):
            if poly.contains(pt):
                self.mask, self.index, self.rings, self.selected = mask, k, _rings_of(poly), set()
                self._geom = mask.geometry
                ctx.project.select(mask)
                self._show(ctx)
                return True
        return False

    def _commit(self, ctx, rings, text, keep_selection: bool = False) -> None:
        """Replace the active polygon by one built from ``rings``, as one undo step."""
        mask = self.mask
        polys = list(mask.geometry.geoms)
        edited = ann.as_multipolygon(Polygon(rings[0], rings[1:]))
        others = polys[:self.index] + polys[self.index + 1:]
        out = others + list(edited.geoms)
        new = MultiPolygon(out) if out else ann.EMPTY
        old = mask.geometry
        # Exact coordinate comparison: a point inserted on a straight edge does not change the
        # shape (``equals`` would say "same") but it is a real edit that must be kept
        if not shapely.equals_exact(new, old, 0.0):
            ctx.undo.push(set_geometry_command(ctx.project, mask, old, new, text))
        if edited.is_empty:
            self._leave(ctx)
            return
        # Keep editing the same object (its largest piece if the edit split it)
        target = max(edited.geoms, key=lambda g: g.area)
        self.index = next((k for k, g in enumerate(mask.geometry.geoms) if g.equals(target)), -1)
        if self.index < 0:
            self._leave(ctx)
            return
        before = [len(r) for r in rings]
        self.rings = _rings_of(mask.geometry.geoms[self.index])
        self._geom = mask.geometry
        # Indices only stay meaningful if the rings kept their shape (no repair, no reordering)
        if not (keep_selection and [len(r) for r in self.rings] == before
                and all(np.allclose(a, b) for a, b in zip(self.rings, rings))):
            self.selected = set()
        self._show(ctx)

    def _move_selected(self, ctx, delta_local, originals) -> None:
        for (i, j), p in originals.items():
            self.rings[i][j] = p + delta_local

    # ------------------------------------------------------------ events
    def activate(self, ctx) -> None:
        self._leave(ctx)

    def deactivate(self, ctx) -> None:
        self._leave(ctx)

    def _resync(self, ctx) -> None:
        """Follow changes made elsewhere (undo/redo, other tools): re-find the active object."""
        if self.mask is None:
            return
        if self.mask.parent is None:          # the mask was deleted
            self._leave(ctx)
            return
        if self.mask.geometry is self._geom:
            return
        # A point inside the object (holes respected: the centre of a ring can sit in a hole)
        centre = Polygon(self.rings[0], self.rings[1:]).representative_point() if len(self.rings[0]) >= 3 else None
        for k, poly in enumerate(self.mask.geometry.geoms):
            if centre is not None and poly.contains(centre):
                self.index, self.rings, self.selected = k, _rings_of(poly), set()
                self._geom = self.mask.geometry
                self._show(ctx)
                return
        self._leave(ctx)

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        self._resync(ctx)
        mods = ev.get("modifiers", ())
        additive = "Control" in mods or "Shift" in mods
        if ev["button"] == RIGHT:
            hit = self._pick_vertex(ctx, ev) if self.mask is not None else None
            if hit is None:
                return False
            targets = self.selected if hit in self.selected else {hit}
            self._delete(ctx, targets)
            return True
        if ev["button"] != LEFT:
            return False
        if self.mask is not None:
            # 1. A point: select it (or toggle with Ctrl/Shift), then drag the selection
            hit = self._pick_vertex(ctx, ev)
            if hit is not None:
                # Ctrl/Shift on an already-selected point: remove it only if released without
                # dragging (dragging moves the whole selection, as in most editors)
                toggle_off = None
                if additive:
                    if hit in self.selected:
                        toggle_off = hit
                    else:
                        self.selected.add(hit)
                elif hit not in self.selected:
                    self.selected = {hit}
                self._drag = {"kind": "move", "screen": (ev["x"], ev["y"]), "moving": False,
                              "start": np.asarray(self._local(wx, wy)), "toggle_off": toggle_off,
                              "inserted": False,
                              "originals": {k: self.rings[k[0]][k[1]].copy() for k in self.selected}}
                self._show(ctx)
                return True
            # 2. An edge: insert a point and drag it
            edge = self._pick_edge(ctx, ev) if not additive else None
            if edge is not None:
                i, j = edge
                self.rings[i] = np.insert(self.rings[i], j + 1, self._local(wx, wy), axis=0)
                self.selected = {(i, j + 1)}
                self._drag = {"kind": "move", "screen": (ev["x"], ev["y"]), "moving": False,
                              "start": np.asarray(self._local(wx, wy)), "toggle_off": None, "inserted": True,
                              "originals": {(i, j + 1): self.rings[i][j + 1].copy()}}
                self._show(ctx)
                return True
            # 3. Empty space: maybe a rubber band (decided on move), else pick another object on release
            self._drag = {"kind": "box", "screen": (ev["x"], ev["y"]), "world": (wx, wy), "additive": additive}
            return True
        node = node_at(ctx.project, wx, wy)
        if isinstance(node, MaskNode) and self._activate_object(ctx, node, wx, wy):
            return True
        ctx.project.select(node)
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self.mask is None:
            return False
        d = self._drag
        if d is not None:
            sx, sy = d["screen"]
            # Below the threshold a press is still a click (hands move 1–2 px while clicking)
            if not d.get("moving") and np.hypot(ev["x"] - sx, ev["y"] - sy) < DRAG_START_PX:
                return True
            d["moving"] = True
            if d["kind"] == "move":
                self._move_selected(ctx, np.asarray(self._local(wx, wy)) - d["start"], d["originals"])
                self._show(ctx)
            else:
                ctx.preview.set_box(d["world"], (wx, wy))
                ctx.request_draw()
            return True
        hot = self._pick_vertex(ctx, ev)
        if hot != self.hot:
            self.hot = hot
            self._show(ctx)
        return False

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        d, self._drag = self._drag, None
        if d["kind"] == "move":
            if d["moving"] or d["inserted"]:
                n = len(d["originals"])
                text = f"Move {n} point{'s' if n > 1 else ''}" if d["moving"] else "Insert point"
                self._commit(ctx, [r.copy() for r in self.rings], text, keep_selection=True)
            elif d["toggle_off"] is not None:
                self.selected.discard(d["toggle_off"])        # Ctrl/Shift-click on a selected point
                self._show(ctx)
            return True
        # Rubber band (or a plain click on empty space)
        (x0, y0), additive = d["world"], d["additive"]
        ctx.preview.set_box()
        if not d["moving"]:
            # A click: another object, or leave
            node = node_at(ctx.project, wx, wy)
            self._leave(ctx)
            if isinstance(node, MaskNode):
                self._activate_object(ctx, node, wx, wy)
            else:
                ctx.project.select(node)
            return True
        lo, hi = np.minimum((x0, y0), (wx, wy)), np.maximum((x0, y0), (wx, wy))
        inside = {(i, j) for i, r in enumerate(self.rings) for j, p in enumerate(self._world(r))
                  if np.all(p >= lo) and np.all(p <= hi)}
        self.selected = (self.selected | inside) if additive else inside
        self._show(ctx)
        return True

    def _delete(self, ctx, targets) -> None:
        rings = delete_vertices(self.rings, set(targets))
        if rings is not None:
            n = len(targets)
            self.selected = set()
            self.hot = None
            self._commit(ctx, rings, f"Delete {n} point{'s' if n > 1 else ''}")

    def on_key(self, ctx, ev) -> bool:
        key = ev.get("key", "")
        mods = ev.get("modifiers", ())
        self._resync(ctx)
        if self.mask is None:
            return False
        if key == "Escape":
            if self.selected:
                self.selected = set()
                self._show(ctx)
            else:
                self._leave(ctx)
            return True
        if key in ("Delete", "Backspace"):
            # While an object is being edited, Delete only ever removes its selected points
            # (never the whole mask), even when no point is selected
            if self.selected:
                self._delete(ctx, self.selected)
            return True
        if key in ("a", "A") and "Control" in mods:
            self.selected = {(i, j) for i, r in enumerate(self.rings) for j in range(len(r))}
            self._show(ctx)
            return True
        if key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown") and self.selected:
            step = (10 if "Shift" in mods else 1) * ctx.view.um_per_px
            dx, dy = {"ArrowLeft": (-step, 0), "ArrowRight": (step, 0),
                      "ArrowUp": (0, -step), "ArrowDown": (0, step)}[key]
            # Move in world µm, converted to the image's local frame (rotation/scale aware)
            inv = np.linalg.inv(self._m())
            delta = inv[:2, :2] @ np.array([dx, dy])
            for i, j in self.selected:
                self.rings[i][j] = self.rings[i][j] + delta
            n = len(self.selected)
            self._commit(ctx, [r.copy() for r in self.rings], f"Nudge {n} point{'s' if n > 1 else ''}",
                         keep_selection=True)
            return True
        if key in ("s", "S"):
            poly = Polygon(self.rings[0], self.rings[1:]).simplify(self.value("tolerance"), preserve_topology=True)
            if not poly.is_empty:
                self.selected = set()
                self._commit(ctx, _rings_of(poly), "Simplify")
            return True
        return False
