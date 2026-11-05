"""
Manual annotation tools: Polygon, Freehand, Brush, Eraser, Knife.

All tools edit **vector** masks (shapely MultiPolygons in the target image's
local µm frame):

* Polygon and Freehand add their shape to the active class (or cut it out in
  Subtract mode, or while holding Alt). A shape that only touches an object of
  the class (less than a fifth of the shape over it) stays a separate object —
  two touching glands are two glands; one drawn mostly over an object extends it.
* Brush adds a round stroke (it always extends what it paints over); Eraser cuts
  a round stroke out of the active class or of every class on the image.
* Knife: a freehand line across objects cuts each one it crosses from side to
  side into pieces (suggestions waiting for review too).

Target image: the image under the start of the gesture — the selected one if it
is there, else the top-most visible one. One gesture = one undo step.
"""

from __future__ import annotations

import numpy as np
import shapely

from modalytics.core import annotations as ann
from modalytics.core import geometry as geo
from modalytics.core.commands import Command
from modalytics.tools.base import Setting, Tool

LEFT, RIGHT = 1, 2
CLOSE_SNAP_PX = 8          # clicking this close to the first vertex closes the polygon
SAMPLE_PX = 2.0            # freehand/brush: record a point every 2 screen px
MIN_PRESSURE_SIZE = 0.2    # a pen's lightest touch paints a fifth of the brush diameter
MIN_AREA_UM2 = 0.5         # drop slivers smaller than this after an edit
JOIN_FRACTION = 0.2        # a drawn shape at least this much over an object extends it; less: a separate object
GAP_UM = 0.05              # between objects kept separate (touching glands; the knife's cut): a hair's breadth

LABEL = Setting("label", "Class", "label", default=("Gland", "Normal"), shared=True,
                tooltip="Group › subgroup the drawn shapes belong to")
EXCLUSIVE = Setting("exclusive", "Overwrite other classes", "bool", True, shared=True,
                        tooltip="Drawing a class removes that area from the other classes of the image")


def separate_from(geometry, shape):
    """``shape`` without the objects of ``geometry`` it only touches (less than ``JOIN_FRACTION`` of the shape
    over each), with a hair's breadth gap, so it stays a separate object; objects it is mostly over are left
    for it to join (an extension)."""
    parts = shapely.get_parts(geometry)
    if not len(parts) or shape.is_empty:
        return shape
    hit = shapely.STRtree(parts).query(shape, predicate="intersects")
    touched = [p for p in parts[hit] if p.intersection(shape).area < JOIN_FRACTION * shape.area]
    if not touched:
        return shape
    return shape.difference(shapely.union_all(touched).buffer(GAP_UM))


def apply_shape(ctx, image, shape_local, label, subtract: bool, scope_all: bool = False,
                exclusive: bool = False, text: str = "", separate: bool = False) -> bool:
    """Add a shape to (or cut it from) the class mask(s) of an image, as one undo step.

    Args:
        ctx: Tool context.
        image: Target ImageNode.
        shape_local: MultiPolygon in the image's local µm.
        label: (group, subgroup) of the active class.
        subtract: Cut instead of add.
        scope_all: When subtracting, cut from every class on the image.
        exclusive: When adding, also cut the shape from every *other* class on the
            image, so classes never overlap ("Overwrite other classes").
        text: Undo text.
        separate: When adding, objects of the class the shape only touches stay separate (``separate_from``).

    Returns:
        bool: True if anything changed.
    """
    project = ctx.project
    if shape_local.is_empty:
        return False
    group, subgroup = label
    # 1. Which masks change, and how ("add" to the active class, "cut" from others)
    own = project.mask_for(image, group, subgroup, create=False)
    if subtract:
        ops = [(m, "cut") for m in (list(image.masks) if scope_all else [own] if own else [])]
    else:
        ops = [(own, "add")]
        if exclusive:
            ops += [(m, "cut") for m in image.masks if m is not own]
    created = own is None and not subtract
    if created:
        ops[0] = (project.mask_for(image, group, subgroup, create=True), "add")
    masks = [m for m, _op in ops]
    # 2. New geometries
    changes = []
    for m, op in ops:
        old = m.geometry
        shape = shape_local
        if separate and op == "add":
            shape = ann.as_multipolygon(_polys(separate_from(old, shape_local)))
        new, changed = ann.edit(old, shape, op, MIN_AREA_UM2)
        if changed:
            changes.append((m, old, new))
    if not changes:
        if created:
            project.remove(masks[0])
        return False
    new_mask = masks[0] if created else None

    # 3. One command for all of it (re-adding the mask on redo if it was created here)
    def do():
        if new_mask is not None and new_mask.parent is None:
            project.add_mask(image, new_mask)
        for m, _o, n in changes:
            project.set_mask_geometry(m, n)

    def undo():
        for m, o, _n in changes:
            project.set_mask_geometry(m, o)
        if new_mask is not None:
            project.remove(new_mask)

    verb = "Erase" if subtract else "Draw"
    ctx.undo.push(Command(text or f"{verb} {ann.label_text(group, subgroup)}", do, undo))
    return True


PRESSURE = Setting("pressure", "Pen pressure sets the size", "bool", True,
                   tooltip="With a graphics tablet: lighter pressure paints (or erases) thinner, down to a fifth "
                           "of the diameter; a mouse always uses the full diameter")


def drawing_target(project, wx: float, wy: float):
    """The image a drawing starting at world (wx, wy) goes on (see ``_AnnotateTool._target``)."""
    under = [img for img in reversed(project.images) if img.visible and img.contains_world(wx, wy)]
    if not under:
        return None
    chosen = [img for img in project.selected_images() if img in under]
    return chosen[0] if chosen else under[0]


class _AnnotateTool(Tool):
    pointer = "crosshair"                         # drawing: the exact point
    """Shared helpers: target image and world → local conversion."""

    tabs = ("Annotate",)
    group = "Draw"
    target_verb = "Draws on"

    def _target(self, ctx, wx, wy, quiet: bool = False):
        """The image a drawing that starts at (wx, wy) goes on: the selected image if it is there (so a lower image —
        a slide under a phase map — is drawn on by selecting it), else the top visible image there; None (and a
        word in the status bar) outside every image."""
        image = drawing_target(ctx.project, wx, wy)
        if image is None and not quiet and ctx.tasks is not None:
            ctx.tasks.message("ℹ️ Not over an image: nothing is drawn (start on the image to annotate)")
        return image

    def _show_target(self, ctx, wx, wy) -> None:
        """Tool Settings says which image a drawing started here would go on."""
        image = drawing_target(ctx.project, wx, wy)
        text = f"{self.target_verb}: {image.name}" if image is not None else \
            "Not over an image: nothing would be drawn"
        if text != getattr(self, "_target_text", None):
            self._target_text = text
            self.set_info(text)

    @staticmethod
    def _to_local(image, pts_world) -> np.ndarray:
        return geo.apply(np.linalg.inv(image.local_to_world()), np.asarray(pts_world, dtype=float))

    @staticmethod
    def _subtract(tool, ev) -> bool:
        alt = "Alt" in ev.get("modifiers", ())
        return (tool.value("mode") == "Subtract") != alt


class PolygonTool(_AnnotateTool):
    """Click vertices; double-click, Enter, right-click or clicking the first vertex closes."""

    id = "polygon"
    label = "Polygon"
    icon = "mdi6.vector-polygon"
    shortcut = "P"
    hint = ("Click to add points. Double-click, Enter, right-click or click the first point to close. "
            "Backspace removes the last point, Esc cancels. Hold Alt to subtract.")
    settings = [LABEL, Setting("mode", "Mode", "choice", "Add", choices=["Add", "Subtract"]), EXCLUSIVE]

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.points: list[tuple[float, float]] = []
        self.image = None
        self.cursor = None

    def deactivate(self, ctx) -> None:
        self._reset(ctx)

    def _reset(self, ctx) -> None:
        self.points, self.image, self.cursor = [], None, None
        ctx.preview.clear()
        ctx.request_draw()

    def _redraw(self, ctx) -> None:
        pts = self.points + ([self.cursor] if self.cursor else [])
        ctx.preview.set_path(np.array(pts) if pts else np.zeros((0, 2)), closed=len(pts) > 2)
        ctx.request_draw()

    def _finish(self, ctx, ev) -> None:
        if len(self.points) >= 3 and self.image is not None:
            shape = ann.as_multipolygon([self._to_local(self.image, self.points)])
            subtract = self._subtract(self, ev)
            apply_shape(ctx, self.image, shape, self.value("label"), subtract,
                        exclusive=self.value("exclusive"), separate=not subtract)
        self._reset(ctx)

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] == RIGHT:
            if self.points:
                self._finish(ctx, ev)
                return True
            return False
        if ev["button"] != LEFT:
            return False
        if not self.points:
            self.image = self._target(ctx, wx, wy)
            if self.image is None:
                return True
        elif len(self.points) >= 3:
            fx, fy = ctx.view.world_to_screen(*self.points[0])
            if np.hypot(fx - ev["x"], fy - ev["y"]) <= CLOSE_SNAP_PX:
                self._finish(ctx, ev)
                return True
        self.points.append((wx, wy))
        self._redraw(ctx)
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self.points:
            self.cursor = (wx, wy)
            self._redraw(ctx)
        else:
            self._show_target(ctx, wx, wy)
        return False

    def on_double_click(self, ctx, ev, wx, wy) -> bool:
        if self.points:
            # The double-click's first click already added a point at the same place
            if len(self.points) > 1 and np.allclose(self.points[-1], self.points[-2]):
                self.points.pop()
            self._finish(ctx, ev)
            return True
        return False

    def on_key(self, ctx, ev) -> bool:
        key = ev.get("key", "")
        if not self.points:
            return False
        if key == "Escape":
            self._reset(ctx)
        elif key in ("Enter", "Return"):
            self._finish(ctx, ev)
        elif key == "Backspace":
            self.points.pop()
            self._redraw(ctx) if self.points else self._reset(ctx)
        else:
            return False
        return True


class FreehandTool(_AnnotateTool):
    """Press, draw the outline, release to close."""

    id = "freehand"
    label = "Freehand"
    icon = "mdi6.draw"
    shortcut = "L"
    hint = "Press and draw an outline; release to close it. Hold Alt to subtract."
    settings = [LABEL, Setting("mode", "Mode", "choice", "Add", choices=["Add", "Subtract"]), EXCLUSIVE]

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.points, self.image, self._last = [], None, None

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        self.image = self._target(ctx, wx, wy)
        if self.image is not None:
            self.points, self._last = [(wx, wy)], (ev["x"], ev["y"])
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if not self.points:
            self._show_target(ctx, wx, wy)
            return False
        if np.hypot(ev["x"] - self._last[0], ev["y"] - self._last[1]) >= SAMPLE_PX:
            self.points.append((wx, wy))
            self._last = (ev["x"], ev["y"])
            ctx.preview.set_path(np.array(self.points), closed=True)
            ctx.request_draw()
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if not self.points:
            return False
        if len(self.points) >= 3:
            shape = ann.as_multipolygon([self._to_local(self.image, self.points)])
            subtract = self._subtract(self, ev)
            apply_shape(ctx, self.image, shape, self.value("label"), subtract,
                        exclusive=self.value("exclusive"), separate=not subtract)
        self.points, self.image = [], None
        ctx.preview.clear()
        ctx.request_draw()
        return True


class BrushTool(_AnnotateTool):
    """Paint a round brush; the stroke is merged into the class on release."""

    id = "brush"
    label = "Brush"
    icon = "mdi6.brush"
    shortcut = "B"
    hint = ("Drag to paint. [ and ] change the size. Hold Alt to erase from this class. With a pen: lighter "
            "pressure paints thinner; the pen's eraser end erases.")
    settings = [LABEL, Setting("size", "Diameter", "float", 20.0, 0.5, 2000.0, 1.0, " µm"), EXCLUSIVE, PRESSURE]
    subtract = False
    pointer = "none"                              # its ring is the pointer (Eraser's too)

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.points, self.radii, self.image, self._last = [], [], None, None

    def _radius(self, ev: dict | None = None) -> float:
        """Half the diameter; with a pen (and "Pen pressure sets the size") scaled by its pressure."""
        r = self.value("size") / 2
        pressure = ev.get("pressure") if ev is not None else None
        if pressure is not None and self.value("pressure"):
            r *= MIN_PRESSURE_SIZE + (1 - MIN_PRESSURE_SIZE) * max(0.0, min(1.0, pressure))
        return r

    def _preview(self, ctx) -> None:
        if len(set(self.radii)) > 1:                         # a pen stroke of varying width
            ctx.preview.set_stroke(np.zeros((0, 2)), 0)
            ctx.preview.set_dabs(np.array(self.points), self.radii)
        else:
            ctx.preview.set_stroke(np.array(self.points), self.radii[0] if self.radii else self._radius())

    def _hover(self, ctx, wx, wy) -> None:
        ctx.preview.set_circle((wx, wy), self._radius())
        if not self.points:
            self._show_target(ctx, wx, wy)
        ctx.request_draw()

    def activate(self, ctx) -> None:
        ctx.preview.clear()

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        self.image = self._target(ctx, wx, wy)
        if self.image is not None:
            self.points, self.radii, self._last = [(wx, wy)], [self._radius(ev)], (ev["x"], ev["y"])
            self._preview(ctx)
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        self._hover(ctx, wx, wy)
        if not self.points:
            return False
        if np.hypot(ev["x"] - self._last[0], ev["y"] - self._last[1]) >= SAMPLE_PX:
            self.points.append((wx, wy))
            self.radii.append(self._radius(ev))
            self._last = (ev["x"], ev["y"])
            self._preview(ctx)
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if not self.points:
            return False
        local = self._to_local(self.image, self.points)
        scale = self.image.placement.scale                          # brush size is in world µm
        shape = ann.stroke_polygon(local, self.radii[0] / scale, [r / scale for r in self.radii])
        subtract = self.subtract != ("Alt" in ev.get("modifiers", ()))
        scope_all = self.subtract and self.value("scope") == "All classes"
        exclusive = not self.subtract and self.value("exclusive")
        apply_shape(ctx, self.image, shape, self.value("label"), subtract, scope_all, exclusive)
        self.points, self.radii, self.image = [], [], None
        ctx.preview.set_stroke(np.zeros((0, 2)), 0)
        ctx.preview.set_dabs([], [])
        ctx.request_draw()
        return True

    def on_key(self, ctx, ev) -> bool:
        key = ev.get("key", "")
        if key in ("[", "]"):
            size = self.value("size") * (1 / 1.25 if key == "[" else 1.25)
            self.manager.set_value(self, "size", round(size, 2))
            return True
        return False


class EraserTool(BrushTool):
    """Erase with a round brush from the active class or from every class."""

    id = "eraser"
    label = "Eraser"
    icon = "mdi6.eraser"
    shortcut = "E"
    hint = "Drag to erase. [ and ] change the size. Erasing inside a shape makes a hole."
    settings = [LABEL, Setting("size", "Diameter", "float", 20.0, 0.5, 2000.0, 1.0, " µm"),
                Setting("scope", "Erase from", "choice", "Active class", choices=["Active class", "All classes"]),
                PRESSURE]
    subtract = True


class KnifeTool(_AnnotateTool):
    """Draw a freehand line across objects: each one it crosses from side to side is cut into pieces."""

    id = "knife"
    label = "Knife"
    icon = "mdi6.knife"
    shortcut = "K"
    hint = ("Draw a line (any shape) across an object, from outside it on one side to outside it on the other: "
            "it is cut in two, e.g. two glands outlined as one. Suggestions waiting for review are cut too.")
    settings = [LABEL, Setting("scope", "Cut", "choice", "All classes", choices=["All classes", "Active class"])]
    target_verb = "Cuts on"

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self.points, self.image, self._last = [], None, None

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        self.image = self._target(ctx, wx, wy)
        if self.image is not None:
            self.points, self._last = [(wx, wy)], (ev["x"], ev["y"])
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if not self.points:
            self._show_target(ctx, wx, wy)
            return False
        if np.hypot(ev["x"] - self._last[0], ev["y"] - self._last[1]) >= SAMPLE_PX:
            self.points.append((wx, wy))
            self._last = (ev["x"], ev["y"])
            ctx.preview.set_path(np.array(self.points), closed=False)
            ctx.request_draw()
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if not self.points:
            return False
        points, image = self.points + [(wx, wy)], self.image
        self.points, self.image = [], None
        ctx.preview.clear()
        ctx.request_draw()
        if len(points) >= 2 and not cut_along(ctx, image, self._to_local(image, points), self.value("label"),
                                              self.value("scope") == "All classes") and ctx.tasks is not None:
            ctx.tasks.message("ℹ️ Nothing was cut: draw the line across the object, from outside it on one side "
                              "to outside it on the other")
        return True

    def on_key(self, ctx, ev) -> bool:
        if self.points and ev.get("key") == "Escape":
            self.points, self.image = [], None
            ctx.preview.clear()
            ctx.request_draw()
            return True
        return False


def _polys(geom) -> list:
    """The polygons of a geometry (a Polygon, a MultiPolygon or a collection)."""
    return [p for p in shapely.get_parts(geom) if isinstance(p, shapely.Polygon) and not p.is_empty]


def split_polygon(poly, blade) -> list:
    """The pieces of ``poly`` once ``blade`` (the cut line, widened by a hair) is taken out; [poly] when the line
    does not cross it from side to side (a cut that ends inside leaves it whole, not slit)."""
    if not poly.intersects(blade):
        return [poly]
    pieces = [p for p in _polys(poly.difference(blade)) if p.area >= MIN_AREA_UM2]
    return pieces if len(pieces) >= 2 else [poly]


def cut_along(ctx, image, line_local, label, all_classes: bool) -> bool:
    """Cut the objects (and the suggestions) a line crosses on ``image``, as one undo step.

    Args:
        ctx: Tool context.
        image: The ImageNode.
        line_local: (n, 2) the line's points in the image's local µm.
        label: The active class (group, subgroup), when only it is cut.
        all_classes: Cut every class's objects (else only the active class's).

    Returns:
        bool: True if anything was cut.
    """
    from modalytics.core.model import Suggestion
    project = ctx.project
    blade = shapely.LineString(np.asarray(line_local, float)).buffer(GAP_UM / 2, cap_style="flat")
    # 1. Masks: each object the line crosses becomes its pieces
    masks = list(image.masks) if all_classes else         [m for m in [project.mask_for(image, *label, create=False)] if m is not None]
    changes = []
    for m in masks:
        parts = _polys(m.geometry)
        out = [q for p in parts for q in split_polygon(p, blade)]
        if len(out) > len(parts):
            changes.append((m, m.geometry, ann.as_multipolygon(out)))
    # 2. Suggestions waiting for review: likewise (each piece keeps the suggested class and reason)
    node = image.suggestions
    before = list(node.items) if node is not None else []
    after = []
    for it in before:
        if not all_classes and it.group != label[0]:
            after.append(it)
            continue
        pieces = split_polygon(it.geometry, blade)
        after += [it] if len(pieces) == 1 else [Suggestion(p, it.group, it.subgroup, it.score, it.reason,
                                                           it.regrade) for p in pieces]
    cut_suggestions = len(after) > len(before)
    if not changes and not cut_suggestions:
        return False

    def do():
        for m, _o, n in changes:
            project.set_mask_geometry(m, n)
        if cut_suggestions:
            project.set_suggestions(node, after)

    def undo():
        for m, o, _n in changes:
            project.set_mask_geometry(m, o)
        if cut_suggestions:
            project.set_suggestions(node, before)

    n = sum(len(_polys(new)) - len(_polys(old)) for _m, old, new in changes) + len(after) - len(before)
    ctx.undo.push(Command(f"Cut with the knife ({n} more object{'s' if n != 1 else ''})", do, undo))
    return True


def all_tools():
    """Tool classes in ribbon order."""
    from modalytics.tools.arrange import RegisterTool, RotateTool
    from modalytics.tools.edit_points import EditPointsTool
    from modalytics.tools.review import ReviewTool
    from modalytics.tools.segment import SegmentTool
    from modalytics.tools.select import MoveTool, SelectTool
    return [SelectTool, EditPointsTool, MoveTool, RotateTool, RegisterTool, PolygonTool, FreehandTool,
            BrushTool, EraserTool, KnifeTool, SegmentTool, ReviewTool]
