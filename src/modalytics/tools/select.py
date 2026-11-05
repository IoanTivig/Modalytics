"""
Select and Move tools.
"""

from __future__ import annotations

import numpy as np
import shapely

from modalytics.core import geometry as geo
from modalytics.core.commands import set_placement_command
from modalytics.core.model import ImageNode, MaskNode
from modalytics.tools.base import Tool

LEFT = 1
DRAG_PX = 4                # a press that moves less than this is a click, not a selection rectangle


def node_at(project, wx: float, wy: float):
    """Top-most visible mask under a world point, else the top-most image.

    Args:
        project: The project.
        wx, wy: World µm.

    Returns:
        MaskNode | ImageNode | None: What a click there selects.
    """
    for img in reversed(project.images):
        if not img.visible:
            continue
        lx, ly = geo.apply(np.linalg.inv(img.local_to_world()), (wx, wy))
        pt = shapely.Point(lx, ly)
        for mask in reversed(img.masks):
            if mask.visible and not mask.geometry.is_empty and mask.geometry.contains(pt):
                return mask
        if img.contains_world(wx, wy):
            return img
    return None


def object_at(project, wx: float, wy: float):
    """The object (one polygon of a mask) under a world point, top-most first: (mask, polygon), or None."""
    for img in reversed(project.images):
        if not img.visible:
            continue
        lx, ly = geo.apply(np.linalg.inv(img.local_to_world()), (wx, wy))
        pt = shapely.Point(lx, ly)
        for mask in reversed(img.masks):
            if not mask.visible or mask.geometry.is_empty:
                continue
            for poly in mask.geometry.geoms:
                if poly.contains(pt):
                    return mask, poly
        if img.contains_world(wx, wy):
            return None
    return None


def images_in_rect(project, a, b) -> list[ImageNode]:
    """The visible images a world rectangle between corners ``a`` and ``b`` touches (bottom → top).

    Touching is enough: large images are often only partly in view.
    """
    box = shapely.box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
    return [img for img in project.images
            if img.visible and shapely.Polygon(img.world_corners()).intersects(box)]


class RectSelect:
    """A selection made with the pointer: a click (one layer) or a dragged rectangle (the images it touches).

    Used by the Select tool (plain: replaces the selection; Ctrl+click: adds or removes one) and by the canvas
    for Shift + click / Shift + drag in every tool (adds to the selection, the tool stays active).
    """

    def __init__(self, ctx, ev, wx, wy, add: bool = False, toggle: bool = False, clicked: bool = False) -> None:
        self.start, self.screen = (wx, wy), (ev["x"], ev["y"])
        self.add, self.toggle, self.dragging = add, toggle, False
        self.clicked = clicked           # the press already selected something (a suggestion): only a drag adds

    def move(self, ctx, ev, wx, wy) -> None:
        if not self.dragging and np.hypot(ev["x"] - self.screen[0], ev["y"] - self.screen[1]) >= DRAG_PX:
            self.dragging = True
        if self.dragging:
            ctx.preview.set_box(self.start, (wx, wy))
            ctx.request_draw()

    def finish(self, ctx, wx, wy) -> None:
        project = ctx.project
        before = list(project.selection)
        if self.dragging:
            ctx.preview.set_box()
            ctx.request_draw()
            found = images_in_rect(project, self.start, (wx, wy))
            if self.add:
                project.select_many(before + found, primary=found[-1] if found else project.selected)
            else:
                project.select_many(found)
            return
        if self.clicked:
            return
        obj = object_at(project, *self.start)
        if obj is not None:                                  # a single object (one gland), not its whole layer
            have = list(project.objects)
            there = any(m is obj[0] and p.equals(obj[1]) for m, p in have)
            if self.toggle and there:
                rest = [(m, p) for m, p in have if not (m is obj[0] and p.equals(obj[1]))]
                project.select_objects(rest) if rest else project.select(None)
            elif self.add or self.toggle:
                project.select_objects(have + [obj])
            else:
                project.select_objects([obj])
            return
        node = node_at(project, *self.start)
        if node is None:
            if not (self.add or self.toggle):
                project.select(None)
        elif self.toggle and node in before:
            project.select_many([n for n in before if n is not node])
        elif self.add or self.toggle:
            project.select_many(before + [node], primary=node)
        else:
            project.select(node)


class SelectTool(Tool):
    """Click to select a mask (or the image under it); drag a rectangle to select the images it touches."""

    id = "select"
    label = "Select"
    icon = "mdi6.cursor-default-outline"
    shortcut = "V"
    tabs = ("Arrange", "Annotate")
    group = "Tools"
    hint = ("Click an object (one gland), a suggestion or an image to select it; drag a rectangle to select "
            "the images it touches. Ctrl+click adds or removes one; Shift + click / drag adds (in every tool). "
            "Selected objects: right-click › Class, or 1–9 gives them that grade; Delete deletes them. "
            "Esc clears the selection.")

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._pick = None

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        ctrl = "Control" in ev.get("modifiers", ())
        from modalytics.tools.review import suggestion_at
        node, item = suggestion_at(ctx.project, wx, wy)          # a suggestion (drawn above the masks) first
        on_suggestion = node is not None and not ctrl
        if on_suggestion:
            ctx.project.set_current_suggestion(node, item.id)
        self._pick = RectSelect(ctx, ev, wx, wy, toggle=ctrl, clicked=on_suggestion)   # a drag: a rectangle
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self._pick is None:
            return False
        self._pick.move(ctx, ev, wx, wy)
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._pick is None:
            return False
        pick, self._pick = self._pick, None
        pick.finish(ctx, wx, wy)
        return True

    def deactivate(self, ctx) -> None:
        if self._pick is not None:
            self._pick = None
            ctx.preview.set_box()
            ctx.request_draw()

    def on_key(self, ctx, ev) -> bool:
        """1–9 with objects selected: give each the n-th grade of its group (as in Review)."""
        key = ev.get("key", "")
        objects = ctx.project.objects
        if not (objects and len(key) == 1 and key.isdigit() and key != "0"):
            return False
        from modalytics.services.classes import change_class
        from modalytics.tools.review import grades_of
        k = int(key) - 1
        by_group: dict = {}
        for m, p in objects:
            by_group.setdefault(m.group, []).append((m, p))
        moved = []
        for group, items in by_group.items():
            grades = grades_of(ctx.project, group)
            if k >= len(grades):
                continue
            cmd, after = change_class(ctx.project, items, group, grades[k])
            if cmd is not None:
                ctx.undo.push(cmd)
            moved += after if cmd is not None else items
        if moved:
            ctx.project.select_objects(moved)
        return True


class MoveTool(Tool):
    """Left-drag an image (same as Ctrl + middle-drag in any tool)."""

    id = "move"
    label = "Move"
    icon = "mdi6.cursor-move"
    shortcut = "M"
    tabs = ("Arrange",)
    group = "Tools"
    hint = "Drag an image to move it; its masks follow. Ctrl + middle-drag does the same in every tool."

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._drag = None

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        sel = ctx.project.selected
        target = sel if isinstance(sel, ImageNode) and sel.contains_world(wx, wy) else ctx.project.image_at(wx, wy)
        if target is None:
            return True
        ctx.project.select(target)
        self._drag = (target, target.placement, wx, wy)
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        img, start, ax, ay = self._drag
        ctx.project.set_placement(img, start.moved_by(wx - ax, wy - ay))
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        img, start, _ax, _ay = self._drag
        self._drag = None
        if img.placement != start:
            ctx.undo.push(set_placement_command(ctx.project, img, start, img.placement), already_done=True)
        return True


def selected_image(project) -> ImageNode | None:
    """The selected image, or the image of the selected mask (or suggestions)."""
    from modalytics.core.model import SuggestionNode
    sel = project.selected
    if isinstance(sel, ImageNode):
        return sel
    if isinstance(sel, (MaskNode, SuggestionNode)):
        return sel.parent
    return None
