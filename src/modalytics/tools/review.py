"""
Review tool: go through the models' suggestions and accept, regrade or delete each.

    Click            pick the suggestion under the cursor (else select as the Select tool)
    Tab / Shift+Tab  next / previous suggestion (brought into view)
    Enter            accept it with its suggested grade
    1 … 9            accept it with that grade (the group's subgroups, in the profile's order)
    Delete           delete it
    Ctrl+Enter       accept every suggestion on the image

Each step is one undo command and moves on to the next suggestion. An accepted
object is part of the annotation: correct its outline with the drawing tools.
What the suggestion says (grade, confidence, reason) is shown here and in
Properties. See ``services.suggest`` for what accepting does.
"""

from __future__ import annotations

import math

import numpy as np
import shapely

from modalytics.core import geometry as geo
from modalytics.services import suggest as sg
from modalytics.tools.base import Tool, ToolAction
from modalytics.tools.select import node_at

LEFT = 1


def suggestion_at(project, wx: float, wy: float):
    """(SuggestionNode, Suggestion) of the top-most visible suggestion under a world point, else (None, None)."""
    for img in reversed(project.images):
        node = img.suggestions
        if not img.visible or node is None or not node.visible or not node.items:
            continue
        lx, ly = geo.apply(np.linalg.inv(img.local_to_world()), (wx, wy))
        pt = shapely.Point(lx, ly)
        for it in node.items:
            if it.geometry.contains(pt):
                return node, it
    return None, None


def grades_of(project, group: str) -> list[str]:
    """The grades a suggestion of ``group`` can be given: the profile's subgroups of it, in its order (else the
    project's classes, for a project without a profile)."""
    for groups in ((project.profile.classes.groups if project.profile is not None else []), project.labels.groups):
        g = next((g for g in groups if g.name == group), None)
        if g is not None and g.subgroups:
            return [s.name for s in g.subgroups]
    return []


def describe(project, node) -> str:
    """One line on the suggestion under review, for Tool Settings and Properties."""
    it = node.current_item if node is not None else None
    if it is None:
        n = len(node.items) if node is not None else 0
        return f"{n} suggestion(s) to review: Tab goes to the first." if n else "No suggestions to review."
    k = [s.id for s in node.items].index(it.id) + 1
    what = f"{it.group} › {it.subgroup}" if it.subgroup else it.group
    # a trained classifier's confidence; for ranges and scores the reason says it all (conditions met, points)
    score = f" · confidence {it.score:.2f}" if not math.isnan(it.score) and "P(" in it.reason else ""
    keys = " · ".join(f"{i + 1} {g}" for i, g in list(enumerate(grades_of(project, it.group)))[:9])
    return (f"Suggestion {k} of {len(node.items)}: {what}{score}" + (" (grade only: drawn by hand)" if it.regrade
                                                                       else "")
            + (f"\n{it.reason}" if it.reason else "") + "\nEnter accept · " + (keys + " · " if keys else "")
            + "Delete · Tab next")


class ReviewTool(Tool):
    """Review the models' suggestions one by one (see the module docstring)."""

    id = "review"
    label = "Review"
    icon = "mdi6.check-decagram-outline"
    shortcut = "W"                     # (R is Rotate)
    tabs = ("Annotate",)
    group = "Automatic"
    hint = ("Go through the suggestions: Enter accepts, 1–9 accepts with that grade, Delete deletes, Tab / "
            "Shift+Tab next / previous, Ctrl+Enter accepts all on the image. Accepted objects are ordinary "
            "masks: correct their outlines with the drawing tools.")
    actions = [
        ToolAction("previous", "Previous", "mdi6.skip-previous", "The previous suggestion (Shift+Tab)"),
        ToolAction("next", "Next", "mdi6.skip-next", "The next suggestion (Tab)"),
        ToolAction("accept", "Accept", "mdi6.check-bold", "Accept with the suggested grade (Enter; 1–9: another "
                                                          "grade)"),
        ToolAction("delete", "Delete", "mdi6.close-thick", "Delete the suggestion (Delete)"),
        ToolAction("accept_all", "Accept all", "mdi6.check-all", "Accept every suggestion on the image (Ctrl+Enter)"),
    ]

    # ------------------------------------------------------------ helpers
    def node(self, ctx):
        """The suggestions being reviewed: the selected layer, else the selected image's, else the first with any."""
        sel = ctx.project.selected
        from modalytics.core.model import ImageNode, MaskNode, SuggestionNode
        if isinstance(sel, SuggestionNode):
            return sel
        img = sel if isinstance(sel, ImageNode) else sel.parent if isinstance(sel, MaskNode) else None
        if img is not None and img.suggestions is not None and img.suggestions.items:
            return img.suggestions
        return next((i.suggestions for i in ctx.project.images if i.suggestions is not None and i.suggestions.items),
                    None)

    def go(self, ctx, node, item) -> None:
        """Make ``item`` the suggestion under review and bring it into view."""
        if node is None:
            return
        ctx.project.set_current_suggestion(node, item.id if item is not None else "")
        if item is not None and ctx.view is not None:
            ring = sg.bounds_world(node, item)
            x0, y0 = np.nanmin(ring, axis=0)
            x1, y1 = np.nanmax(ring, axis=0)
            w, h = x1 - x0, y1 - y0
            if not (ctx.view.cx - ctx.view.width * ctx.view.um_per_px / 2 < x0 and
                    x1 < ctx.view.cx + ctx.view.width * ctx.view.um_per_px / 2 and
                    ctx.view.cy - ctx.view.height * ctx.view.um_per_px / 2 < y0 and
                    y1 < ctx.view.cy + ctx.view.height * ctx.view.um_per_px / 2) or \
                    max(w / max(ctx.view.width, 1), h / max(ctx.view.height, 1)) / ctx.view.um_per_px < 0.05:
                ctx.view.fit((x0 - w, y0 - h, x1 + w, y1 + h))
        self.set_info(describe(ctx.project, node))
        self.actions_changed()
        ctx.request_draw()

    def step(self, ctx, direction: int = 1) -> None:
        node = self.node(ctx)
        if node is not None and node.items:
            self.go(ctx, node, sg.neighbour(node, direction))

    def accept(self, ctx, grade: str | None = None, every: bool = False) -> None:
        """Accept the current suggestion (or all on its image) and move on."""
        node = self.node(ctx)
        if node is None or not node.items:
            return
        items = list(node.items) if every else [node.current_item] if node.current_item else []
        if not items:
            self.step(ctx)
            return
        after = sg.neighbour(node) if not every else None
        ctx.undo.push(sg.accept_command(ctx.project, node, items, grade))
        self.go(ctx, node, after if after is not None and after.id in {s.id for s in node.items} else
                (node.items[0] if node.items else None))

    def delete(self, ctx) -> None:
        node = self.node(ctx)
        it = node.current_item if node is not None else None
        if it is None:
            return
        after = sg.neighbour(node)
        ctx.undo.push(sg.reject_command(ctx.project, node, [it]))
        self.go(ctx, node, after if after is not None and after.id != it.id else None)

    def delete_all(self, ctx) -> None:
        """Delete every suggestion on the image under review."""
        node = self.node(ctx)
        if node is not None and node.items:
            ctx.undo.push(sg.reject_command(ctx.project, node, list(node.items)))
            self.go(ctx, node, None)

    # ------------------------------------------------------------ events
    def activate(self, ctx) -> None:
        node = self.node(ctx)
        if node is not None and node.items and node.current_item is None:
            self.go(ctx, node, node.items[0])
        else:
            self.set_info(describe(ctx.project, node))

    def action_enabled(self, key: str) -> bool:
        return True

    def on_action(self, ctx, key: str) -> None:
        if key in ("previous", "next"):
            self.step(ctx, -1 if key == "previous" else 1)
        elif key == "accept":
            self.accept(ctx)
        elif key == "accept_all":
            self.accept(ctx, every=True)
        elif key == "delete":
            self.delete(ctx)

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        node, it = suggestion_at(ctx.project, wx, wy)
        if node is not None:
            ctx.project.set_current_suggestion(node, it.id)
            self.set_info(describe(ctx.project, node))
        else:
            ctx.project.select(node_at(ctx.project, wx, wy))
        return True

    def on_key(self, ctx, ev) -> bool:
        key, mods = ev.get("key", ""), ev.get("modifiers", ())
        if key == "Tab":
            self.step(ctx, -1 if "Shift" in mods else 1)
            return True
        if key in ("Enter", "Return"):
            self.accept(ctx, every="Control" in mods)
            return True
        if key in ("Delete", "Backspace"):
            self.delete(ctx)
            return True
        if len(key) == 1 and key.isdigit() and key != "0":
            node = self.node(ctx)
            it = node.current_item if node is not None else None
            grades = grades_of(ctx.project, it.group) if it is not None else []
            k = int(key) - 1
            if k < len(grades):
                self.accept(ctx, grades[k])
            return True
        return False
