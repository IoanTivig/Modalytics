"""
Arrange tools: Rotate and Register (automatic registration).

Rotate
    Drag around the image: it turns about its centre. Shift snaps to 15°.

Register
    The selected image moves (one a method can move, e.g. a phase map; a slide
    never moves): drag it roughly to where it belongs (the circle is the search
    radius), press Start (ribbon) or Enter. The reference is the top-most visible
    image the circle touches that the method accepts (e.g. the slide). The method
    (a script, ``modalytics.registration``: the profile's default, or the one
    chosen in Tool Settings) finds the candidates; the best is applied, Previous
    / Next (Shift+Tab / Tab) step through the others (outlined), Save keeps the
    one shown and drops the rest, Ctrl+Z undoes. Method options: the method's own
    settings (kept in the app settings).
"""

from __future__ import annotations

import math

import numpy as np

from modalytics.core import geometry as geo
from modalytics.core.commands import set_placement_command
from modalytics.core.model import ImageNode
from modalytics.services.registration import rotated_about
from modalytics.tools.base import Setting, Tool, ToolAction, device_key, device_setting

LEFT = 1
SNAP_DEG = 15.0


class RotateTool(Tool):
    uses_shift_click = True        # Shift snaps the angle
    """Drag to rotate an image about its centre."""

    id = "rotate"
    label = "Rotate"
    icon = "mdi6.rotate-left-variant"
    shortcut = "R"
    tabs = ("Arrange",)
    group = "Tools"
    hint = "Drag around an image to rotate it about its centre; hold Shift to snap to 15°. Its masks follow."

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._drag = None

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        sel = ctx.project.selected
        image = sel if isinstance(sel, ImageNode) else ctx.project.image_at(wx, wy)
        if image is None:
            return True
        ctx.project.select(image)
        pivot = image.world_corners().mean(axis=0)
        self._drag = (image, image.placement, pivot, math.atan2(-(wy - pivot[1]), wx - pivot[0]))
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        image, start, pivot, a0 = self._drag
        # Angle counter-clockwise on screen (y points down)
        delta = math.degrees(math.atan2(-(wy - pivot[1]), wx - pivot[0]) - a0)
        if "Shift" in ev.get("modifiers", ()):
            target = round((start.rotation_deg + delta) / SNAP_DEG) * SNAP_DEG
            delta = target - start.rotation_deg
        ctx.project.set_placement(image, rotated_about(start, pivot, delta))
        if ctx.tasks is not None:
            ctx.tasks.message(f"Rotation {(image.placement.rotation_deg + 180) % 360 - 180:+.1f}°", 1500)
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        image, start, _pivot, _a0 = self._drag
        self._drag = None
        if image.placement != start:
            ctx.undo.push(set_placement_command(ctx.project, image, start, image.placement), already_done=True)
        return True


PROFILE_DEFAULT = "The profile's default"


def _methods():
    """The registration methods (``registration.registry``), read once per session (Settings › Registration
    reloads them)."""
    global _LIBRARY
    if _LIBRARY is None:
        from modalytics.registration.registry import RegistrationLibrary
        _LIBRARY = RegistrationLibrary()
    return _LIBRARY


def reload_methods() -> None:
    """Read the method scripts again (after Settings › Registration changed one)."""
    global _LIBRARY
    _LIBRARY = None


_LIBRARY = None


def offered_methods(profile) -> list:
    """The methods a project's profile offers (all validated ones without a profile), its default first."""
    specs = list(_methods().specs.values())
    if profile is None:
        return [s for s in specs if not s.experimental]
    default = profile.registration_default()
    out = [s for s in specs if profile.registration_offered(s.key, s.experimental)]
    return sorted(out, key=lambda s: s.key != default)


def method_label(spec) -> str:
    return spec.name + (" (experimental)" if spec.experimental else "")


class RegisterTool(Tool):
    """Automatic registration of an image onto a reference near a rough position, with a method script."""

    id = "register"
    label = "Register"
    icon = "mdi6.image-move"
    shortcut = "T"
    tabs = ("Arrange",)
    group = "Registration"
    hint = ("Select the image to place (e.g. a phase map) and drag it roughly to where it belongs; the circle shows "
            "how far the search reaches, and the top image under it that the method accepts is the reference (e.g. "
            "the slide). Press Start (Enter). Previous / Next (Shift+Tab / Tab) show the other candidates, Save "
            "keeps the one shown; Esc cancels.")
    actions = [
        ToolAction("start", "Start", "mdi6.play", "Find the exact position of the selected image (Enter)"),
        ToolAction("previous", "Previous", "mdi6.skip-previous", "Show the previous candidate (Shift+Tab)"),
        ToolAction("next", "Next", "mdi6.skip-next", "Show the next candidate (Tab)"),
        ToolAction("save", "Save", "mdi6.check-bold",
                   "Keep the position shown and finish: the other candidates are dropped (Ctrl+Z still undoes it)"),
        ToolAction("options", "Method options", "mdi6.tune-variant", "The chosen method's own settings"),
    ]
    settings = [
        Setting("method", "Method", "choice", PROFILE_DEFAULT, choices=[PROFILE_DEFAULT],
                tooltip="How the position is found: the methods the profile offers (Settings › Profiles › "
                        "Registration; Settings › Registration lists and tests them all)"),
        Setting("radius", "Search radius", "float", 400.0, 20.0, 5000.0, 50.0, " µm",
                tooltip="How far from its current position the phase map may end up"),
        Setting("rotation", "Rotation", "choice", "Any", choices=["Any", "±30°", "±10°", "Keep current"]),
        Setting("mirror", "Try mirrored", "bool", True, tooltip="Also consider the phase map mirrored"),
        Setting("scale", "Scale correction", "choice", "None", choices=["None", "±2 %", "±5 %"]),
        Setting("candidates", "Candidates", "int", 5, 1, 20, 1,
                tooltip="How many of the best positions are refined and kept to step through (more is slower)"),
        device_setting(),
    ]

    def __init__(self, manager) -> None:
        super().__init__(manager)
        # The Method list depends on the project's profile: computed whenever Tool Settings shows it
        from dataclasses import replace
        self.settings = [replace(s, choices=lambda: [PROFILE_DEFAULT] + [method_label(m) for m in
                                                                         offered_methods(self.manager.profile())])
                         if s.key == "method" else s for s in type(self).settings]
        self.moving = None         # the image being placed (one a method can move: never a slide)
        self.candidates = []       # list[Candidate] of the last run for ``moving``, best first
        self.current = 0           # index of the candidate now applied
        self._drag = None          # (start placement, grab point) while dragging
        self._ctx = None

    # ------------------------------------------------------------ helpers
    def _say(self, ctx, text, ms=8000):
        if ctx.tasks is not None:
            ctx.tasks.message(text, ms)

    def method(self, ctx):
        """The chosen method (the profile's default, else the library's)."""
        offered = offered_methods(ctx.project.profile)        # its default first
        chosen = self.value("method")
        spec = next((s for s in offered if chosen == method_label(s)), None)   # a method no longer offered: the default
        return spec or (offered[0] if offered else None)

    def movable(self, image) -> bool:
        """Whether some method can move this image (a slide never moves)."""
        return image is not None and image.modality != "wsi" and \
            any(s.can_move(image.modality) for s in _methods().specs.values())

    def reference(self, ctx, moving):
        """The reference: the top-most visible image (other than ``moving``) the search circle touches that the
        method accepts; None if there is none."""
        import shapely
        spec = self.method(ctx)
        if moving is None or spec is None:
            return None
        circle = shapely.Point(*moving.world_corners().mean(axis=0)).buffer(self.value("radius"))
        for img in reversed(ctx.project.images):
            if img is not moving and img.visible and spec.can_reference(img.modality) and \
                    shapely.Polygon(img.world_corners()).intersects(circle):
                return img
        return None

    def _pick(self, image) -> None:
        """Make ``image`` the one being placed (old candidates belong to the previous one)."""
        if image is not self.moving:
            self.moving, self.candidates = image, []

    def _update(self, ctx) -> None:
        """Redraw the search circle and candidate outlines, the info line and the ribbon actions."""
        self.actions_changed()
        if self.moving is not None and self.moving not in ctx.project.images:
            self._pick(None)                             # deleted
        moving = self.moving
        if moving is None:
            ctx.preview.clear()
            self.set_info("Click the image to place (e.g. a phase map).")
            ctx.request_draw()
            return
        ctx.preview.set_circle(moving.world_corners().mean(axis=0), self.value("radius"))
        rows = []                                        # the other candidates' footprints
        w, h = moving.size_um
        for i, c in enumerate(self.candidates):
            if i != self.current:
                cc = geo.apply(c.placement.local_to_world(), geo.rect_corners(w, h))
                rows += [cc, cc[:1], np.full((1, 2), np.nan)]
        ctx.preview.set_path(np.vstack(rows) if rows else np.zeros((0, 2)))
        ref = self.reference(ctx, moving)
        spec = self.method(ctx)
        self.set_info(f"Places: {moving.name}\nOn: " + (ref.name if ref is not None
                                                         else "— (drag it over an image the method accepts)")
                      + (f"\nMethod: {spec.name}" + (" — refines the current position, no search"
                                                      if spec.refines_only else "") if spec else ""))
        ctx.request_draw()

    def _apply(self, ctx, index: int) -> None:
        cand = self.candidates[index]
        self.current = index
        old = self.moving.placement
        ctx.undo.push(set_placement_command(ctx.project, self.moving, old, cand.placement))
        n = len(self.candidates)
        best, nxt = self.candidates[0].score, (self.candidates[1].score if n > 1 else None)
        verdict = ""
        if index == 0 and nxt is not None:
            verdict = (" — clear match" if best - nxt >= 0.1 else
                       " — ambiguous: check it, Next shows the alternatives")
        self._say(ctx, f"Candidate {index + 1}/{n}: score {cand.score:.2f}"
                       + (f" (next best {nxt:.2f})" if index == 0 and nxt is not None else "") + verdict, 20000)
        self._update(ctx)

    # ------------------------------------------------------------ model → tool
    def _on_node_changed(self, node, field) -> None:
        """The phase map moved by any means (Ctrl+middle drag, Properties, undo): follow it."""
        if node is self.moving and field == "placement" and self._ctx is not None:
            if self.candidates and self._drag is None and \
                    not any(c.placement == node.placement for c in self.candidates):
                self.candidates = []                     # a new rough position: old candidates are stale
            self._update(self._ctx)

    def _on_selection(self, node) -> None:
        """Choosing an image elsewhere (e.g. the layer list) makes it the one to place."""
        if self.movable(node) and self._ctx is not None:
            self._pick(node)
            self._update(self._ctx)

    # ------------------------------------------------------------ events
    def activate(self, ctx) -> None:
        self._ctx = ctx
        sel = ctx.project.selected
        if self.movable(sel):
            self._pick(sel)
        ctx.project.node_changed.connect(self._on_node_changed)
        ctx.project.selection_changed.connect(self._on_selection)
        self._update(ctx)

    def deactivate(self, ctx) -> None:
        ctx.project.node_changed.disconnect(self._on_node_changed)
        ctx.project.selection_changed.disconnect(self._on_selection)
        self.candidates, self._drag, self._ctx = [], None, None
        ctx.preview.clear()

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        image = next((i for i in reversed(ctx.project.images)
                      if i.visible and self.movable(i) and i.contains_world(wx, wy)), None)
        if image is None:
            # A slide is a reference: it never moves here
            if ctx.project.image_at(wx, wy) is not None:
                self._say(ctx, "ℹ️ That image is a reference and stays in place — click the image to place on it")
            return True
        self._pick(image)
        ctx.project.select(image)
        self._drag = (image.placement, np.array([wx, wy]))
        self._update(ctx)
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        start, grab = self._drag
        dx, dy = np.array([wx, wy]) - grab
        ctx.project.set_placement(self.moving, start.moved_by(dx, dy))
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._drag is None:
            return False
        start, _grab = self._drag
        self._drag = None
        if self.moving.placement != start:
            self.candidates = []                         # a new rough position: old candidates no longer apply
            ctx.undo.push(set_placement_command(ctx.project, self.moving, start, self.moving.placement),
                          already_done=True)
        self._update(ctx)
        return True

    def _step(self, ctx, step: int) -> None:
        if self.candidates and self.moving is not None:
            self._apply(ctx, (self.current + step) % len(self.candidates))
        else:
            self._say(ctx, "ℹ️ No other candidates — press Start first")

    def on_key(self, ctx, ev) -> bool:
        key = ev.get("key", "")
        if key == "Escape" and ctx.tasks is not None and ctx.tasks.busy:
            ctx.tasks.cancel()
            return True
        if key == "Escape" and self.candidates:
            self.candidates = []
            self._update(ctx)
            return True
        if key == "Tab" and len(self.candidates) > 1:
            self._step(ctx, -1 if "Shift" in ev.get("modifiers", ()) else 1)
            return True
        if key in ("Enter", "Return"):
            self.run(ctx)
            return True
        return False

    def action_enabled(self, key: str) -> bool:
        if key in ("previous", "next"):
            return len(self.candidates) > 1
        if key == "save":
            return bool(self.candidates)
        return True

    def on_action(self, ctx, key: str) -> None:
        if key == "start":
            self.run(ctx)
        elif key in ("previous", "next"):
            self._step(ctx, -1 if key == "previous" else 1)
        elif key == "save":
            self.save(ctx)
        elif key == "options":
            self.edit_options(ctx)

    def edit_options(self, ctx) -> None:
        """The chosen method's own options (kept for the session and in the app settings)."""
        from modalytics import app_settings
        spec = self.method(ctx)
        if spec is None or ctx.tasks is None:
            return
        stored = dict(app_settings.get("registration_options") or {})
        values = ctx.tasks.edit_options(f"{spec.name} — options", spec.options, stored.get(spec.key, {}),
                                        spec.description)
        if values is not None:
            stored[spec.key] = values
            app_settings.put("registration_options", stored)

    def on_setting_changed(self, ctx, key: str, value) -> None:
        if key in ("radius", "method"):
            self._update(ctx)                            # resize the search circle, name the reference now

    def save(self, ctx) -> None:
        """Keep the candidate shown and finish (drops the alternatives; the placement is already applied)."""
        if not self.candidates or self.moving is None:
            self._say(ctx, "ℹ️ Nothing to save yet — press Start first")
            return
        cand, n = self.candidates[self.current], len(self.candidates)
        ref = self.reference(ctx, self.moving)
        self.candidates = []
        self._say(ctx, f"✅ Saved the position of {self.moving.name}" + (f" on {ref.name}" if ref else "")
                       + f" (candidate {self.current + 1}/{n}, score {cand.score:.2f})")
        self._update(ctx)

    def run(self, ctx) -> None:
        """Register the image being placed onto the reference under the search circle, with the chosen method."""
        from modalytics import app_settings
        from modalytics.io.sources import is_missing
        from modalytics.registration.context import Search
        from modalytics.registration.run import run_method
        moving = self.moving
        if moving is None:
            self._say(ctx, "ℹ️ Select the image to place first, and drag it roughly to where it belongs")
            return
        spec = self.method(ctx)
        if spec is None:
            self._say(ctx, "⚠️ No registration method is installed (Settings › Registration)")
            return
        if not spec.can_move(moving.modality):
            self._say(ctx, f"⚠️ {spec.name} cannot move {moving.modality} images — choose another method")
            return
        fixed = self.reference(ctx, moving)
        if fixed is None:
            self._say(ctx, f"⚠️ No image {spec.name} accepts as reference under the circle — drag {moving.name} "
                           f"over it first")
            return
        gone = next((i for i in (moving, fixed) if is_missing(i)), None)
        if gone is not None:
            self._say(ctx, f"⚠️ {gone.name}: its file is not found — relink it in Properties first")
            return
        rot = {"Any": 180.0, "±30°": 30.0, "±10°": 10.0, "Keep current": 0.0}[self.value("rotation")]
        scale = {"None": 0.0, "±2 %": 0.02, "±5 %": 0.05}[self.value("scale")]
        search = Search(tuple(moving.world_corners().mean(axis=0)), self.value("radius"), rot, scale,
                        bool(self.value("mirror")), int(self.value("candidates")))
        options = dict(app_settings.get("registration_options") or {}).get(spec.key, {})
        device, profile = device_key(self.value("device")), ctx.project.profile

        def job(progress, cancelled):
            return run_method(spec, moving, fixed, search, options, device, profile, progress, cancelled)

        def done(cands):
            if not cands:
                self._say(ctx, "⚠️ No match found within the search radius")
                return
            if moving is not self.moving:                # the user picked another image meanwhile
                return
            self.candidates = cands
            self._apply(ctx, 0)

        ctx.tasks.run(f"Registering {moving.name} on {fixed.name} ({spec.name})", job, done)
