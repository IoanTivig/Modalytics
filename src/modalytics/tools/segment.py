"""
Segment tool: run a segmentation model on a region and get vector masks.

    Drag a box      segment that region of the target image
    Enter           segment what is visible
    Ctrl+Enter      segment the whole image (large images run block by block)
    Esc             cancel a running segmentation

Target image: the selected image (or the image of the selected mask), else
the image under the cursor. The model is chosen in Tool Settings ("Auto"
picks one trained for the image's type and resamples the image to the µm/px
the model was trained at).

Results are **suggestions** to review (default): objects already drawn by hand are
not suggested again, and if the profile classifies the group each suggestion gets
its grade, confidence and reason (``services.suggest``); the Review tool then goes
through them. With "Result: Masks directly" they land in the output class as
ordinary masks instead. Either way it is one undo step, and with "Replace in
region" a re-run replaces the previous result inside the region instead of piling up.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from modalytics.core import annotations as ann
from modalytics.core import geometry as geo
from modalytics.core.commands import Command
from modalytics.core.model import MaskNode
from modalytics.io.sources import is_missing
from modalytics.tools.base import Setting, Tool, device_key, device_setting

LEFT = 1
MIN_BOX_PX = 8
AUTO = "Auto (by image type)"
SUGGEST = "Suggestions to review"
DIRECT = "Masks directly"
CONFIRM_ABOVE_S = 60          # whole-image runs estimated longer than this ask first
SMALL_MPX = 20                # below this (at model resolution) no estimate is needed


def duration_text(seconds: float) -> str:
    """'under a minute', 'about 12 minutes', 'about 1 hour 20 minutes'."""
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''}"
    h, m = divmod(minutes, 60)
    m = int(round(m / 5) * 5)
    return f"about {h} hour{'s' if h != 1 else ''}" + (f" {m} minutes" if m else "")


class SegmentTool(Tool):
    """Automatic segmentation of a region into vector masks."""

    id = "segment"
    label = "Segment"
    icon = "mdi6.auto-fix"
    shortcut = "G"
    tabs = ("Annotate",)
    group = "Automatic"
    hint = ("Drag a box to segment that region. Enter segments the visible area, Ctrl+Enter the whole image. "
            "Esc cancels. The objects found become suggestions, graded by the profile's classifier: review them "
            "with the Review tool (or choose Result: Masks directly).")

    def __init__(self, manager) -> None:
        from modalytics.services.segmentation import find_models
        self.models = find_models()
        names = [AUTO] + [m.name for m in self.models]
        # Settings are per instance: the model list depends on the installed packages
        self.settings = [
            Setting("model", "Model", "choice", AUTO, choices=names),
            Setting("result", "Result", "choice", SUGGEST, choices=[SUGGEST, DIRECT],
                    tooltip="Suggestions: the expert accepts, regrades or deletes each object (Review tool); they "
                            "are classified with the profile's classifier. Masks directly: straight into the "
                            "annotation."),
            Setting("classify", "Classify when segmenting", "bool", True,
                    tooltip="Suggestions graded by the profile's classifier: from the minimum confidence a grade, "
                            "below it the plain group (Ranges and Score: always their grade, to check). Off: "
                            "ungraded. Suggestions only — with Masks directly, nothing is classified."),
            Setting("confidence", "Minimum confidence", "float", 0.80, 0.5, 1.0, 0.05,
                    tooltip="The confidence from which the classifier's grade is given (classifiers giving a "
                            "probability: logistic regression, random forest, gradient boosting). Kept in the "
                            "profile, for the group segmented — Annotate › Classify selected uses it too; also in "
                            "Settings › Profiles."),
            Setting("output", "Output class", "label", ("", ""),
                    tooltip="Class the objects are added to (empty: the model's own class)"),
            Setting("threshold", "Threshold", "float", 0.5, 0.05, 0.95, 0.05),
            Setting("min_area", "Smallest object", "float", 20.0, 0.0, 100000.0, 5.0, " µm²"),
            Setting("background", "Ignore empty background", "bool", True,
                    tooltip="RGB images: drop objects that are mostly bare glass (bright, colourless)"),
            Setting("replace", "Replace in region", "bool", True,
                    tooltip="Remove this class inside the region before adding the new result"),
            device_setting(),
        ]
        super().__init__(manager)
        self._box = None

    # ------------------------------------------------------------ classification settings
    def _class_group(self) -> str:
        """The group whose classifier grades what is segmented: the output class's, else the profile's first
        group with a classifier."""
        group = (self.value("output") or ("", ""))[0]
        profile = self.manager.profile()
        if group or profile is None:
            return group
        return next((g.name for g in profile.classes.groups if profile.classifier_for(g.name) is not None), "")

    def activate(self, ctx) -> None:
        """Show the remembered Classify toggle and the profile's minimum confidence."""
        from modalytics import app_settings
        profile, group = self.manager.profile(), self._class_group()
        self._syncing = True
        try:
            self.manager.set_value(self, "classify", bool(app_settings.get("classify_segmenting")))
            if profile is not None and group:
                self.manager.set_value(self, "confidence", profile.confidence_for(group))
        finally:
            self._syncing = False

    def on_setting_changed(self, ctx, key: str, value) -> None:
        """Classify: remembered for next time. Minimum confidence: into the project's profile (one undo step) and
        the profile in your library."""
        if getattr(self, "_syncing", False):
            return
        if key == "classify":
            from modalytics import app_settings
            app_settings.put("classify_segmenting", bool(value))
        elif key == "output":
            self.activate(ctx)                                  # another group: its own confidence
        elif key == "confidence":
            profile, group = ctx.project.profile, self._class_group()
            if profile is None or not group or abs(profile.confidence_for(group) - float(value)) < 1e-9:
                return
            old, new = dict(profile.min_confidence), dict(profile.min_confidence, **{group: float(value)})

            def do():
                profile.min_confidence = dict(new)

            def undo():
                profile.min_confidence = dict(old)
            ctx.undo.push(Command(f"Minimum confidence {group} {float(value):.2f}", do, undo))
            try:                                                # the profile in your library too
                from modalytics.profiles.library import ProfileLibrary
                pid = getattr(ctx.project, "profile_id", "")
                lib = ProfileLibrary()
                mine = lib.get(pid) if pid else None
                if mine is not None:
                    mine.min_confidence[group] = float(value)
                    lib.save(mine, pid)
            except Exception:                                   # noqa: BLE001 — a built-in or unreadable profile
                pass

    # ------------------------------------------------------------ helpers
    def _model_for(self, image):
        from modalytics.services.segmentation import pick_model
        choice = self.value("model")
        if choice != AUTO:
            return next((m for m in self.models if m.name == choice), None)
        return pick_model(self.models, image.modality, self.manager.profile())

    def _local_rect(self, image, corners_world):
        local = geo.apply(np.linalg.inv(image.local_to_world()), np.asarray(corners_world, float))
        return float(local[:, 0].min()), float(local[:, 1].min()), float(local[:, 0].max()), float(local[:, 1].max())

    def _run(self, ctx, image, rect_local) -> None:
        """Segment a region of one image (see ``_run_many``)."""
        self._run_many(ctx, [(image, rect_local)])

    def _run_many(self, ctx, jobs: list) -> None:
        """Segment regions of images, one after the other in one background task, each with its model.

        Args:
            ctx: Tool context.
            jobs: [(image, (x0, y0, x1, y1) local µm)].
        """
        tasks = ctx.tasks
        if tasks is None:
            return
        # 1. What can be done: the image's file is here, a model fits it, the region overlaps it
        group, sub = self.value("output") or ("", "")
        work, said = [], []
        for image, rect_local in jobs:
            if is_missing(image):
                said.append(f"⚠️ {image.name}: its file is not found — relink it in Properties first")
                continue
            pkg = self._model_for(image)
            if pkg is None:
                said.append(f"⚠️ No segmentation model for {image.modality} images ({image.name})")
                continue
            if pkg.modality != image.modality:
                said.append(f"⚠️ {pkg.name} was trained on {pkg.modality} images, {image.name} is {image.modality}")
            w, h = image.size_um
            x0, y0 = max(0.0, rect_local[0]), max(0.0, rect_local[1])
            x1, y1 = min(w, rect_local[2]), min(h, rect_local[3])
            if x1 > x0 and y1 > y0:
                work.append((image, (x0, y0, x1, y1), pkg, (group, sub) if group else pkg.output_class))
        if not work:
            tasks.message(said[0] if said else "⚠️ The region does not overlap an image")
            return
        threshold, min_area = self.value("threshold"), self.value("min_area")
        device = device_key(self.value("device"))
        replace = self.value("replace")
        background = self.value("background")
        suggest = self.value("result") == SUGGEST
        profile = self.manager.profile()
        classify_on = bool(self.value("classify"))          # Classify when segmenting (Tool Settings)
        everything = list(ctx.project.images)
        n = len(work)

        def job(progress, cancelled):
            from modalytics.services.segmentation import segment
            out = []
            for k, (image, rect, pkg, label) in enumerate(work):
                def sub_progress(f, k=k):
                    progress((k + (0.9 * f if suggest else f)) / n)
                geom = segment(image, rect, pkg, threshold, min_area, device, sub_progress, cancelled,
                               ignore_background=background)
                if not suggest:
                    out.append(geom)
                    continue
                # Suggestions: what is not drawn yet, graded by the profile's classifier (measured in the background)
                from modalytics.services import suggest as sg
                items = sg.make(geom, image, everything, profile, label[0], cancelled=cancelled,
                                classify_on=classify_on)
                if label[1] and not (classify_on and profile is not None and profile.classifier_for(label[0])):
                    for it in items:
                        it.subgroup = it.subgroup or label[1]
                out.append(items)
            return out

        def done(results):
            ctx.preview.set_box()
            first = None
            found = 0
            for (image, (x0, y0, x1, y1), pkg, label), result in zip(work, results):
                if not suggest:
                    found += apply_result(ctx, image, label, result, box(x0, y0, x1, y1), replace, pkg.name)
                    continue
                node = put_suggestions(ctx, image, result, box(x0, y0, x1, y1), replace, pkg, profile, label[0])
                found += len(result)
                if result and first is None:
                    first = (node, result[0])
            where = f" on {n} images" if n > 1 else ""
            if not suggest:
                tasks.message(f"✅ {found} object(s) → {ann.label_text(*work[0][3])}{where}" if found
                              else "ℹ️ No objects found")
            elif first is not None:
                tasks.message(f"✅ {found} suggestion(s) to review{where} — Review tool (W)")
                ctx.project.set_current_suggestion(first[0], first[1].id)
                self.manager.activate("review")
            else:
                tasks.message("ℹ️ Nothing new to suggest (no objects found, or all already drawn)")
            ctx.request_draw()

        mpx = sum((r[2] - r[0]) * (r[3] - r[1]) / pkg.um_per_px ** 2 for _i, r, pkg, _l in work) / 1e6
        what = work[0][0].name if n == 1 else f"{n} images"
        tasks.run(f"Segmenting {what} · {mpx:,.1f} Mpx", job, done)

    # ------------------------------------------------------------ events
    def deactivate(self, ctx) -> None:
        ctx.preview.set_box()
        super().deactivate(ctx)

    def _images_in(self, ctx, corners_world) -> list:
        """The visible images a world region covers; if some of them are selected, only those."""
        from shapely.geometry import Polygon
        region = Polygon(corners_world)
        under = [img for img in ctx.project.images
                 if img.visible and Polygon(img.world_corners()).intersects(region)]
        chosen = [img for img in ctx.project.selected_images() if img in under]
        return chosen or under

    def on_pointer_down(self, ctx, ev, wx, wy) -> bool:
        if ev["button"] != LEFT:
            return False
        self._box = ((wx, wy), (ev["x"], ev["y"]))
        return True

    def on_pointer_move(self, ctx, ev, wx, wy) -> bool:
        if self._box is None:
            return False
        ctx.preview.set_box(self._box[0], (wx, wy))
        ctx.request_draw()
        return True

    def on_pointer_up(self, ctx, ev, wx, wy) -> bool:
        if self._box is None:
            return False
        (ax, ay), (sx, sy) = self._box
        self._box = None
        if np.hypot(ev["x"] - sx, ev["y"] - sy) < MIN_BOX_PX:
            ctx.preview.set_box()
            ctx.request_draw()
            return True
        corners = [(ax, ay), (wx, ay), (wx, wy), (ax, wy)]
        images = self._images_in(ctx, corners)
        if not images:
            ctx.preview.set_box()
            ctx.request_draw()
            return True
        self._run_many(ctx, [(img, self._local_rect(img, corners)) for img in images])
        return True

    def on_key(self, ctx, ev) -> bool:
        key, mods = ev.get("key", ""), ev.get("modifiers", ())
        if key == "Escape" and ctx.tasks is not None and ctx.tasks.busy:
            ctx.tasks.cancel()
            return True
        if key in ("Enter", "Return"):
            if "Control" in mods:
                cx, cy = ctx.view.cx, ctx.view.cy
                images = ctx.project.selected_images() or [i for i in [ctx.project.image_at(cx, cy)] if i]
                self.run_whole_images(ctx, images)
            else:
                corners = ctx.view.corners()
                images = self._images_in(ctx, corners)
                if images:
                    self._run_many(ctx, [(img, self._local_rect(img, corners)) for img in images])
            return True
        return False

    def run_whole_image(self, ctx, image) -> None:
        """Segment an entire image (see ``run_whole_images``)."""
        self.run_whole_images(ctx, [image])

    def run_whole_images(self, ctx, images: list) -> None:
        """Segment entire images, one after the other; when that will take long (large images), the time is
        estimated and must be confirmed.

        Args:
            ctx: Tool context (``ctx.tasks.confirm`` shows the question).
            images: The images to segment.
        """
        jobs, seconds, large = [], 0.0, []
        for image in images:
            if is_missing(image):
                if ctx.tasks is not None:
                    ctx.tasks.message(f"⚠️ {image.name}: its file is not found — relink it in Properties first")
                continue
            pkg = self._model_for(image)
            if pkg is None:
                if ctx.tasks is not None:
                    ctx.tasks.message(f"⚠️ No segmentation model for {image.modality} images")
                continue
            w, h = image.size_um
            rect = (0.0, 0.0, w, h)
            jobs.append((image, rect))
            if w * h / pkg.um_per_px ** 2 / 1e6 > SMALL_MPX and ctx.tasks is not None:
                from modalytics.services.segmentation import estimate
                e = estimate(image, rect, pkg, device=device_key(self.value("device")),
                             ignore_background=self.value("background"))
                seconds += e["seconds"]
                large.append((image, pkg, e))
        if not jobs:
            return
        if large and seconds > CONFIRM_ABOVE_S and ctx.tasks is not None:
            if len(jobs) == 1:
                image, pkg, e = large[0]
                w, h = image.size_um
                kind = "a whole-slide image" if image.modality == "wsi" else "a large image"
                what = (f"The model will analyse its {e['tissue_mm2']:,.0f} mm² of tissue"
                        if e["tissue_mm2"] < e["area_mm2"] * 0.98 else f"The model will analyse all {e['area_mm2']:,.0f} mm²")
                text = (f"{image.name} is {kind} ({w / 1000:.1f} × {h / 1000:.1f} mm). {what} at full resolution "
                        f"({pkg.um_per_px:g} µm per pixel), roughly {e['megapixels']:,.0f} megapixels.\n\n")
                title = f"Segment the whole {'slide' if image.modality == 'wsi' else 'image'}?"
                ok_text = "Segment whole slide" if image.modality == "wsi" else "Segment"
            else:
                mpx = sum(e["megapixels"] for _i, _p, e in large)
                text = (f"{len(jobs)} images, {len(large)} of them large (about {mpx:,.0f} megapixels at the models' "
                        f"resolution).\n\n")
                title, ok_text = f"Segment {len(jobs)} images?", "Segment"
            text += (f"This will take {duration_text(seconds)} on this computer. You can keep working "
                     f"while it runs and stop it at any time with Esc or the Cancel button in the status bar. "
                     f"The result is kept, so running it again later (for example with another threshold) "
                     f"is much faster.\n\n"
                     f"To try the settings first, drag a box over a smaller area with the Segment tool.")
            if not ctx.tasks.confirm(title, text, ok_text):
                return
        self._run_many(ctx, jobs)


def put_suggestions(ctx, image, items: list, region, replace: bool, pkg, profile, group: str):
    """Put new suggestions on an image's Suggestions layer, as one undo step.

    Args:
        ctx: Tool context.
        image: Target image.
        items: The new suggestions.
        region: Polygon of the segmented region (local µm): with ``replace``, earlier suggestions of the group in
            it are dropped first.
        pkg: The model package (recorded as the source).
        profile: The profile (its classifier, recorded as the source).
        group: The class group.

    Returns:
        SuggestionNode: The layer.
    """
    import datetime as dt
    project = ctx.project
    node = project.suggestions_for(image, create=False)
    created = node is None
    if created:
        from modalytics.core.model import SuggestionNode
        node = SuggestionNode()
    old_items, old_source = list(node.items), dict(node.source)
    kept = [s for s in old_items if not (replace and s.group == group and region.contains(
        s.geometry.representative_point()))]
    clf = profile.classifier_for(group) if profile is not None else None
    source = {"model": pkg.id, "classifier": f"{clf.kind}: {clf.note}" if clf is not None else "",
              "made": dt.datetime.now().isoformat(timespec="seconds")}
    new_items = kept + list(items)

    def do():
        if created and node.parent is None:
            node.parent = image
            image.children.append(node)
            project.structure_changed.emit()
        project.set_suggestions(node, new_items, source)

    def undo():
        project.set_suggestions(node, old_items, old_source)
        if created and node.parent is not None:
            project.remove(node)

    ctx.undo.push(Command(f"Suggest {group} ({pkg.name})", do, undo))
    return node


def apply_result(ctx, image, label, geom, region, replace: bool, source: str = "") -> int:
    """Put a segmentation result into the image's class mask, as one undo step.

    Args:
        ctx: Tool context.
        image: Target image.
        label: (group, subgroup).
        geom: MultiPolygon in the image's local µm.
        region: Polygon of the segmented region (local µm).
        replace: Remove the class inside the region first.
        source: Model name (undo text).

    Returns:
        int: Number of objects found.
    """
    project = ctx.project
    group, sub = label
    mask = project.mask_for(image, group, sub, create=False)
    created = None
    if mask is None:
        if geom.is_empty:
            return 0
        created = MaskNode(geometry=None, color=project.labels.color(group, sub), group=group, subgroup=sub)
        mask = created
    old = mask.geometry
    new = old
    if replace:
        new, _ = ann.edit(new, ann.as_multipolygon(region), "cut")
    if not geom.is_empty:
        new, _ = ann.edit(new, geom, "add")
    if created is None and new is old:
        return 0

    def do():
        if created is not None and created.parent is None:
            project.add_mask(image, created)
        project.set_mask_geometry(mask, new)

    def undo():
        project.set_mask_geometry(mask, old)
        if created is not None:
            project.remove(created)

    ctx.undo.push(Command(f"Segment {ann.label_text(group, sub)}" + (f" ({source})" if source else ""), do, undo))
    return len(geom.geoms)
