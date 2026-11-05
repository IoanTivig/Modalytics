"""
Inspector for the selected node: calibration, placement, opacity, display.

Every field writes through ``Project`` (with undo where it is a user edit) and
every model change refreshes the fields, so the panel, the canvas and the
layer list can never disagree.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.core.commands import Command, UndoStack, set_opacity_command, set_placement_command
from modalytics.core.model import ImageNode, MaskNode, Project, SuggestionNode
from modalytics.render.visuals import COLORMAP_NAMES
from modalytics.ui.layer_panel import MODALITY_LABELS


def _spin(lo: float, hi: float, decimals: int, suffix: str = "", step: float = 1.0) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)        # apply on Enter / focus-out, not on every keystroke
    return s


def polygon_area(p: np.ndarray) -> float:
    """Shoelace area of a closed polygon (same units², always positive)."""
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


class PropertiesPanel(QtWidgets.QWidget):
    """Shows and edits the selected image or mask.

    Args:
        project: The project.
        undo: Undo stack for edits.
    """

    def __init__(self, project: Project, undo: UndoStack, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.project = project
        self.undo = undo
        self.node = None
        self._opacity_start: float | None = None

        # 1. Widgets
        self.title = QtWidgets.QLabel("Nothing selected")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600; font-size: 11pt;")
        self.subtitle = QtWidgets.QLabel("")
        self.subtitle.setObjectName("Muted")
        self.subtitle.setWordWrap(True)
        # An image whose file was not found: say so, offer Relink…
        self.missing = QtWidgets.QLabel("")
        self.missing.setWordWrap(True)
        self.relink_btn = QtWidgets.QPushButton("Relink…")
        self.relink_btn.setToolTip("Point the image at its file again (where it is now); the masks are kept")
        self.relink_btn.clicked.connect(self._relink)
        # Suggestions: the one under review, and what to do with it (the main window runs it via the Review tool)
        self.on_review = None                         # callback(key, grade=None), set by the main window
        self.review_box = QtWidgets.QWidget()
        rcol = QtWidgets.QVBoxLayout(self.review_box)
        rcol.setContentsMargins(0, 6, 0, 0)
        self.review_text = QtWidgets.QLabel("")
        self.review_text.setWordWrap(True)
        rcol.addWidget(self.review_text)
        self.grade_buttons = QtWidgets.QHBoxLayout()
        rcol.addLayout(self.grade_buttons)
        self._grade_key = None
        for row_keys in ((("previous", "◀ Previous"), ("next", "Next ▶")),
                         (("accept", "Accept"), ("delete", "Delete")),
                         (("accept_all", "Accept all"), ("delete_all", "Delete all"))):
            row = QtWidgets.QHBoxLayout()
            for key, text in row_keys:
                b = QtWidgets.QPushButton(text)
                b.setToolTip({"accept": "Accept with the suggested grade (Enter in the Review tool)",
                              "delete": "Delete this suggestion (Delete)",
                              "accept_all": "Accept every suggestion on this image, each with its suggested grade",
                              "delete_all": "Delete every suggestion on this image",
                              "next": "The next suggestion (Tab)", "previous": "The previous one (Shift+Tab)"}[key])
                b.clicked.connect(lambda _c=False, k=key: self._review(k))
                row.addWidget(b)
            rcol.addLayout(row)

        self.pixel_size = _spin(1e-4, 1e4, 5, " µm/px", 0.001)
        self.x = _spin(-1e7, 1e7, 2, " µm", 10)
        self.y = _spin(-1e7, 1e7, 2, " µm", 10)
        self.rotation = _spin(-360, 360, 2, " °", 1)
        self.scale = _spin(0.01, 100, 4, " ×", 0.01)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity_label = QtWidgets.QLabel("100 %")
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(COLORMAP_NAMES)
        self.clim_lo = _spin(-1e6, 1e6, 3, "", 0.05)
        self.clim_hi = _spin(-1e6, 1e6, 3, "", 0.05)

        self.object_class = QtWidgets.QComboBox()           # the class of the selected objects (single glands)
        self.object_class.setToolTip("Give the selected objects another class (also: right-click › Class, or the "
                                     "keys 1–9 for the grades); one undo step")
        self.object_class.activated.connect(self._set_object_class)
        self.partial = QtWidgets.QCheckBox("Partial: shape incomplete")
        self.partial.setToolTip("Mark the selected objects whose shape is incomplete — cut by the section plane, "
                                "torn, folded, out of focus… Those cut by the image edge are so by themselves. Their "
                                "shape parameters are left out of classifier training and the shape statistics; "
                                "segmentation uses them as any other.")
        self.partial.clicked.connect(self._set_partial)
        self.partial_note = QtWidgets.QLabel()
        self.partial_note.setObjectName("Muted")
        self.partial_note.setWordWrap(True)

        # 2. Layout: sections that are shown per node type
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.sec_objects = self._section(form, "SELECTED OBJECTS")
        form.addRow("Class", self.object_class)
        form.addRow("Shape", self.partial)
        form.addRow("", self.partial_note)
        self.sec_calib = self._section(form, "CALIBRATION")
        form.addRow("Pixel size", self.pixel_size)
        self.sec_place = self._section(form, "PLACEMENT (WORLD µm)")
        form.addRow("X", self.x)
        form.addRow("Y", self.y)
        form.addRow("Rotation", self.rotation)
        form.addRow("Scale", self.scale)
        self.sec_display = self._section(form, "DISPLAY")
        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(self.opacity)
        op_row.addWidget(self.opacity_label)
        form.addRow("Opacity", op_row)
        self._opacity_row = op_row
        form.addRow("Colormap", self.colormap)
        form.addRow("Low", self.clim_lo)
        form.addRow("High", self.clim_hi)
        self.sec_annot = self._section(form, "ANNOTATION")
        self.outlined_box = QtWidgets.QWidget()       # one tick per group: "Annotation complete: every Gland drawn"
        self.outlined_col = QtWidgets.QVBoxLayout(self.outlined_box)
        self.outlined_col.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.outlined_box)
        self.form = form
        self._image_only = [self.pixel_size, self.x, self.y, self.rotation, self.scale, self.sec_annot,
                            self.outlined_box]
        self._phase_only = [self.colormap, self.clim_lo, self.clim_hi]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.missing)
        layout.addWidget(self.relink_btn)
        layout.addWidget(self.review_box)
        layout.addLayout(form)
        layout.addStretch(1)

        # 3. Edits → model
        self.pixel_size.valueChanged.connect(self._set_pixel_size)
        for w in (self.x, self.y, self.rotation, self.scale):
            w.valueChanged.connect(self._set_placement)
        self.opacity.sliderPressed.connect(lambda: setattr(self, "_opacity_start", self.node.opacity))
        self.opacity.valueChanged.connect(self._opacity_live)
        self.opacity.sliderReleased.connect(self._opacity_commit)
        self.colormap.currentTextChanged.connect(lambda name: self._set_display(colormap=name))
        self.clim_lo.valueChanged.connect(lambda _v: self._set_display(clim=(self.clim_lo.value(), self.clim_hi.value())))
        self.clim_hi.valueChanged.connect(lambda _v: self._set_display(clim=(self.clim_lo.value(), self.clim_hi.value())))

        # 4. Model → widgets. Live edits (drags, animations) can change the node
        #    every frame; refreshing ~9 widgets each time costs several ms, so
        #    refreshes are coalesced to at most one per 66 ms.
        self._refresh_timer = QtCore.QTimer(self, singleShot=True, interval=66, timeout=self.refresh)
        project.selection_changed.connect(self.show_node)
        project.node_changed.connect(lambda n, _f: n is self.node and not self._refresh_timer.isActive()
                                     and self._refresh_timer.start())
        project.structure_changed.connect(lambda: self.show_node(project.selected))
        # Another view shown (or a relinked file): which rows apply may change (colormap: single channel only)
        project.node_changed.connect(lambda n, f: f == "source" and n is self.node and self.show_node(n))
        self.show_node(None)

    def _review(self, key: str, grade: str | None = None) -> None:
        if self.on_review is not None:
            self.on_review(key, grade)

    def _show_review(self, node) -> None:
        """The suggestion under review: what it says, a button per grade."""
        from modalytics.services import suggest as sg
        from modalytics.tools.review import describe, grades_of
        it = node.current_item
        per = ", ".join(f"{n} {g or 'not graded'}" for g, n in sg.counts(node).items())
        src = node.source
        self.subtitle.setText(f"On {node.parent.name if node.parent else '—'} · {len(node.items)} to review"
                              + (f" ({per})" if per else "") + (f"\nModel: {src['model']}" if src.get("model") else "")
                              + (f"\nClassifier: {src['classifier']}" if src.get("classifier") else ""))
        self.review_text.setText(describe(self.project, node).split("\nEnter")[0])
        grades = grades_of(self.project, it.group) if it is not None else []
        key = (it.group if it is not None else None, tuple(grades))
        if key != self._grade_key:                     # rebuild the grade buttons only when they differ
            self._grade_key = key
            while self.grade_buttons.count():
                w = self.grade_buttons.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            for i, g in enumerate(grades[:9]):
                b = QtWidgets.QPushButton(f"{i + 1} {g}")
                b.setToolTip(f"Accept as {g} (key {i + 1} in the Review tool)")
                b.clicked.connect(lambda _c=False, g=g: self._review("accept", g))
                self.grade_buttons.addWidget(b)
        for i in range(self.grade_buttons.count()):
            w = self.grade_buttons.itemAt(i).widget()
            if w is not None:
                w.setEnabled(it is not None)

    def _section(self, form: QtWidgets.QFormLayout, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("SectionTitle")
        form.addRow(label)
        return label

    def _set_row_visible(self, widget, visible: bool) -> None:
        self.form.setRowVisible(widget, visible) if hasattr(self.form, "setRowVisible") else widget.setVisible(visible)

    # 1. Model → widgets
    def show_node(self, node) -> None:
        self.node = node
        self.review_box.setVisible(isinstance(node, SuggestionNode))
        is_img = isinstance(node, ImageNode)
        is_phase = is_img and not node.view_source.is_rgb        # single-channel: colormap and contrast
        for w in self._image_only:
            self._set_row_visible(w, is_img)
        for w in self._phase_only:
            self._set_row_visible(w, is_phase)
        has_objects = bool(self.project.objects)
        self._set_row_visible(self.object_class, has_objects)
        self._set_row_visible(self.partial, has_objects)
        self._set_row_visible(self.partial_note, has_objects)
        for sec, vis in ((self.sec_calib, is_img), (self.sec_place, is_img), (self.sec_display, node is not None),
                         (self.sec_annot, is_img), (self.sec_objects, has_objects)):
            self._set_row_visible(sec, vis)
        self._set_row_visible(self._opacity_row, node is not None)
        self.refresh()

    def refresh(self) -> None:
        node = self.node
        widgets = [self.pixel_size, self.x, self.y, self.rotation, self.scale, self.opacity,
                   self.colormap, self.clim_lo, self.clim_hi]
        for w in widgets:
            w.blockSignals(True)
        try:
            if node is None:
                self.missing.setVisible(False)
                self.relink_btn.setVisible(False)
                self.title.setText("Nothing selected")
                self.subtitle.setText("Click an image on the canvas or in Layers.")
                return
            self.title.setText(node.name)
            from modalytics.io.sources import is_missing
            gone = isinstance(node, ImageNode) and is_missing(node)
            self.missing.setVisible(gone)
            self.relink_btn.setVisible(gone)
            if gone:
                from modalytics.ui.settings.script_editor import current_tokens
                self.missing.setStyleSheet(f"color: {current_tokens()['warning']}; font-weight: 600;")
                self.missing.setText(f"⚠ File not found: {node.source.path}\nShown as stripes; its masks are kept. "
                                     f"Analysis needs the file.")
            self.opacity.setValue(round(node.opacity * 100))
            self.opacity_label.setText(f"{round(node.opacity * 100)} %")
            if isinstance(node, ImageNode):
                h, w = node.source.shape
                wu, hu = node.size_um
                from modalytics.image_import.contracts import describe_source
                source = describe_source(node.metadata)          # e.g. "Values in rad, converted from wv × 6.283"
                shown = node.shown
                showing = f"\nShowing: {shown.name}" + (f" ({shown.unit})" if shown.unit else "") \
                    if shown is not None else ""
                self.subtitle.setText(f"{MODALITY_LABELS.get(node.modality, node.modality)} · {w}×{h} px · "
                                      f"{wu:,.1f}×{hu:,.1f} µm · {len(node.source.levels)} level(s)"
                                      + (f"\n{source}" if source else "") + showing)
                self.pixel_size.setValue(node.pixel_size_um[0])
                self._fill_outlined(node)
                p = node.placement
                self.x.setValue(p.x_um)
                self.y.setValue(p.y_um)
                self.rotation.setValue(p.rotation_deg)
                self.scale.setValue(p.scale)
                if not node.view_source.is_rgb:
                    d = node.view_display                         # the shown view's own colormap / contrast
                    self.colormap.setCurrentText(d.colormap)
                    self.clim_lo.setValue(d.clim[0])
                    self.clim_hi.setValue(d.clim[1])
            elif isinstance(node, MaskNode) and self.project.objects:
                objs = self.project.objects
                images = {m.parent.name for m, _p in objs if m.parent is not None}
                classes = {m.name for m, _p in objs}
                n = len(objs)
                self.title.setText(f"{n} object{'s' if n != 1 else ''}" if n > 1 else f"Object · {node.name}")
                self.subtitle.setText(f"{', '.join(sorted(classes))} · on {', '.join(sorted(images))} · "
                                      f"{sum(p.area for _m, p in objs):,.0f} µm²")
                self._fill_object_class()
                self._fill_partial()
            elif isinstance(node, MaskNode):
                self.subtitle.setText(f"Mask on {node.parent.name if node.parent else '—'} · "
                                      f"{node.object_count} object(s) · {node.area_um2:,.0f} µm²")
            elif isinstance(node, SuggestionNode):
                self._show_review(node)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _groups(self) -> list[str]:
        """The class groups: the profile's, else the project's."""
        prof = self.project.profile
        groups = prof.classes.groups if prof is not None else self.project.labels.groups
        return [g.name for g in groups]

    def _fill_outlined(self, image) -> None:
        """One tick per group: the annotation of this image is complete — every object of it is drawn (for training
        and evaluation)."""
        while self.outlined_col.count():
            w = self.outlined_col.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for group in self._groups():
            box = QtWidgets.QCheckBox(f"Annotation complete: every {group} drawn")
            box.setChecked(image.fully_outlined(group))
            box.setToolTip(f"Tick when every {group} on this image is drawn (or accepted) — those cut by the image "
                           f"edge too, as far as they are visible; also on an image with no {group} at all. "
                           f"Segmentation training and evaluation then use the whole image: what is not drawn is "
                           f"surely not {group} (stroma, glass…) and teaches the model so. Leave it off while some "
                           f"are missing: only the area around the drawn ones is used.")
            box.toggled.connect(lambda on, g=group, img=image: self._set_outlined(img, g, on))
            self.outlined_col.addWidget(box)

    def _fill_object_class(self) -> None:
        """The class list, on the selected objects' class (blank when they have several)."""
        from modalytics.services import classes as cls
        combo = self.object_class
        combo.blockSignals(True)
        combo.clear()
        current = {(m.group, m.subgroup) for m, _p in self.project.objects}
        if len(current) > 1:
            combo.addItem("— several classes —", None)
        for group, sub in cls.class_choices(self.project):
            r, g, b, _a = cls.class_color(self.project, group, sub)
            pix = QtGui.QPixmap(12, 12)
            pix.fill(QtGui.QColor.fromRgbF(r, g, b))
            combo.addItem(QtGui.QIcon(pix), f"{group} › {sub}" if sub else group, (group, sub))
        if len(current) == 1:
            want = next(iter(current))
            combo.setCurrentIndex(max(0, next((i for i in range(combo.count())
                                               if combo.itemData(i) is not None
                                               and tuple(combo.itemData(i)) == want), 0)))
        combo.blockSignals(False)

    def _fill_partial(self) -> None:
        """The selected objects' shape: complete, cut by the image edge (automatic), or marked partial."""
        from modalytics.core.model import EDGE_SHARE, PARTIAL_EDGE, PARTIAL_MARKED
        share = (self.project.profile.edge_share if self.project.profile is not None else EDGE_SHARE)
        reasons = [m.parent.partial_reason(p, share) for m, p in self.project.objects if m.parent is not None]
        edge, marked = reasons.count(PARTIAL_EDGE), reasons.count(PARTIAL_MARKED)
        self.partial.blockSignals(True)
        self.partial.setTristate(False)
        if reasons and all(reasons):
            self.partial.setCheckState(QtCore.Qt.CheckState.Checked)
        elif any(reasons):                                  # some of them: shown "partly"; a click marks all
            self.partial.setTristate(True)
            self.partial.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
        else:
            self.partial.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.partial.setEnabled(bool(reasons) and edge < len(reasons))   # cut by the edge: nothing to mark
        self.partial.blockSignals(False)
        bits = ([f"{edge} cut by the image edge (found by itself)"] if edge else []) + \
            ([f"{marked} marked partial"] if marked else [])
        self.partial_note.setText(" · ".join(bits))
        self.partial_note.setVisible(bool(bits))

    # 2. Widgets → model
    def _set_partial(self, checked: bool) -> None:
        from modalytics.services.partial import set_partial
        objects = list(self.project.objects)
        cmd = set_partial(self.project, objects, bool(checked))
        if cmd is not None:
            self.undo.push(cmd)
        self._fill_partial()

    def _set_object_class(self, index: int) -> None:
        from modalytics.services.classes import change_class
        data = self.object_class.itemData(index)
        if data is None or not self.project.objects:
            return
        cmd, moved = change_class(self.project, list(self.project.objects), *data)
        if cmd is not None:
            self.undo.push(cmd)
            self.project.select_objects(moved)

    def _set_outlined(self, image, group: str, on: bool) -> None:
        from modalytics.core.commands import Command
        if image.fully_outlined(group) == on:
            return
        self.undo.push(Command(f"{'Mark' if on else 'Unmark'} all {group} outlined on {image.name}",
                               lambda: self.project.set_fully_outlined(image, group, on),
                               lambda: self.project.set_fully_outlined(image, group, not on)))

    def _relink(self) -> None:
        from modalytics.ui.relink import relink
        if isinstance(self.node, ImageNode):
            relink(self, self.project, self.undo, self.node)

    def _set_pixel_size(self, value: float) -> None:
        img = self.node
        if not isinstance(img, ImageNode) or value <= 0:
            return
        old, new = img.pixel_size_um, (value, value)
        self.undo.push(Command(f"Pixel size of {img.name}",
                               lambda: self.project.set_pixel_size(img, new),
                               lambda: self.project.set_pixel_size(img, old)))

    def _set_placement(self, _value=None) -> None:
        img = self.node
        if not isinstance(img, ImageNode):
            return
        old = img.placement
        new = replace(old, x_um=self.x.value(), y_um=self.y.value(),
                      rotation_deg=self.rotation.value(), scale=self.scale.value())
        if new != old:
            self.undo.push(set_placement_command(self.project, img, old, new))

    def _opacity_live(self, value: int) -> None:
        if self.node is None:
            return
        self.opacity_label.setText(f"{value} %")
        if self._opacity_start is None:      # keyboard / click on the groove: one undo step each
            self.undo.push(set_opacity_command(self.project, self.node, self.node.opacity, value / 100))
        else:
            self.project.set_opacity(self.node, value / 100)

    def _opacity_commit(self) -> None:
        if self.node is not None and self._opacity_start is not None and self._opacity_start != self.node.opacity:
            self.undo.push(set_opacity_command(self.project, self.node, self._opacity_start, self.node.opacity),
                           already_done=True)
        self._opacity_start = None

    def _set_display(self, **changes) -> None:
        if isinstance(self.node, ImageNode) and not self.node.view_source.is_rgb:
            self.project.set_display(self.node, **changes)
