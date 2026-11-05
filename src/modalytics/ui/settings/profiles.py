"""
Settings › Profiles: create, edit, share and remove analysis profiles.

    ┌ Profiles ─────────────┬ Colonic tissue ───────────────────────────────┐
    │ TISSUE                │ Name, kind, description                        │
    │   Colonic tissue      │ Image types   ☑ Whole-slide ☑ Phase ☐ Bright…  │
    │ CELLS                 │ Classes       Gland › Normal, Low grade…       │
    │   Blood cells         │ Segmentation  model per class and image type   │
    │                       │ Parameters    ☑ Area ☑ Disorder… + constants   │
    │ [New][Duplicate]…     │ [Save] [Revert]                                │
    └───────────────────────┴────────────────────────────────────────────────┘

Edits apply to a working copy; Save writes it to the library (its revision
goes up), Revert drops them. Nothing is locked: built-in profiles are edited
like any other, and "Reset to built-in" restores the shipped version.
Projects are not affected by edits: each keeps its own copy of its profile.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import paths
from modalytics.core.annotations import Group, Label
from modalytics.core.model import ACQUIRED_TYPES, IMAGE_TYPES
from modalytics.parameters.registry import ANY, SCOPES, TYPE_NAMES, ParameterLibrary
from modalytics.profiles import bundle
from modalytics.profiles.library import ProfileLibrary
from modalytics.profiles.model import KINDS, Profile, ProfileError
from modalytics.ui.settings.widgets import (clear_tree, group_item, is_group, keep_on, list_tree, muted, section,
                                            setting_widget, swatch)

UID = QtCore.Qt.ItemDataRole.UserRole
NONE = "—"
DEFAULT_IMAGES = {"tissue": ["wsi", "phase"], "cells": ["phase"], "kinetics": ["phase"]}


def _installed_models():
    from modalytics.services.segmentation import find_models
    return find_models()


class NewProfileDialog(QtWidgets.QDialog):
    """Name and kind of a new profile."""

    def __init__(self, library: ProfileLibrary, kind: str = "tissue", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New profile")
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. Gastric tissue, Blood cells")
        self.kind = QtWidgets.QComboBox()
        for k, v in KINDS.items():
            self.kind.addItem(v, k)
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        form.addRow("Name", self.name)
        form.addRow("Kind", self.kind)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(lambda: self.accept() if self.name.text().strip() else None)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.library = library


class ProfilesPage(QtWidgets.QWidget):
    """The profile library page.

    Args:
        library: Profile library (default: the user's).
        parameters: Parameter library (shared with the Parameters page).

    Signals:
        profiles_changed(): a profile was saved, added or removed (the start screen refreshes).
    """

    profiles_changed = QtCore.Signal()

    def __init__(self, library: ProfileLibrary | None = None, parameters: ParameterLibrary | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.library = library or ProfileLibrary()
        self.parameters = parameters or ParameterLibrary()
        self.models = _installed_models()
        self.pid: str | None = None              # library id of the profile being edited
        self.profile: Profile | None = None      # its working copy
        self.dirty = False
        self._loading = False

        split = QtWidgets.QSplitter()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        # 1. Left: the profiles, grouped by kind, and the library actions
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.list = list_tree()
        self.list.currentItemChanged.connect(self._list_clicked)
        lc.addWidget(self.list, 1)
        grid = QtWidgets.QGridLayout()
        self.btn = {}
        for i, (key, text, slot) in enumerate((("new", "New…", self._new), ("dup", "Duplicate", self._duplicate),
                                              ("del", "Delete", self._delete), ("imp", "Import…", self._import),
                                              ("exp", "Export…", self._export),
                                              ("reset", "Reset to built-in", self._reset))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 2, i % 2)
            self.btn[key] = b
        lc.addLayout(grid)
        split.addWidget(left)

        # 2. Right: the editor (scrolls) and Save / Revert
        right = QtWidgets.QWidget()
        rc = QtWidgets.QVBoxLayout(right)
        rc.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.editor = QtWidgets.QWidget()
        scroll.setWidget(self.editor)
        rc.addWidget(scroll, 1)
        self._build_editor()
        bottom = QtWidgets.QHBoxLayout()
        self.problems = QtWidgets.QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #e5a03c;")
        bottom.addWidget(self.problems, 1)
        self.revert_btn = QtWidgets.QPushButton("Revert")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setDefault(True)
        self.revert_btn.clicked.connect(lambda: self._open(self.pid))
        self.save_btn.clicked.connect(self.save)
        bottom.addWidget(self.revert_btn)
        bottom.addWidget(self.save_btn)
        rc.addLayout(bottom)
        split.addWidget(right)
        split.setSizes([240, 660])
        self.refresh()

    # ------------------------------------------------------------ editor layout
    def _build_editor(self) -> None:
        col = QtWidgets.QVBoxLayout(self.editor)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit()
        self.kind = QtWidgets.QComboBox()
        for k, v in KINDS.items():
            self.kind.addItem(v, k)
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(54)
        self.description.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form.addRow("Name", self.name)
        form.addRow("Kind", self.kind)
        form.addRow("Description", self.description)
        types_box = QtWidgets.QWidget()
        types_grid = QtWidgets.QGridLayout(types_box)
        types_grid.setContentsMargins(0, 0, 0, 0)
        self.types = {}
        maps = [t for t in IMAGE_TYPES if t not in ACQUIRED_TYPES]
        for i, t in enumerate(list(ACQUIRED_TYPES) + maps):       # the usual imports first, then maps
            label = IMAGE_TYPES[t]
            b = QtWidgets.QCheckBox(label.split(" (")[0])
            b.setToolTip(label if t in ACQUIRED_TYPES else
                         f"{label}: usually made by processing; tick it to import such maps directly "
                         "(computed by another program)")
            self.types[t] = b
            row, col_ = (0, i) if t in ACQUIRED_TYPES else (1 + (i - 3) // 4, (i - 3) % 4)
            types_grid.addWidget(b, row, col_)
        types_grid.setColumnStretch(4, 1)
        form.addRow("Image types", types_box)
        col.addLayout(form)
        self.name.textEdited.connect(self._edited)
        self.kind.currentIndexChanged.connect(self._edited)
        self.description.textChanged.connect(self._edited)
        for b in self.types.values():
            b.toggled.connect(self._types_changed)

        # Classes
        col.addWidget(section("Classes"))
        self.classes = QtWidgets.QTreeWidget()
        self.classes.setHeaderHidden(True)
        self.classes.setFixedHeight(150)
        self.classes.itemChanged.connect(self._class_renamed)
        self.classes.currentItemChanged.connect(lambda *_a: self._show_order())
        col.addWidget(self.classes)
        crow = QtWidgets.QHBoxLayout()
        for text, slot in (("Add class", self._add_group), ("Add subclass", self._add_sub),
                           ("Colour…", self._recolor), ("Remove", self._remove_class)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            crow.addWidget(b)
        self.up_btn, self.down_btn = QtWidgets.QPushButton("▲"), QtWidgets.QPushButton("▼")
        for b, step in ((self.up_btn, -1), (self.down_btn, 1)):
            b.setFixedWidth(34)
            b.setToolTip("Move the subclass up or down: the order of grades on their scale (lowest first)")
            b.clicked.connect(lambda _c=False, st=step: self._move_sub(st))
            crow.addWidget(b)
        crow.addStretch(1)
        col.addLayout(crow)
        self.ordered = QtWidgets.QCheckBox("Its subclasses are grades in order (a scale, lowest first)")
        self.ordered.setToolTip("On: grades on a scale (e.g. low < high grade dysplasia) — trend tests, and "
                                "classifiers that climb the scale (Ranges, Score). Off: categories with no order (e.g. "
                                "four pathologies) — each class against the others, the learned classifiers only.")
        self.ordered.toggled.connect(self._ordered_changed)
        col.addWidget(self.ordered)

        # Segmentation: a model per class and image type
        col.addWidget(section("Segmentation"))
        col.addWidget(muted("The model that segments each class on each image type (installed models only)."))
        self.seg = QtWidgets.QTableWidget()
        self.seg.verticalHeader().setVisible(True)
        self.seg.setFixedHeight(120)
        col.addWidget(self.seg)

        # Registration: the Register tool's default method in this profile's projects
        col.addWidget(section("Registration"))
        col.addWidget(muted("The methods the Register tool offers in its projects (Arrange tab) and the one it "
                            "starts with. Experimental methods are off until ticked: test them on your images first "
                            "(Settings › Registration › Test)."))
        self.reg_methods = list_tree(["Method", "Default", "Available"])
        self.reg_methods.setColumnWidth(0, 300)
        self.reg_methods.setColumnWidth(1, 70)
        self.reg_methods.setFixedHeight(190)
        self.reg_methods.itemChanged.connect(self._reg_toggled)
        col.addWidget(self.reg_methods)

        # Processing at import: per image type, the chains that make views of each new image
        col.addWidget(section("Processing at import"))
        col.addWidget(muted("Processing chains applied to every image of that type when it is imported: each makes "
                            "a view of the image (same masks). Ticked = measured with parameters: its results are "
                            "the standard ones for its image type."))
        from modalytics.processing.registry import ProcessingLibrary
        self.processing_library = ProcessingLibrary()
        self.proc = list_tree(["Chain", "Steps"])
        self.proc.setColumnWidth(0, 230)
        self.proc.setFixedHeight(130)
        self.proc.itemChanged.connect(self._chain_toggled)
        self.proc.currentItemChanged.connect(lambda cur, prev: keep_on(self.proc, prev) if is_group(cur) else None)
        self.proc.itemDoubleClicked.connect(lambda item, _c: self._edit_chain())
        col.addWidget(self.proc)
        prow2 = QtWidgets.QHBoxLayout()
        self.add_chain_btn = QtWidgets.QPushButton("Add chain ▾")
        self.add_chain_btn.clicked.connect(self._add_chain_menu)
        prow2.addWidget(self.add_chain_btn)
        for text, slot in (("Edit…", self._edit_chain), ("Remove", self._remove_chain)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            prow2.addWidget(b)
        prow2.addStretch(1)
        col.addLayout(prow2)

        # Processing steps: which steps and chains its projects offer, and which get a button on the Process tab
        col.addWidget(section("Processing steps"))
        col.addWidget(muted("The steps and your chains its projects offer. Toolbar: a button on the Process tab "
                            "(and in Quick step). Available only: in Quick step, Apply chain and the chain editor. "
                            "Neither: hidden. Only steps for the profile's image types are listed; a button's "
                            "label and icon are the step's (Settings › Processing)."))
        srow = QtWidgets.QHBoxLayout()
        self.steps_count = muted("", wrap=False)
        srow.addWidget(self.steps_count, 1)
        b = QtWidgets.QPushButton("Defaults")
        b.setToolTip("Every step available; the default buttons in the toolbar")
        b.clicked.connect(self._steps_defaults)
        srow.addWidget(b)
        col.addLayout(srow)
        self.steps = list_tree(["Step", "Toolbar", "Available"])
        self.steps.setColumnWidth(0, 300)
        self.steps.setColumnWidth(1, 70)
        self.steps.setMinimumHeight(240)
        self.steps.itemChanged.connect(self._step_toggled)
        self.steps.currentItemChanged.connect(lambda cur, prev: keep_on(self.steps, prev) if is_group(cur) else None)
        col.addWidget(self.steps)

        # Parameters, with the constants of the selected one
        col.addWidget(section("Parameters"))
        prow = QtWidgets.QHBoxLayout()
        self.param_count = muted("", wrap=False)
        prow.addWidget(self.param_count, 1)
        for text, which in (("Defaults", "defaults"), ("All", "all"), ("None", "none")):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, w=which: self._select_params(w))
            prow.addWidget(b)
        col.addLayout(prow)
        self.params = list_tree(["Parameter", "Unit", "Report", "In PDF"])
        self.params.setColumnWidth(0, 300)
        self.params.setColumnWidth(1, 70)
        self.params.setColumnWidth(2, 70)
        self.params.headerItem().setToolTip(2, "Key parameters of the report: summarised per class and plotted "
                                               "(the full tables stay in Results and the CSV export)")
        self.params.headerItem().setToolTip(3, "Parameters of the report's PDF summary (paper cannot scroll: keep "
                                               "them few). None ticked: the classifier's, then the key parameters, "
                                               "at most six")
        self.params.setMinimumHeight(260)
        self.params.itemChanged.connect(self._param_toggled)
        self.params.currentItemChanged.connect(
            lambda cur, prev: keep_on(self.params, prev) if is_group(cur) else self._show_param_constants(cur))
        col.addWidget(self.params)
        self.param_constants_title = muted("Select a parameter to see its constants.")
        col.addWidget(self.param_constants_title)
        self.param_form = QtWidgets.QFormLayout()
        col.addLayout(self.param_form)
        erow = QtWidgets.QHBoxLayout()
        erow.addWidget(QtWidgets.QLabel("Cut by the image edge when more than"))
        self.edge_percent = QtWidgets.QDoubleSpinBox()
        self.edge_percent.setRange(0, 50)
        self.edge_percent.setDecimals(1)
        self.edge_percent.setSingleStep(1)
        self.edge_percent.setSuffix(" %")
        self.edge_percent.setToolTip("An object touching the image border has an incomplete shape (left out of "
                                     "classifier training and the shape statistics) only when more than this share "
                                     "of its outline lies on the border; one grazing it keeps nearly all its shape. "
                                     "0 %: any touch.")
        self.edge_percent.valueChanged.connect(self._edge_changed)
        erow.addWidget(self.edge_percent)
        erow.addWidget(QtWidgets.QLabel("of its outline lies on it"))
        erow.addStretch(1)
        col.addLayout(erow)

        col.addWidget(section("Scripts"))
        col.addWidget(muted("External scripts its projects offer in the Scripts menu (between Tools and Help): "
                            "tasks beside the analysis, e.g. gathering an instrument's files. None ticked: no "
                            "Scripts menu."))
        self.scripts = QtWidgets.QListWidget()
        self.scripts.setMaximumHeight(110)
        self.scripts.itemChanged.connect(self._script_toggled)
        col.addWidget(self.scripts)
        col.addWidget(section("Classification"))
        self.classifiers = list_tree(["Group", "Classifier", "Made from"])
        self.classifiers.setColumnWidth(0, 120)
        self.classifiers.setColumnWidth(1, 300)
        self.classifiers.setMaximumHeight(120)
        col.addWidget(self.classifiers)
        self.classifiers.currentItemChanged.connect(lambda *_a: self._show_confidence())
        crow = QtWidgets.QHBoxLayout()
        remove = QtWidgets.QPushButton("Remove classifier")
        remove.clicked.connect(self._remove_classifier)
        crow.addWidget(remove)
        crow.addWidget(QtWidgets.QLabel("Minimum confidence"))
        self.confidence = QtWidgets.QDoubleSpinBox()
        self.confidence.setRange(0.5, 1.0)
        self.confidence.setDecimals(2)
        self.confidence.setSingleStep(0.05)
        self.confidence.setToolTip("For the selected group: a new object (Segment, Annotate › Classify selected) is "
                                   "given its classifier's grade from this confidence; below it, it stays the plain "
                                   "group. Classifiers giving a probability only (logistic regression, random "
                                   "forest, gradient boosting); Ranges' and Score's grades are always suggested.")
        self.confidence.valueChanged.connect(self._confidence_changed)
        crow.addWidget(self.confidence)
        crow.addWidget(muted("Classifiers are made in Training (Build, then Save to profile)."), 1)
        col.addLayout(crow)
        col.addWidget(section("Report"))
        col.addWidget(muted("Report layouts are added in stage 6."))
        col.addStretch(1)

    # ------------------------------------------------------------ list
    def refresh(self, select: str | None = None) -> None:
        """Rebuild the profile list (keeps or sets the selection)."""
        select = select or self.pid
        self.list.blockSignals(True)
        self.list.clear()
        item_to_select = None
        for kind, kind_name in KINDS.items():
            top = group_item(self.list, kind_name)
            for e in self.library.entries(kind=kind):
                it = QtWidgets.QTreeWidgetItem(top, [e.profile.name])
                it.setData(0, UID, e.id)
                it.setToolTip(0, e.profile.description)
                if e.id == select:
                    item_to_select = it
            top.setExpanded(True)
        broken = [e for e in self.library.entries() if e.profile is None]
        if broken:
            top = group_item(self.list, "Could not read")
            for e in broken:
                it = QtWidgets.QTreeWidgetItem(top, [e.path.name])
                it.setToolTip(0, e.error)
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            top.setExpanded(True)
        self.list.blockSignals(False)
        if item_to_select is None:
            it = QtWidgets.QTreeWidgetItemIterator(self.list)
            while it.value() and it.value().data(0, UID) is None:
                it += 1
            item_to_select = it.value()
        if item_to_select is not None:
            self.list.setCurrentItem(item_to_select)
        else:
            self._open(None)

    def _list_clicked(self, cur, prev) -> None:
        if is_group(cur):                                # a group row only opens / closes
            keep_on(self.list, prev)
            return
        pid = cur.data(0, UID) if cur else None
        if pid is None or pid == self.pid:
            return
        if not self.confirm_leave():
            self.list.blockSignals(True)
            self.list.setCurrentItem(prev)
            self.list.blockSignals(False)
            return
        self._open(pid)

    def confirm_leave(self) -> bool:
        """Ask what to do with unsaved edits. Returns False to stay."""
        if not self.dirty:
            return True
        btn = QtWidgets.QMessageBox.question(
            self, "Unsaved profile", f"Save the changes to “{self.profile.name}”?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if btn == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return btn == QtWidgets.QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------ open / save
    def _open(self, pid: str | None) -> None:
        """Load a profile into the editor (None: empty, disabled)."""
        self.pid, self.dirty = pid, False
        try:
            self.profile = self.library.get(pid) if pid else None
            if self.profile is not None:                 # ids from before image types → current ids
                self.profile.parameters = self.parameters.migrate(self.profile.parameters)
        except (KeyError, ProfileError) as exc:
            QtWidgets.QMessageBox.warning(self, "Profile", str(exc))
            self.profile = None
        self.editor.setEnabled(self.profile is not None)
        for k in ("dup", "del", "exp"):
            self.btn[k].setEnabled(self.profile is not None)
        self.btn["reset"].setEnabled(bool(pid) and self.library.is_builtin(pid))
        self._loading = True
        p = self.profile or Profile("")
        self.name.setText(p.name)
        self.kind.setCurrentIndex(max(0, self.kind.findData(p.kind)))
        self.description.setPlainText(p.description)
        for t, b in self.types.items():
            b.setChecked(t in p.images)
        self._fill_classes()
        self._fill_segmentation()
        self._fill_processing()
        self._fill_parameters()
        self._fill_steps()
        self._fill_registration()
        self._fill_classifiers()
        self._fill_scripts()
        self.edge_percent.setValue(100 * p.edge_share)
        self._loading = False
        self._update_state()

    def reload_current(self) -> None:
        """Read the open profile again (when it has no unsaved edits): steps or chains may have changed."""
        self.processing_library.reload()
        if not self.dirty:
            self._open(self.pid)

    def _collect(self) -> None:
        """Copy the simple fields from the widgets into the working copy."""
        p = self.profile
        p.name = self.name.text().strip()
        p.kind = self.kind.currentData()
        p.description = self.description.toPlainText().strip()
        p.images = [t for t, b in self.types.items() if b.isChecked()]

    def save(self) -> bool:
        """Write the working copy to the library. Returns True if saved."""
        if self.profile is None:
            return False
        self._collect()
        if not self.profile.name:
            QtWidgets.QMessageBox.warning(self, "Profile", "The profile needs a name.")
            return False
        clash = self.library.find(self.profile.name)
        if clash and clash != self.pid:
            QtWidgets.QMessageBox.warning(self, "Profile", f"Another profile is called “{self.profile.name}”.")
            return False
        self.pid = self.library.save(self.profile, self.pid)
        self.dirty = False
        self.refresh(self.pid)
        self._update_state()
        self.profiles_changed.emit()
        return True

    def _edited(self, *_args) -> None:
        if not self._loading and self.profile is not None:
            self.dirty = True
            self._update_state()

    def _update_state(self) -> None:
        self.save_btn.setEnabled(self.dirty)
        self.revert_btn.setEnabled(self.dirty)
        if self.profile is None:
            self.problems.setText("")
            return
        self._collect()
        found = self.profile.problems({m.id for m in self.models}, set(self.parameters.specs)) + \
            self.profile.chain_problems(self.processing_library)
        self.problems.setText(("⚠ " + " · ".join(found)) if found else "")
        n = sum(1 for u in self.profile.parameters if u in self.parameters.specs)
        self.param_count.setText(f"{n} of {len(self.parameters.specs)} parameters measured")

    # ------------------------------------------------------------ image types
    def _types_changed(self, _on) -> None:
        if self._loading or self.profile is None:
            return
        self._collect()
        self.profile.processing = {t: s for t, s in self.profile.processing.items() if t in self.profile.images}
        self._fill_segmentation()
        self._fill_processing()
        self._fill_parameters()
        self._fill_steps()
        self._fill_registration()
        self._edited()

    # ------------------------------------------------------------ classes
    def _fill_classes(self) -> None:
        self.classes.blockSignals(True)
        clear_tree(self.classes)
        for g in (self.profile.classes.groups if self.profile else []):
            gi = QtWidgets.QTreeWidgetItem([g.name])
            if len(g.subgroups) >= 2:                        # what its subclasses are, on hover (the name is editable)
                gi.setToolTip(0, ("Grades in order: " + " < ".join(s.name for s in g.subgroups)) if g.ordered
                              else "Categories (no order): " + ", ".join(s.name for s in g.subgroups))
            gi.setIcon(0, swatch(g.color))
            gi.setFlags(gi.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            gi.setData(0, UID, (g.name, ""))
            self.classes.addTopLevelItem(gi)
            for s in g.subgroups:
                si = QtWidgets.QTreeWidgetItem(gi, [s.name])
                si.setIcon(0, swatch(s.color))
                si.setFlags(si.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                si.setData(0, UID, (g.name, s.name))
            gi.setExpanded(True)
        self.classes.blockSignals(False)
        self._show_order()

    def _selected_class(self):
        it = self.classes.currentItem()
        if it is None:
            return None, None
        group, sub = it.data(0, UID)
        g = next((g for g in self.profile.classes.groups if g.name == group), None)
        s = next((s for s in g.subgroups if s.name == sub), None) if g and sub else None
        return g, s

    def _unique(self, names, base: str) -> str:
        out, n = base, 2
        while out in names:
            out, n = f"{base} {n}", n + 1
        return out

    def _add_group(self) -> None:
        name = self._unique({g.name for g in self.profile.classes.groups}, "New class")
        self.profile.classes.groups.append(Group(name, (0.7, 0.7, 0.7, 1.0), []))
        self._classes_changed()

    def _add_sub(self) -> None:
        g, _s = self._selected_class()
        if g is None:
            QtWidgets.QMessageBox.information(self, "Add subclass", "Select the class it belongs to first.")
            return
        g.subgroups.append(Label(self._unique({s.name for s in g.subgroups}, "New subclass"), g.color))
        self._classes_changed()

    def _recolor(self) -> None:
        g, s = self._selected_class()
        target = s or g
        if target is None:
            return
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*target.color[:3]), self, "Class colour")
        if c.isValid():
            target.color = (round(c.redF(), 4), round(c.greenF(), 4), round(c.blueF(), 4), 1.0)
            self._classes_changed()

    def _remove_class(self) -> None:
        g, s = self._selected_class()
        if s is not None:
            g.subgroups.remove(s)
        elif g is not None:
            self.profile.classes.groups.remove(g)
            self.profile.segmentation.pop(g.name, None)
            self.profile.classification.pop(g.name, None)
        self._classes_changed()

    def _class_renamed(self, item, _col) -> None:
        group, sub = item.data(0, UID)
        new = item.text(0).strip()
        g = next((g for g in self.profile.classes.groups if g.name == group), None)
        if g is None or not new:
            self._fill_classes()
            return
        if sub:
            s = next(s for s in g.subgroups if s.name == sub)
            if any(x.name == new for x in g.subgroups if x is not s):
                self._fill_classes()
                return
            s.name = new
        else:
            if any(x.name == new for x in self.profile.classes.groups if x is not g):
                self._fill_classes()
                return
            g.name = new
            for d in (self.profile.segmentation, self.profile.classification):
                if group in d:
                    d[new] = d.pop(group)
        self._classes_changed()

    def _fill_scripts(self) -> None:
        """Every external script here, ticked if the profile offers it (one it names but is not here: shown)."""
        from modalytics.scripts.registry import ScriptLibrary
        self.scripts.blockSignals(True)
        self.scripts.clear()
        chosen = list(self.profile.scripts) if self.profile else []
        specs = ScriptLibrary().specs
        for uid in list(specs) + [u for u in chosen if u not in specs]:
            spec = specs.get(uid)
            item = QtWidgets.QListWidgetItem(f"{spec.name}  ({spec.group})" if spec else f"{uid}  (not here)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, uid)
            item.setToolTip(spec.description if spec else "This profile offers a script that is not installed here")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if uid in chosen else QtCore.Qt.CheckState.Unchecked)
            self.scripts.addItem(item)
        self.scripts.blockSignals(False)

    def _script_toggled(self, item) -> None:
        if self.profile is None:
            return
        uid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        on = item.checkState() == QtCore.Qt.CheckState.Checked
        self.profile.scripts = [u for u in self.profile.scripts if u != uid] + ([uid] if on else [])
        self._edited()

    def _fill_classifiers(self) -> None:
        """Each group's classifier: what it is and where it comes from."""
        clear_tree(self.classifiers)
        if self.profile is None:
            return
        for g in self.profile.classes.groups:
            clf = self.profile.classifier_for(g.name)
            if clf is None:
                if g.name in self.profile.classification:
                    QtWidgets.QTreeWidgetItem(self.classifiers, [g.name, f"model {self.profile.classification[g.name]}", ""])
                continue
            from modalytics.training.classifiers import describe
            what = describe(clf)                             # every kind: its parameters, weights, rules or points
            item = QtWidgets.QTreeWidgetItem(self.classifiers, [g.name, what, clf.note])
            item.setData(0, UID, g.name)
            item.setToolTip(1, what)
            item.setToolTip(2, clf.note)
        self._show_confidence()

    def _show_confidence(self) -> None:
        """The minimum confidence of the selected group (enabled for a classifier giving a probability)."""
        item = self.classifiers.currentItem() or (self.classifiers.topLevelItem(0)
                                                   if self.classifiers.topLevelItemCount() else None)
        clf = self.profile.classifier_for(item.text(0)) if item is not None and self.profile is not None else None
        self.confidence.blockSignals(True)
        self.confidence.setValue(self.profile.confidence_for(item.text(0)) if clf is not None else 0.8)
        self.confidence.blockSignals(False)
        self.confidence.setEnabled(clf is not None and clf.learned)

    def _confidence_changed(self, value: float) -> None:
        item = self.classifiers.currentItem() or (self.classifiers.topLevelItem(0)
                                                   if self.classifiers.topLevelItemCount() else None)
        if item is None or self.profile is None:
            return
        self.profile.min_confidence[item.text(0)] = float(value)
        self._edited()

    def _remove_classifier(self) -> None:
        item = self.classifiers.currentItem()
        if item is None or self.profile is None:
            return
        self.profile.classification.pop(item.text(0), None)
        self._fill_classifiers()
        self._edited()

    def _show_order(self) -> None:
        """The ordered tick and the up / down buttons follow the selected class (or subclass)."""
        g, s = self._selected_class() if self.profile is not None else (None, None)
        self.ordered.blockSignals(True)
        self.ordered.setChecked(bool(g and g.ordered))
        self.ordered.blockSignals(False)
        self.ordered.setEnabled(g is not None and len(g.subgroups) >= 2)
        self.ordered.setText(f"{g.name}: its subclasses are grades in order (a scale, lowest first)" if g else
                             "Its subclasses are grades in order (a scale, lowest first)")
        for b in (self.up_btn, self.down_btn):
            b.setEnabled(s is not None)

    def _ordered_changed(self, on: bool) -> None:
        g, _s = self._selected_class()
        if g is not None and g.ordered != on:
            g.ordered = bool(on)
            self._classes_changed()

    def _move_sub(self, step: int) -> None:
        g, s = self._selected_class()
        if g is None or s is None:
            return
        i = g.subgroups.index(s)
        j = i + step
        if 0 <= j < len(g.subgroups):
            g.subgroups[i], g.subgroups[j] = g.subgroups[j], g.subgroups[i]
            self._classes_changed()
            for k in range(self.classes.topLevelItemCount()):        # the moved subclass stays selected
                top = self.classes.topLevelItem(k)
                for c in range(top.childCount()):
                    if top.child(c).data(0, UID) == (g.name, s.name):
                        self.classes.setCurrentItem(top.child(c))

    def _classes_changed(self) -> None:
        self._fill_classes()
        self._fill_segmentation()
        self._fill_classifiers()
        self._edited()

    # ------------------------------------------------------------ segmentation
    def _fill_segmentation(self) -> None:
        p = self.profile
        groups = [g.name for g in p.classes.groups] if p else []
        types = [t for t in IMAGE_TYPES if p and t in p.images]
        self.seg.clear()
        self.seg.setRowCount(len(groups))
        self.seg.setColumnCount(len(types))
        self.seg.setHorizontalHeaderLabels([IMAGE_TYPES[t].split(" (")[0] for t in types])
        self.seg.setVerticalHeaderLabels(groups)
        self.seg.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r, group in enumerate(groups):
            for c, t in enumerate(types):
                combo = QtWidgets.QComboBox()
                combo.addItem(NONE, "")
                for m in self.models:
                    if m.modality == t:
                        combo.addItem(m.name, m.id)
                current = p.model_for(group, t) or ""
                i = combo.findData(current)
                if i < 0 and current:                       # named but not installed here
                    combo.addItem(f"{current} (not installed)", current)
                    i = combo.count() - 1
                combo.setCurrentIndex(max(i, 0))
                combo.currentIndexChanged.connect(
                    lambda _i, g=group, ty=t, cb=combo: self._set_model(g, ty, cb.currentData()))
                self.seg.setCellWidget(r, c, combo)

    def _set_model(self, group: str, image_type: str, model_id: str) -> None:
        per = self.profile.segmentation.setdefault(group, {})
        if model_id:
            per[image_type] = model_id
        else:
            per.pop(image_type, None)
            if not per:
                self.profile.segmentation.pop(group, None)
        self._edited()

    # ------------------------------------------------------------ parameters
    def _fill_parameters(self) -> None:
        self.params.blockSignals(True)
        clear_tree(self.params)
        p = self.profile
        images = set(p.images) | set(p.made_types(self.processing_library)) if p else set()
        # Scope → image type (Any image + the types this profile accepts or makes) → category → parameter
        for scope, scope_name in SCOPES.items():
            if not self.parameters.of_scope(scope):
                continue
            top = group_item(self.params, scope_name)
            for image_type, type_name in TYPE_NAMES.items():
                if image_type != ANY and image_type not in images:
                    continue                             # the profile does not take these images
                of_type = self.parameters.of_type(scope, image_type)
                if not of_type:
                    continue
                tnode = group_item(top, type_name)
                cats = {}
                for spec in of_type:
                    cat = cats.get(spec.category)
                    if cat is None:
                        cat = cats[spec.category] = group_item(tnode, spec.category.capitalize())
                    it = QtWidgets.QTreeWidgetItem(cat, [spec.label, spec.unit])
                    it.setData(0, UID, spec.uid)
                    it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(0, QtCore.Qt.CheckState.Checked if p and spec.uid in p.parameters
                                     else QtCore.Qt.CheckState.Unchecked)
                    it.setCheckState(2, QtCore.Qt.CheckState.Checked if p and spec.uid in p.report_parameters
                                     else QtCore.Qt.CheckState.Unchecked)
                    it.setCheckState(3, QtCore.Qt.CheckState.Checked if p and spec.uid in p.pdf_parameters
                                     else QtCore.Qt.CheckState.Unchecked)
                    it.setToolTip(0, spec.help_text())
                tnode.setExpanded(True)
            top.setExpanded(True)
        self.params.blockSignals(False)
        self._show_param_constants(None)

    # ------------------------------------------------------------ registration
    def _fill_registration(self) -> None:
        """The methods that can move one of the profile's image types (validated first): Default, Available."""
        from modalytics.registration.registry import RegistrationLibrary
        self._reg_library = RegistrationLibrary()
        p = self.profile
        tree = self.reg_methods
        tree.blockSignals(True)
        clear_tree(tree)
        if p is not None:
            on, off = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked
            default = p.registration_default()
            for spec in sorted(self._reg_library.specs.values(), key=lambda s_: s_.experimental):
                if not any(spec.can_move(t) for t in p.images):
                    continue
                it = QtWidgets.QTreeWidgetItem(tree, [spec.name + (" (experimental)" if spec.experimental else ""),
                                                      "", ""])
                it.setData(0, UID, spec.key)
                it.setToolTip(0, spec.description + (f"\n\n{spec.references_text()}" if spec.references else ""))
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(1, on if spec.key == default else off)
                it.setCheckState(2, on if p.registration_offered(spec.key, spec.experimental) else off)
        tree.blockSignals(False)

    def _reg_toggled(self, item, col) -> None:
        """Default: one method (it is available too); Available: the default stays available."""
        key = item.data(0, UID)
        spec = self._reg_library.get(key) if key else None
        if spec is None or self.profile is None or col not in (1, 2):
            return
        on = item.checkState(col) == QtCore.Qt.CheckState.Checked
        p = self.profile
        if col == 1:
            p.set_registration_default(key if on else "", spec.experimental)
        elif key != p.registration_default():
            p.set_registration_offered(key, on, spec.experimental)
        QtCore.QTimer.singleShot(0, self._fill_registration)    # not inside the item's own signal
        self._edited()

    # ------------------------------------------------------------ processing steps
    def _fill_steps(self) -> None:
        """Category › step (then your chains): Toolbar and Available ticks, from the profile's choice."""
        from modalytics.processing.chains import ChainLibrary
        from modalytics.processing.registry import WORKFLOW_ORDER
        from modalytics.ui.ribbon import icon
        self.steps.blockSignals(True)
        clear_tree(self.steps)
        p = self.profile
        if p is None:
            self.steps.blockSignals(False)
            return
        types = set(p.images) | set(p.made_types(self.processing_library))
        color = self.palette().color(self.foregroundRole()).name()
        on, off = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked
        counts = {"toolbar": 0, "available": 0, "hidden": 0}

        def add(parent, item_id, text, icon_name, tip):
            place = p.step_place(item_id)
            counts[place] += 1
            it = QtWidgets.QTreeWidgetItem(parent, [text, "", ""])
            it.setData(0, UID, item_id)
            it.setIcon(0, icon(icon_name, color))
            it.setToolTip(0, tip)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(1, on if place == "toolbar" else off)
            it.setCheckState(2, on if place != "hidden" else off)
        groups: dict = {}
        for spec in self.processing_library.specs.values():
            if any(spec.accepts(t) for t in types):
                groups.setdefault(spec.group, []).append(spec)
        for title in [*WORKFLOW_ORDER, *[g for g in groups if g not in WORKFLOW_ORDER]]:
            if not groups.get(title):
                continue
            top = group_item(self.steps, title)
            for spec in sorted(groups[title], key=lambda s_: s_.name.lower()):
                add(top, spec.uid, spec.name, spec.button_icon,
                    f"{spec.description}\nButton: “{spec.button_label}”")
            top.setExpanded(True)
        mine = [c for c in ChainLibrary().chains()
                if any(not c.problems(t, self.processing_library) for t in types)]
        if mine:
            top = group_item(self.steps, "My chains")
            for chain in mine:
                add(top, f"chain:{chain.name}", chain.name, chain.button_icon,
                    f"{chain.summary(self.processing_library)}\nButton: “{chain.button_label}”")
            top.setExpanded(True)
        self.steps_count.setText(f"{counts['toolbar']} in the toolbar · {counts['available']} in Quick step only · "
                                 f"{counts['hidden']} hidden")
        self.steps.blockSignals(False)

    def _step_toggled(self, item, col) -> None:
        """Toolbar implies available; not available means not in the toolbar either."""
        item_id = item.data(0, UID)
        if item_id is None or self.profile is None or col not in (1, 2):
            return
        on = QtCore.Qt.CheckState.Checked
        toolbar, available = item.checkState(1) == on, item.checkState(2) == on
        if col == 1 and toolbar:
            available = True
        if col == 2 and not available:
            toolbar = False
        self.profile.set_step_place(item_id, "toolbar" if toolbar else "available" if available else "hidden")
        QtCore.QTimer.singleShot(0, self._fill_steps)    # not inside the item's own signal
        self._edited()

    def _steps_defaults(self) -> None:
        if self.profile is not None:
            self.profile.steps = {}
            self._fill_steps()
            self._edited()

    # ------------------------------------------------------------ processing at import
    def _fill_processing(self) -> None:
        """Accepted image type › its chains (ticked = measured), in the order they are applied."""
        self.proc.blockSignals(True)
        clear_tree(self.proc)
        p = self.profile
        for image_type in (t for t in IMAGE_TYPES if p and t in p.images):
            chains = p.chains_for(image_type)
            top = group_item(self.proc, f"{IMAGE_TYPES[image_type].split(' (')[0]} ({len(chains)})")
            for i, chain in enumerate(chains):
                it = QtWidgets.QTreeWidgetItem(top, [chain.name, chain.summary(self.processing_library)])
                it.setData(0, UID, (image_type, i))
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.CheckState.Checked if chain.measure else QtCore.Qt.CheckState.Unchecked)
                problems = chain.problems(image_type, self.processing_library)
                it.setToolTip(0, "⚠ " + " · ".join(problems) if problems else
                              "Ticked: measured with parameters (the standard results). Double-click to edit.")
            top.setExpanded(True)
        self.proc.blockSignals(False)

    def _selected_chain(self):
        """(image type, index) of the selected chain row, or None."""
        item = self.proc.currentItem()
        data = item.data(0, UID) if item is not None else None
        return tuple(data) if data else None

    def _set_chains(self, image_type: str, chains: list) -> None:
        if chains:
            self.profile.processing[image_type] = [c.to_dict() for c in chains]
        else:
            self.profile.processing.pop(image_type, None)
        self._fill_processing()
        self._fill_parameters()
        self._fill_steps()
        self._fill_registration()                          # the types its chains make have parameters too
        self._edited()

    def _chain_toggled(self, item, _col) -> None:
        data = item.data(0, UID)
        if data is None or self.profile is None:
            return
        image_type, i = data
        chains = self.profile.chains_for(image_type)
        chains[i].measure = item.checkState(0) == QtCore.Qt.CheckState.Checked
        self._set_chains(image_type, chains)

    def _add_chain_menu(self) -> None:
        """New chain…, one of your chains, or a single step, for each accepted image type."""
        from modalytics.processing.chains import Chain, ChainLibrary
        if self.profile is None:
            return
        menu = QtWidgets.QMenu(self)
        mine = ChainLibrary().chains()
        for image_type in (t for t in IMAGE_TYPES if t in self.profile.images):
            sub = menu.addMenu(f"For {IMAGE_TYPES[image_type].split(' (')[0].lower()} images")
            sub.addAction("New chain…", lambda t=image_type: self._edit_chain(new_for=t))
            fits = [c for c in mine if not c.problems(image_type, self.processing_library)]
            ms = sub.addMenu("From my chains")
            for c in fits:
                ms.addAction(c.name, lambda t=image_type, c=c: self._set_chains(t, self.profile.chains_for(t) + [c]))
            ms.setEnabled(bool(fits))
            one = sub.addMenu("One step")
            for spec in self.processing_library.for_type(image_type):
                one.addAction(spec.name, lambda t=image_type, sp=spec: self._set_chains(
                    t, self.profile.chains_for(t) + [Chain(sp.name, [{"step": sp.uid, "options": sp.resolve()}])]))
        menu.exec(self.add_chain_btn.mapToGlobal(self.add_chain_btn.rect().bottomLeft()))

    def _edit_chain(self, new_for: str | None = None) -> None:
        """The chain editor on the selected chain (or a new one for ``new_for``), on a sample image."""
        from modalytics.processing.chains import Chain
        from modalytics.ui.chain_editor import ChainEditor
        if self.profile is None:
            return
        if new_for is not None:
            image_type, index, chain = new_for, None, Chain("New chain", [])
        else:
            sel = self._selected_chain()
            if sel is None:
                return
            image_type, index = sel
            chain = self.profile.chains_for(image_type)[index]
        dlg = ChainEditor(None, chain, for_type=image_type, library=self.processing_library, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        chains = self.profile.chains_for(image_type)
        if index is None:
            chains.append(dlg.chain)
        else:
            chains[index] = dlg.chain
        self._set_chains(image_type, chains)

    def _remove_chain(self) -> None:
        sel = self._selected_chain()
        if sel is None or self.profile is None:
            return
        image_type, index = sel
        chains = self.profile.chains_for(image_type)
        del chains[index]
        self._set_chains(image_type, chains)

    def _param_toggled(self, item, col) -> None:
        uid = item.data(0, UID)
        if uid is None or self.profile is None:
            return
        on = QtCore.Qt.CheckState.Checked
        report = list(self.profile.report.get("parameters", []))
        pdf = list(self.profile.report.get("pdf", []))
        if col in (2, 3):                                # in the report / its PDF: it is measured too
            chosen = report if col == 2 else pdf
            if item.checkState(col) == on:
                chosen += [uid] if uid not in chosen else []
                if item.checkState(0) != on:
                    self.params.blockSignals(True)
                    item.setCheckState(0, on)
                    self.params.blockSignals(False)
                    self.profile.parameters.setdefault(uid, self.parameters.get(uid).resolve())
            else:
                chosen[:] = [u for u in chosen if u != uid]
        elif item.checkState(0) == on:
            self.profile.parameters.setdefault(uid, self.parameters.get(uid).resolve())
        else:                                            # not measured: not in the report (nor its PDF) either
            self.profile.parameters.pop(uid, None)
            report = [u for u in report if u != uid]
            pdf = [u for u in pdf if u != uid]
            self.params.blockSignals(True)
            item.setCheckState(2, QtCore.Qt.CheckState.Unchecked)
            item.setCheckState(3, QtCore.Qt.CheckState.Unchecked)
            self.params.blockSignals(False)
        self.profile.report = dict(self.profile.report, parameters=report, **({"pdf": pdf} if pdf else {}))
        if not pdf:
            self.profile.report.pop("pdf", None)
        self._show_param_constants(item)
        self._edited()

    def _edge_changed(self, value: float) -> None:
        if self._loading or self.profile is None:
            return
        self.profile.measuring = dict(self.profile.measuring, edge_percent=round(float(value), 2))
        self._edited()

    def _select_params(self, which: str) -> None:
        if self.profile is None:
            return
        images = set(self.profile.images)
        chosen = {}
        for uid, spec in self.parameters.specs.items():
            usable = spec.image_type == ANY or spec.image_type in images
            if usable and (which == "all" or (which == "defaults" and spec.default_on)):
                chosen[uid] = self.profile.parameters.get(uid) or spec.resolve()
        # Keep the order of the library (columns follow it)
        self.profile.parameters = chosen
        self._fill_parameters()
        self._edited()

    def _show_param_constants(self, item) -> None:
        """The selected parameter's constants, with this profile's values (editable)."""
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        uid = item.data(0, UID) if item else None
        spec = self.parameters.get(uid) if uid else None
        if spec is None:
            self.param_constants_title.setText("Select a parameter to see its constants.")
            return
        if not spec.constants:
            self.param_constants_title.setText(f"{spec.label}: no constants.")
            return
        if uid not in self.profile.parameters:
            self.param_constants_title.setText(f"{spec.label}: tick it to measure it and set its constants.")
            return
        self.param_constants_title.setText(f"Constants of {spec.label} in this profile:")
        values = self.profile.parameters[uid]
        for s in spec.constants:
            def changed(v, key=s.key, setting=s):
                self.profile.parameters[uid][key] = setting.clean(v)
                self._edited()
            self.param_form.addRow(s.label, setting_widget(s, values.get(s.key, s.default), changed,
                                                           classes=self.profile.classes))

    def parameters_changed(self) -> None:
        """The parameter library changed (Settings › Parameters): refresh the list and the checks."""
        if self.profile is not None:
            self._fill_parameters()
            self._update_state()

    # ------------------------------------------------------------ library actions
    def _new(self) -> None:
        if not self.confirm_leave():
            return
        dlg = NewProfileDialog(self.library, self.profile.kind if self.profile else "tissue", self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        kind = dlg.kind.currentData()
        p = Profile(self.library.unique_name(dlg.name.text().strip()), kind=kind, images=list(DEFAULT_IMAGES[kind]))
        images = set(p.images)
        p.parameters = {u: s.resolve() for u, s in self.parameters.specs.items()
                        if s.default_on and (s.image_type == ANY or s.image_type in images)}
        pid = self.library.save(p)
        self.refresh(pid)
        self.profiles_changed.emit()

    def _duplicate(self) -> None:
        if self.pid and self.confirm_leave():
            pid = self.library.duplicate(self.pid)
            self.refresh(pid)
            self.profiles_changed.emit()

    def _delete(self) -> None:
        if not self.pid:
            return
        extra = ("\n\n“Reset to built-in” can bring the shipped version back later."
                 if self.library.is_builtin(self.pid) else "")
        if QtWidgets.QMessageBox.question(
                self, "Delete profile", f"Delete the profile “{self.profile.name}”?\n\nProjects made with it keep "
                f"their own copy and are not affected.{extra}") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.library.delete(self.pid)
        self.pid, self.dirty = None, False
        self.refresh()
        self.profiles_changed.emit()

    def _reset(self) -> None:
        if not self.pid or not self.library.is_builtin(self.pid):
            return
        if QtWidgets.QMessageBox.question(
                self, "Reset to built-in", f"Replace “{self.profile.name}” with the version shipped with "
                f"Modalytics? Your changes to it are lost.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.library.reset_builtin(self.pid)
        self.dirty = False
        self._open(self.pid)
        self.refresh(self.pid)
        self.profiles_changed.emit()

    def _bundle_sources(self) -> dict:
        """What is here, for a bundle: {kind: {id: file}} and the built-in ids, with the libraries' registries."""
        from modalytics.image_import.registry import ImporterLibrary
        from modalytics.processing.registry import ProcessingLibrary
        from modalytics.scripts.registry import ScriptLibrary
        from modalytics.services.segmentation import installed_models
        steps, importers, scripts = ProcessingLibrary(), ImporterLibrary(), ScriptLibrary()
        return {"known": {"models": installed_models(), "parameters": self.parameters.all_files(),
                          "steps": {u: Path(x.path) for u, x in steps.specs.items()},
                          "importers": {u: Path(x.path) for u, x in importers.specs.items()},
                          "scripts": scripts.all_files()},
                "builtin": {"parameters": {u for u, x in self.parameters.specs.items() if x.source != "user"},
                            "steps": {u for u, x in steps.specs.items() if x.source != "user"},
                            "scripts": {u for u, x in scripts.specs.items() if x.source != "user"}},
                "steps": steps, "importers": importers, "scripts": scripts}

    def _export(self) -> None:
        if self.profile is None or (self.dirty and not self.save()):
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export profile", str(Path.home() / f"{self.profile.name}{bundle.EXTENSION}"),
            f"Modalytics profile (*{bundle.EXTENSION})")
        if not path:
            return
        from modalytics import __version__
        from modalytics.services.segmentation import installed_models
        src = self._bundle_sources()
        steps, importers = src["steps"], src["importers"]
        # 1. What travels: your parameters, steps (of its chains) and importers (of its image types); the models
        user_params = {u: f for u, f in self.parameters.user_files().items() if u in self.profile.parameters}
        user_steps = {u: f for u, f in steps.user_files().items() if u in bundle.steps_used(self.profile)}
        user_importers = {u: f for u, f in importers.user_files().items()
                          if importers.get(u).image_type in self.profile.images}
        user_scripts = {u: f for u, f in src["scripts"].user_files().items() if u in self.profile.scripts}
        # 2. The libraries their scripts (and the built-in ones the profile uses) need, with the version here
        needed = set()
        for uid in self.profile.parameters:
            spec = self.parameters.specs.get(uid)
            needed |= set(spec.requires) if spec is not None else set()
        for uid in bundle.steps_used(self.profile):
            spec = steps.specs.get(uid)
            needed |= set(spec.requires) if spec is not None else set()
        for uid in user_importers:
            needed |= set(importers.get(uid).requires)
        for uid in self.profile.scripts:
            spec = src["scripts"].get(uid)
            needed |= set(spec.requires) if spec is not None else set()
        import importlib.metadata as md
        requires = {}
        for name in sorted(needed):
            try:
                requires[name] = md.version(name)
            except md.PackageNotFoundError:
                requires[name] = ""
        # 3. The models' training weights: others can fine-tune them, but the bundle is about twice as big
        models = installed_models()
        heavy = [m for m in self.profile.models() if m in models and _has_weights(models[m])]
        weights = True
        if heavy:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "Export profile",
                                        f"{len(heavy)} model(s) carry their training weights, so whoever receives "
                                        f"the profile can fine-tune them to their own images (the full edition). "
                                        f"Without them the file is about half the size and the models still run.",
                                        parent=self)
            keep = box.addButton("Include the weights", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Leave them out", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.exec()
            weights = box.clickedButton() is keep
        warnings = bundle.export_profile(self.profile, path, models, user_params, user_importers, user_steps,
                                         requires, weights, __version__, user_scripts)
        n_models = len([m for m in self.profile.models() if m in models])
        size = Path(path if path.endswith(bundle.EXTENSION) else path + bundle.EXTENSION).stat().st_size / 1e6
        msg = (f"Exported “{self.profile.name}” ({size:,.0f} MB) with {n_models} model(s)"
               + (" and their training weights" if heavy and weights else "")
               + f", {len(user_params)} of your parameter(s), {len(user_steps)} of your processing step(s), "
               f"{len(user_importers)} of your importer(s) and {len(user_scripts)} of your script(s), to\n{path}"
               + (f"\n\nIts scripts need: {', '.join(f'{n} {v}'.strip() for n, v in requires.items())} — "
                  "offered for installation when it is imported." if requires else ""))
        QtWidgets.QMessageBox.information(self, "Export profile", msg + ("\n\n" + "\n".join(warnings)
                                                                          if warnings else ""))

    def _import(self) -> None:
        if not self.confirm_leave():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import profile", str(Path.home()),
                                                        f"Modalytics profile (*{bundle.EXTENSION})")
        if not path:
            return
        self.import_bundle(path)

    def import_bundle(self, path: str) -> None:
        """Say what importing a bundle would do, then do it as chosen (new profile, or replacing one)."""
        src = self._bundle_sources()
        # 1. The plan
        try:
            plan = bundle.inspect(path, self.library, src["known"], src["builtin"])
        except ProfileError as exc:
            QtWidgets.QMessageBox.warning(self, "Import profile", str(exc))
            return
        kinds = {"models": "model", "parameters": "parameter", "steps": "processing step", "importers": "importer",
                 "scripts": "script"}
        lines = [f"“{plan.name}” ({plan.size_mb:,.0f} MB" + (f", from Modalytics {plan.app_version}"
                                                               if plan.app_version else "") + ")."]
        for status, text in (("new", "Will be added"), ("same", "Already here (identical)"),
                             ("different", "Differ from yours, added under a new name (yours stay as they are)")):
            items = [f"{kinds[k]} {u}" for k, u, st in plan.items if st == status]
            if items:
                lines.append(f"{text}: " + ", ".join(items))
        if any(k == "models" for k, _u, _s in plan.items):
            lines.append("The models carry their training weights (they can be fine-tuned)." if plan.weights
                         else "The models come without their training weights (they run, but cannot be fine-tuned).")
        if plan.requires:
            lines.append("Its scripts need libraries not installed here: "
                         + ", ".join(f"{n} {v}".strip() for n, v in plan.requires.items())
                         + " — you can install them right after.")
        if plan.missing:
            lines.append("⚠ Neither in the file nor here (that part will not work): " + ", ".join(plan.missing))
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "Import profile", "\n\n".join(lines),
                                    parent=self)
        new_btn = box.addButton("Import as a new profile" if plan.existing else "Import",
                                QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        replace_btn = None
        if plan.existing:
            box.setInformativeText(f"You already have a profile “{plan.name}”. Replace it with this one (e.g. a newer "
                                   f"release; your projects keep their own copy), or import it as a new profile?")
            replace_btn = box.addButton(f"Replace “{plan.name}”", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() not in (new_btn, replace_btn) or box.clickedButton() is None:
            return
        replace = plan.existing if box.clickedButton() is replace_btn else ""
        # 2. Import
        try:
            r = bundle.import_profile(path, self.library, paths.user_dir("models"), paths.user_dir("parameters"),
                                      dict(src["known"]["models"]), dict(src["known"]["parameters"]),
                                      paths.user_dir("importers"), dict(src["known"]["importers"]),
                                      paths.user_dir("processing"), dict(src["known"]["steps"]), replace,
                                      scripts_dir=paths.user_dir("scripts"),
                                      known_scripts=dict(src["known"]["scripts"]))
        except ProfileError as exc:
            QtWidgets.QMessageBox.warning(self, "Import profile", str(exc))
            return
        self.parameters.reload()
        self.models = _installed_models()
        done = [f"{'Replaced' if r.replaced else 'Imported'} “{r.name}”."]
        if r.renamed:
            done.append("Added under a new name (yours were different): "
                        + ", ".join(f"{a} → {b}" for a, b in r.renamed.items()))
        done += r.warnings
        # 3. Libraries its scripts need: install them now (Settings › Parameters › Libraries, same rules)
        if r.requires:
            ask = QtWidgets.QMessageBox.question(
                self, "Import profile", "\n\n".join(done) + f"\n\nIts scripts need {', '.join(r.requires)}, not "
                "installed here. Install now? (Ready-built packages only, into your library folder; the libraries "
                "included with Modalytics stay as they are.)")
            if ask == QtWidgets.QMessageBox.StandardButton.Yes:
                from modalytics.ui.settings.libraries import LibrariesDialog
                dlg = LibrariesDialog("", self)
                dlg.changed.connect(self.parameters.reload)
                dlg.install_many(r.requires)
                dlg.exec()
        else:
            QtWidgets.QMessageBox.information(self, "Import profile", "\n\n".join(done))
        self.refresh(r.profile_id)
        self.profiles_changed.emit()


def _has_weights(path) -> bool:
    """Whether a model package carries its training weights."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return bundle.WEIGHTS in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False
