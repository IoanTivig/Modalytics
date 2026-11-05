"""
Settings › Processing: the steps that make new versions of images (browse, add, edit, remove).

    ┌ Step                       Input ┬ (ProcessingEditor) ────────────────────┐
    │ OD map (1)                       │ [Properties] [Options] [Code]          │
    │   OD map              wsi brightf│                                        │
    │ OPD map (1)                      │                                        │
    │ Dry-mass map (1)                 │                                        │
    │ Filters (2)                      │                                        │
    │   Gaussian blur       *          │                                        │
    │ [New…] [Remove] [Reload] [Folder]│ [Test] [External]       [Revert] [Save]│
    └──────────────────────────────────┴────────────────────────────────────────┘

Steps are scripts, like parameters and importers: built-in ones ship with
Modalytics and are never changed in place (saving one writes your own copy with
the same id, which replaces it until removed); new ones start from a commented
template. A step's result is a view of the image it is applied to (Process tab,
or a profile's processing at import).
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics.core.model import ACQUIRED_TYPES, IMAGE_TYPES
from modalytics.processing.registry import SAME, ProcessingLibrary, StepSpec
from modalytics.ui.settings.processing_editor import ProcessingEditor
from modalytics.ui.settings.widgets import (clear_tree, group_item, is_group, keep_on, list_tree, muted,
                                            open_in_system)

TEMPLATE = '''"""{name}: {description}

Your processing step (made in Settings › Processing). It turns an image's pixels
into a new version of the image (a view: same size, placement and masks). See
modalytics/processing/registry.py for the format of this script.
"""

import numpy as np

KEY = {key!r}
NAME = {name!r}
INPUT_TYPES = {inputs!r}
OUTPUT_TYPE = {output!r}
UNIT = {unit!r}
KIND = 'pixel'          # 'pixel', 'neighbourhood' (then define margin) or 'whole'
DESCRIPTION = {description!r}
OPTIONS = [
    {{'key': 'factor', 'label': 'Factor', 'kind': 'float', 'default': 1.0, 'minimum': 0.0, 'maximum': 1000.0,
     'tooltip': 'An example option: the code reads it as options["factor"]'}},
]


def prepare(image, options):
    # Optional: computed once per image (e.g. a background level from image.metadata or its pixels).
    return {{}}


def process(pixels, options, state, um_per_px):
    # pixels: a tile of the input (H, W) or (H, W, 3); return an array of the same height and width
    # (2D float for another image type). um_per_px converts sizes given in µm at every zoom level.
    values = pixels.astype(np.float32)
    if values.ndim == 3:
        values = values.mean(axis=2)
    return values * options["factor"]
'''


def _key_from(name: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return key if key and not key[0].isdigit() else f"s_{key}"


class NewStepDialog(QtWidgets.QDialog):
    """Name, input and output types of a new step."""

    def __init__(self, library: ProcessingLibrary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New processing step")
        self.library = library
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. Background subtraction")
        self.inputs = QtWidgets.QLineEdit("phase")
        self.inputs.setToolTip(", ".join(IMAGE_TYPES) + ", or * for any")
        self.output = QtWidgets.QComboBox()
        self.output.addItem("Same type (a filter)", SAME)
        for k, v in IMAGE_TYPES.items():
            if k not in ACQUIRED_TYPES:
                self.output.addItem(v, k)
        self.description = QtWidgets.QLineEdit()
        form.addRow("Name", self.name)
        form.addRow("Input types", self.inputs)
        form.addRow("Makes", self.output)
        form.addRow("Description", self.description)
        form.addRow(muted("It starts from a template that multiplies the pixel values by a factor; change "
                          "process() for your step."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self) -> None:
        if not self.name.text().strip():
            QtWidgets.QMessageBox.warning(self, "New processing step", "Give it a name.")
            return
        if self.uid() in self.library.specs:
            QtWidgets.QMessageBox.warning(self, "New processing step", "A step with this name exists already.")
            return
        self.accept()

    def folder(self) -> str:
        return "filters" if self.output.currentData() == SAME else self.output.currentData()

    def uid(self) -> str:
        return f"{self.folder()}:{_key_from(self.name.text())}"

    def script(self) -> tuple[str, str]:
        """(file path relative to your processing folder, script text)."""
        name, out = self.name.text().strip(), self.output.currentData()
        inputs = [t.strip() for t in self.inputs.text().split(",") if t.strip()] or ["*"]
        from modalytics.processing.types import MADE
        unit = MADE.get(out, ("",))[0] if out != SAME else ""
        text = TEMPLATE.format(name=name, key=_key_from(name), inputs=inputs, output=out, unit=unit,
                               description=self.description.text().strip() or "makes …")
        return f"{self.folder()}/{_key_from(name)}.py", text


class ProcessingPage(QtWidgets.QWidget):
    """The processing library page.

    Signals:
        library_changed(): steps were added, changed or removed (profiles refresh their lists).
    """

    library_changed = QtCore.Signal()

    def __init__(self, library: ProcessingLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        self.library = library or ProcessingLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split)

        # 1. Left: the tree and the library actions
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Step", "Input"])
        self.tree.setColumnWidth(0, 210)
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your step (a copy of a built-in: the built-in comes back)")
        for b, slot in ((QtWidgets.QPushButton("New…"), self._new), (self.remove_btn, self._remove),
                        (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        libraries_btn = QtWidgets.QPushButton("Libraries…")
        libraries_btn.setToolTip("What scripts can import: the included libraries, and yours")
        libraries_btn.clicked.connect(lambda: self.open_libraries())
        lc.addWidget(libraries_btn)
        split.addWidget(left)

        # 2. Right: the editor of the selected step
        self.editor = ProcessingEditor(self.library)
        self.editor.saved.connect(self._saved)
        self.editor.install_requested.connect(self.open_libraries)
        split.addWidget(self.editor)
        split.setSizes([320, 740])
        self.reload()

    # ------------------------------------------------------------ the tree
    def reload(self, select: str | None = None) -> None:
        """Rescan the scripts and rebuild the tree (keeps the selection, or selects ``select``)."""
        select = select or (self._spec().uid if self._spec() else None)
        self.library.reload()
        self._built = getattr(self, "_built", 0) + 1     # rows from an earlier build are gone
        self.tree.blockSignals(True)
        clear_tree(self.tree)
        chosen = None
        role = QtCore.Qt.ItemDataRole.UserRole
        groups = [(t, IMAGE_TYPES[t].split(" (")[0]) for t in IMAGE_TYPES if t not in ACQUIRED_TYPES] + \
            [("filters", "Filters (same type)")]
        for folder, title in groups:
            specs = [s for s in self.library.specs.values() if s.folder == folder]
            top = group_item(self.tree, f"{title} ({len(specs)})")
            # Filters (many) are listed by category: Correction, Denoise, Enhance…
            parts = [(c, g, ) for c, g in self.library.grouped() if any(s.folder == folder for s in g)] \
                if folder == "filters" else [(None, specs)]
            for category, members in parts:
                parent = top
                if category is not None:
                    members = [s for s in members if s.folder == folder]
                    parent = group_item(top, f"{category} ({len(members)})")
                    parent.setExpanded(True)
                for spec in members:
                    mark = "  · yours" if spec.source == "user" else ""
                    item = QtWidgets.QTreeWidgetItem(parent, [spec.name + mark, " ".join(spec.input_types)])
                    item.setData(0, role, spec.uid)
                    item.setToolTip(0, spec.description)
                    if spec.uid == select:
                        chosen = item
            top.setExpanded(True)
        if self.library.errors:
            bad = group_item(self.tree, f"Could not load ({len(self.library.errors)})")
            for path, why in self.library.errors.items():
                item = QtWidgets.QTreeWidgetItem(bad, [Path(path).name])
                item.setToolTip(0, why)
                item.setData(0, role + 1, path)
                if path == select:
                    chosen = item
            bad.setExpanded(True)
        self.tree.blockSignals(False)
        if chosen is None:
            it = QtWidgets.QTreeWidgetItemIterator(self.tree)
            while it.value() and not it.value().data(0, role):
                it += 1
            chosen = it.value()
        self.tree.setCurrentItem(chosen)
        self._show(chosen)

    def _spec(self) -> StepSpec | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(uid) if uid else None

    def _selected(self, cur, prev) -> None:
        if cur is prev:
            return
        if is_group(cur):
            keep_on(self.tree, prev)
            return
        role = QtCore.Qt.ItemDataRole.UserRole
        wanted = [cur.data(0, role + k) for k in range(2)]
        built = self._built
        if not self.editor.confirm_leave():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(prev)
            self.tree.blockSignals(False)
            return
        if self._built != built:                          # saving rebuilt the tree: ``cur`` is gone
            self.select(wanted[0] or wanted[1])
            return
        self._show(cur)

    def _show(self, item) -> None:
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        self.editor.load(spec, broken if broken else None)
        self.editor.set_missing(self.library.missing_libraries().get(broken or (spec.uid if spec else ""), []))
        self.remove_btn.setEnabled(bool(broken) or (spec is not None and spec.source == "user"))

    def select(self, uid: str) -> None:
        """Select a step by id (or a broken script by path)."""
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            v = it.value()
            if uid in (v.data(0, QtCore.Qt.ItemDataRole.UserRole), v.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)):
                self.tree.setCurrentItem(v)
                return
            it += 1

    def confirm_leave(self) -> bool:
        return self.editor.confirm_leave()

    # ------------------------------------------------------------ actions
    def open_libraries(self, install: str = "") -> None:
        from modalytics.ui.settings.libraries import LibrariesDialog
        dlg = LibrariesDialog(install, self)
        dlg.changed.connect(lambda: (self.reload(), self.library_changed.emit()))
        dlg.exec()

    def _saved(self, uid: str) -> None:
        path = self.editor.path
        self.reload(uid or path)
        if not uid:
            self.select(path)
        self.library_changed.emit()

    def _new(self) -> None:
        if not self.editor.confirm_leave():
            return
        dlg = NewStepDialog(self.library, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, text = dlg.script()
        target = self.library.user / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.reload(dlg.uid())
        self.editor.tabs.setCurrentIndex(2)
        self.library_changed.emit()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        path = broken or (spec.path if spec and spec.source == "user" else None)
        if path is None:
            return
        replaces = spec is not None and spec.uid in self.library.overridden
        msg = (f"Delete “{spec.name if spec else Path(path).name}”?\n\n{path}"
               + ("\n\nThe built-in version comes back." if replaces
                  else "\n\nProfiles that apply it will report it as missing." if spec else ""))
        if QtWidgets.QMessageBox.question(self, "Remove processing step", msg) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.uid if replaces else None)
        self.library_changed.emit()
