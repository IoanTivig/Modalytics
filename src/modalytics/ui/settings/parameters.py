"""
Settings › Parameters: the parameter library (browse, add, edit, remove).

    ┌ Parameter            Unit ┬ (ParameterEditor) ─────────────────────────┐
    │ Per object (31)           │ [Properties] [Constants] [Code]             │
    │   Any image (12)          │                                             │
    │     Shape                 │                                             │
    │       Area           µm²  │                                             │
    │   Phase map (19)          │                                             │
    │     Intensity …           │                                             │
    │ [New…] [Remove] [Reload]  │ [Test] [External editor]   [Revert] [Save]  │
    └───────────────────────────┴─────────────────────────────────────────────┘

The tree lists the library; the right side edits the selected parameter
(``parameter_editor``). Built-in parameters ship with Modalytics and are never
changed in place: saving one writes your own copy with the same id, which
replaces the built-in everywhere until you remove it. New parameters start
from a commented template.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics import paths
from modalytics.parameters.registry import SCOPES, TYPE_NAMES, ParameterLibrary, ParameterSpec, type_word
from modalytics.ui.settings.parameter_editor import ParameterEditor
from modalytics.ui.settings.widgets import clear_tree, group_item, is_group, keep_on, list_tree, open_in_system

TEMPLATE = '''"""{label}: {description}

Your parameter (made in Settings › Parameters). Edit freely: Settings ›
Parameters › Reload picks up the changes. See modalytics/parameters/registry.py
for the format and modalytics/parameters/context.py for everything ctx offers.
"""

KEY = {key!r}
LABEL = {label!r}
UNIT = {unit!r}
CATEGORY = {category!r}
SCOPE = {scope!r}
DEFAULT_ON = True
WHOLE_OBJECT = {whole!r}     # object scope: needs the whole object (shape, totals) — left empty on partial objects
ORDER = 1000
IMAGE_TYPE = {image_type!r}
DESCRIPTION = {description!r}
CONSTANTS = []           # values a profile may change, e.g. [{{"key": "window", "label": "Window", "kind": "int", "default": 3}}]


def compute(ctx):
    # Object scope: ctx.polygon, ctx.area_um2, ctx.perimeter_um, ctx.major_axis_um,
    #   ctx.minor_axis_um, ctx.curvature_um(spacing), ctx.mask, ctx.pixels (values inside),
    #   ctx.image_crop, ctx.glcm_features(distance), ctx.fluctuation_image(degree, window)…
    # Whole-image scope: ctx.phase (2D array), ctx.flat, ctx.glcm_features(distance)…
    # ctx.constants holds this parameter's constants (declared in CONSTANTS / on the Constants tab).
    {body}
'''


def _breakable(path: str) -> str:
    """A path that may wrap after its separators (a long path must not widen the window)."""
    return path.replace("\\", "\\\u200b").replace("/", "/\u200b")


def _key_from(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    return key if key and not key[0].isdigit() else f"p_{key}"


class NewParameterDialog(QtWidgets.QDialog):
    """Name, scope and image type of a new parameter (the code is written afterwards).

    Args:
        library: The parameter library (for categories and taken ids).
        scope, image_type: Preselected (e.g. the group selected in the tree).
    """

    def __init__(self, library: ParameterLibrary, scope: str = "object", image_type: str = "phase",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New parameter")
        self.library = library
        form = QtWidgets.QFormLayout(self)
        self.label = QtWidgets.QLineEdit()
        self.label.setPlaceholderText("e.g. Aspect ratio")
        self.key = QtWidgets.QLineEdit()
        self.label.textChanged.connect(lambda t: self.key.setText(_key_from(t)))
        self.scope = QtWidgets.QComboBox()
        for k, v in SCOPES.items():
            self.scope.addItem(v, k)
        self.scope.setCurrentIndex(max(0, self.scope.findData(scope)))
        self.image_type = QtWidgets.QComboBox()
        for k, v in TYPE_NAMES.items():
            self.image_type.addItem(v, k)
        self.image_type.setCurrentIndex(max(0, self.image_type.findData(image_type)))
        self.image_type.setToolTip("Any image: it uses only the outline (shape). Otherwise the one image "
                                   "type whose pixels it measures.")
        self.category = QtWidgets.QComboBox()
        self.category.setEditable(True)
        self.scope.currentIndexChanged.connect(self._scope_changed)
        self.image_type.currentIndexChanged.connect(self._scope_changed)
        self._scope_changed()
        self.unit = QtWidgets.QLineEdit()
        self.unit.setPlaceholderText("e.g. µm, rad (empty if none)")
        self.description = QtWidgets.QLineEdit()
        self.description.setPlaceholderText("What it measures and how to read it")
        form.addRow("Name", self.label)
        form.addRow("Id", self.key)
        form.addRow("Scope", self.scope)
        form.addRow("Image type", self.image_type)
        form.addRow("Category", self.category)
        form.addRow("Unit", self.unit)
        form.addRow("Description", self.description)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _scope_changed(self, *_args) -> None:
        """Offer the categories already used for the chosen scope and image type."""
        current = self.category.currentText()
        self.category.clear()
        self.category.addItems(self.library.categories(self.scope.currentData(), self.image_type.currentData())
                               or self.library.categories(self.scope.currentData()) or ["other"])
        if current:
            self.category.setCurrentText(current)

    def _accept(self) -> None:
        key, scope = self.key.text().strip(), self.scope.currentData()
        if not self.label.text().strip() or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            QtWidgets.QMessageBox.warning(self, "New parameter", "Give it a name (the id is made from it).")
            return
        if self.uid() in self.library.specs:
            QtWidgets.QMessageBox.warning(self, "New parameter", f"A parameter '{key}' already exists for "
                                          f"{self.image_type.currentText().lower()} ({SCOPES[scope].lower()}).")
            return
        self.accept()

    def uid(self) -> str:
        return f"{self.scope.currentData()}:{type_word(self.image_type.currentData())}:{self.key.text().strip()}"

    def script(self) -> tuple[str, str]:
        """(file path inside the parameters folder, script text) for the chosen values."""
        from modalytics.parameters.registry import whole_object_default
        scope, image_type = self.scope.currentData(), self.image_type.currentData()
        body = ("return ctx.area_um2   # replace with your measurement" if scope == "object"
                else "return float(ctx.flat.mean())   # replace with your measurement")
        text = TEMPLATE.format(key=self.key.text().strip(), label=self.label.text().strip(), unit=self.unit.text().strip(),
                               category=self.category.currentText().strip() or "other", scope=scope,
                               image_type=image_type, description=self.description.text().strip(), body=body,
                               whole=whole_object_default(self.category.currentText().strip(), ""))
        return f"{scope}/{type_word(image_type)}/{self.key.text().strip()}.py", text


class ParametersPage(QtWidgets.QWidget):
    """The parameter library page.

    Signals:
        library_changed(): parameters were added, changed or removed (profiles refresh their lists).
    """

    library_changed = QtCore.Signal()

    def __init__(self, library: ParameterLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        self.library = library or ParameterLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split)

        # 1. Left: the tree and the library actions
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Parameter", "Unit"])
        self.tree.setColumnWidth(0, 210)
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your parameter (a copy of a built-in: the built-in comes back)")
        for b, slot in ((QtWidgets.QPushButton("New…"), self._new), (self.remove_btn, self._remove),
                        (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        libraries_btn = QtWidgets.QPushButton("Libraries…")
        libraries_btn.setToolTip("What parameter scripts can import: the included libraries, and yours")
        libraries_btn.clicked.connect(lambda: self.open_libraries())
        lc.addWidget(libraries_btn)
        split.addWidget(left)

        # 2. Right: the editor of the selected parameter
        self.editor = ParameterEditor(self.library)
        self.editor.saved.connect(self._saved)
        self.editor.install_requested.connect(self.open_libraries)
        split.addWidget(self.editor)
        split.setSizes([300, 760])
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
        for scope, scope_name in SCOPES.items():
            specs = self.library.of_scope(scope)
            if not specs:
                continue
            top = group_item(self.tree, f"{scope_name} ({len(specs)})")
            for image_type, type_name in TYPE_NAMES.items():
                of_type = self.library.of_type(scope, image_type)
                tnode = group_item(top, f"{type_name} ({len(of_type)})")
                if not of_type:
                    tnode.setToolTip(0, "None yet: New… adds one, or Duplicate… copies a parameter of another "
                                        "image type")
                cats = {}
                for spec in of_type:
                    cat = cats.get(spec.category)
                    if cat is None:
                        cat = cats[spec.category] = group_item(tnode, spec.category.capitalize())
                    mark = "  · yours" if spec.source == "user" else ""
                    item = QtWidgets.QTreeWidgetItem(cat, [spec.label + mark, spec.unit])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, spec.uid)
                    item.setToolTip(0, spec.help_text())
                    if spec.uid == select:
                        chosen = item
                tnode.setExpanded(bool(of_type))
            top.setExpanded(True)
        if self.library.errors:
            bad = group_item(self.tree, f"Could not load ({len(self.library.errors)})")
            for path, why in self.library.errors.items():
                item = QtWidgets.QTreeWidgetItem(bad, [Path(path).name])
                item.setToolTip(0, why)
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, path)
                if path == select:
                    chosen = item
            bad.setExpanded(True)
        self.tree.blockSignals(False)
        if chosen is None:                              # first parameter
            it = QtWidgets.QTreeWidgetItemIterator(self.tree)
            while it.value() and not it.value().data(0, QtCore.Qt.ItemDataRole.UserRole):
                it += 1
            chosen = it.value()
        self.tree.setCurrentItem(chosen)
        if chosen is not None:
            self.tree.scrollToItem(chosen)               # opens its category
        self._show(chosen)

    def _spec(self) -> ParameterSpec | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(uid) if uid else None

    def _selected(self, cur, prev) -> None:
        if cur is prev:
            return
        if is_group(cur):                                # a group row only opens / closes
            keep_on(self.tree, prev)
            return
        role = QtCore.Qt.ItemDataRole.UserRole
        wanted = [cur.data(0, role + k) for k in range(2)]      # the row, by its ids
        built = self._built
        if not self.editor.confirm_leave():              # stay on the parameter with unsaved edits
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(prev)
            self.tree.blockSignals(False)
            return
        if self._built != built:                         # saving rebuilt the tree: ``cur`` is gone
            self._select_row(wanted)
            return
        self._show(cur)

    def _select_row(self, wanted: list) -> None:
        """Select the row whose ids (UserRole, UserRole + 1…) are ``wanted``."""
        role = QtCore.Qt.ItemDataRole.UserRole
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            if [it.value().data(0, role + k) for k in range(len(wanted))] == wanted:
                self.tree.setCurrentItem(it.value())
                return
            it += 1

    def _show(self, item) -> None:
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        self.editor.load(spec, broken if broken else None)
        self.editor.set_missing(self.library.missing_libraries().get(broken or (spec.uid if spec else ""), []))
        self.remove_btn.setEnabled(bool(broken) or (spec is not None and spec.source == "user"))

    def select(self, uid: str) -> None:
        """Select a parameter by id, an id from before image types too (or a broken script by path)."""
        uid = self.library.find(uid) or uid
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
        """The Libraries window (``install``: a package to put in its install field)."""
        from modalytics.ui.settings.libraries import LibrariesDialog
        dlg = LibrariesDialog(install, self)
        dlg.changed.connect(lambda: (self.reload(), self.library_changed.emit()))
        dlg.exec()

    def _saved(self, uid: str) -> None:
        """The editor wrote a script: rescan, keep it selected, tell the profiles."""
        path = self.editor.path
        self.reload(uid or path)
        if not uid:                                      # a broken script: select it again by path
            self.select(path)
        self.library_changed.emit()

    def _new(self) -> None:
        if not self.editor.confirm_leave():
            return
        spec = self._spec()                              # preselect the scope and image type in view
        dlg = NewParameterDialog(self.library, spec.scope if spec else "object",
                                 spec.image_type if spec else "phase", self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, text = dlg.script()
        target = paths.user_dir("parameters") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.reload(dlg.uid())
        self.editor.tabs.setCurrentIndex(2)              # straight to the code
        self.library_changed.emit()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        path = broken or (spec.path if spec and spec.source == "user" else None)
        if path is None:
            return
        replaces = spec is not None and spec.uid in self.library.overridden
        msg = (f"Delete “{spec.label if spec else Path(path).name}”?\n\n{path}"
               + ("\n\nThe built-in version comes back." if replaces
                  else "\n\nProfiles that use it will report it as missing." if spec else ""))
        if QtWidgets.QMessageBox.question(self, "Remove parameter", msg) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.uid if replaces else None)
        self.library_changed.emit()
