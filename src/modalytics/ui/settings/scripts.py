"""
Settings › Scripts: external scripts — tasks beside the analysis (browse, add, edit, run, remove).

    ┌ Script                          ┬ (ScriptsEditor) ───────────────────────────┐
    │ Data preparation (1)            │ [Properties] [Options] [Code]              │
    │   DHM phase map extractor       │                                            │
    │ [New…] [Remove] [Reload] [Folder]│ [Run…] [External]           [Revert] [Save]│
    │ [Libraries…]                    │                                            │
    └─────────────────────────────────┴────────────────────────────────────────────┘

Scripts are files like parameters, importers and processing steps (see
``scripts.registry`` for the format): built-in ones are never changed in place
(Save writes your own copy with the same id, which replaces it until removed),
new ones start from a commented template. *Run…* runs the script as edited.
A profile chooses which scripts its projects offer in the Scripts menu (Settings ›
Profiles).
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics import paths
from modalytics.parameters.script_file import ExternalScript
from modalytics.scripts.registry import ScriptLibrary, ScriptSpec, load_script
from modalytics.ui.settings.script_editor import ScriptEditor
from modalytics.ui.settings.widgets import (clear_tree, group_item, is_group, keep_on, list_tree, muted,
                                            open_in_system)

TEMPLATE = '''"""{name}: {description}

Your external script (made in Settings › Scripts). It does a task beside the analysis —
e.g. gathering or renaming an instrument's files. See modalytics/scripts/registry.py
for the format of this script.
"""

import os

KEY = {key!r}
NAME = {name!r}
CATEGORY = {category!r}
DESCRIPTION = {description!r}
OPTIONS = [
    {{'key': 'folder', 'label': 'Folder', 'kind': 'folder', 'default': '',
     'tooltip': 'An example option: the code reads it as options["folder"]'}},
]


def run(options, ctx):
    # ctx.log(text) writes to the log · ctx.progress(0-1) · ctx.check() stops if the user asked
    # ctx.import_files(paths or a folder) offers files for import into the open project at the end
    folder = options["folder"]
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose a folder")
    names = sorted(os.listdir(folder))
    for i, name in enumerate(names, start=1):
        ctx.check()
        ctx.log(name)
        ctx.progress(i / len(names))
    return f"{{len(names)}} item(s) in {{folder}}"
'''

HELP = [
    ("Context", "ctx.log(text)", "ctx.log()", "Write a line to the run's log."),
    ("Context", "ctx.progress(fraction)", "ctx.progress()", "Show progress (0–1)."),
    ("Context", "ctx.check()", "ctx.check()", "Stop here if the user pressed Stop (call it in loops)."),
    ("Context", "ctx.cancelled()", "ctx.cancelled()", "True once the user pressed Stop."),
    ("Context", "ctx.import_files(paths)", "ctx.import_files()",
     "Files, or a folder, to offer for import into the open project at the end."),
    ("Arguments", 'options["key"]', 'options[""]', "An option's value (Options tab): folders and files as text."),
    ("Helpers", "os.walk(folder)", "os.walk()", "Every folder and file below a folder.\nNeeds: import os"),
    ("Helpers", "shutil.copy2(src, dst)", "shutil.copy2()", "Copy a file with its dates.\nNeeds: import shutil"),
    ("Helpers", "Path(p).glob('*.txt')", "Path().glob()", "Files matching a pattern.\nNeeds: from pathlib import Path"),
]


def _key_from(name: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_") or "script"
    return key if not key[0].isdigit() else f"s_{key}"


class ScriptsEditor(ScriptEditor):
    """Edits the selected external script."""

    SCRIPT = ExternalScript
    WHAT = "script"
    VALUES_TAB = "Options"
    VALUES_ROW = "Option"
    VALUES_PROPERTY = "OPTIONS"
    VALUES_HINT = ("What the script asks before it runs (folders, files, text, numbers, choices); the values used "
                   "last are offered the next time. The code reads them as options[\"key\"].")
    TEST_TIP = "Run the script as edited (Ctrl+T)"

    run_requested = QtCore.Signal(object)          # a ScriptSpec loaded from the edited text

    def __init__(self, library, parent=None) -> None:
        super().__init__(library, parent)
        self.test_btn.setText("Run…")                # a script's test is running it

    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.name = QtWidgets.QLineEdit()
        self.key = QtWidgets.QLineEdit()
        self.category = QtWidgets.QComboBox()
        self.category.setEditable(True)
        self.category.setToolTip("Where the Scripts menu lists it (pick one, or type your own)")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones")
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(80)
        for label, w in (("Name", self.name), ("Id", self.key), ("Category", self.category),
                         ("Requires", self.requires), ("Description", self.description)):
            form.addRow(label, w)
        for sig in (self.name.textEdited, self.key.textEdited, self.requires.textEdited,
                    self.description.textChanged):
            sig.connect(self._changed)
        self.category.currentTextChanged.connect(self._changed)
        self.name.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("NAME", p.get("KEY", "")))
        self.name.setText(str(p.get("NAME", "")))
        self.key.setText(str(p.get("KEY", "")))
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItems(sorted({s.group for s in self.library.specs.values()} | {"Data preparation"}))
        self.category.setCurrentText(str(p.get("CATEGORY", "") or "Other"))
        self.category.blockSignals(False)
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        self.helper.set_entries("scripts", HELP)

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.key.setToolTip("Built-in scripts keep their id (your copy replaces the built-in)." if builtin else
                            "Changing the id renames the file; profiles that offer the old id will report it missing.")

    def properties(self) -> dict:
        p = dict(self._known)
        p.update({"KEY": self.key.text().strip(), "NAME": self.name.text().strip(),
                  "DESCRIPTION": " ".join(self.description.toPlainText().split())})
        p.pop("CATEGORY", None)
        if self.category.currentText().strip() not in ("", "Other"):
            p["CATEGORY"] = self.category.currentText().strip()
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["NAME"]:
            raise ValueError("The script needs a name")

    def target(self, p: dict) -> Path:
        return paths.user_dir("scripts") / f"{p['KEY']}.py"

    def run_test(self, f: Path) -> None:
        """Load the script as edited and open its run window."""
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        self.result.setText("")
        self.run_requested.emit(spec)


class NewScriptDialog(QtWidgets.QDialog):
    """Name, category and description of a new script."""

    def __init__(self, library: ScriptLibrary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New script")
        self.library = library
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. Rename microscope exports")
        self.category = QtWidgets.QComboBox()
        self.category.setEditable(True)
        self.category.addItems(sorted({s.group for s in library.specs.values()} | {"Data preparation"}))
        self.description = QtWidgets.QLineEdit()
        form.addRow("Name", self.name)
        form.addRow("Category", self.category)
        form.addRow("Description", self.description)
        form.addRow(muted("It starts from a template that lists a folder's contents; change run() for your task."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self) -> None:
        if not self.name.text().strip():
            QtWidgets.QMessageBox.warning(self, "New script", "Give it a name.")
            return
        if self.uid() in self.library.specs:
            QtWidgets.QMessageBox.warning(self, "New script", "A script with this name exists already.")
            return
        self.accept()

    def uid(self) -> str:
        return _key_from(self.name.text())

    def script(self) -> str:
        return TEMPLATE.format(name=self.name.text().strip(), key=self.uid(),
                               category=self.category.currentText().strip() or "Other",
                               description=self.description.text().strip() or "does …")


class ScriptsPage(QtWidgets.QWidget):
    """The external scripts page.

    Args:
        on_import: Given to the run window: imports the files a script hands over (None: not offered).

    Signals:
        library_changed(): scripts were added, changed or removed (profiles refresh their lists).
    """

    library_changed = QtCore.Signal()

    def __init__(self, library: ScriptLibrary | None = None, parent=None, on_import=None) -> None:
        super().__init__(parent)
        self.library = library or ScriptLibrary()
        self.on_import = on_import
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split)
        # 1. Left: the scripts by category, and the library actions
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        intro = muted("Tasks beside the analysis (e.g. gathering an instrument's files for import). A profile "
                      "chooses which ones its projects offer in the Scripts menu.")
        intro.setWordWrap(True)
        lc.addWidget(intro)
        self.tree = list_tree(["Script"])
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your script (a copy of a built-in: the built-in comes back)")
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
        # 2. Right: the editor of the selected script
        self.editor = ScriptsEditor(self.library)
        self.editor.saved.connect(self._saved)
        self.editor.install_requested.connect(self.open_libraries)
        self.editor.run_requested.connect(self.run_script)
        split.addWidget(self.editor)
        split.setSizes([320, 740])
        self.reload()

    # ------------------------------------------------------------ the tree
    def reload(self, select: str | None = None) -> None:
        """Rescan the scripts and rebuild the tree (keeps the selection, or selects ``select``)."""
        select = select or (self._spec().uid if self._spec() else None)
        self.library.reload()
        self._built = getattr(self, "_built", 0) + 1
        self.tree.blockSignals(True)
        clear_tree(self.tree)
        chosen = None
        role = QtCore.Qt.ItemDataRole.UserRole
        for category, specs in self.library.grouped():
            top = group_item(self.tree, f"{category} ({len(specs)})")
            for spec in specs:
                item = QtWidgets.QTreeWidgetItem(top, [spec.name + ("  · yours" if spec.source == "user" else "")])
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
            while it.value() and not (it.value().data(0, role) or it.value().data(0, role + 1)):
                it += 1
            chosen = it.value()
        self.tree.setCurrentItem(chosen)
        self._show(chosen)

    def _spec(self) -> ScriptSpec | None:
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
        if self._built != built:
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
    def run_script(self, spec: ScriptSpec) -> None:
        """The run window of a script (as edited)."""
        from modalytics.ui.scripts import RunScriptDialog
        self._run = RunScriptDialog(spec, self, self.on_import)
        self._run.show()

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
        dlg = NewScriptDialog(self.library, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        target = self.library.user / f"{dlg.uid()}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dlg.script(), encoding="utf-8")
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
                  else "\n\nProfiles that offer it will report it as missing." if spec else ""))
        if QtWidgets.QMessageBox.question(self, "Remove script", msg) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.uid if replaces else None)
        self.library_changed.emit()
