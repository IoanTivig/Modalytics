"""
The editor shared by every kind of user script (parameters, importers).

    ┌ Name ──────────────────────────────── built in · Save makes your own copy ┐
    │ [Properties] [Constants / Options] [Code]                                 │
    │  Properties  a form, given by the kind of script                          │
    │  values      one row per adjustable value: key, label, kind, default…    │
    │  Code        editor | helper list (double-click to insert)                │
    │ result of Test                                                            │
    │ [Test] [Open in external editor] [extra…]                [Revert] [Save] │
    └───────────────────────────────────────────────────────────────────────────┘

The script file stays the only source of truth: Save writes the properties
back into it and keeps the code as typed (``parameters.script_file``).
Built-in scripts are never changed: saving one writes your own copy (same
id), which replaces it until you remove the copy. The external editor and this
one can be used together: a change on disk is loaded when there are no
unsaved edits here.

A kind of script subclasses ``ScriptEditor`` and gives: its ``ScriptFile``
class, what it is called (``WHAT``), the values tab's name and hint, the
properties form (``_build_form``, ``_fill``, ``properties``, ``validate``),
where a script is saved (``target``, ``uid_of``) and what Test does
(``run_test``).
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import fileio
from modalytics.core.settings import KINDS as SETTING_KINDS
from modalytics.core.settings import Setting
from modalytics.parameters.script_file import ScriptFile
from modalytics.ui.settings.code_editor import CodeEditor, HelperPanel
from modalytics.ui.settings.widgets import open_in_system

COLUMNS = ("Key", "Label", "Kind", "Default", "Min", "Max", "Step", "Suffix", "Choices", "Help")


def current_tokens() -> dict:
    """Theme tokens of the running application (dark or light, from its palette)."""
    from modalytics.ui.theme import TOKENS
    bg = QtWidgets.QApplication.instance().palette().color(QtGui.QPalette.ColorRole.Window)
    return TOKENS["dark" if bg.lightness() < 128 else "light"]


class SettingsTable(QtWidgets.QWidget):
    """A table of adjustable values (``core.settings.Setting`` as dicts) with Add / Remove / Up / Down.

    Args:
        hint: Explanation shown above the table.
        what: What one row is called in error messages ("Constant", "Option").

    Attributes:
        table: The QTableWidget (one row per value; the Kind column is a combo box).

    Signals:
        changed(): a row was edited, added, removed or moved.
    """

    changed = QtCore.Signal()

    def __init__(self, hint: str, what: str, parent=None) -> None:
        super().__init__(parent)
        self.what = what
        self._filling = False
        col = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(hint)
        label.setObjectName("Muted")
        label.setWordWrap(True)
        col.addWidget(label)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        col.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add", lambda: self.add()), ("Remove", self.remove),
                           ("Up", lambda: self.move(-1)), ("Down", lambda: self.move(1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)

    def set_rows(self, rows: list[dict]) -> None:
        """Show these values (without emitting ``changed``)."""
        self._filling = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for s in rows or []:
            self.add(s)
        self.table.blockSignals(False)
        self._filling = False

    def add(self, s: dict | None = None) -> None:
        """Append a row (a new value named value<n> when ``s`` is None)."""
        s = s if isinstance(s, dict) else {"key": f"value{self.table.rowCount() + 1}", "label": "Value",
                                           "kind": "float", "default": 1.0}
        r = self.table.rowCount()
        self.table.insertRow(r)
        kind = QtWidgets.QComboBox()
        kind.addItems(list(SETTING_KINDS))                     # "label": an annotation class, "Group › Subgroup"
        kind.setCurrentText(s.get("kind", "float"))
        kind.currentTextChanged.connect(lambda *_: self.changed.emit())
        default = s.get("default", "")
        if s.get("kind") == "label" and isinstance(default, (list, tuple)):
            default = " › ".join(str(x) for x in default if x)
        values = [s.get("key", ""), s.get("label", ""), None, default, s.get("minimum", ""),
                  s.get("maximum", ""), s.get("step", ""), s.get("suffix", ""),
                  ", ".join(s.get("choices", []) or []), s.get("tooltip", "")]
        for c, v in enumerate(values):
            if c == 2:
                self.table.setCellWidget(r, c, kind)
            else:
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem("" if v is None else str(v)))
        if not self._filling:
            self.changed.emit()

    def remove(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self.changed.emit()

    def move(self, step: int) -> None:
        r = self.table.currentRow()
        rows = self.rows(validate=False)
        if r < 0 or not 0 <= r + step < len(rows):
            return
        rows[r], rows[r + step] = rows[r + step], rows[r]
        self.set_rows(rows)
        self.table.setCurrentCell(r + step, 0)
        self.changed.emit()

    def rows(self, validate: bool = True) -> list[dict]:
        """The table as a list of setting dicts.

        Raises:
            ValueError: (validate) with the row and the problem.
        """
        out = []
        for r in range(self.table.rowCount()):
            cell = lambda c: (self.table.item(r, c).text().strip() if self.table.item(r, c) else "")  # noqa: E731
            kind = self.table.cellWidget(r, 2).currentText()
            d = {"key": cell(0), "label": cell(1), "kind": kind}
            raw = cell(3)
            try:
                if kind == "int":
                    d["default"] = int(float(raw or 0))
                elif kind == "float":
                    d["default"] = float(raw or 0)
                elif kind == "bool":
                    d["default"] = raw.lower() in ("1", "true", "yes", "on")
                elif kind == "label":
                    from modalytics.core.settings import _label_value
                    d["default"] = list(_label_value(raw))
                else:
                    d["default"] = raw
                for c, name in ((4, "minimum"), (5, "maximum"), (6, "step")):
                    if cell(c):
                        d[name] = int(float(cell(c))) if kind == "int" else float(cell(c))
            except ValueError as exc:
                if validate:
                    raise ValueError(f"{self.what} row {r + 1}: {exc}") from exc
            if cell(7):
                d["suffix"] = self.table.item(r, 7).text()      # keep its leading space
            if kind == "choice":
                d["choices"] = [c.strip() for c in cell(8).split(",") if c.strip()]
            if cell(9):
                d["tooltip"] = cell(9)
            if validate:
                try:
                    Setting.from_dict(d)
                except ValueError as exc:
                    raise ValueError(f"{self.what} row {r + 1}: {exc}") from exc
                if not d["key"].isidentifier():
                    raise ValueError(f"{self.what} row {r + 1}: the key must be a name (letters, digits, _)")
            out.append(d)
        if validate and len({d["key"] for d in out}) != len(out):
            raise ValueError(f"Two {self.what.lower()}s have the same key")
        return out


class ScriptEditor(QtWidgets.QWidget):
    """Edits one script (see the module docstring); subclassed per kind of script.

    Signals:
        saved(str): the id of the script just saved (the page reloads its library).
        install_requested(str): the user wants to install a missing library (the page opens Libraries).
    """

    saved = QtCore.Signal(str)
    install_requested = QtCore.Signal(str)

    SCRIPT: type[ScriptFile] = ScriptFile
    WHAT = "script"                          # "parameter", "importer"
    VALUES_TAB = "Values"                    # name of the adjustable-values tab
    VALUES_HINT = ""
    VALUES_ROW = "Value"                     # one row, in error messages
    VALUES_PROPERTY = "VALUES"               # the property holding them
    TEST_TIP = "Try it (Ctrl+T)"

    def __init__(self, library, parent=None) -> None:
        super().__init__(parent)
        self.library = library
        self.spec = None
        self.path: str | None = None           # the file shown
        self.raw = False                       # the file does not parse: whole text in the code editor
        self.dirty = False
        self._loading = False
        self._known: dict = {}                 # properties the form does not show (kept as they are)
        self.watcher = QtCore.QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._changed_on_disk)

        # 1. Title and origin
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.origin = QtWidgets.QLabel()
        self.origin.setObjectName("Muted")
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.origin)
        col.addLayout(head)

        # 2. Properties | values | Code
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_properties(), "Properties")
        self.values = SettingsTable(self.VALUES_HINT, self.VALUES_ROW)
        self.values.changed.connect(self._changed)
        self.table = self.values.table
        self.tabs.addTab(self.values, self.VALUES_TAB)
        self.tabs.addTab(self._build_code(), "Code")
        col.addWidget(self.tabs, 1)

        # 3. Result line and buttons
        self.result = QtWidgets.QLabel()
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.result_row = QtWidgets.QHBoxLayout()           # a subclass may add a preview beside the result
        self.result_row.addWidget(self.result, 1)
        col.addLayout(self.result_row)
        buttons = QtWidgets.QHBoxLayout()
        self.needs_btn = QtWidgets.QPushButton("Install missing…")
        self.needs_btn.setVisible(False)
        self.needs_btn.clicked.connect(lambda: self.install_requested.emit(self._missing[0] if self._missing else ""))
        self._missing: list[str] = []
        self.test_btn = QtWidgets.QPushButton("Test")
        self.test_btn.setToolTip(self.TEST_TIP)
        self.external_btn = QtWidgets.QPushButton("Open in external editor")
        self.revert_btn = QtWidgets.QPushButton("Revert")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setToolTip("Write the script (Ctrl+S)")
        self.test_btn.clicked.connect(self.test)
        self.external_btn.clicked.connect(self._open_external)
        self.revert_btn.clicked.connect(lambda: self.load(self.spec, self.path))
        self.save_btn.clicked.connect(self.save)
        for b in (self.test_btn, self.external_btn, *self._extra_buttons(), self.needs_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.revert_btn)
        buttons.addWidget(self.save_btn)
        col.addLayout(buttons)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self, self.save)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self, self.test)
        self.setEnabled(False)

    # ------------------------------------------------------------ for subclasses
    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        """Add the property fields to the form (connect their edits to ``self._changed``)."""

    def _extra_buttons(self) -> list[QtWidgets.QPushButton]:
        """Buttons after Test / Open in external editor."""
        return []

    def _fill(self, p: dict) -> None:
        """Show the properties ``p`` in the form."""

    def properties(self) -> dict:
        """The properties from the form (plus any the form does not show)."""
        return dict(self._known)

    def validate(self, p: dict) -> None:
        """Raise ValueError with a message if the properties cannot be saved."""
        if not str(p.get("KEY", "")).isidentifier():
            raise ValueError("The id must be a name (letters, digits, _; not starting with a digit)")

    def target(self, p: dict) -> Path:
        """Where your script with these properties lives."""
        raise NotImplementedError

    def uid_of(self, p: dict) -> str:
        """The id of a script with these properties."""
        return str(p.get("KEY", ""))

    def lock_builtin(self, builtin: bool) -> None:
        """Make the fields a built-in must keep (its id…) read-only."""

    def run_test(self, path: Path) -> None:
        """Load the script at ``path`` (the edited text) and try it; report in ``self.result``."""

    # ------------------------------------------------------------ layout
    def _build_properties(self) -> QtWidgets.QWidget:
        # The form keeps its natural height at the top and a readable width (it would otherwise spread
        # its rows over the whole tab and stretch every field across it)
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        box = QtWidgets.QWidget()
        box.setMaximumWidth(680)
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(0, 4, 0, 0)
        outer.addWidget(box)
        outer.addStretch(1)
        self._build_form(form)
        return w

    def _build_code(self) -> QtWidgets.QWidget:
        split = QtWidgets.QSplitter()
        self.code = CodeEditor(current_tokens())
        self.code.textChanged.connect(self._changed)
        self.helper = HelperPanel()
        self.helper.insert_requested.connect(self.code.insert_snippet)
        split.addWidget(self.code)
        split.addWidget(self.helper)
        split.setSizes([620, 280])
        return split

    # ------------------------------------------------------------ loading
    def load(self, spec, path: str | None = None) -> None:
        """Show a script (or, with ``spec`` None and a ``path``, a script that failed to load)."""
        self.spec, self.path = spec, (path or (spec.path if spec else None))
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        self.setEnabled(self.path is not None)
        self.result.setText("")
        if self.path is None:
            self.title.setText("")
            self.origin.setText("")
            return
        self.watcher.addPath(self.path)
        text = Path(self.path).read_text(encoding="utf-8")
        self._loading = True
        try:
            script = self.SCRIPT.parse(text)
            self.raw = False
        except SyntaxError as exc:
            script, self.raw = None, True
            self.result.setText(f"✖ The script does not parse (line {exc.lineno}): {exc.msg}. "
                                f"Fix it in the Code tab.")
        self.tabs.setTabEnabled(0, not self.raw)
        self.tabs.setTabEnabled(1, not self.raw)
        if self.raw:
            self.code.setPlainText(text)
            self.tabs.setCurrentIndex(2)
        else:
            self._known = dict(script.properties)
            self._fill(script.properties)
            self.values.set_rows(script.properties.get(self.VALUES_PROPERTY, []) or [])
            self.code.setPlainText(script.code)
        builtin = spec is not None and spec.source == "builtin"
        self.lock_builtin(builtin)
        self.origin.setText("built in · Save makes your own copy" if builtin else
                            ("yours · replaces the built-in" if spec and spec.uid in self.library.overridden else
                             "yours" if spec else "could not be loaded"))
        self._loading = False
        self._set_dirty(False)

    def set_missing(self, names: list[str]) -> None:
        """Libraries the shown script needs but are not installed (shows the Install button)."""
        self._missing = list(names)
        self.needs_btn.setVisible(bool(names))
        if names:
            self.result.setText(f"ℹ This {self.WHAT} needs {', '.join(names)}, which is not installed here.")

    # ------------------------------------------------------------ state
    def _changed(self, *_args) -> None:
        if not self._loading and self.path:
            self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self.dirty = dirty
        self.save_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)

    def confirm_leave(self) -> bool:
        """Ask about unsaved edits. Returns False to stay."""
        if not self.dirty:
            return True
        btn = QtWidgets.QMessageBox.question(
            self, f"Unsaved {self.WHAT}", f"Save the changes to this {self.WHAT}?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if btn == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        if btn == QtWidgets.QMessageBox.StandardButton.Discard:
            self._set_dirty(False)
            return True
        return False

    def _changed_on_disk(self, path: str) -> None:
        """The file was changed elsewhere (external editor): load it unless there are edits here."""
        if os.path.exists(path) and path not in self.watcher.files():
            self.watcher.addPath(path)                 # editors that save by renaming drop the watch
        if path != self.path:
            return
        if self.dirty:
            self.result.setText("ℹ The file changed on disk. Revert loads that version (your edits here are lost).")
        else:
            self.load(self.spec, self.path)
            self.result.setText("ℹ Loaded the version saved by the external editor.")

    # ------------------------------------------------------------ the script text
    def all_properties(self) -> dict:
        """The form's properties with the values table."""
        p = self.properties()
        p[self.VALUES_PROPERTY] = self.values.rows()
        return p

    def text(self) -> str:
        """The script as it would be saved.

        Raises:
            ValueError: If a property or value is invalid.
        """
        if self.raw:
            return self.code.toPlainText()
        p = self.all_properties()
        self.validate(p)
        return self.SCRIPT(p, self.code.toPlainText()).render()

    def _code_line(self, file_line: int) -> int:
        """Line in the code editor of a line of the saved file (the properties block sits at the marker)."""
        if self.raw:
            return file_line
        lines = self.code.toPlainText().split("\n")
        marker = next((i for i, ln in enumerate(lines) if ln.strip() == self.SCRIPT.MARKER), None)
        if marker is None:
            return file_line
        try:
            block = self.SCRIPT.block(self.all_properties()).count("\n")
        except ValueError:
            return file_line
        return file_line if file_line <= marker else max(marker + 1, file_line - block + 1)

    # ------------------------------------------------------------ actions
    def save(self) -> bool:
        """Write the script (a built-in becomes your own copy). Returns True if written."""
        if not self.path or not self.dirty and not self.raw:
            return False
        try:
            text = self.text()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, f"Save {self.WHAT}", str(exc))
            return False
        try:
            ast.parse(text)
        except SyntaxError as exc:
            if QtWidgets.QMessageBox.question(
                    self, f"Save {self.WHAT}", f"Line {self._code_line(exc.lineno or 1)}: {exc.msg}.\n\nSave "
                    f"anyway? It will be listed under “Could not load” until fixed.") != \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        # 1. Where: the place for its properties in your library (a built-in is copied there;
        #    a script whose id or type changed moves there)
        old = Path(self.path)
        p = None if self.raw else self.all_properties()
        if self.raw:
            target = old
        else:
            target = self.target(p)
            if self.spec is not None and self.spec.source == "user" and old.resolve() != target.resolve() \
                    and target.exists():
                QtWidgets.QMessageBox.warning(self, f"Save {self.WHAT}", f"{target} already exists.")
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
        # 2. Write (atomically), drop the old file after a rename
        fileio.write_text(target, text)
        if target.resolve() != old.resolve() and (self.spec is None or self.spec.source == "user"):
            old.unlink(missing_ok=True)                  # moved (renamed, or another type)
        uid = "" if self.raw else self.uid_of(p)
        self._set_dirty(False)
        if target != old and self.spec is not None and self.spec.source == "builtin":
            self.result.setText(f"✔ Saved as your own copy, which now replaces the built-in:\n{target}")
        else:
            self.result.setText(f"✔ Saved {target.name}")
        self.path = str(target)
        self.saved.emit(uid)
        return True

    def test(self) -> None:
        """Try the script as edited (unsaved): written to a temporary file and handed to ``run_test``."""
        if not self.path:
            return
        try:
            text = self.text()
        except ValueError as exc:
            self.result.setText(f"✖ {exc}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / f"{self.WHAT}_under_test.py"
            f.write_text(text, encoding="utf-8")
            self.run_test(f)

    def report_error(self, exc: BaseException, script: Path, loading: bool = False) -> None:
        """Show an error of the script under test with its line, and put the cursor on that line.

        Args:
            exc: The exception (a ValueError from ``load_script`` when ``loading``).
            script: The temporary file the script was run from (to find its frame).
            loading: The script failed to load (else: it failed while running).
        """
        import traceback
        if loading:
            line = getattr(exc.__cause__, "lineno", None)
            where = f" (line {self._code_line(line)})" if line else ""
            self.result.setText(f"✖ It does not load: {exc}{where}")
        else:
            tb = traceback.extract_tb(exc.__traceback__)
            frame = next((fr for fr in reversed(tb) if fr.filename == str(script)), None)
            line = frame.lineno if frame else None
            where = f" (line {self._code_line(line)}: {frame.line})" if frame else ""
            self.result.setText(f"✖ {type(exc).__name__}: {exc}{where}")
        if line:
            self.code.go_to_line(self._code_line(line))

    def _open_external(self) -> None:
        """Open the file in the system's editor (a built-in is first copied: it is never edited in place)."""
        if not self.path:
            return
        if self.spec is not None and self.spec.source == "builtin":
            target = self.target(self._known)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(Path(self.path).read_text(encoding="utf-8"), encoding="utf-8")
            self.result.setText(f"ℹ Opened your own copy (it replaces the built-in):\n{target}")
            self.saved.emit(self.spec.uid)               # the page reloads and selects the copy
            open_in_system(target)
            return
        open_in_system(self.path)
