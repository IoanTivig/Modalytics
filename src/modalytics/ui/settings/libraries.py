"""
Libraries (Settings › Parameters or Importers › Libraries…): what scripts can import.

    ┌ Libraries ────────────────────────────────────────────────────────────┐
    │ INCLUDED WITH MODALYTICS (locked)        [filter…]                    │
    │   numpy 2.x · scipy 1.x · scikit-image · shapely · opencv …           │
    │ YOURS                                                                 │
    │   Library      Version   Latest       What it is                      │
    │   PyWavelets   1.8.0     1.10.0 ↑     wavelet transform module        │
    │   [Folder]   [Update selected] [Update all]   [Remove]                │
    │ [package name……………………]  [Install]                                    │
    │ log                                                                   │
    └───────────────────────────────────────────────────────────────────────┘

"Latest" is the newest version that works with the included (locked)
libraries, i.e. what Update would install; it is looked up in the background
(one pip dry run per library) when the window opens and after every change.

Installing runs pip in two steps (``parameters.libraries`` has the rules):
plan (resolve with the included libraries pinned; refuse what would change
them), then install only what is missing into your libraries folder. Both
run in the background (QProcess); the log shows pip's output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.parameters import libraries as libs
from modalytics.ui.settings.widgets import muted, open_in_system, section


def _friendly(stderr: str) -> str:
    """pip's error, in one sentence a user can act on."""
    if "ResolutionImpossible" in stderr or "conflict" in stderr.lower():
        return ("Refused: it needs a different version of a library included with Modalytics, and those are "
                "locked (they are the versions Modalytics is tested with).")
    if "No matching distribution" in stderr or "Could not find a version" in stderr:
        return ("Not found as a ready-built package for this computer (or not on PyPI). Packages that need a "
                "compiler are not installed.")
    if "Could not fetch" in stderr or "ConnectionError" in stderr or "NewConnectionError" in stderr:
        return "No connection to PyPI (pypi.org): check the internet connection."
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "pip failed."


class LibrariesDialog(QtWidgets.QDialog):
    """The libraries window.

    Args:
        install: A requirement to put in the install field (e.g. what a parameter needs).

    Signals:
        changed(): a library was installed or removed (parameters are loaded again).
    """

    changed = QtCore.Signal()

    def __init__(self, install: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Libraries for parameters")
        self.resize(860, 720)
        self.proc: QtCore.QProcess | None = None      # an install running
        self.checker: QtCore.QProcess | None = None   # a latest-version lookup running
        self._latest: dict[str, str] = {}
        self._latest_queue: list[str] = []
        self.tmp = tempfile.TemporaryDirectory()
        col = QtWidgets.QVBoxLayout(self)
        col.addWidget(muted(
            "Parameter and importer scripts can import the libraries included with Modalytics, and small helper, "
            "statistics or image-processing libraries you add here. Included libraries are locked at the "
            "versions Modalytics is tested with. Yours go into your library folder, never into Modalytics "
            "itself. Ready-built packages only; large machine-learning frameworks belong to Training."))

        # 1. Included (locked)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(section("Included with Modalytics (locked)"))
        row.addStretch(1)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter…")
        self.filter.setMaximumWidth(220)
        self.filter.textChanged.connect(self._filter)
        row.addWidget(self.filter)
        col.addLayout(row)
        col.addWidget(muted("The ones parameters use most, by import name: numpy (np), scipy (scipy.stats, "
                            "scipy.ndimage), scikit-image (skimage), shapely, OpenCV (cv2), and Python's math "
                            "and statistics. The rest support Modalytics itself."))
        self.included = self._table(["Library", "Version", "What it is"])
        col.addWidget(self.included, 3)

        # 2. Yours: the table, then what acts on it
        col.addWidget(section("Yours"))
        self.yours = self._table(["Library", "Version", "Latest", "What it is"])
        self.yours.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.yours.itemSelectionChanged.connect(self._selection_changed)
        col.addWidget(self.yours, 1)
        row = QtWidgets.QHBoxLayout()
        folder = QtWidgets.QPushButton("Folder")
        folder.setToolTip(str(libs.folder()))
        folder.clicked.connect(lambda: open_in_system(libs.folder()))
        self.update_btn = QtWidgets.QPushButton("Update selected")
        self.update_btn.setToolTip("Install the newest version that works with the included libraries, for "
                                   "the selected rows (Ctrl / Shift-click to select several)")
        self.update_btn.clicked.connect(lambda: self._update(selected_only=True))
        self.update_all_btn = QtWidgets.QPushButton("Update all")
        self.update_all_btn.setToolTip("Update every library of yours that has a newer version")
        self.update_all_btn.clicked.connect(lambda: self._update(selected_only=False))
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove)
        row.addWidget(folder)
        row.addWidget(self.update_btn)
        row.addWidget(self.update_all_btn)
        row.addStretch(1)
        row.addWidget(self.remove_btn)
        col.addLayout(row)

        # 3. Install
        row = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLineEdit(install)
        self.name.setPlaceholderText("Package name on PyPI, e.g. PyWavelets, statsmodels, mahotas")
        self.name.returnPressed.connect(self._install)
        self.install_btn = QtWidgets.QPushButton("Install")
        self.install_btn.clicked.connect(self._install)
        row.addWidget(self.name, 1)
        row.addWidget(self.install_btn)
        col.addLayout(row)
        if not libs.pip_available():
            self.install_btn.setEnabled(False)
            self.name.setEnabled(False)
            col.addWidget(muted("Installing needs pip, which this installation does not include. Libraries "
                                "copied into your library folder can still be used."))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log.setMaximumHeight(150)
        col.addWidget(self.log)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        col.addWidget(close)
        self.refresh()

    @staticmethod
    def _table(headers: list[str]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        for c in range(len(headers) - 1):
            t.horizontalHeader().setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def refresh(self) -> None:
        """Re-read both lists, then look up the latest versions of yours in the background."""
        self.included.setRowCount(0)
        for lib in libs.included():
            r = self.included.rowCount()
            self.included.insertRow(r)
            for c, v in enumerate((lib.name, lib.version, lib.summary)):
                self.included.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.yours.setRowCount(0)
        for lib in libs.yours():
            r = self.yours.rowCount()
            self.yours.insertRow(r)
            latest = QtWidgets.QTableWidgetItem("checking…" if libs.pip_available() else "—")
            for c, item in enumerate((QtWidgets.QTableWidgetItem(lib.name), QtWidgets.QTableWidgetItem(lib.version),
                                      latest, QtWidgets.QTableWidgetItem(lib.summary))):
                self.yours.setItem(r, c, item)
        self._filter(self.filter.text())
        self._selection_changed()
        self._check_latest()

    def _selection_changed(self) -> None:
        busy = self.proc is not None
        has = self.yours.rowCount() > 0
        selected = bool(self.yours.selectionModel().selectedRows())
        self.update_btn.setEnabled(not busy and selected and libs.pip_available())
        self.update_all_btn.setEnabled(not busy and has and libs.pip_available())
        self.remove_btn.setEnabled(not busy and selected)

    # ------------------------------------------------------------ latest versions (background)
    def _check_latest(self) -> None:
        """Look up, one library after another, the newest version that works here."""
        if not libs.pip_available():
            return
        self._latest_queue = [self.yours.item(r, 0).text() for r in range(self.yours.rowCount())]
        self._latest: dict[str, str] = {}
        if self.checker is None:
            self._next_latest()

    def _next_latest(self) -> None:
        if not self._latest_queue:
            self.checker = None
            return
        name = self._latest_queue.pop(0)
        report = Path(self.tmp.name) / "latest.json"
        report.unlink(missing_ok=True)
        cmd = libs.plan_command(name, report, libs.constraints_file(Path(self.tmp.name) / "constraints.txt"))
        self.checker = QtCore.QProcess(self)
        self.checker.setProgram(cmd[0])
        self.checker.setArguments(cmd[1:])

        def finished(code, _status):
            err = bytes(self.checker.readAllStandardError()).decode("utf-8", "replace") if self.checker else ""
            version, tip = "", ""
            if code == 0 and report.exists():
                version = next((i["metadata"]["version"] for i in libs.read_report(report).get("install", [])
                                if libs.canonical(i["metadata"]["name"]) == libs.canonical(name)), "")
            else:
                tip = _friendly(err)
            self._show_latest(name, version, tip)
            self._next_latest()
        self.checker.finished.connect(finished)
        self.checker.start()

    def _show_latest(self, name: str, latest: str, problem: str) -> None:
        """Fill the Latest cell of one library ("↑" when newer than the installed one)."""
        for r in range(self.yours.rowCount()):
            if libs.canonical(self.yours.item(r, 0).text()) != libs.canonical(name):
                continue
            cell = self.yours.item(r, 2)
            installed = self.yours.item(r, 1).text()
            if not latest:
                cell.setText("no connection" if "connection" in problem else "unknown")
                cell.setToolTip(problem)
            elif latest == installed:
                cell.setText("up to date")
                cell.setToolTip(f"{latest} is the newest version that works with the included libraries")
            else:
                cell.setText(f"{latest}  ↑")
                cell.setForeground(self.palette().brush(QtGui.QPalette.ColorRole.Highlight))
                cell.setToolTip(f"Update selected / Update all installs {latest}")
            self._latest[libs.canonical(name)] = latest

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        for r in range(self.included.rowCount()):
            row = " ".join(self.included.item(r, c).text().lower() for c in range(3))
            self.included.setRowHidden(r, bool(text) and text not in row)

    def _say(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ------------------------------------------------------------ install: plan, then install
    def _update(self, selected_only: bool) -> None:
        """Update the selected libraries of yours, or all of them."""
        if self.proc is not None:
            return
        if selected_only:
            rows = sorted(i.row() for i in self.yours.selectionModel().selectedRows())
        else:
            rows = range(self.yours.rowCount())
        if not selected_only:                        # Update all: skip the ones known to be up to date
            rows = [r for r in rows if self.yours.item(r, 2).text() != "up to date"]
        names = [self.yours.item(r, 0).text() for r in rows]
        if not names:
            self._say("✔ Everything of yours is up to date.")
            return
        self._queue = names
        self._next_update()

    def install_many(self, names: list[str]) -> None:
        """Install several libraries one after the other (e.g. what an imported profile's scripts need)."""
        self._queue = [n for n in names if n]
        self._next_update()

    def _next_update(self) -> None:
        if getattr(self, "_queue", None):
            self.name.setText(self._queue.pop(0))
            self._install(then=self._next_update)

    def _install(self, then=None) -> None:
        req = self.name.text().strip()
        if not req or self.proc is not None:
            return
        self._then = then if callable(then) else None
        why = libs.check_name(req)
        if why:
            self._say(f"✖ {req}: {why}")
            return
        self._busy(True)
        self._say(f"… {req}: finding what it needs (included libraries pinned)")
        report = Path(self.tmp.name) / "plan.json"
        report.unlink(missing_ok=True)
        cmd = libs.plan_command(req, report, libs.constraints_file(Path(self.tmp.name) / "constraints.txt"))
        self._run(cmd, lambda ok, err: self._planned(req, report, ok, err))

    def _planned(self, req: str, report: Path, ok: bool, err: str) -> None:
        if not ok:
            self._say(f"✖ {req}: {_friendly(err)}")
            self._done()
            return
        todo, problems = libs.plan_from_report(libs.read_report(report))
        if problems:
            self._say(f"✖ {req}: refused, {'; '.join(problems)} (included libraries are locked).")
            self._done()
            return
        have = {libs.canonical(lib.name): lib.version for lib in libs.yours()}
        todo = [t for t in todo if have.get(libs.canonical(t.split("==")[0])) != t.split("==")[1]]
        if not todo:
            self._say(f"✔ {req}: already the newest version that works here; nothing to install.")
            self._done()
            return
        self._say(f"… installing into your libraries: {', '.join(todo)}")
        stage = libs.staging()
        self._run(libs.install_command(todo, stage), lambda ok2, err2: self._installed(req, stage, ok2, err2))

    def _installed(self, req: str, stage: Path, ok: bool, err: str) -> None:
        if ok:
            changes = libs.finish_install(stage)
            text = ", ".join(f"{n} {old} → {new}" if old else f"{n} {new}" for n, old, new in changes)
            self._say(f"✔ {'Updated' if any(old for _n, old, _v in changes) else 'Installed'} {text}. "
                      f"Scripts can import it now (restart Modalytics if one had already imported an older "
                      f"version).")
            self.name.clear()
            self.refresh()
            self.changed.emit()
        else:
            import shutil
            shutil.rmtree(stage, ignore_errors=True)
            self._say(f"✖ {req}: {_friendly(err)} Nothing was changed.")
        self._done()

    def _done(self) -> None:
        """One install finished: next in an update queue, or ready."""
        self._busy(False)
        then, self._then = getattr(self, "_then", None), None
        if then is not None:
            then()

    def _run(self, cmd: list[str], done) -> None:
        """Run a command in the background; ``done(ok, stderr)`` when it ends."""
        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(cmd[0])
        self.proc.setArguments(cmd[1:])

        def finished(code, _status):
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace")
            self.proc = None
            done(code == 0, err)
        self.proc.finished.connect(finished)
        self.proc.start()

    def _busy(self, busy: bool) -> None:
        if busy:
            for b in (self.update_btn, self.update_all_btn, self.remove_btn):
                b.setEnabled(False)
        else:
            self._selection_changed()
        self.install_btn.setEnabled(not busy and libs.pip_available())
        self.name.setEnabled(not busy and libs.pip_available())
        self.setCursor(QtCore.Qt.CursorShape.BusyCursor if busy else QtCore.Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------ remove
    def _remove(self) -> None:
        names = [self.yours.item(i.row(), 0).text() for i in self.yours.selectionModel().selectedRows()]
        if not names:
            self._say("ℹ Select one of your libraries first.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Remove library", f"Remove {', '.join(names)}?\n\nParameters that import it will stop "
                f"loading until it is installed again.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for name in names:
            try:
                libs.remove(name)
                self._say(f"✔ Removed {name}. Restart Modalytics if a parameter had already imported it.")
            except KeyError:
                self._say(f"✖ {name} is not in your libraries folder.")
        self.refresh()
        self.changed.emit()

    def reject(self) -> None:
        if self.proc is not None:
            self._say("ℹ Still working; wait until it finishes.")
            return
        if self.checker is not None:                 # only looking up versions: stop that
            self._latest_queue = []
            self.checker.kill()
            self.checker.waitForFinished(2000)
            self.checker = None
        super().reject()
