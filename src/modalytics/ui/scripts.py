"""
Running an external script (Scripts menu, or Settings › Scripts › Run).

    ┌ DHM phase map extractor ────────────────────────────────────────────┐
    │ Gather every Phase.txt of a DHM database folder into one folder…    │
    │ Input folder   [C:/data/dhm/sample01                        ] […]  │
    │ Output folder  [C:/data/phase maps                          ] […]  │
    │ …                                                                   │
    │ [██████████░░░░░░░░░░]                                              │
    │ log: ✅ z1/1/Phase.txt → sample01-1-1.txt …                         │
    │                                              [Run] [Stop] [Close]   │
    └─────────────────────────────────────────────────────────────────────┘

The form is built from the script's OPTIONS; the values used last are kept for the
next run (``app_settings`` "script_options"). The script runs in a background
thread; its log lines, progress and result come back through Qt signals. Files it
hands over (``ctx.import_files``) are offered for import into the open project.
"""

from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from modalytics import app_settings, diagnostics
from modalytics.scripts import registry as sr
from modalytics.ui.settings.widgets import muted, setting_widget


class _Bridge(QtCore.QObject):
    line = QtCore.Signal(str)
    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object, object)          # (result, error)


class RunScriptDialog(QtWidgets.QDialog):
    """Ask a script's options, run it, show its log (see the module docstring).

    Args:
        spec: The script.
        on_import: Called with the files the script hands over, if the user wants them imported (None: not offered).
    """

    def __init__(self, spec: sr.ScriptSpec, parent=None, on_import=None) -> None:
        super().__init__(parent)
        self.spec, self.on_import = spec, on_import
        self.setWindowTitle(f"{spec.name} — Modalytics")
        self.resize(760, 560)
        saved = app_settings.get("script_options").get(spec.uid, {})
        self.values = spec.resolve(saved)
        self._cancel = threading.Event()
        self._thread = None
        self.bridge = _Bridge()
        self.bridge.line.connect(self._line)
        self.bridge.progress.connect(lambda f: self.bar.setValue(int(round(f * 1000))))
        self.bridge.finished.connect(self._finished)
        col = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(f"<b>{spec.name}</b>")
        col.addWidget(title)
        if spec.description:
            d = muted(spec.description)
            d.setWordWrap(True)
            col.addWidget(d)
        # 1. The options
        form = QtWidgets.QFormLayout()
        self.widgets = {}
        for opt in spec.options:
            w = setting_widget(opt, self.values[opt.key], lambda v, k=opt.key: self.values.__setitem__(k, v))
            self.widgets[opt.key] = w
            form.addRow(opt.label, w)
        col.addLayout(form)
        # 2. Progress and log
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        col.addWidget(self.bar)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("What the script does appears here.")
        col.addWidget(self.log, 1)
        # 3. Buttons
        row = QtWidgets.QHBoxLayout()
        self.result = muted("")
        self.result.setWordWrap(True)
        row.addWidget(self.result, 1)
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._cancel.set)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (self.run_btn, self.stop_btn, close):
            row.addWidget(b)
        col.addLayout(row)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Run the script with the values in the form (in the background)."""
        if self.busy:
            return
        # A typed path is only taken when its field loses focus: read every text field now
        for opt in self.spec.options:
            w = self.widgets[opt.key]
            edit = w if isinstance(w, QtWidgets.QLineEdit) else w.findChild(QtWidgets.QLineEdit)
            if opt.kind in ("file", "folder", "text") and edit is not None:
                self.values[opt.key] = edit.text().strip()
        stored = dict(app_settings.get("script_options"))
        stored[self.spec.uid] = dict(self.values)
        app_settings.put("script_options", stored)
        self._cancel.clear()
        self.log.clear()
        self.result.setText("")
        self.bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        spec, values, bridge, cancel = self.spec, dict(self.values), self.bridge, self._cancel

        def work():
            ctx = sr.ScriptContext(bridge.line.emit, bridge.progress.emit, cancel.is_set)
            try:
                summary = sr.run(spec, values, ctx)
                bridge.finished.emit((summary, ctx.to_import), None)
            except BaseException as exc:                      # delivered to the window, never lost
                bridge.finished.emit(None, exc)

        self._thread = threading.Thread(target=work, name=f"script {spec.uid}", daemon=True)
        self._thread.start()

    def _line(self, text: str) -> None:
        self.log.appendPlainText(text)
        diagnostics.event("script", line=text)

    def _finished(self, result, error) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if error is not None:
            if isinstance(error, sr.Cancelled):
                self.result.setText("Stopped.")
                return
            diagnostics.error(f"script {self.spec.uid} failed", error)
            self.result.setText(f"❌ {error}")
            self.log.appendPlainText(f"❌ {type(error).__name__}: {error}")
            return
        summary, files = result
        self.bar.setValue(1000)
        self.result.setText(f"✅ {summary}" if summary else "✅ Done")
        if files and self.on_import is not None:
            ask = QtWidgets.QMessageBox.question(
                self, self.spec.name, f"Import the {len(files)} file(s) it made into the open project?")
            if ask == QtWidgets.QMessageBox.StandardButton.Yes:
                self.on_import(files)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self.busy:
            self._cancel.set()
        super().closeEvent(event)
