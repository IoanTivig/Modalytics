"""
Background tasks with progress and cancel, shown in the status bar.

Long jobs (segmentation, later registration and training) run in a worker
thread; the UI stays responsive. Results are delivered back on the UI thread,
where they are applied to the project as undoable commands.

Tools reach this through ``ToolContext.tasks`` (``run``, ``cancel``,
``message``, ``busy``), so they never import Qt.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6 import QtCore, QtWidgets


class TaskRunner(QtCore.QObject):
    """Runs one background task at a time.

    Args:
        status_bar: Where the progress widget and messages appear.
    """

    _progress = QtCore.Signal(float)
    _finished = QtCore.Signal(object, object)      # (result, error)

    def __init__(self, status_bar: QtWidgets.QStatusBar) -> None:
        super().__init__(status_bar)
        self.status_bar = status_bar
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_done = None
        self._on_error = None
        self._on_cancel = None
        # Status-bar widget: label · progress bar · Cancel
        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(self.widget)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(180)
        self.bar.setTextVisible(False)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        for w in (self.label, self.bar, self.cancel_button):
            lay.addWidget(w)
        status_bar.addPermanentWidget(self.widget)
        self.widget.hide()
        self._progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        self._finished.connect(self._done)

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def run(self, label: str, fn: Callable[[Callable[[float], None], Callable[[], bool]], Any],
            on_done: Callable[[Any], None], on_error: Callable[[BaseException], None] | None = None,
            on_cancel: Callable[[], None] | None = None) -> bool:
        """Start ``fn(progress, cancelled)`` in a worker thread.

        Args:
            label: Text shown next to the progress bar.
            fn: The job; call ``progress(0–1)`` and check ``cancelled()``.
            on_done: Called on the UI thread with the result.
            on_error: Called on the UI thread with the exception (default: status message).
            on_cancel: Called on the UI thread when the job was cancelled (after the status message).

        Returns:
            bool: False if another task is still running.
        """
        if self.busy:
            self.message("⚠️ Another task is still running")
            return False
        self._cancel.clear()
        self._on_done, self._on_error, self._on_cancel = on_done, on_error, on_cancel
        import time
        from modalytics import diagnostics
        self._started = (label, time.perf_counter())
        diagnostics.event("task started", task=label)
        self.label.setText(label)
        self.bar.setValue(0)
        self.widget.show()

        def work():
            try:
                result = fn(self._progress.emit, self._cancel.is_set)
                self._finished.emit(result, None)
            except BaseException as exc:          # delivered to the UI thread, never lost
                from modalytics.services.segmentation import Cancelled
                if not isinstance(exc, Cancelled):
                    traceback.print_exc()
                self._finished.emit(None, exc)

        self._thread = threading.Thread(target=work, name="task", daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Ask the running task to stop at its next check."""
        if self.busy:
            self._cancel.set()
            self.label.setText(self.label.text() + " — cancelling…")

    def confirm_box(self, title: str, text: str, ok_text: str = "Continue"):
        """The question dialog (separate from ``confirm`` so it can be shown and grabbed in tests)."""
        box = QtWidgets.QMessageBox(self.status_bar.window())
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(f"<b>{title}</b>")
        box.setInformativeText(text)
        ok = box.addButton(ok_text, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(ok)
        return box, ok

    def confirm(self, title: str, text: str, ok_text: str = "Continue") -> bool:
        """Ask before a long job. Returns True if the user agrees."""
        box, ok = self.confirm_box(title, text, ok_text)
        box.exec()
        return box.clickedButton() is ok

    def edit_options(self, title: str, settings: list, values: dict, note: str = "") -> dict | None:
        """A dialog with one field per setting (``core.settings.Setting``); the values if OK, else None."""
        from modalytics.ui.settings.widgets import muted, setting_widget
        dlg = QtWidgets.QDialog(self.status_bar.window())
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(420)
        form = QtWidgets.QFormLayout(dlg)
        if note:
            form.addRow(muted(note))
        chosen = {s.key: values.get(s.key, s.default) for s in settings}
        for s in settings:
            w = setting_widget(s, chosen[s.key], lambda v, k=s.key: chosen.__setitem__(k, v))
            if s.tooltip:
                w.setToolTip(s.tooltip)
            form.addRow(s.label, w)
        if not settings:
            form.addRow(muted("This method has no options of its own."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        return chosen if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted else None

    def message(self, text: str, ms: int = 6000) -> None:
        self.status_bar.showMessage(text, ms)

    def _done(self, result, error) -> None:
        import time
        from modalytics import diagnostics
        label, t0 = getattr(self, "_started", ("?", time.perf_counter()))
        seconds = round(time.perf_counter() - t0, 2)
        if error is None:
            diagnostics.event("task finished", task=label, seconds=seconds)
        elif type(error).__name__ == "Cancelled":
            diagnostics.event("task cancelled", task=label, seconds=seconds)
        else:
            diagnostics.error(f"task failed: {label} ({seconds} s)", error)
        self._thread = None
        self.widget.hide()
        if error is not None:
            from modalytics.services.segmentation import Cancelled
            if isinstance(error, Cancelled):
                self.message("ℹ️ Cancelled")
                if self._on_cancel is not None:
                    self._on_cancel()
            elif self._on_error is not None:
                self._on_error(error)
            else:
                self.message(f"❌ {error}", 10000)
            return
        self._on_done(result)
