"""
The training pack in Training › Segmentation models › Train: installing it, or switching when the hardware changed.

    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Training needs the training pack (PyTorch and its libraries). For this computer (NVIDIA RTX   │
    │ A2000 12GB): the pack for an NVIDIA GPU (CUDA), about 2.6 GB to download.                      │
    │ [Install the training pack]                                                     ▓▓▓▓░░░░░░░  │
    │ ┌ log ────────────────────────────────────────────────────────────────────────────────────┐ │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

Hidden when PyTorch is there and fits the hardware. The work is ``modalytics-cli --install-training VARIANT``
(``training.pack.install``) in a separate process, its output shown as it comes: the same as the installer's
*Download training support now*. A first pack is used at once; a switch takes effect at the next start (PyTorch,
once loaded, stays).
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from modalytics.ui.settings.widgets import muted


class PackPanel(QtWidgets.QFrame):
    """The training pack's state and its Install / Switch button (see the module docstring).

    Signals:
        ready(): a first pack was installed and loaded — training can start.
    """

    ready = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.proc: QtCore.QProcess | None = None
        self.state: dict = {}
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 6)
        self.message = muted("")
        self.message.setWordWrap(True)
        col.addWidget(self.message)
        row = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton("Install the training pack")
        self.button.clicked.connect(self.install)
        row.addWidget(self.button)
        self.keep = QtWidgets.QPushButton("Keep the current one")
        self.keep.clicked.connect(self.hide)
        row.addWidget(self.keep)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setVisible(False)
        row.addWidget(self.busy, 1)
        row.addStretch(0)
        col.addLayout(row)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFixedHeight(140)
        self.log.setVisible(False)
        col.addWidget(self.log)
        self.refresh()

    def refresh(self) -> None:
        """Ask ``training.pack.status`` again and show what it says."""
        from modalytics.training import pack
        self.state = pack.status()
        state = self.state["state"]
        self.setVisible(state in ("none", "changed"))
        self.message.setText(self.state["message"])
        variant = pack.variants()[self.state["recommended"]] if self.state["recommended"] else None
        if variant is not None:
            verb = "Install" if state == "none" else "Switch to"
            self.button.setText(f"{verb} the training pack for {variant.label}")
        self.keep.setVisible(state == "changed")

    def install(self) -> None:
        """Install (or switch to) the pack the hardware asks for, in a separate process."""
        from modalytics import paths
        if self.proc is not None:
            return
        cmd = [*paths.helper_command(), "--install-training", self.state.get("recommended") or "cpu"]
        self.log.clear()
        self.log.setVisible(True)
        self.busy.setVisible(True)
        self.button.setEnabled(False)
        self.keep.setEnabled(False)
        self.proc = QtCore.QProcess(self)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._output)
        self.proc.finished.connect(self._finished)
        self.proc.start(cmd[0], cmd[1:])

    def _output(self) -> None:
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            if line.strip():
                self.log.appendPlainText(line)

    def _finished(self, code: int, _status) -> None:
        from modalytics.training import pack
        self.proc = None
        self.busy.setVisible(False)
        self.button.setEnabled(True)
        self.keep.setEnabled(True)
        if code != 0:
            self.log.appendPlainText("❌ The training pack was not installed (see above); nothing was changed.")
            return
        was = self.state["state"]
        if was == "none" and pack.activate():
            self.refresh()
            self.ready.emit()
        else:                                                # a switch: PyTorch already loaded stays until a restart
            self.refresh()
            self.setVisible(True)
            self.message.setText("✅ The new training pack is installed: it is used from the next start of "
                                 "Modalytics.")
            self.button.setVisible(False)
            self.keep.setVisible(False)
