"""
Training › … › XLSX…: a run's results saved as an Excel workbook (``report.xlsx``) and opened.

The workbook is written in the background; the save dialog proposes the run's folder and its report's name.
XlsxWriter is part of Modalytics (bundled by the installer); without it — a source checkout set up before it was
added, or a damaged installation — the message says what to do.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

INSTALL = "python -m pip install XlsxWriter"


def save_xlsx(panel, html_report: Path, write: Callable[[str], str], what: str) -> None:
    """Ask where, write the workbook in the background (``write(path)``), then open it.

    Args:
        panel: The training panel (its ``win``: tasks and status bar).
        html_report: The run's report (the workbook is proposed beside it, under its name).
        write: Writes the workbook to the path given (in the background thread).
        what: For the messages, e.g. "the analysis".
    """
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        from modalytics.paths import installed
        QtWidgets.QMessageBox.information(
            panel, "XLSX", "Saving workbooks needs XlsxWriter, a part of Modalytics this installation is missing: "
            "reinstall Modalytics." if installed() else "Saving workbooks needs XlsxWriter, which this environment "
            f"does not have yet. In the Modalytics environment, run:\n\n    {INSTALL}\n\nthen try again.")
        return
    start = str(Path(html_report).with_suffix(".xlsx")).replace("Report - ", "Results - ") if html_report else ""
    path, _ = QtWidgets.QFileDialog.getSaveFileName(panel, f"Save {what} as a workbook", start, "Excel workbook (*.xlsx)")
    if not path:
        return
    if not path.lower().endswith(".xlsx"):
        path += ".xlsx"

    def done(written) -> None:
        panel.win.statusBar().showMessage(f"✅ Workbook saved: {written}", 10000)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(written))

    def failed(exc) -> None:
        why = ("it is open in another program (close it there first)" if isinstance(exc, PermissionError)
               else str(exc))
        QtWidgets.QMessageBox.warning(panel, "XLSX", f"The workbook could not be saved: {why}.")

    panel.win.tasks.run(f"Saving {what} as a workbook…", lambda _p, _c: write(path), done, failed)


def exists(path) -> bool:
    return bool(path) and os.path.exists(str(path))
