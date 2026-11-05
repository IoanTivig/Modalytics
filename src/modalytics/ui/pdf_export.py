"""
An HTML report printed to PDF inside Modalytics, with Qt WebEngine (Chromium): the same page a browser shows,
laid out by the report's own print rules (``@page`` A4, ``@media print``: no page break inside a table, a figure
or a plot). Nothing is installed or opened for it.

The page is loaded from the HTML file (a report with its pictures embedded can be far larger than Qt's limit for
HTML given as text), printed to PDF on A4 with the report's margins, then dropped. It runs on the UI thread,
asynchronously: ``print_pdf`` returns at once and calls back when the file is written.
"""

from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui

TIMEOUT_MS = 180_000        # a report that is still not printed after 3 minutes: given up
MARGIN_MM = 14.0            # as the report's ``@page`` rule
# Landscape: a report's own ``@page`` rule wins over the PDF's page setting, so it is overruled in the page itself
# (reports written before their print rules were landscape print right too); wide tables wrap and shrink on paper
LANDSCAPE_CSS = ("@page { size: A4 landscape; margin: 12mm; } @media print { table { font-size: 9.5px; } "
                 "th, td { white-space: normal !important; padding: 3px 5px !important; } "
                 ".scroll { overflow: visible !important; max-height: none !important; border: none !important; } "
                 ".scroll table { width: 100%; } td .muted, th .muted { font-size: 8.5px !important; } }")


class _Job(QtCore.QObject):
    """One HTML file printed to one PDF (kept alive until it has called back)."""

    def __init__(self, html_path: str, pdf_path: str, done: Callable[[str], None],
                 failed: Callable[[str], None], landscape: bool = False, parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWebEngineCore import QWebEnginePage
        self.pdf_path, self._done, self._failed, self._over = pdf_path, done, failed, False
        self.landscape = landscape
        self.page = QWebEnginePage(self)
        self.page.loadFinished.connect(self._loaded)
        self.page.pdfPrintingFinished.connect(self._printed)
        self.timer = QtCore.QTimer(self, singleShot=True, interval=TIMEOUT_MS)
        self.timer.timeout.connect(lambda: self._end(False, "it took too long"))
        self.timer.start()
        self.page.load(QtCore.QUrl.fromLocalFile(html_path))

    def _loaded(self, ok: bool) -> None:
        if not ok:
            self._end(False, "the report page could not be loaded")
            return
        if self.landscape:                                  # the page's own @page rule overruled first
            import json
            script = (f"var s = document.createElement('style'); s.textContent = {json.dumps(LANDSCAPE_CSS)}; "
                      f"document.head.appendChild(s); true;")
            self.page.runJavaScript(script, 0, lambda _r: self._print())
        else:
            self._print()

    def _print(self) -> None:
        layout = QtGui.QPageLayout(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4),
                                   QtGui.QPageLayout.Orientation.Landscape if self.landscape
                                   else QtGui.QPageLayout.Orientation.Portrait,
                                   QtCore.QMarginsF(MARGIN_MM, MARGIN_MM, MARGIN_MM, MARGIN_MM),
                                   QtGui.QPageLayout.Unit.Millimeter)
        self.page.printToPdf(self.pdf_path, layout)

    def _printed(self, path: str, ok: bool) -> None:
        self._end(ok, "" if ok else "printing to PDF failed (is the file open in another program?)")

    def _end(self, ok: bool, why: str) -> None:
        if self._over:
            return
        self._over = True
        self.timer.stop()
        try:
            (self._done(self.pdf_path) if ok else self._failed(why))
        finally:
            self.page.deleteLater()
            self.deleteLater()
            _JOBS.discard(self)


_JOBS: set = set()


def print_pdf(html_path: str, pdf_path: str, done: Callable[[str], None], failed: Callable[[str], None],
              landscape: bool = False) -> None:
    """Print an HTML file to a PDF (A4, the report's print rules), calling ``done(pdf_path)`` or ``failed(why)``
    on the UI thread when it is over.

    Args:
        landscape: A4 across (reports with wide tables: the training reports).

    Raises:
        ImportError: If Qt WebEngine is not available (a PySide6 without it).
    """
    _JOBS.add(_Job(html_path, pdf_path, done, failed, landscape))


def report_pdf(parent, html_path, landscape: bool = True, status=None) -> None:
    """A training report as PDF: printed beside its HTML (same name, ``.pdf``) and opened; a warning if it cannot
    be made. ``status``: a status bar for the messages."""
    import os

    from PySide6 import QtWidgets
    html_path = str(html_path)
    if not html_path or not os.path.exists(html_path):
        QtWidgets.QMessageBox.information(parent, "PDF", "Write the report first.")
        return
    pdf_path = os.path.splitext(html_path)[0] + ".pdf"

    def done(path: str) -> None:
        if status is not None:
            status.showMessage(f"✅ PDF saved: {path}", 10000)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def failed(why: str) -> None:
        QtWidgets.QMessageBox.warning(parent, "PDF", f"The PDF could not be made: {why}.")

    if status is not None:
        status.showMessage("Printing the report to PDF…")
    try:
        print_pdf(html_path, pdf_path, done, failed, landscape)
    except ImportError as exc:
        failed(f"Qt WebEngine is not available ({exc}); open the report and print it from the browser")
