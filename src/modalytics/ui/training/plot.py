"""
A small distribution plot: one parameter's values per grade, overlaid, with a cut-off.

    ┌ Irregularity (BAM)                                  ■ Low grade (n 50)  ■ High grade (n 30) ┐
    │   ▁▃▇█▅▂▁                                                                                    │
    │          ▁▂▃▅▆▅▃▂▁   ┊ cut-off 0.031                                                         │
    │ 0.0002            0.02             0.04                                                       │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘

Drawn with Qt itself (no plotting library): each grade's histogram as a filled
step outline in its class colour, as a share of that grade's objects (so grades
of different sizes compare), over the 1st–99th percentile of all values.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class DistributionPlot(QtWidgets.QWidget):
    """Overlaid per-grade histograms of one parameter, with an optional cut-off line."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.series: list[tuple[str, np.ndarray, QtGui.QColor]] = []
        self.title, self.cutoff, self.note = "", None, ""

    def show_values(self, title: str, series, cutoff: float | None = None, note: str = "") -> None:
        """Show these values.

        Args:
            title: The parameter's name and unit.
            series: [(grade name, values, QColor)].
            cutoff: A value to mark with a dashed line (None: none).
            note: A line under the title (e.g. "AUC 0.93 · cut-off ≥ 0.031").
        """
        self.title, self.cutoff, self.note = title, cutoff, note
        self.series = [(n, np.asarray(v, float)[np.isfinite(np.asarray(v, float))], c) for n, v, c in series]
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.color(QtGui.QPalette.ColorRole.Text)
        muted = QtGui.QColor(text)
        muted.setAlpha(140)
        r = self.rect().adjusted(12, 8, -12, -8)
        p.setPen(text)
        f = p.font()
        p.drawText(r.adjusted(0, 0, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                   self.title)
        p.setPen(muted)
        p.drawText(r.adjusted(0, 18, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                   self.note)
        values = [v for _n, v, _c in self.series if v.size]
        if not values:
            p.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, "No values to show")
            return
        allv = np.concatenate(values)
        lo, hi = np.percentile(allv, [1, 99])
        if self.cutoff is not None and np.isfinite(self.cutoff):
            lo, hi = min(lo, self.cutoff), max(hi, self.cutoff)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 31)
        plot = QtCore.QRectF(r.left() + 4, r.top() + 44, r.width() - 8, r.height() - 44 - 22)
        # 1. Histograms as shares of each grade
        hists = [(n, np.histogram(np.clip(v, lo, hi), bins)[0] / max(v.size, 1), c) for n, v, c in self.series]
        top = max((h.max() for _n, h, _c in hists if h.size), default=1.0) or 1.0

        def x_of(v):
            return plot.left() + (v - lo) / (hi - lo) * plot.width()

        for name, h, colour in hists:
            path = QtGui.QPainterPath(QtCore.QPointF(plot.left(), plot.bottom()))
            for i, share in enumerate(h):
                y = plot.bottom() - share / top * plot.height()
                path.lineTo(x_of(bins[i]), y)
                path.lineTo(x_of(bins[i + 1]), y)
            path.lineTo(plot.right(), plot.bottom())
            fill = QtGui.QColor(colour)
            fill.setAlpha(70)
            p.fillPath(path, fill)
            p.setPen(QtGui.QPen(colour, 1.6))
            p.drawPath(path)
        # 2. Axis and cut-off
        p.setPen(muted)
        p.drawLine(QtCore.QPointF(plot.left(), plot.bottom()), QtCore.QPointF(plot.right(), plot.bottom()))
        for v, align in ((lo, QtCore.Qt.AlignmentFlag.AlignLeft), ((lo + hi) / 2, QtCore.Qt.AlignmentFlag.AlignHCenter),
                         (hi, QtCore.Qt.AlignmentFlag.AlignRight)):
            x = x_of(v)
            box = QtCore.QRectF(x - 60, plot.bottom() + 3, 120, 16)
            if align == QtCore.Qt.AlignmentFlag.AlignLeft:
                box.moveLeft(x)
            elif align == QtCore.Qt.AlignmentFlag.AlignRight:
                box.moveRight(x)
            p.drawText(box, align | QtCore.Qt.AlignmentFlag.AlignTop, f"{v:.4g}")
        if self.cutoff is not None and np.isfinite(self.cutoff):
            x = x_of(self.cutoff)
            pen = QtGui.QPen(text, 1.3, QtCore.Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
            p.drawText(QtCore.QRectF(x + 4, plot.top(), 160, 16), QtCore.Qt.AlignmentFlag.AlignLeft,
                       f"cut-off {self.cutoff:.4g}")
        # 3. Legend
        f.setPointSizeF(f.pointSizeF() * 0.95)
        p.setFont(f)
        x = r.right()
        for name, v, colour in reversed(self.series):
            label = f"{name} (n {v.size})"
            w = p.fontMetrics().horizontalAdvance(label) + 22
            x -= w
            p.fillRect(QtCore.QRectF(x, r.top() + 4, 10, 10), colour)
            p.setPen(text)
            p.drawText(QtCore.QRectF(x + 14, r.top(), w, 18), QtCore.Qt.AlignmentFlag.AlignLeft, label)
        p.end()
