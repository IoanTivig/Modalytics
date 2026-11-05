"""
Arranging imported images (File › Import › Images…, scripts' imports): asked when two or more come in.

    ┌ Arrange 12 images ─────────────────────────────────────────────────────────────┐
    │ (•) Grid  ( ) By name  ( ) Custom  ( ) Stacked     ┌ sketch to scale ─────────┐ │
    │ Columns [Auto]  Rows [Auto]  Gap [10 %]            │ ▭ ▭ ▭ ▭                  │ │
    │ Order [By name]  Fill [Row by row]                 │ ▭ ▭ ▭ ▭                  │ │
    │ Place them  (•) next to the images in the project  │ ▭ ▭ ▭ ▭                  │ │
    │             ( ) at the centre of the view          └──────────────────────────┘ │
    │                                                                 [OK] [Cancel]  │
    └─────────────────────────────────────────────────────────────────────────────────┘

The arrangement itself is ``core.arrange``; the choices are kept for the next
import (``app_settings`` "arrange").
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import app_settings
from modalytics.core import arrange as ar
from modalytics.ui.settings.widgets import muted

MODE_NAMES = {"grid": "Grid", "names": "By name", "custom": "Custom", "stack": "Stacked"}


class ArrangePreview(QtWidgets.QWidget):
    """A sketch of the arrangement, to scale: one box per image with its name."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes, self.names, self.spots = [], [], []
        self.setMinimumSize(380, 300)

    def show_layout(self, sizes, names, spots) -> None:
        self.sizes, self.names, self.spots = sizes, names, spots
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QtGui.QPalette.ColorRole.Base))
        if not self.spots:
            return
        w, h = ar.bounds(self.sizes, self.spots)
        pad = 12
        k = min((self.width() - 2 * pad) / max(w, 1e-9), (self.height() - 2 * pad) / max(h, 1e-9))
        accent = QtGui.QColor("#4fc3a1")
        font = p.font()
        font.setPointSizeF(7.5)
        p.setFont(font)
        for (x, y), (iw, ih), name in zip(self.spots, self.sizes, self.names):
            r = QtCore.QRectF(pad + x * k, pad + y * k, max(iw * k, 2), max(ih * k, 2))
            p.setPen(QtGui.QPen(accent, 1))
            p.setBrush(QtGui.QColor(79, 195, 161, 40))
            p.drawRect(r)
            if r.width() > 30 and r.height() > 12:
                p.setPen(pal.color(QtGui.QPalette.ColorRole.Text))
                p.drawText(r.adjusted(3, 2, -3, -2), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                           p.fontMetrics().elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle, int(r.width() - 6)))


class ArrangeDialog(QtWidgets.QDialog):
    """How to arrange the images being imported (see the module docstring).

    Args:
        names: The images' names.
        sizes: Their (width, height) in µm.
        has_others: Whether the project already holds images (placing next to them is then offered).
    """

    def __init__(self, names: list[str], sizes: list[tuple[float, float]], has_others: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Arrange {len(names)} images — Modalytics")
        self.names, self.sizes = list(names), list(sizes)
        saved = dict(app_settings.get("arrange"))
        self._filling = True
        outer = QtWidgets.QHBoxLayout(self)
        col = QtWidgets.QVBoxLayout()
        outer.addLayout(col, 1)
        # 1. The way
        row = QtWidgets.QHBoxLayout()
        self.mode_group = QtWidgets.QButtonGroup(self)
        for i, (mode, label) in enumerate(MODE_NAMES.items()):
            b = QtWidgets.QRadioButton(label)
            b.setProperty("mode", mode)
            self.mode_group.addButton(b, i)
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)
        self.pages = QtWidgets.QStackedWidget()
        col.addWidget(self.pages)
        # Grid
        grid = QtWidgets.QWidget()
        g = QtWidgets.QFormLayout(grid)
        self.columns = self._count(saved.get("columns", 0), "How many images side by side (Auto: from the rows, or "
                                                            "a square-ish grid)")
        self.rows = self._count(saved.get("rows", 0), "How many rows (Auto: from the columns)")
        self.gap = QtWidgets.QDoubleSpinBox()
        self.gap.setRange(0, 1000)
        self.gap.setSuffix(" %")
        self.gap.setValue(float(saved.get("gap_pct", 10.0)))
        self.gap.setToolTip("Space between images: % of the largest image's width (between columns) and height "
                            "(between rows); 100 % = one image")
        self.order = QtWidgets.QComboBox()
        self.order.addItem("By name (2 before 10)", "name")
        self.order.addItem("As selected", "given")
        self.order.setCurrentIndex(max(0, self.order.findData(saved.get("order", "name"))))
        self.fill = QtWidgets.QComboBox()
        self.fill.addItem("Row by row", "rows")
        self.fill.addItem("Column by column", "columns")
        self.fill.setCurrentIndex(max(0, self.fill.findData(saved.get("fill", "rows"))))
        for label, w in (("Columns", self.columns), ("Rows", self.rows), ("Gap", self.gap), ("Order", self.order),
                         ("Fill", self.fill)):
            g.addRow(label, w)
        self.pages.addWidget(grid)
        # By name
        names_page = QtWidgets.QWidget()
        n = QtWidgets.QFormLayout(names_page)
        fields = ar.name_fields(self.names[0]) if self.names else []
        n.addRow(muted(f"A name is split at - _ . and spaces: {self.names[0]!r} → "
                       + " · ".join(f"{i + 1}: {f}" for i, f in enumerate(fields)) if self.names else ""))
        self.row_field, self.column_field = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        longest = max((len(ar.name_fields(nm)) for nm in self.names), default=0)
        for i in range(max(longest, 1)):
            example = fields[i] if i < len(fields) else "—"
            for combo in (self.row_field, self.column_field):
                combo.addItem(f"part {i + 1} (e.g. {example})", i)
        default_row, default_col = (1, 2) if longest >= 3 else (0, max(0, longest - 1))
        self.row_field.setCurrentIndex(max(0, self.row_field.findData(saved.get("row_field", default_row))))
        self.column_field.setCurrentIndex(max(0, self.column_field.findData(saved.get("column_field", default_col))))
        self.gap_names = QtWidgets.QDoubleSpinBox()
        self.gap_names.setRange(0, 1000)
        self.gap_names.setSuffix(" %")
        self.gap_names.setValue(float(saved.get("gap_pct", 10.0)))
        n.addRow("Row from", self.row_field)
        n.addRow("Column from", self.column_field)
        n.addRow("Gap", self.gap_names)
        n.addRow(muted("E.g. sample01-1-2 (DHM phase map extractor): row = area 1, column = image 2. Names without "
                       "those parts go in a last row."))
        self.pages.addWidget(names_page)
        # Custom
        custom = QtWidgets.QWidget()
        c = QtWidgets.QVBoxLayout(custom)
        c.addWidget(muted("Each image's row, its place in the row and the gap after it (% of the largest width). "
                          "Rows are a largest height plus the row gap apart."))
        self.table = QtWidgets.QTableWidget(len(self.names), 4)
        self.table.setHorizontalHeaderLabels(["Image", "Row", "Place", "Gap after"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        start = ar.cells(self.names, ar.Layout("grid", columns=int(saved.get("columns", 0) or 0)))
        for i, name in enumerate(self.names):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            for k, (lo, hi, val, suffix) in enumerate(((1, 999, start[i][0] + 1, ""), (1, 999, start[i][1] + 1, ""),
                                                       (0, 1000, self.gap.value(), " %")), start=1):
                spin = QtWidgets.QDoubleSpinBox() if k == 3 else QtWidgets.QSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(val)
                spin.setSuffix(suffix)
                spin.valueChanged.connect(self._update)
                self.table.setCellWidget(i, k, spin)
        c.addWidget(self.table)
        row_gap = QtWidgets.QHBoxLayout()
        row_gap.addWidget(QtWidgets.QLabel("Gap between rows"))
        self.gap_rows = QtWidgets.QDoubleSpinBox()
        self.gap_rows.setRange(0, 1000)
        self.gap_rows.setSuffix(" %")
        self.gap_rows.setValue(self.gap.value())
        row_gap.addWidget(self.gap_rows)
        row_gap.addStretch(1)
        c.addLayout(row_gap)
        self.pages.addWidget(custom)
        # Stacked
        stack = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(stack)
        s.addWidget(muted("All at the same place, one over the other — e.g. phase maps each to be registered onto "
                          "a slide (Arrange › Register). Layers sets which one is on top."))
        s.addStretch(1)
        self.pages.addWidget(stack)
        # 2. Where
        where = QtWidgets.QGroupBox("Place them")
        w = QtWidgets.QVBoxLayout(where)
        self.next_to = QtWidgets.QRadioButton("next to the images already in the project (to their right)")
        self.at_view = QtWidgets.QRadioButton("at the centre of the view")
        self.next_to.setEnabled(has_others)
        (self.next_to if has_others and saved.get("where", "next") == "next" else self.at_view).setChecked(True)
        w.addWidget(self.next_to)
        w.addWidget(self.at_view)
        col.addWidget(where)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)
        # 3. The sketch
        self.preview = ArrangePreview()
        outer.addWidget(self.preview, 1)
        mode = saved.get("mode", "grid")
        self.mode_group.button(list(MODE_NAMES).index(mode) if mode in MODE_NAMES else 0).setChecked(True)
        self.mode_group.idToggled.connect(lambda _i, on: on and self._update())
        for wdg in (self.columns, self.rows, self.gap, self.gap_names, self.gap_rows):
            wdg.valueChanged.connect(self._update)
        for wdg in (self.order, self.fill, self.row_field, self.column_field):
            wdg.currentIndexChanged.connect(self._update)
        self._filling = False
        self._update()
        self.resize(900, 520)

    @staticmethod
    def _count(value: int, tip: str) -> QtWidgets.QSpinBox:
        s = QtWidgets.QSpinBox()
        s.setRange(0, 999)
        s.setSpecialValueText("Auto")
        s.setValue(int(value))
        s.setToolTip(tip)
        return s

    def mode(self) -> str:
        return self.mode_group.checkedButton().property("mode")

    def layout_chosen(self) -> ar.Layout:
        """The arrangement as set in the dialog."""
        mode = self.mode()
        if mode == "custom":
            custom = {i: (self.table.cellWidget(i, 1).value() - 1, self.table.cellWidget(i, 2).value(),
                          self.table.cellWidget(i, 3).value()) for i in range(self.table.rowCount())}
            return ar.Layout("custom", gap_pct=self.gap_rows.value(), custom=custom)
        if mode == "names":
            return ar.Layout("names", gap_pct=self.gap_names.value(), row_field=self.row_field.currentData(),
                             column_field=self.column_field.currentData())
        return ar.Layout(mode, columns=self.columns.value(), rows=self.rows.value(), gap_pct=self.gap.value(),
                         order=self.order.currentData(), fill=self.fill.currentData())

    def placement(self) -> str:
        """"next" (to the right of the project's images) or "view" (centred in the view)."""
        return "next" if self.next_to.isChecked() else "view"

    def _update(self, *_args) -> None:
        if self._filling:
            return
        self.pages.setCurrentIndex(list(MODE_NAMES).index(self.mode()))
        spots = ar.positions(self.sizes, self.names, self.layout_chosen())
        self.preview.show_layout(self.sizes, self.names, spots)

    def _accept(self) -> None:
        lay = self.layout_chosen()
        app_settings.put("arrange", {"mode": lay.mode, "columns": self.columns.value(), "rows": self.rows.value(),
                                     "gap_pct": self.gap.value() if lay.mode == "grid" else lay.gap_pct,
                                     "order": self.order.currentData(), "fill": self.fill.currentData(),
                                     "row_field": self.row_field.currentData(),
                                     "column_field": self.column_field.currentData(),
                                     # the choice of where, only when there was one (an empty project has nothing to
                                     # place next to: that does not change the preference)
                                     "where": self.placement() if self.next_to.isEnabled()
                                     else dict(app_settings.get("arrange")).get("where", "next")})
        self.accept()
