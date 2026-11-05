"""
The Results panel: the latest measurement of the project, in three tables.

    ┌ Measured 7 objects on 1 image with Colonic tissue · 20 Oct 14:32 · 1.9 s   [Measure ▾] [Export CSV…] ┐
    │ ⚠ Out of date: 1 mask(s) were edited… — Measure again                                                │
    │ Image [All ▾]  Class [All ▾]  [filter…]                                                              │
    │ [Objects (7)] [Classes (6)] [Images (1)]                                                             │
    │ Image   Class            #  Area [µm²]  Circularity  …                                               │
    │ S2635…  Gland › Normal   1  4083        0.21                                                         │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────┘

* **Objects**: one row per object (each separate outline of a class), the
  profile's per-object parameters; a click selects it on the canvas.
* **Classes**: one row per class on each image (each group as a whole, then
  each subgroup), the per-class parameters (counts, densities, summaries).
* **Images**: one row per image, the whole-image parameters.

Columns are sortable; a header's tooltip is the parameter's description, formula
and references; an empty cell's tooltip says why it has no value. When masks,
images or the profile change after measuring, a banner says the values are out
of date (``Results.outdated``).
"""

from __future__ import annotations

import datetime as _dt
import math

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.parameters.results import FIXED, SCOPE_ORDER, Results, fixed_columns, fixed_values

TAB_TITLES = {"object": "Objects", "class": "Classes", "image": "Images"}
SORT = QtCore.Qt.ItemDataRole.UserRole + 1          # the numeric value (or text) a column sorts by
ROW = QtCore.Qt.ItemDataRole.UserRole + 2           # the Row behind a table row


def fmt(v) -> str:
    """A value as a table shows it: 5 significant digits, "—" when not computed."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e9:
        return f"{int(v)}"
    return f"{v:.5g}"


KIND_NAMES = {"ranges": "ranges", "score": "score", "logistic": "logistic regression", "forest": "random forest",
              "boosting": "gradient boosting"}


class ResultsModel(QtCore.QAbstractTableModel):
    """One kind of rows as a table: identifying columns, then one column per parameter."""

    def __init__(self, scope: str, results: Results | None = None, parent=None) -> None:
        super().__init__(parent)
        self.scope = scope
        self.rows, self.columns = [], []
        self.set_results(results)

    def set_results(self, results: Results | None) -> None:
        self.beginResetModel()
        self.rows = results.of_scope(self.scope) if results else []
        self.columns = results.columns.get(self.scope, []) if results else []
        self.fixed = fixed_columns(results, self.scope)
        self.predicted = len(self.fixed) > len(FIXED[self.scope])          # Predicted, Score, Matches annotation
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.fixed) + len(self.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, c = self.rows[index.row()], index.column()
        if role == ROW:
            return row
        if c < len(self.fixed):
            v = fixed_values(row, self.predicted)[c]
            if role in (QtCore.Qt.ItemDataRole.DisplayRole, SORT):
                return v
            if role == QtCore.Qt.ItemDataRole.ToolTipRole and self.fixed[c] in ("Predicted", "Score") and v != "":
                return f"{row.prediction.get('grade')}: {row.prediction.get('reason', '')}"
            if role == QtCore.Qt.ItemDataRole.ForegroundRole and self.fixed[c] == "Matches annotation" and v == "no":
                return QtGui.QColor("#eb6f6f")
            return None
        col = self.columns[c - len(self.fixed)]
        v = row.values.get(col.uid, math.nan)
        missing = v is None or (isinstance(v, float) and math.isnan(v))
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return fmt(v)
        if role == SORT:
            return -math.inf if missing else float(v)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if missing:
                return f"{col.label}: no value — {row.problems.get(col.uid, 'not computed')}"
            return f"{col.title}: {v:.10g}"
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and missing:
            return QtWidgets.QApplication.palette().color(QtWidgets.QApplication.palette().ColorRole.PlaceholderText)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation != QtCore.Qt.Orientation.Horizontal:
            return None
        if section < len(self.fixed):
            return self.fixed[section] if role == QtCore.Qt.ItemDataRole.DisplayRole else None
        col = self.columns[section - len(self.fixed)]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return col.title
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return col.help or col.label
        return None


class ResultsFilter(QtCore.QSortFilterProxyModel):
    """Sorting by value, and keeping only one image / class / matching text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT)
        self.image_id, self.group, self.text = None, None, ""

    def set_filter(self, image_id=None, group=None, text: str = "") -> None:
        self.image_id, self.group, self.text = image_id, group, text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        row = self.sourceModel().rows[source_row]
        if self.image_id and row.image_id != self.image_id:
            return False
        if self.group and row.scope != "image" and row.group != self.group:
            return False
        if self.text and self.text not in f"{row.image_name} {row.class_name}".lower():
            return False
        return True

    def lessThan(self, left, right) -> bool:
        a, b = left.data(SORT), right.data(SORT)
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return str(a).lower() < str(b).lower()


class ResultsPanel(QtWidgets.QWidget):
    """The latest measurement (see the module docstring).

    Signals:
        measure_requested(str): "project" or "image" (the selected image).
        export_requested(str): the scope of the table shown.
        object_activated(str, str, int): image id, mask id, object index — a row of Objects was picked.
        image_activated(str): image id — a row of Classes or Images was picked.
    """

    measure_requested = QtCore.Signal(str)
    export_requested = QtCore.Signal(str)
    object_activated = QtCore.Signal(str, str, int)
    image_activated = QtCore.Signal(str)

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(6, 4, 6, 4)
        # 1. What was measured, and the actions
        top = QtWidgets.QHBoxLayout()
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("Muted")
        top.addWidget(self.summary, 1)
        self.measure_btn = QtWidgets.QToolButton()
        self.measure_btn.setText("Measure")
        self.measure_btn.setToolTip("Measure every image with the profile's parameters")
        self.measure_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        menu = QtWidgets.QMenu(self.measure_btn)
        menu.addAction("Measure the whole project", lambda: self.measure_requested.emit("project"))
        menu.addAction("Measure the selected image", lambda: self.measure_requested.emit("image"))
        self.measure_btn.setMenu(menu)
        self.measure_btn.clicked.connect(lambda: self.measure_requested.emit("project"))
        top.addWidget(self.measure_btn)
        self.export_btn = QtWidgets.QPushButton("Export CSV…")
        self.export_btn.setToolTip("Save the table shown as a CSV file (opens in Excel, LibreOffice, R, Python…)")
        self.export_btn.clicked.connect(lambda: self.export_requested.emit(self.current_scope()))
        top.addWidget(self.export_btn)
        col.addLayout(top)
        # 2. Out of date?
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setObjectName("Warning")
        self.banner.setStyleSheet("QLabel#Warning { color: #e8b04b; }")
        self.banner.linkActivated.connect(lambda _l: self.measure_requested.emit("project"))
        self.banner.hide()
        col.addWidget(self.banner)
        # 3. Filters
        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(QtWidgets.QLabel("Image"))
        self.image_filter = QtWidgets.QComboBox()
        self.image_filter.currentIndexChanged.connect(self._apply_filter)
        filters.addWidget(self.image_filter)
        filters.addWidget(QtWidgets.QLabel("Class"))
        self.class_filter = QtWidgets.QComboBox()
        self.class_filter.currentIndexChanged.connect(self._apply_filter)
        filters.addWidget(self.class_filter)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter by image or class name…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        filters.addWidget(self.search, 1)
        col.addLayout(filters)
        # 4. The three tables
        self.tabs = QtWidgets.QTabWidget()
        self.models, self.proxies, self.views = {}, {}, {}
        for scope in SCOPE_ORDER:
            model = ResultsModel(scope, None, self)
            proxy = ResultsFilter(self)
            proxy.setSourceModel(model)
            view = QtWidgets.QTableView()
            view.setModel(proxy)
            view.setSortingEnabled(True)
            view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            view.setAlternatingRowColors(True)
            view.verticalHeader().setVisible(False)
            view.verticalHeader().setDefaultSectionSize(22)
            view.horizontalHeader().setDefaultSectionSize(96)
            view.horizontalHeader().setSectionsMovable(True)
            view.selectionModel().currentRowChanged.connect(lambda cur, _prev, s=scope: self._picked(s, cur))
            self.models[scope], self.proxies[scope], self.views[scope] = model, proxy, view
            self.tabs.addTab(view, TAB_TITLES[scope])
        self.empty = QtWidgets.QLabel("Not measured yet. <b>Measure</b> measures every image of the project with "
                                      "the parameters its profile chose: each object, each class on each image, "
                                      "and each whole image.")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.tabs)
        col.addWidget(self.stack, 1)
        # 5. Checks whether the results still match the project (shortly after each change)
        self._check = QtCore.QTimer(self, singleShot=True, interval=400, timeout=self.check_outdated)
        project.results_changed.connect(self.refresh)
        project.structure_changed.connect(self._check.start)
        project.node_changed.connect(lambda node, field: self._check.start()
                                     if field in ("geometry", "pixel_size") else None)
        self.refresh()

    # ------------------------------------------------------------ showing
    def refresh(self) -> None:
        """Show the project's results (or the empty message)."""
        results = self.project.results
        for scope in SCOPE_ORDER:
            self.models[scope].set_results(results)
            view = self.views[scope]
            header = view.horizontalHeader()
            for c in range(self.models[scope].columnCount()):     # wide enough for its title and values
                view.resizeColumnToContents(c)
                header.resizeSection(c, min(max(header.sectionSize(c), 70), 220))
            n = len(self.models[scope].rows)
            self.tabs.setTabText(SCOPE_ORDER.index(scope), f"{TAB_TITLES[scope]} ({n})")
        self.stack.setCurrentIndex(1 if results else 0)
        self.export_btn.setEnabled(results is not None)
        self._fill_filters(results)
        if results is None:
            self.summary.setText("No results yet")
        else:
            c = results.counts()
            when = _when(results.measured)
            self.summary.setText(f"Measured {c['object']} objects on {c['image']} image"
                                 f"{'s' if c['image'] != 1 else ''} with {results.profile_name or 'the profile'}"
                                 f" · {when} · {results.seconds:.1f} s"
                                 + (f" · includes {results.suggested} unreviewed suggestion(s)"
                                    if results.suggested else "")
                                 + "".join(f" · {g} classified ({KIND_NAMES.get(how.split(':')[0], how)})"
                                           for g, how in results.classified.items()))
            self.summary.setToolTip("\n".join(f"{g}: {how}" for g, how in results.classified.items()))
        self.check_outdated()

    def _fill_filters(self, results) -> None:
        for combo in (self.image_filter, self.class_filter):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
        if results is not None:
            seen = []
            for r in results.of_scope("image"):
                self.image_filter.addItem(r.image_name, r.image_id)
            for r in results.rows:
                if r.group and r.group not in seen:
                    seen.append(r.group)
            for g in seen:
                self.class_filter.addItem(g, g)
        for combo in (self.image_filter, self.class_filter):
            combo.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self, *_args) -> None:
        for proxy in self.proxies.values():
            proxy.set_filter(self.image_filter.currentData(), self.class_filter.currentData(), self.search.text())

    def check_outdated(self) -> list[str]:
        """Show the banner if the results no longer match the project; returns the reasons."""
        results = self.project.results
        reasons = results.outdated(self.project) if results is not None else []
        if reasons:
            self.banner.setText("⚠ Out of date — " + "; ".join(reasons) + ". The values are those of the earlier "
                                "state. <a href='measure'>Measure again</a>")
        self.banner.setVisible(bool(reasons))
        return reasons

    def current_scope(self) -> str:
        return SCOPE_ORDER[self.tabs.currentIndex()]

    def show_scope(self, scope: str) -> None:
        self.tabs.setCurrentIndex(SCOPE_ORDER.index(scope))

    # ------------------------------------------------------------ picking
    def _picked(self, scope: str, index) -> None:
        if not index.isValid():
            return
        row = self.proxies[scope].data(index, ROW)
        if row is None:
            return
        if scope == "object":
            self.object_activated.emit(row.image_id, row.mask_id, row.index)
        else:
            self.image_activated.emit(row.image_id)


def _when(iso: str) -> str:
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%d %b %H:%M")
    except ValueError:
        return iso
