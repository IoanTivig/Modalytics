"""
The Training section: the data, classifiers from parameters, and segmentation models.

    ┌ Training sets ──┐ ┌ [Data] [Classifiers] [Segmentation models] ─────────────────────────────────────────┐
    │ Colon 2026      │ │ Name [Colon 2026]  Profile [Colonic tissue ▾]  Objects [Gland ▾]                     │
    │ Pilot           │ │ Grades: Low grade dysplasia < High grade dysplasia (also read: Normal → Low…)        │
    │                 │ │ Project  Role        Images  µm/px  Outlined  Fully  Objects  Low  High  Ungraded …  │
    │ [New…] [Delete] │ │ case_01  Training ▾  16      0.314  4         0      16       12   4               ✔  │
    │ [Folder]        │ │ [Add projects…] [Remove]      Evaluation ◉ marked projects ○ 20 %                    │
    └─────────────────┘ └─────────────────────────────────────────────────────────────────────────────────────┘

* **Data** (shared by classifiers and segmentation models): the profile, the class
  group the set is about (classifiers grade its subgroups; segmentation models
  outline it), the projects with their role (Training / Evaluation / Test, the test
  projects in named groups — e.g. one per microscope setup or lab — each a final check of
  its own), and how the evaluation part is chosen: the projects marked Evaluation, or an
  automatic share of whole projects. The table says what each project holds, read from its
  masks in the background (nothing measured; ``training.dataset.project_overview``):
  images per type, their pixel sizes (⚠ one far from the others'), images with
  outlines, images whose annotation is marked complete, objects, per grade, ungraded, unreviewed
  suggestions.
* **Parameters** (``ui.training.parameters``): **Measure** gathers every graded
  object with the profile's parameters (in the background; cached per project);
  **Analyse** finds which parameters separate the grades (intervals over projects,
  corrected for the many parameters) and which repeat each other, proposes a
  selection and makes the ticked ones the profile's key parameters, with a report.
* **Classifiers**, from the measured objects:
* **Explore** (the range finder): parameters ranked by how well they separate the
  grades (AUC), with the best cut-off and its sensitivity and specificity, on the
  training projects only (never the evaluation ones; test projects are never measured), with each one's verdict
  and redundancy group from Parameters; a click shows its distribution per grade.
* **Segmentation models › Train** (``ui.training.train_seg``, full edition): new or
  fine-tuned models trained, judged and compared, with a report. **Evaluate**
  (``ui.training.segmentation``): existing models — kept, released or another
  person's file — judged and compared the same way without training, with a report
  and the objects to look at.
* **Build** (``ui.training.build``): Ranges and a points Score from the parameters,
  compared on the evaluation projects, and saved to the profile (the trained
  classifier needs scikit-learn).

Training sets are saved as you change them (``training.dataset``).
"""

from __future__ import annotations

import math
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.profiles.library import ProfileLibrary
from modalytics.training import dataset as ds
from modalytics.training import ranges
from modalytics.training import relevance as rv
from modalytics.ui.settings.widgets import clear_tree, list_tree, muted, open_in_system
from modalytics.ui.training.build import BuildPanel
from modalytics.ui.training.parameters import ParametersPanel
from modalytics.ui.training.segmentation import EvaluatePanel
from modalytics.ui.training.train_seg import TrainPanel
from modalytics.ui.training.plot import DistributionPlot
from modalytics.ui.training.sorting import SortItem, sortable

ROLE_NAMES = {"training": "Training", "evaluation": "Evaluation", "test": "Test"}
NEW_GROUP = "new"                      # the role box's "New test group…"


def role_text(entry) -> str:
    """A project's role as shown: "Training", "Evaluation", "Test" or "Test: <group>"."""
    if entry.role == "test":
        return f"Test: {entry.group}" if entry.group else ROLE_NAMES["test"]
    return ROLE_NAMES.get(entry.role, "Training")


def role_key(entry) -> str:
    """A project's role as the role box's data: "training", "evaluation" or "test:<group>"."""
    return f"test:{entry.group}" if entry.role == "test" else entry.role


def _measured_by(tset) -> str:
    """What a training set's measurements depend on (profile, group, projects and roles, partial objects): fresh
    measurements are kept unless one of these changed meanwhile — not for a classifier built in the meantime."""
    import json
    return json.dumps([tset.profile.to_dict(), tset.group, [(p.path, p.role) for p in tset.projects],
                       tset.include_partial], sort_keys=True, default=str)


class TrainingWindow(QtWidgets.QMainWindow):
    """The Training section (see the module docstring)."""

    overview_ready = QtCore.Signal(object)               # (request, {path: ProjectOverview}) from the reading thread

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Modalytics — Training")
        self.resize(1400, 880)
        self.profiles = ProfileLibrary()
        self.tset: ds.TrainingSet | None = None
        self.path: str | None = None
        self.data: ds.Dataset | None = None
        self._filling = False
        from modalytics.ui.tasks import TaskRunner
        self.tasks = TaskRunner(self.statusBar())

        central = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(central)
        # 1. Training sets
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("<b>Training sets</b>"))
        self.sets = QtWidgets.QListWidget()
        self.sets.currentItemChanged.connect(self._open_selected)
        left.addWidget(self.sets, 1)
        buttons = QtWidgets.QGridLayout()
        for i, (text, slot) in enumerate((("New…", self._new), ("Duplicate", self._duplicate),
                                          ("Delete", self._delete),
                                          ("Folder", lambda: open_in_system(ds.library_folder())))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b, i // 2, i % 2)
        left.addLayout(buttons)
        side = QtWidgets.QWidget()
        side.setLayout(left)
        side.setFixedWidth(240)
        row.addWidget(side)
        # 2. The data, then what is trained from it
        self.overviews: dict = {}                          # {project path: ProjectOverview}
        self._overview_key = None
        self.overview_ready.connect(self._overview_found)
        self.kinds = QtWidgets.QTabWidget()
        self.data_page = self._build_data()
        self.kinds.addTab(self.data_page, "Data")
        self.measure_note = muted("")
        self.measure_note.setWordWrap(True)
        self.measure_btn = QtWidgets.QPushButton("Measure")
        self.measure_btn.setToolTip("Measure every graded object of the projects with the profile's parameters (in the "
                                    "background; kept for next time)")
        self.measure_btn.clicked.connect(lambda: self.gather())
        self.parameters = ParametersPanel(self)           # holds Measure: Parameters and Classifiers read its objects
        self.kinds.addTab(self.parameters, "Parameters")
        self.classifiers_page = QtWidgets.QWidget()
        cc = QtWidgets.QVBoxLayout(self.classifiers_page)
        self.classifiers = QtWidgets.QTabWidget()
        self.classifiers.addTab(self._build_explore(), "Explore")
        self.build = BuildPanel(self)
        if parent is not None and hasattr(parent, "show_graded_object"):
            self.build.show_object.connect(parent.show_graded_object)
        self.classifiers.addTab(self.build, "Build")
        cc.addWidget(self.classifiers, 1)
        self.kinds.addTab(self.classifiers_page, "Classifiers")
        self.segmentation = QtWidgets.QTabWidget()
        self.evaluate = EvaluatePanel(self)
        if parent is not None and hasattr(parent, "show_graded_object"):
            self.evaluate.show_object.connect(parent.show_graded_object)
        self.segmentation.addTab(self.evaluate, "Evaluate")
        self.train_seg = TrainPanel(self, self.evaluate)
        self.segmentation.addTab(self.train_seg, "Train")
        self.kinds.addTab(self.segmentation, "Segmentation models")
        row.addWidget(self.kinds, 1)
        self.setCentralWidget(central)
        help_btn = QtWidgets.QToolButton()
        help_btn.setText("Help")
        help_btn.setToolTip("The help on the tab shown (F1)")
        help_btn.clicked.connect(self.show_help)
        self.kinds.setCornerWidget(help_btn)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F1), self, activated=self.show_help)
        self.empty = muted("Make a training set with New…: a profile, the projects whose objects were graded, "
                           "and which of them are kept for evaluation.")
        self.reload()

    # ------------------------------------------------------------ building the tabs
    def _placeholder(self, text: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        label = muted(text)
        label.setWordWrap(True)
        col.addWidget(label)
        col.addStretch(1)
        return w

    def _build_data(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit()
        self.name.editingFinished.connect(self._renamed)
        self.profile_box = QtWidgets.QComboBox()
        self.profile_box.currentIndexChanged.connect(self._profile_changed)
        self.group_box = QtWidgets.QComboBox()
        self.group_box.setToolTip("The class group the set is about (e.g. Gland): classifiers grade its subgroups, "
                                  "segmentation models outline it (all its subgroups)")
        self.group_box.currentIndexChanged.connect(self._group_changed)
        self.grades = muted("")
        self.grades.setWordWrap(True)
        form.addRow("Name", self.name)
        form.addRow("Profile", self.profile_box)
        form.addRow("Objects", self.group_box)
        form.addRow("Grades", self.grades)
        col.addLayout(form)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._data_sort: tuple[str, QtCore.Qt.SortOrder] | None = None   # (column title, order); None: the set's
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_data)
        col.addWidget(self.table, 1)
        actions = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Add projects…", self._add_projects,
                                 "Projects whose objects were outlined or graded (.modalytics, or HoloPath .hpro)"),
                                ("Remove", self._remove_project, "Remove the selected project from the set"),
                                ("Reload", self.reload_data, "Read the projects again (e.g. after they were changed "
                                 "and saved in the main window) and measure what changed")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            actions.addWidget(b)
        actions.addStretch(1)
        col.addLayout(actions)
        ev = QtWidgets.QGroupBox("Evaluation (never used for training)")
        evl = QtWidgets.QHBoxLayout(ev)
        self.mode_marked = QtWidgets.QRadioButton("The projects marked Evaluation")
        self.mode_share = QtWidgets.QRadioButton("An automatic share of the projects:")
        self.share = QtWidgets.QSpinBox()
        self.share.setRange(5, 90)
        self.share.setSuffix(" %")
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(1, 9999)
        self.seed.setPrefix("draw ")
        self.seed.setToolTip("Another number draws other projects (the same number always the same ones)")
        for wdg in (self.mode_marked, self.mode_share, self.share, self.seed):
            evl.addWidget(wdg)
        evl.addStretch(1)
        self.mode_marked.toggled.connect(self._evaluation_changed)
        self.share.valueChanged.connect(self._evaluation_changed)
        self.seed.valueChanged.connect(self._evaluation_changed)
        col.addWidget(ev)
        self.include_partial = QtWidgets.QCheckBox("Include partial objects where the parameter allows it")
        self.include_partial.setToolTip(
            "Objects whose shape is incomplete — cut by the image edge, or marked partial — are measured for the "
            "parameters that do not need the whole object (pixel statistics, texture: Settings › Parameters, "
            "\"Needs the whole object\"); their values for the others stay empty.\nParameters and Explore then "
            "count them where they have a value; a classifier uses them only when it reads none of the "
            "whole-object parameters.\nOff: they are left out altogether.")
        self.include_partial.toggled.connect(self._partial_changed)
        col.addWidget(self.include_partial)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        col.addWidget(self.summary)
        return w

    def _build_explore(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        self.explore_note = muted("")
        self.explore_note.setWordWrap(True)
        col.addWidget(self.explore_note)
        top = QtWidgets.QHBoxLayout()
        self.pair_box = QtWidgets.QComboBox()
        self.pair_box.setToolTip("Grades: which two neighbouring grades are compared. Categories: any two classes, "
                                 "or one against all the others")
        self.pair_box.currentIndexChanged.connect(self._fill_explore)
        top.addWidget(QtWidgets.QLabel("Grades compared"))
        top.addWidget(self.pair_box)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter parameters…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill_explore)
        top.addWidget(self.search, 1)
        col.addLayout(top)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.ranked = sortable(list_tree(["Parameter", "AUC", "Cut-off", "Sensitivity", "Specificity", "Objects",
                                          "Verdict", "Group"]))
        self.ranked.setColumnWidth(0, 330)
        self.ranked.setColumnWidth(1, 70)
        self.ranked.setColumnWidth(2, 330)
        self.ranked.currentItemChanged.connect(self._show_parameter)
        split.addWidget(self.ranked)
        self.plot = DistributionPlot()
        split.addWidget(self.plot)
        split.setSizes([420, 280])
        col.addWidget(split, 1)
        return w

    # ------------------------------------------------------------ training sets
    def reload(self, select: str | None = None) -> None:
        """List the training sets in your library (keeps or makes the selection)."""
        select = select or self.path
        self.sets.blockSignals(True)
        self.sets.clear()
        folder = ds.library_folder()
        chosen = None
        for p in sorted(folder.glob(f"*{ds.SUFFIX}")) if folder.exists() else []:
            try:
                name = ds.TrainingSet.load(p).name
            except (OSError, ValueError, KeyError) as exc:
                name = f"{p.stem} (unreadable: {exc})"
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(p))
            self.sets.addItem(item)
            if str(p) == select:
                chosen = item
        self.sets.blockSignals(False)
        if chosen is None and self.sets.count():
            chosen = self.sets.item(0)
        if chosen is not None:
            self.sets.setCurrentItem(chosen)
        else:
            self._show_set(None, None)

    def _open_selected(self, item, _prev=None) -> None:
        if item is None:
            return
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        try:
            tset = ds.TrainingSet.load(path)
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "Training set", f"{path}\n\n{exc}")
            return
        self._show_set(tset, path)
        if tset.projects:
            self.gather()                                 # quick from the cache; measures what is new

    def _new(self) -> None:
        entries = [e for e in self.profiles.entries() if e.profile is not None]
        if not entries:
            QtWidgets.QMessageBox.information(self, "New training set", "Make a profile first (Settings › Profiles).")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New training set", "Name:", text="New training set")
        if not ok or not name.strip():
            return
        entry = entries[0]
        groups = [g.name for g in entry.profile.classes.groups if len(g.subgroups) >= 2]
        tset = ds.TrainingSet(name.strip(), entry.profile.copy(), entry.id, groups[0] if groups else "")
        path = self._free_path(name)
        tset.save(path)
        self.reload(str(path))

    def _duplicate(self) -> None:
        if self.tset is None:
            return
        copy = ds.TrainingSet.from_dict(self.tset.to_dict())
        copy.name = f"{self.tset.name} (copy)"
        path = self._free_path(copy.name)
        copy.save(path)
        self.reload(str(path))

    def _delete(self) -> None:
        if self.path is None:
            return
        if QtWidgets.QMessageBox.question(self, "Delete training set",
                                          f"Delete “{self.tset.name}”? (The projects are not touched.)") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        os.remove(self.path)
        self.path, self.tset = None, None
        self.reload()

    def _free_path(self, name: str):
        import re
        folder = ds.library_folder()
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower()).strip("_") or "training_set"
        path, n = folder / f"{stem}{ds.SUFFIX}", 2
        while path.exists():
            path, n = folder / f"{stem}_{n}{ds.SUFFIX}", n + 1
        return path

    def _save(self) -> None:
        if self.tset is not None and self.path is not None:
            self.tset.save(self.path)

    # ------------------------------------------------------------ showing a set
    def _show_set(self, tset, path) -> None:
        self.tset, self.path, self.data = tset, path, None
        self.kinds.setEnabled(tset is not None)
        self._filling = True
        self.name.setText(tset.name if tset else "")
        self.profile_box.clear()
        for e in self.profiles.entries():
            if e.profile is not None:
                self.profile_box.addItem(e.profile.name, e.id)
        if tset is not None:
            idx = self.profile_box.findData(tset.profile_id)
            if idx < 0:                                        # a profile no longer in the library: its copy
                self.profile_box.addItem(f"{tset.profile.name} (kept with the set)", tset.profile_id)
                idx = self.profile_box.count() - 1
            self.profile_box.setCurrentIndex(idx)
            self.mode_marked.setChecked(tset.mode == "projects")
            self.mode_share.setChecked(tset.mode == "share")
            self.share.setValue(int(round(tset.share * 100)))
            self.seed.setValue(tset.seed)
            self.include_partial.setChecked(tset.include_partial)
        self._fill_groups()
        self._filling = False
        self.overviews = {}
        self._fill_projects()
        self._refresh_views()
        self._read_overview()

    def _fill_groups(self) -> None:
        self.group_box.blockSignals(True)
        self.group_box.clear()
        if self.tset is not None:
            for g in self.tset.profile.classes.groups:
                self.group_box.addItem(g.name + ("" if len(g.subgroups) >= 2 else "  (no grades)"), g.name)
            self.group_box.setCurrentIndex(max(0, self.group_box.findData(self.tset.group)))
        self.group_box.blockSignals(False)
        self._show_grades()

    def graded(self) -> bool:
        """Whether the set's group has grades (two or more subgroups): what classifiers need."""
        return self.tset is not None and len(self.tset.grades) >= 2

    def _show_grades(self) -> None:
        if self.tset is None or not self.tset.grades:
            self.grades.setText("The group has no subgroups.")
            return
        if not self.graded():
            self.grades.setText(f"{self.tset.group} has no grades (one subgroup): segmentation models can be trained "
                                f"on it; classifiers need two or more.")
            return
        ordered = self.tset.profile.classes.ordered(self.tset.group)
        text = (" < ".join(self.tset.grades) + "   (grades in order)") if ordered else \
            (", ".join(self.tset.grades) + "   (categories, no order — Settings › Profiles › Classes)")
        older = self.tset.profile.aliases.get(self.tset.group, {})
        if older:
            text += "   ·   also read: " + ", ".join(f"{o} → {n}" for o, n in older.items())
        self.grades.setText(text)

    # ------------------------------------------------------------ what the projects hold
    def _read_overview(self) -> None:
        """Read what each project holds (its masks, nothing measured) in the background; the table follows."""
        import threading
        tset = self.tset
        if tset is None or not tset.projects:
            return
        key = (tuple(p.path for p in tset.projects), tset.group, tset.profile_id)
        self._overview_key = key
        profile = tset.profile.copy()

        def work():
            out = {}
            for path in key[0]:
                try:
                    out[path] = ds.project_overview(path, profile, key[1])
                except Exception as exc:                            # noqa: BLE001 (said in the table)
                    out[path] = ds.ProjectOverview(problem=str(exc))
            self.overview_ready.emit((key, out))
        threading.Thread(target=work, daemon=True).start()

    def _overview_found(self, result) -> None:
        key, overviews = result
        if key != self._overview_key:                       # an older request: the set changed since
            return
        self.overviews = overviews
        self._fill_projects()

    def _fill_projects(self) -> None:
        grades = self.tset.grades if self.tset and self.graded() else []
        headers = ["Project", "Role", "Images", "µm/px", "Outlined", "Complete", "Objects", "Partial"] + grades + \
            ["Ungraded", "Unreviewed", "Status"]
        tips = {"Images": "Images per type", "µm/px": "The images' pixel sizes (⚠: far from the other projects')",
                "Outlined": "Images with at least one object of the group drawn",
                "Complete": "Images whose annotation is marked complete (Properties › Annotation: every object "
                            "of the group drawn): everything not drawn is not the group — used whole by "
                            "segmentation training, background included",
                "Role": "Training: built and trained on. Evaluation: judged on (and, for segmentation, each "
                        "model's epoch chosen on them). Test: never trained on, never used to choose — each test "
                        "group (New test group…: e.g. one per microscope setup or lab) is a final check of its own",
                "Objects": "Objects of the group drawn",
                "Partial": "Objects whose shape is incomplete — cut by the image edge (found by itself) or marked "
                           "partial: left out of classifier training (their shape parameters would mislead); "
                           "segmentation uses them as any other",
                "Ungraded": "Objects of the group without a grade",
                "Unreviewed": "Unreviewed suggestions of the group: left out of training until reviewed"}
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for c, h in enumerate(headers):
            if h in tips:
                self.table.horizontalHeaderItem(c).setToolTip(tips[h])
        projects = self.tset.projects if self.tset else []
        self.table.setRowCount(0)                           # also drops the role boxes of the rows shown before
        self.table.setRowCount(len(projects))
        held = ds.evaluation_projects(self.tset, self.data) if self.tset else set()
        outliers = ds.resolution_outliers(self.overviews)
        last = len(headers) - 1
        rows = []                                           # per project: its cells (column → item), its role text
        for pi, entry in enumerate(projects):
            shown = role_text(entry) if entry.role == "test" or self.tset.mode == "projects" else (
                "Evaluation (drawn)" if pi in held else "Training")
            cells = {0: _cell(entry.name), 1: _cell(shown)}
            cells[0].setToolTip(entry.path)
            cells[0].setData(QtCore.Qt.ItemDataRole.UserRole, pi)      # which project, whatever the order shown
            ov = self.overviews.get(entry.path)
            if ov is None:
                cells[last] = _cell("reading…")
            else:
                sizes = " · ".join((f"{t} " if len(ov.um_per_px) > 1 else "")
                                   + "/".join(f"{v:g}" for v in vs) for t, vs in ov.um_per_px.items())
                size_cell = _cell(("⚠ " if entry.path in outliers else "") + sizes)
                size_cell.setToolTip(outliers.get(entry.path, ""))
                values = [", ".join(f"{n} {t}" for t, n in ov.images.items()), size_cell, ov.outlined, ov.fully,
                          ov.objects, ov.partial or "", *[ov.grades.get(g, 0) for g in grades], ov.ungraded or "",
                          ov.unreviewed or ""]
                for c, v in enumerate(values, start=2):
                    cells[c] = v if isinstance(v, QtWidgets.QTableWidgetItem) else _cell(v)
                problem = ov.problem or (self.data.problems.get(entry.name) if self.data is not None else "")
                cells[last] = _cell("⚠ " + problem if problem else "✔")
                cells[last].setToolTip(problem or "")
            rows.append((pi, entry, cells))
        # The order shown: the set's, or by the column clicked (numbers as numbers; the set's order is kept)
        header = self.table.horizontalHeader()
        col = headers.index(self._data_sort[0]) if self._data_sort and self._data_sort[0] in headers else -1
        if col >= 0:
            from modalytics.ui.training.sorting import sort_key
            rows.sort(key=lambda row: sort_key(row[2][col].text() if col in row[2] else ""),
                      reverse=self._data_sort[1] == QtCore.Qt.SortOrder.DescendingOrder)
        header.setSortIndicatorShown(col >= 0)
        if col >= 0:
            header.setSortIndicator(col, self._data_sort[1])
        groups = [""] + [g for g in dict.fromkeys(p.group for p in projects if p.role == "test") if g]
        for r, (pi, entry, cells) in enumerate(rows):
            for c, item in cells.items():
                if c == 1 and self.tset.mode == "projects":
                    continue
                self.table.setItem(r, c, item)
            if self.tset.mode == "projects":                 # the role: chosen here
                role = QtWidgets.QComboBox()
                role.addItem(ROLE_NAMES["training"], "training")
                role.addItem(ROLE_NAMES["evaluation"], "evaluation")
                for g in groups:                             # every test group of the set
                    role.addItem(f"Test: {g}" if g else ROLE_NAMES["test"], f"test:{g}")
                role.addItem("New test group…", NEW_GROUP)
                role.setCurrentIndex(max(0, role.findData(role_key(entry))))
                role.currentIndexChanged.connect(lambda _i, p=pi, box=role: self._role_changed(p, box))
                self.table.setCellWidget(r, 1, role)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, max(self.table.columnWidth(0), 200))
        self._update_summary()

    def _update_summary(self) -> None:
        """Per role: projects, images outlined, objects (per grade) — from what the projects hold."""
        self._update_measure_note()
        if self.tset is None or not self.tset.projects:
            self.summary.setText("")
            return
        if not self.overviews:
            self.summary.setText("Reading what the projects hold…")
            return
        grades = self.tset.grades if self.graded() else []
        roles = [("Training", ds.training_projects(self.tset, self.data)),
                 ("Evaluation", ds.evaluation_projects(self.tset, self.data))]
        roles += [(f"Test: {g}" if g != ds.TEST else g, idx) for g, idx in ds.test_groups(self.tset).items()]
        parts, missing = [], False
        for name, idx in roles:
            ovs = [self.overviews.get(p.path) for i, p in enumerate(self.tset.projects) if i in idx]
            ovs = [o for o in ovs if o is not None]
            test = name.startswith(ds.TEST)
            per = {g: sum(o.grades.get(g, 0) for o in ovs) for g in grades}
            missing |= not test and bool(grades) and any(v == 0 for v in per.values())
            parts.append(f"<b>{name}</b>: {len(idx)} project(s), {sum(o.outlined for o in ovs)} image(s) outlined, "
                         f"{sum(o.objects for o in ovs)} objects"
                         + (f" ({', '.join(f'{g} {n}' for g, n in per.items())})" if grades else ""))
        text = " · ".join(parts) + "."
        if not ds.evaluation_projects(self.tset, self.data):
            text += " ⚠ No project is kept for evaluation."
        elif missing:
            text += " ⚠ A grade is missing on one side: add projects or change the evaluation choice."
        self.summary.setText(text)

    def _update_measure_note(self) -> None:
        """The Classifiers bar: what is measured (classifiers need the parameters; segmentation does not)."""
        graded = self.graded()
        self.measure_btn.setEnabled(graded and self.tset is not None and bool(self.tset.projects))
        self.classifiers.setEnabled(graded)
        if self.tset is None:
            self.measure_note.setText("")
        elif not graded:
            self.measure_note.setText(f"Classifiers need a group with grades (two or more subgroups); "
                                      f"{self.tset.group or 'the group'} has none. Choose another in Data.")
        elif self.data is None:
            self.measure_note.setText("Not measured yet: Measure measures every graded object with the profile's "
                                      "parameters (in the background; kept for next time).")
        else:
            n = len({int(p) for p in self.data.project}) if len(self.data.project) else 0
            problems = "; ".join(f"{k}: {v}" for k, v in self.data.problems.items())
            partial = sum(self.data.partial.values())
            kept = int(self.data.partial_rows.sum()) if self.data.partial_rows is not None else 0
            self.measure_note.setText(f"Measured: {len(self.data.grade)} graded objects from {n} project(s) with "
                                      f"{len(self.data.columns)} parameters (test projects are never used)."
                                      + (f" {partial} with an incomplete shape (cut by the image edge, or marked "
                                         f"partial) left out (Data: include them where the parameter allows it)."
                                         if partial else "")
                                      + (f" {kept} of them with an incomplete shape, counted only for the parameters "
                                         f"that do not need the whole object." if kept else "")
                                      + (f" ⚠ {problems}" if problems else ""))

    # ------------------------------------------------------------ edits (saved at once)
    def _renamed(self) -> None:
        if self.tset is not None and self.name.text().strip() and self.name.text().strip() != self.tset.name:
            from modalytics.training import runs
            runs.rename_set(self.tset, self.name.text().strip())        # its folder of runs follows
            self.tset.name = self.name.text().strip()
            self._save()
            self.train_seg._list_runs()                                  # the run shown, from its new place
            self.evaluate._list_runs()                                   # and the evaluation shown
            item = self.sets.currentItem()
            if item is not None:
                item.setText(self.tset.name)

    def _profile_changed(self) -> None:
        if self._filling or self.tset is None:
            return
        pid = self.profile_box.currentData()
        try:
            profile = self.profiles.get(pid)
        except (KeyError, ValueError):
            return
        self.tset.profile, self.tset.profile_id = profile.copy(), pid
        groups = [g.name for g in profile.classes.groups if len(g.subgroups) >= 2]
        if self.tset.group not in groups:
            self.tset.group = groups[0] if groups else ""
        self._save()
        self.data = None
        self._fill_groups()
        self._fill_projects()
        self._refresh_views()
        self._read_overview()

    def _group_changed(self) -> None:
        if self.tset is None or self._filling:
            return
        self.tset.group = self.group_box.currentData() or ""
        self._save()
        self.data = None
        self._show_grades()
        self._fill_projects()
        self._refresh_views()
        self._read_overview()
        if self.graded():
            self.gather()

    def _sort_data(self, col: int) -> None:
        """A click on a column title: sort by it (again: the other way)."""
        title = self.table.horizontalHeaderItem(col).text() if self.table.horizontalHeaderItem(col) else ""
        asc, desc = QtCore.Qt.SortOrder.AscendingOrder, QtCore.Qt.SortOrder.DescendingOrder
        same = self._data_sort is not None and self._data_sort[0] == title
        self._data_sort = (title, desc if same and self._data_sort[1] == asc else asc)
        self._fill_projects()

    def _project_at(self, row: int) -> int:
        """The project shown on a row of the table (its index in the set)."""
        item = self.table.item(row, 0)
        return int(item.data(QtCore.Qt.ItemDataRole.UserRole)) if item is not None else row

    def reload_data(self) -> None:
        """Read the projects again (even if their files look unchanged) and measure what changed."""
        if self.tset is None or not self.tset.projects:
            return
        ds.forget_overviews([p.path for p in self.tset.projects])
        self.overviews = {}
        self._fill_projects()
        self._read_overview()
        self.gather()

    def _role_changed(self, project: int, box) -> None:
        """A project's role chosen: Training, Evaluation, a test group, or a new test group (named here)."""
        entry, key = self.tset.projects[project], box.currentData()
        if key == NEW_GROUP:
            name, ok = QtWidgets.QInputDialog.getText(
                self, "New test group", "Name of the test group (e.g. Naive, External – Hospital B, Setup 20×):")
            name = name.strip()
            if not ok or not name:
                QtCore.QTimer.singleShot(0, self._fill_projects)           # the box back as it was
                return
            entry.role, entry.group = "test", "" if name == ds.TEST else name
        elif key.startswith("test:"):
            entry.role, entry.group = "test", key[len("test:"):]
        else:
            entry.role, entry.group = key, ""
        self._save()
        QtCore.QTimer.singleShot(0, self._fill_projects)    # every box offers the groups as they are now
        self._refresh_views()

    def _evaluation_changed(self, *_args) -> None:
        if self._filling or self.tset is None:
            return
        self.tset.mode = "projects" if self.mode_marked.isChecked() else "share"
        self.tset.share = self.share.value() / 100.0
        self.tset.seed = self.seed.value()
        self._save()
        self._fill_projects()
        self._refresh_views()

    def _partial_changed(self, on: bool) -> None:
        """Partial objects in or out: the objects gathered again (from the cache: quick)."""
        if self._filling or self.tset is None or self.tset.include_partial == on:
            return
        self.tset.include_partial = bool(on)
        self._save()
        self.gather()

    def _add_projects(self) -> None:
        if self.tset is None:
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add graded projects", "",
                                                          "Projects (*.modalytics *.hpro)")
        known = {os.path.abspath(p.path) for p in self.tset.projects}
        added = [p for p in paths if os.path.abspath(p) not in known]
        self.tset.projects += [ds.ProjectEntry(p) for p in added]
        self._save()
        if added:
            self._fill_projects()
            self._read_overview()
            self.gather()

    def _remove_project(self) -> None:
        r = self.table.currentRow()
        if self.tset is None or r < 0:
            return
        del self.tset.projects[self._project_at(r)]
        self._save()
        self._fill_projects()
        self._read_overview()
        self.gather() if self.tset.projects else self._show_set(self.tset, self.path)

    # ------------------------------------------------------------ measuring
    def gather(self) -> None:
        """Measure the projects' graded objects in the background (quick for those measured before)."""
        if self.tset is None or not self.tset.projects or not self.graded():
            return
        tset = ds.TrainingSet.from_dict(self.tset.to_dict())
        key = _measured_by(tset)

        def done(data) -> None:
            if self.tset is not None and _measured_by(self.tset) == key:     # what measuring reads: unchanged
                self.data = data
                self._fill_projects()
                self._refresh_views()
                self.statusBar().showMessage(f"✅ {len(data.grade)} graded objects from {len(tset.projects)} "
                                             f"project(s)", 8000)

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            try:
                return ds.gather(tset, progress, cancelled)
            except ds.Cancelled:
                raise Cancelled() from None

        self.tasks.run(f"Measuring the graded objects of {len(tset.projects)} project(s)…", job, done,
                       lambda exc: QtWidgets.QMessageBox.warning(self, "Measure", str(exc)))

    # ------------------------------------------------------------ explore
    def _refresh_views(self) -> None:
        """Parameters, Explore and Build again, after the set or its measurements changed."""
        self.parameters.refresh()
        self._fill_explore()
        self.build.refresh()
        self.evaluate.refresh()
        self.train_seg.refresh()

    def analysis_changed(self) -> None:
        """The Parameters tab shows another analysis: Explore's verdicts follow it."""
        self._fill_explore()

    def _verdicts(self) -> dict:
        """{parameter id: Relevance} of the analysis Parameters shows, if it was made on the training projects."""
        a = self.parameters.analysis
        return {r.uid: r for r in a.rows} if a is not None and a.on == "training" else {}

    def _fill_explore(self, *_args) -> None:
        clear_tree(self.ranked)
        if self.data is None or self.tset is None or not len(self.data.grade):
            self.explore_note.setText("Measure the projects (Parameters tab) to see which parameters separate the "
                                      "grades.")
            self.plot.show_values("", [])
            self.pair_box.clear()
            return
        held = ds.split(self.tset, self.data)
        train = ds.training_rows(self.tset, self.data)    # neither evaluation nor test projects
        grades = self.data.grades
        # Grades: each neighbouring pair (a scale). Categories: every two classes, and each one against the rest
        if self.data.ordered:
            choices = [(f"{grades[k]} vs {grades[k + 1]}", ([k], [k + 1], grades[k], grades[k + 1]))
                       for k in range(len(grades) - 1)]
        else:
            n = len(grades)
            choices = [(f"{grades[a]} vs {grades[b]}", ([a], [b], grades[a], grades[b]))
                       for a in range(n) for b in range(a + 1, n)]
            if n > 2:
                choices += [(f"{grades[c]} vs the rest", ([x for x in range(n) if x != c], [c], "the rest",
                                                         grades[c])) for c in range(n)]
        if [self.pair_box.itemText(i) for i in range(self.pair_box.count())] != [t for t, _c in choices]:
            self.pair_box.blockSignals(True)
            self.pair_box.clear()
            for text, _c in choices:
                self.pair_box.addItem(text)
            self.pair_box.blockSignals(False)
        self.pair_box.setVisible(len(choices) > 1)
        lower, higher, lower_name, higher_name = choices[max(0, self.pair_box.currentIndex())][1]
        self._sides = (lower, higher)
        n_tr = int(train.sum())
        verdicts = self._verdicts()
        self.explore_note.setText(
            f"On the {n_tr} training objects ({', '.join(f'{g} {n}' for g, n in self.data.counts(train).items())}); "
            f"the {int(held.sum())} evaluation objects (and the test projects') are kept for the final check. "
            "<b>AUC</b>: how well the parameter separates the grades (0.5 = not at all, 1 = perfectly). "
            "<b>Cut-off</b>: the value that separates them best (Youden's index), with its sensitivity and "
            "specificity. "
            + ("<b>Verdict</b> and <b>group</b>: from the analysis shown in Parameters." if verdicts else
               "Parameters › Analyse adds each one's verdict (does it separate beyond chance?) and redundancy group."))
        text = self.search.text().strip().lower()
        every = ranges.between(self.data, train, lower, higher, lower_name, higher_name)
        self.separations = [s for s in every if s.n_lower and s.n_higher]      # parameters with values here
        unused = len(every) - len(self.separations)
        if unused:
            self.explore_note.setText(self.explore_note.text() + f" {unused} parameter(s) of the profile have no "
                                      "values in these projects (another image type) and are not listed.")
        role = QtCore.Qt.ItemDataRole.UserRole
        for s in self.separations:
            if text and text not in s.label.lower():
                continue
            title = f"{s.label} [{s.unit}]" if s.unit else s.label
            v = verdicts.get(s.uid)
            item = SortItem(self.ranked, [
                title, _f(s.auc, 3), s.rule, _pct(s.sensitivity), _pct(s.specificity), f"{s.n_lower} / {s.n_higher}",
                rv.VERDICTS[v.verdict] if v else "", str(v.group or "") if v else ""])
            item.setData(0, role, s.uid)
            item.setToolTip(0, s.help)
            for c in range(1, 6):
                item.setTextAlignment(c, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if self.ranked.topLevelItemCount():
            self.ranked.setCurrentItem(self.ranked.topLevelItem(0))

    def _show_parameter(self, item, _prev=None) -> None:
        if item is None or self.data is None:
            return
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        s = next((x for x in self.separations if x.uid == uid), None)
        j = next(i for i, c in enumerate(self.data.columns) if c[0] == uid)
        train = ds.training_rows(self.tset, self.data)
        series = []
        lower, higher = getattr(self, "_sides", ([0], [1]))
        for side, name, classes in ((0, s.lower, lower), (1, s.higher, higher)):
            rgba = (self.tset.profile.classes.color(self.tset.group, self.data.grades[classes[0]])
                    if len(classes) == 1 else (0.55, 0.6, 0.66, 1.0))          # "the rest": grey
            series.append((name, self.data.values[train & np.isin(self.data.grade, classes), j],
                           QtGui.QColor.fromRgbF(*rgba)))
        title = f"{s.label} [{s.unit}]" if s.unit else s.label
        note = (f"AUC {_f(s.auc, 3)} · {s.higher} has {'larger' if s.direction == 'higher' else 'smaller'} values · "
                f"cut-off {s.rule}: sensitivity {_pct(s.sensitivity)}, specificity {_pct(s.specificity)}")
        self.plot.show_values(title, series, s.cutoff, note)

    def help_context(self) -> str:
        """The tab shown, for F1: the inner one (Explore, Build, Evaluate, Train) when there is one."""
        inner = self.kinds.currentWidget()
        if isinstance(inner, QtWidgets.QTabWidget):
            return f"training:{inner.tabText(inner.currentIndex())}"
        if inner is self.classifiers_page and self.classifiers is not None:
            return f"training:{self.classifiers.tabText(self.classifiers.currentIndex())}"
        return f"training:{self.kinds.tabText(self.kinds.currentIndex())}"

    def show_help(self) -> None:
        """The help on the tab shown (F1)."""
        from modalytics import app_settings
        from modalytics import help as hp
        from modalytics.ui.help_window import show_help
        show_help(self, hp.page_for(self.help_context()), app_settings.get("theme") or "dark")

    def closeEvent(self, event) -> None:
        if self.tasks.busy:
            self.tasks.cancel()
        super().closeEvent(event)


def _cell(value) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
    return item


def _f(v: float, digits: int) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.0f} %"

