"""
Training › Build: make a classifier from the parameters, check it on the evaluation projects, save it to the profile.

    ┌ Parameters ───────────────┐ ┌ [Ranges] [Score] [Logistic regression] [Random forest] [Gradient boosting] ┐
    │ Filter…                   │ │ Ranges: High grade dysplasia when [all ▾] of these hold             │
    │ ☑ Curvature (mean)  1.000 │ │ Grade            Parameter          Condition  Value                 │
    │ ☐ Uniformity Coef.  0.998 │ │ High grade dysp. Curvature (mean)   ≥          0.09742               │
    │ ☑ Irregularity (BAM) 0.979│ │ …                                                                    │
    └───────────────────────────┘ └──────────────────────────────────────────────────────────────────────┘
    ┌ Validation ────────────────────────────────────────────────────────────────────────────────────────┐
    │ Classifier  Training  Evaluation (95 % CI)  κ  Sensitivity (CI)  Specificity (CI)  Cross-validation  κ │
    │ Ranges      …                                                                                     │
    │ [Cross-validate]  leave-one-project-out on the training projects                                    │
    │ [Confusion matrices] [Mistakes: project, image, object, hand / predicted grade, why — double-click]   │
    └────────────────────────────────────────────────────────────────────────────────────────────────────┘
    [Save to profile]  Colonic tissue › Gland is classified by: Score (21 Oct 2026, …)

* **Ranges**: ticking a parameter adds a condition at its best cut-off (Explore's
  Youden cut-off, for each grade against the one below); the direction and value
  can be changed; the conditions of a grade are combined with all (AND) or any (OR).
* **Score**: the ticked parameters become criteria at their best cut-offs; their
  points are fitted to the grades (logistic regression, in whole points) and can be
  changed, as can the totals from which each grade is given.
* **Logistic regression**: fitted on the ticked parameters at once (standardised: a
  weight per SD of each, and with two grades an odds ratio per SD) — readable, like
  Score, but using the values themselves instead of yes / no at a cut-off.
* **Random forest**: grown on the ticked parameters (all of them with *All
  parameters*), in the background; it shows each parameter's importance and its
  out-of-bag accuracy, and gives each object a confidence.
* **Gradient boosting**: shallow trees added one after another, each correcting the
  ones before; grown in the background; each parameter's importance.

Ranges and Score climb a scale (each grade from the one below it), so they are for
**ordered grades**; for **categories** (classes with no order, as the profile says) only
the learned classifiers are offered, and κ is unweighted.

The classifiers are worked out on the training projects only; the evaluation
projects judge them, each proportion with its 95 % Wilson interval. **Cross-validate**
judges them again by project on the training projects (``training.validation``):
steadier with few projects. **Mistakes** lists the objects graded differently from
the hand grade; a double-click opens the project at that object. The classifiers
are kept with the training set as you edit them.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.training import classifiers as cl
from modalytics.training import dataset as ds
from modalytics.training import ranges
from modalytics.training import acquisition as aq
from modalytics.training import prep as pp
from modalytics.training import validation as va
from modalytics.training.metrics import CHOOSE_BY
from modalytics.ui.settings.widgets import clear_tree, list_tree, muted
from modalytics.ui.training.sorting import SortItem, sortable

KINDS = {"ranges": "Ranges", "score": "Score", "logistic": "Logistic regression", "forest": "Random forest",
         "boosting": "Gradient boosting"}                       # the tabs, in this order
SCALE = ("ranges", "score")                                     # for ordered grades only (they climb the scale)
PARAMETERS = {"score": "score_parameters", "logistic": "logistic_parameters", "forest": "forest_parameters",
              "boosting": "boosting_parameters"}                # where each kind's ticks are kept (``tset.built``)
UID = QtCore.Qt.ItemDataRole.UserRole


class BuildPanel(QtWidgets.QWidget):
    """The Build tab of a training window (see the module docstring).

    Args:
        window: The ``TrainingWindow`` (its ``tset``, ``data``, ``profiles`` and ``_save``).
    """

    saved = QtCore.Signal(str)                   # the profile id a classifier was saved to
    show_object = QtCore.Signal(str, str, str, float, float)   # project path, image name, group, x, y (µm)

    def __init__(self, window) -> None:
        super().__init__()
        self.win = window
        self.separations: dict[str, list] = {}   # {parameter id: [Separation per neighbouring pair]}
        self.cv: dict[str, va.CrossValidation] = {}   # {kind: cross-validation} of the data shown
        self._cv_data = None
        self._filling = False
        self.data = None                          # the measured objects, normalised as set (judged on)
        self.fit_data, self.fit_rows = None, None  # … with the augmented rows: what classifiers are made from
        self.acquired = None                      # the training projects measured again (acquisition), raw
        self._acquired_key = None
        self._acquired_n = None                   # … normalised
        col = QtWidgets.QVBoxLayout(self)
        self.note = muted("")
        self.note.setWordWrap(True)
        col.addWidget(self.note)
        # 0. How the objects are prepared (every classifier; recorded with it)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel("Normalise per project"))
        self.normalize = QtWidgets.QComboBox()
        for key, text in pp.NORMALIZE.items():
            self.normalize.addItem(text, key)
        self.normalize.setToolTip(
            "Each parameter read relative to its project's complete objects of the group (whatever their grade): "
            "takes out what a whole project shares — illumination, calibration, thickness. Applied the same way when "
            "the classifier grades a new project.\nCaution: a project whose objects are all of one grade loses "
            "that grade's difference — for projects that hold several grades.")
        self.normalize.currentIndexChanged.connect(self._normalize_changed)
        prow.addWidget(self.normalize)
        prow.addWidget(QtWidgets.QLabel("Weigh projects"))
        self.weigh = QtWidgets.QComboBox()
        for key, text in pp.WEIGHTS.items():
            self.weigh.addItem(text, key)
        self.weigh.setToolTip(
            "How much each project's objects weigh together when a classifier is fitted (logistic regression, "
            "forest, boosting, a Score's points; not cut-offs).\nNone: every object the same — a project with many "
            "objects counts more.\n√ objects: a project's total grows with the square root of its objects (a "
            "compromise).\nEqual: every project the same — the one object of a one-object project then weighs as "
            "much as all the objects of another: use with care.")
        self.weigh.currentIndexChanged.connect(self._weigh_changed)
        prow.addWidget(self.weigh)
        self.augment_btn = QtWidgets.QPushButton("Augmentation…")
        self.augment_btn.setToolTip("More training rows made from the training projects' objects (never from the "
                                    "evaluation or test projects): acquisition copies, noisy copies, SMOTE")
        self.augment_btn.clicked.connect(self._edit_augmentation)
        prow.addWidget(self.augment_btn)
        self.prep_note = muted("")
        self.prep_note.setWordWrap(True)
        prow.addWidget(self.prep_note, 1)
        col.addLayout(prow)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # 1. Parameters to use
        left = QtWidgets.QWidget()
        lcol = QtWidgets.QVBoxLayout(left)
        lcol.setContentsMargins(0, 0, 0, 0)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter parameters…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lcol.addWidget(self.search)
        self.params = sortable(list_tree(["Parameter", "AUC"]))
        self.params.setColumnWidth(0, 250)
        self.params.setToolTip("Tick the parameters the classifier uses (ranked by AUC, as in Explore)")
        self.params.itemChanged.connect(self._ticked)
        lcol.addWidget(self.params, 1)
        top.addWidget(left)
        # 2. The classifier kinds
        self.kinds = QtWidgets.QTabWidget()
        self.kinds.addTab(self._build_ranges(), KINDS["ranges"])
        self.kinds.addTab(self._build_score(), KINDS["score"])
        self.kinds.addTab(self._build_logistic(), KINDS["logistic"])
        self.kinds.addTab(self._build_forest(), KINDS["forest"])
        self.kinds.addTab(self._build_boosting(), KINDS["boosting"])
        self.kinds.currentChanged.connect(self._kind_changed)
        top.addWidget(self.kinds)
        top.setSizes([340, 900])
        split.addWidget(top)
        # 3. Validation
        bottom = QtWidgets.QWidget()
        bcol = QtWidgets.QVBoxLayout(bottom)
        bcol.setContentsMargins(0, 0, 0, 0)
        self.compare = sortable(list_tree(["Classifier", "Training", "Evaluation (95 % CI)", "κ", "Sensitivity (95 % CI)",
                                  "Specificity (95 % CI)", "Cross-validation (95 % CI)", "κ", "Per project"]))
        for c, width in enumerate((140, 70, 150, 50, 160, 160, 180, 50)):
            self.compare.setColumnWidth(c, width)
        self.compare.setFixedHeight(172)                             # header + the five classifiers
        self.compare.setToolTip(
            "Accuracy: share of objects given their hand grade, with its 95 % interval — resampling whole projects, "
            "since objects of one project are not independent; with fewer than "
            f"{cl.MIN_PROJECTS} projects Wilson's by objects, marked *: likely too narrow. κ (Cohen's kappa): "
            "agreement beyond chance (1 = perfect, 0 = chance; quadratic-weighted with more than two grades). "
            "Sensitivity / specificity: of the highest grade against the others, on the evaluation projects. "
            "Training: on the objects it was made on (optimistic). Cross-validation: each training project graded "
            "by the classifier made again without it; Per project: its lowest – highest accuracy.")
        bcol.addWidget(self.compare)
        self.coverage = QtWidgets.QLabel()                       # objects left out, a grade never given
        self.coverage.setWordWrap(True)
        bcol.addWidget(self.coverage)
        cvrow = QtWidgets.QHBoxLayout()
        self.cv_btn = QtWidgets.QPushButton("Cross-validate")
        self.cv_btn.setToolTip("Judge the classifiers by project on the training projects: each is made again "
                               "without one project (or fold of projects) and grades it (in the background)")
        self.cv_btn.clicked.connect(self.cross_validate)
        cvrow.addWidget(self.cv_btn)
        self.cv_note = muted("")
        self.cv_note.setWordWrap(True)
        cvrow.addWidget(self.cv_note, 1)
        bcol.addLayout(cvrow)
        self.details_tabs = QtWidgets.QTabWidget()
        self.matrices = QtWidgets.QLabel()
        self.matrices.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.matrices.setWordWrap(True)
        self.matrices.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.details_tabs.addTab(self.matrices, "Confusion matrices")
        wrong = QtWidgets.QWidget()
        wcol = QtWidgets.QVBoxLayout(wrong)
        wcol.setContentsMargins(0, 4, 0, 0)
        wrow = QtWidgets.QHBoxLayout()
        wrow.addWidget(QtWidgets.QLabel("Objects graded differently from the hand grade, on"))
        self.wrong_source = QtWidgets.QComboBox()
        self.wrong_source.addItem("the evaluation projects", "evaluation")
        self.wrong_source.addItem("the training projects, cross-validated", "cv")
        self.wrong_source.currentIndexChanged.connect(self._show_mistakes)
        wrow.addWidget(self.wrong_source)
        self.wrong_note = muted("")
        wrow.addWidget(self.wrong_note, 1)
        wcol.addLayout(wrow)
        self.wrong = sortable(list_tree(["Project", "Image", "Object", "Hand grade", "Predicted", "Score", "Why"]))
        for c, width in enumerate((140, 180, 230, 150, 150, 60)):
            self.wrong.setColumnWidth(c, width)
        self.wrong.setToolTip("Double-click: open the project at that object")
        self.wrong.itemDoubleClicked.connect(self._open_mistake)
        wcol.addWidget(self.wrong, 1)
        self.details_tabs.addTab(wrong, "Mistakes")
        bcol.addWidget(self.details_tabs, 1)
        row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save to profile")
        self.save_btn.setToolTip("Make the classifier shown the profile's classifier for this group "
                                 "(used by Classify in projects with this profile)")
        self.save_btn.clicked.connect(self.save_to_profile)
        row.addWidget(self.save_btn)
        self.report_btn = QtWidgets.QPushButton("Write report")
        self.report_btn.setToolTip("Compare every classifier built in a report — the standard measures with their "
                                   "intervals, confusion matrices, each against the best, and the one recommended "
                                   "(a readable one when it is as good) — saved in the set's runs folder, and open it")
        self.report_btn.clicked.connect(self.write_report)
        row.addWidget(self.report_btn)
        row.addWidget(QtWidgets.QLabel("choosing by"))
        self.choose_by = QtWidgets.QComboBox()
        for key, text in CHOOSE_BY.items():
            self.choose_by.addItem(text, key)
        self.choose_by.setToolTip(
            "Where the report chooses the classifier.\n"
            "Cross-validation: each training project graded by the classifier made again without it — many projects "
            "decide, and the evaluation projects stay untouched: a final check of their own.\n"
            "Evaluation projects: the classifier that does best on them (they then flatter the one chosen a little).\n"
            "Test groups (Data) are never used to choose: each is a final check, apart.")
        self.choose_by.currentIndexChanged.connect(self._choose_by_changed)
        row.addWidget(self.choose_by)
        self.reports_btn = QtWidgets.QPushButton("Reports")
        self.reports_btn.setToolTip("The folder of this set's classifier reports")
        self.reports_btn.clicked.connect(self._open_reports)
        row.addWidget(self.reports_btn)
        self.pdf_btn = QtWidgets.QPushButton("PDF")
        self.pdf_btn.setToolTip("The latest classifier report as a PDF (A4 across), beside it, and open it")
        self.pdf_btn.clicked.connect(self._report_pdf)
        row.addWidget(self.pdf_btn)
        self.xlsx_btn = QtWidgets.QPushButton("XLSX…")
        self.xlsx_btn.setToolTip("The latest classifier report as an Excel workbook: every measure with its interval, "
                                 "against the best, the final checks, per class, confusion matrices, and native charts")
        self.xlsx_btn.clicked.connect(self._export_xlsx)
        row.addWidget(self.xlsx_btn)
        self.in_profile = muted("")
        self.in_profile.setWordWrap(True)
        row.addWidget(self.in_profile, 1)
        bcol.addLayout(row)
        split.addWidget(bottom)
        split.setSizes([400, 360])
        col.addWidget(split, 1)

    # ------------------------------------------------------------ building the editors
    def _build_ranges(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("A grade is given when"))
        self.combine = QtWidgets.QComboBox()
        self.combine.addItem("all its conditions hold (AND)", "all")
        self.combine.addItem("any of its conditions holds (OR)", "any")
        self.combine.currentIndexChanged.connect(self._combine_changed)
        row.addWidget(self.combine)
        row.addStretch(1)
        reset = QtWidgets.QPushButton("Best cut-offs")
        reset.setToolTip("Find every rule's cut-offs again, together: the rule's sensitivity + specificity − 1 as "
                         "large as can be (each cut-off alone is best for its parameter alone, but with AND all must "
                         "hold — found together they balance the rule)")
        reset.clicked.connect(self._reset_ranges)
        row.addWidget(reset)
        col.addLayout(row)
        self.conditions = QtWidgets.QTableWidget(0, 6)
        self.conditions.setHorizontalHeaderLabels(["Grade", "Parameter", "Condition", "Value", "Sensitivity",
                                                   "Specificity"])
        self.conditions.verticalHeader().hide()
        self.conditions.horizontalHeader().setStretchLastSection(True)
        for c, width in enumerate((160, 220, 80, 110, 90)):
            self.conditions.setColumnWidth(c, width)
        col.addWidget(self.conditions, 1)
        hint = muted("Otherwise the lowest grade is given. Sensitivity and specificity: of this condition alone, on "
                     "the training projects, for the grade against the one below. A missing value never counts.")
        hint.setWordWrap(True)
        col.addWidget(hint)
        self.ranges_warn = QtWidgets.QLabel()
        self.ranges_warn.setWordWrap(True)
        col.addWidget(self.ranges_warn)
        return w

    def _build_score(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit points")
        self.fit_btn.setToolTip("Fit the points to the grades of the training projects (logistic regression, "
                                "rounded to whole points) and choose the thresholds")
        self.fit_btn.clicked.connect(self._fit_score)
        row.addWidget(self.fit_btn)
        self.thresholds_row = QtWidgets.QHBoxLayout()
        row.addLayout(self.thresholds_row)
        row.addStretch(1)
        col.addLayout(row)
        self.criteria = QtWidgets.QTableWidget(0, 5)
        self.criteria.setHorizontalHeaderLabels(["Parameter", "Condition", "Value", "Points", "Note"])
        self.criteria.verticalHeader().hide()
        self.criteria.horizontalHeader().setStretchLastSection(True)
        for c, width in enumerate((220, 80, 110, 70)):
            self.criteria.setColumnWidth(c, width)
        col.addWidget(self.criteria, 1)
        hint = muted("Each criterion met adds its points; the total gives the grade. The points are fitted on the "
                     "training projects so that criteria that predict the grade more strongly count more; a "
                     "parameter that adds nothing beyond the others gets no points.")
        hint.setWordWrap(True)
        col.addWidget(hint)
        return w

    def _build_logistic(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Shrinkage"))
        self.strength = QtWidgets.QDoubleSpinBox()
        self.strength.setRange(0.01, 100)
        self.strength.setValue(1.0)
        self.strength.setToolTip("How strongly the weights are pulled towards 0 (L2 penalty = 1 / C): larger is "
                                 "steadier with few objects, smaller follows the training objects more closely")
        self.strength.editingFinished.connect(self._fit_logistic)
        row.addWidget(self.strength)
        self.log_balanced = QtWidgets.QCheckBox("Grades weigh equally")
        self.log_balanced.setChecked(True)
        self.log_balanced.toggled.connect(lambda *_a: self._fit_logistic())
        row.addWidget(self.log_balanced)
        row.addStretch(1)
        col.addLayout(row)
        self.log_state = QtWidgets.QLabel("")
        self.log_state.setWordWrap(True)
        col.addWidget(self.log_state)
        self.log_weights = sortable(list_tree(["Parameter", "Weight per SD", "Odds ratio per SD", ""]))
        self.log_weights.setColumnWidth(0, 260)
        self.log_weights.setColumnWidth(1, 110)
        self.log_weights.setColumnWidth(2, 130)
        col.addWidget(self.log_weights, 1)
        hint = muted("A logistic regression: each grade's log-odds a weighted sum of the parameters, each "
                     "standardised (its training mean and SD), so a weight is per SD and the weights can be "
                     "compared; positive: the higher grade more likely as the value grows. With two grades the "
                     "odds ratio per SD is e^weight. Fitted as the parameters are ticked; missing values count as "
                     "the training median.")
        hint.setWordWrap(True)
        col.addWidget(hint)
        return w

    def _build_boosting(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Trees"))
        self.n_stages = QtWidgets.QSpinBox()
        self.n_stages.setRange(10, 2000)
        self.n_stages.setSingleStep(50)
        self.n_stages.setValue(200)
        self.n_stages.setToolTip("How many trees are added one after another")
        row.addWidget(self.n_stages)
        row.addWidget(QtWidgets.QLabel("Depth"))
        self.depth = QtWidgets.QSpinBox()
        self.depth.setRange(1, 8)
        self.depth.setValue(3)
        self.depth.setToolTip("How deep each tree is (shallow: each corrects a little; 2–4 is usual)")
        row.addWidget(self.depth)
        row.addWidget(QtWidgets.QLabel("Learning rate"))
        self.rate = QtWidgets.QDoubleSpinBox()
        self.rate.setRange(0.01, 1.0)
        self.rate.setSingleStep(0.05)
        self.rate.setValue(0.1)
        self.rate.setToolTip("How much of each tree's correction is taken (smaller: steadier, needs more trees)")
        row.addWidget(self.rate)
        row.addStretch(1)
        self.boost_btn = QtWidgets.QPushButton("Train")
        self.boost_btn.setToolTip("Grow the trees on the training projects (in the background)")
        self.boost_btn.clicked.connect(lambda: self.train_learned("boosting"))
        row.addWidget(self.boost_btn)
        col.addLayout(row)
        self.boost_state = QtWidgets.QLabel("")
        self.boost_state.setWordWrap(True)
        col.addWidget(self.boost_state)
        self.boost_importance = sortable(list_tree(["Parameter", "Importance", ""]))
        self.boost_importance.setColumnWidth(0, 260)
        self.boost_importance.setColumnWidth(1, 90)
        col.addWidget(self.boost_importance, 1)
        hint = muted("Gradient boosting: shallow decision trees added one after another, each correcting what the "
                     "ones before got wrong (80 % of the objects drawn for each); the grade given is the most "
                     "probable. Often the most accurate on tables of measurements; like the forest, not a rule one "
                     "can write down. Missing values count as the training median.")
        hint.setWordWrap(True)
        col.addWidget(hint)
        return w

    def _build_forest(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Trees"))
        self.n_trees = QtWidgets.QSpinBox()
        self.n_trees.setRange(20, 2000)
        self.n_trees.setSingleStep(50)
        self.n_trees.setValue(300)
        self.n_trees.setToolTip("How many decision trees are grown (more: steadier, slower)")
        row.addWidget(self.n_trees)
        row.addWidget(QtWidgets.QLabel("Min. objects per leaf"))
        self.min_leaf = QtWidgets.QSpinBox()
        self.min_leaf.setRange(1, 50)
        self.min_leaf.setValue(2)
        self.min_leaf.setToolTip("A tree stops splitting where fewer objects would remain (larger: smoother, "
                                 "less fitted to the training objects)")
        row.addWidget(self.min_leaf)
        self.balanced = QtWidgets.QCheckBox("Grades weigh equally")
        self.balanced.setChecked(True)
        self.balanced.setToolTip("Each grade counts the same however many objects it has (class weights "
                                 "inversely proportional to the grade's frequency)")
        row.addWidget(self.balanced)
        row.addStretch(1)
        every = QtWidgets.QPushButton("All parameters")
        every.setToolTip("Tick every parameter that has values in these projects")
        every.clicked.connect(self._tick_all)
        row.addWidget(every)
        self.train_btn = QtWidgets.QPushButton("Train")
        self.train_btn.setToolTip("Grow the forest on the training projects (in the background)")
        self.train_btn.clicked.connect(lambda: self.train_learned("forest"))
        row.addWidget(self.train_btn)
        col.addLayout(row)
        self.forest_state = QtWidgets.QLabel("")
        self.forest_state.setWordWrap(True)
        col.addWidget(self.forest_state)
        self.importance = sortable(list_tree(["Parameter", "Importance", ""]))
        self.importance.setColumnWidth(0, 260)
        self.importance.setColumnWidth(1, 90)
        self.importance.setToolTip("How much each parameter helps the trees separate the grades (mean decrease "
                                   "in impurity, sums to 100 %); correlated parameters share their importance")
        col.addWidget(self.importance, 1)
        hint = muted("A random forest: many decision trees, each grown on a random sample of the training objects "
                     "and trying a random part of the parameters at every split; the grade given is the one most "
                     "trees vote for, and the share of votes is the object's confidence. Missing values are "
                     "replaced by the parameter's median on the training objects.")
        hint.setWordWrap(True)
        col.addWidget(hint)
        return w

    # ------------------------------------------------------------ showing a set
    def refresh(self) -> None:
        """Show the training set's classifiers (after it is opened, measured or changed)."""
        tset, data = self.win.tset, self.win.data
        self.separations = {}
        ready = tset is not None and data is not None and len(data.grade) > 0 and len(set(data.grade.tolist())) > 1
        self.setEnabled(ready)
        if not ready:
            self.note.setText("Measure the projects (Data tab) to build classifiers from their parameters.")
            clear_tree(self.params)
            clear_tree(self.compare)
            self.conditions.setRowCount(0)
            self.criteria.setRowCount(0)
            self.matrices.setText("")
            self.in_profile.setText("")
            clear_tree(self.wrong)
            return
        self.held = ds.split(tset, data)
        self.train = ds.training_rows(tset, data)          # neither evaluation nor test projects
        self.choose_by.blockSignals(True)
        self.choose_by.setCurrentIndex(max(0, self.choose_by.findData(tset.choose_by)))
        self.choose_by.blockSignals(False)
        roles = np.stack([self.held, self.train])
        if self._cv_data is not data or getattr(self, "_cv_held", None) is None \
                or not np.array_equal(self._cv_held, roles):
            self.cv, self._cv_data, self._cv_held = {}, data, roles.copy()           # new measurements or roles
        self._prepare()
        for s in ranges.separations(self.fit_data, self.fit_rows):
            if s.n_lower and s.n_higher:
                self.separations.setdefault(s.uid, []).append(s)
        n_ev = int(self.held.sum())
        ordered = data.ordered
        for kind in SCALE:                                   # Ranges, Score: a scale of grades only
            i = list(KINDS).index(kind)
            self.kinds.setTabEnabled(i, ordered)
            self.kinds.setTabToolTip(i, "" if ordered else "For ordered grades only: each grade is given from the one "
                                                           "below it. These classes are categories (the profile).")
        if not ordered and self._kind() in SCALE:
            self.kinds.setCurrentIndex(list(KINDS).index("logistic"))
        added = int(self.fit_rows.sum() - self.train.sum())
        self.note.setText(
            f"Built on the {int(self.train.sum())} training objects"
            + (f" (+ {added} augmented)" if added else "") + f"; judged on the {n_ev} evaluation objects."
            + ("" if n_ev else " <b>No evaluation objects</b>: mark evaluation projects or choose a share (Data tab).")
            + ("" if ordered else " <b>Categories</b> (no order between the classes): the learned classifiers; κ "
                                  "unweighted."))
        self._fill_params()
        self._show_ranges()
        self._show_score()
        self._show_logistic()
        self._show_forest()
        self._show_boosting()
        self._validate()
        self._show_in_profile()
        stale = self._stale()
        key = (self.win.path, str(self._prep().to_dict()), tuple(stale))
        if stale and key != getattr(self, "_healed", None) and not self.win.tasks.busy                 and getattr(self, "_measuring", None) is None:
            self._healed = key                                    # once: one that cannot be made stays as it is
            QtCore.QTimer.singleShot(0, lambda: self._rebuild(
                f"{', '.join(KINDS[k] for k in stale)} made with another preparation"))

    def _built(self, kind: str) -> cl.Classifier | None:
        d = self.win.tset.built.get(kind) if self.win.tset else None
        if not d:
            return None
        clf = cl.Classifier.from_dict(d)
        if clf.group != self.win.tset.group or clf.grades != list(self.win.data.grades):
            return None                                          # made for another group or grades
        if clf.kind in SCALE and not self.win.data.ordered:
            return None                                          # a scale's: the classes are categories now
        return clf

    def _store(self, kind: str, clf: cl.Classifier | None, **extra) -> None:
        if clf is None:
            self.win.tset.built.pop(kind, None)
        else:
            prep = self._prep()
            clf.prep = prep.to_dict() if prep != pp.Prep(seed=prep.seed) else {}
            self.win.tset.built[kind] = clf.to_dict()
            self._remember_selection(clf)
        self.win.tset.built.update(extra)
        self.win._save()

    def _ticked_uids(self) -> list[str]:
        kind = self._kind()
        if kind == "ranges":
            clf = self._built("ranges")
            return clf.parameters if clf else []
        return [u for u in self.win.tset.built.get(PARAMETERS[kind], []) if u in self.separations]

    def _kind(self) -> str:
        return list(KINDS)[self.kinds.currentIndex()]

    def _fill_params(self) -> None:
        self._filling = True
        clear_tree(self.params)
        ticked = set(self._ticked_uids())
        best = sorted(self.separations.items(), key=lambda kv: -max(s.auc for s in kv[1]))
        for uid, seps in best:
            s = max(seps, key=lambda x: x.auc)
            item = SortItem(self.params, [s.label, f"{s.auc:.3f}"])
            item.setData(0, UID, uid)
            item.setToolTip(0, s.help)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if uid in ticked else QtCore.Qt.CheckState.Unchecked)
            item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._filling = False
        self._filter()

    def _filter(self, *_args) -> None:
        text = self.search.text().strip().lower()
        for i in range(self.params.topLevelItemCount()):
            item = self.params.topLevelItem(i)
            item.setHidden(bool(text) and text not in item.text(0).lower())

    def _kind_changed(self, *_args) -> None:
        if self.win.tset is not None and self.win.data is not None:
            self._fill_params()

    def _ticked(self, item, column) -> None:
        if self._filling or column != 0:
            return
        uid = item.data(0, UID)
        on = item.checkState(0) == QtCore.Qt.CheckState.Checked
        if self._kind() == "ranges":
            clf = self._built("ranges") or cl.Classifier("ranges", self.win.tset.group, list(self.win.data.grades))
            if on:
                self._add_conditions(clf, uid)
            else:
                for rule in clf.rules:
                    rule.conditions = [c for c in rule.conditions if c.uid != uid]
                clf.rules = [r for r in clf.rules if r.conditions]
            self._store("ranges", self._joint(clf) if clf.rules else None)       # the rule changed: together again
            self._show_ranges()
        else:
            kind, key = self._kind(), PARAMETERS[self._kind()]
            uids = [u for u in self.win.tset.built.get(key, []) if u != uid] + ([uid] if on else [])
            self._store(kind, self._built(kind), **{key: uids})
            if kind == "score":
                self._fit_score()
            elif kind == "logistic":                                  # quick: fitted at once
                self._fit_logistic()
            else:                                                     # grown on Train only (it takes a while)
                self._show_learned(kind)
        self._validate()

    def selection(self, kind: str) -> list[str]:
        """The parameters a classifier is made from now (Ranges: those in its conditions)."""
        if kind == "ranges":
            clf = self._built("ranges")
            return clf.parameters if clf else []
        return list(self.win.tset.built.get(PARAMETERS[kind], [])) if self.win.tset else []

    def use_parameters(self, kind: str, uids: list[str]) -> list[str]:
        """Make a classifier from these parameters (sent from Training › Parameters), replacing its selection;
        Build then shows it. Ranges: each a condition at its best cut-off; Score: its points fitted; Trained
        classifier: ticked (grown on Train). Returns the parameters left out (no values in these projects)."""
        if not self.separations:
            self.refresh()
        usable = [u for u in uids if u in self.separations]
        if kind == "ranges":
            clf = cl.Classifier("ranges", self.win.tset.group, list(self.win.data.grades))
            for uid in usable:
                self._add_conditions(clf, uid)
            self._store("ranges", self._joint(clf) if clf.rules else None)
        else:
            self._store(kind, self._built(kind), **{PARAMETERS[kind]: usable})
            if kind == "score":
                self._fit_score()
            elif kind == "logistic":
                self._fit_logistic()
        self.kinds.setCurrentIndex(list(KINDS).index(kind))
        self._fill_params()
        self._show_ranges()
        self._show_score()
        self._show_logistic()
        self._show_forest()
        self._show_boosting()
        self._validate()
        return [u for u in uids if u not in usable]

    # ------------------------------------------------------------ ranges
    def _add_conditions(self, clf: cl.Classifier, uid: str) -> None:
        """A condition at the best cut-off, for each grade above the lowest (against the grade below)."""
        combine = self.combine.currentData() or "all"
        for s in self.separations.get(uid, []):
            if not math.isfinite(s.cutoff):
                continue
            rule = next((r for r in clf.rules if r.grade == s.higher), None)
            if rule is None:
                rule = cl.Rule(s.higher, [], combine)
                clf.rules.append(rule)
            rule.conditions = [c for c in rule.conditions if c.uid != uid] + [
                cl.Condition(uid, ">=" if s.direction == "higher" else "<=", s.cutoff, s.label)]
        clf.rules.sort(key=lambda r: clf.grades.index(r.grade))

    def _combine_changed(self, *_args) -> None:
        """AND ↔ OR: the rules change, so their cut-offs are found together again."""
        if self._filling or self.win.tset is None:
            return
        self._ranges_edited()
        clf = self._built("ranges")
        if clf is not None:
            self._store("ranges", self._joint(clf))
            self._show_ranges()
            self._validate()

    def _joint(self, clf: cl.Classifier) -> cl.Classifier:
        """The Ranges' cut-offs found together (``classifiers.fit_ranges_jointly``), on what classifiers are made
        from."""
        return cl.fit_ranges_jointly(clf, self.fit_data, self.fit_rows) if clf.rules else clf

    def _reset_ranges(self) -> None:
        clf = self._built("ranges")
        if clf is None:
            return
        self._store("ranges", self._joint(clf))
        self._show_ranges()
        self._validate()

    def _show_ranges(self) -> None:
        clf = self._built("ranges")
        self._filling = True
        self.conditions.setRowCount(0)
        if clf is not None and clf.rules:
            self.combine.setCurrentIndex(max(0, self.combine.findData(clf.rules[0].combine)))
            for rule in clf.rules:
                for cond in rule.conditions:
                    self._condition_row(self.conditions, rule.grade, cond)
        many = [r for r in (clf.rules if clf is not None else []) if r.combine == "all" and len(r.conditions) >= 3]
        self.ranges_warn.setText(
            "".join(f"⚠ {r.grade}: AND over {len(r.conditions)} conditions — an object must meet every one, so each "
                    f"condition added calls fewer objects {r.grade}. Two or three conditions, or OR, usually do "
                    f"better (Best cut-offs finds them together). " for r in many))
        self.ranges_warn.setVisible(bool(many))
        self._filling = False

    def _condition_row(self, table, grade: str | None, cond: cl.Condition, points: int | None = None) -> None:
        r = table.rowCount()
        table.insertRow(r)
        cells = ([grade] if grade is not None else []) + [cond.label or cond.uid]
        for c, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setData(UID, (grade, cond.uid))
            table.setItem(r, c, item)
        c0 = len(cells)
        op = QtWidgets.QComboBox()
        op.addItem("≥", ">=")
        op.addItem("≤", "<=")
        op.setCurrentIndex(0 if cond.op == ">=" else 1)
        op.currentIndexChanged.connect(self._edited_from(table))
        table.setCellWidget(r, c0, op)
        value = QtWidgets.QLineEdit(f"{cond.value:.6g}")
        value.setValidator(_number_validator(value))
        value.editingFinished.connect(self._edited_from(table))
        table.setCellWidget(r, c0 + 1, value)
        if points is None:                                          # ranges: this condition's own sensitivity
            sens, spec = self._condition_quality(grade, cond)
            for k, v in enumerate((sens, spec)):
                item = QtWidgets.QTableWidgetItem(_pct(v))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c0 + 2 + k, item)
        else:
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 20)
            spin.setValue(points)
            spin.valueChanged.connect(self._edited_from(table))
            table.setCellWidget(r, c0 + 2, spin)

    def _edited_from(self, table):
        return (lambda *_a: self._ranges_edited()) if table is self.conditions else (lambda *_a: self._score_edited())

    def _condition_quality(self, grade: str, cond: cl.Condition) -> tuple[float, float]:
        """Sensitivity and specificity of one condition, for ``grade`` against the grade below (training)."""
        data = self.fit_data
        k = data.grades.index(grade)
        j = next(i for i, c in enumerate(data.columns) if c[0] == cond.uid)
        rows_hi, rows_lo = self.fit_rows & (data.grade == k), self.fit_rows & (data.grade == k - 1)
        hi, lo = data.values[rows_hi, j], data.values[rows_lo, j]
        hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
        if not len(hi) or not len(lo):
            return math.nan, math.nan
        sens = float(np.mean([bool(cond.holds(v)) for v in hi]))
        spec = float(np.mean([not cond.holds(v) for v in lo]))
        return sens, spec

    def _read_conditions(self, table, with_grade: bool) -> list:
        out = []
        c0 = 2 if with_grade else 1
        for r in range(table.rowCount()):
            grade, uid = table.item(r, 0).data(UID)
            op = table.cellWidget(r, c0).currentData()
            try:
                value = float(table.cellWidget(r, c0 + 1).text().replace(",", "."))
            except ValueError:
                value = math.nan
            label = table.item(r, c0 - 1).text()
            points = table.cellWidget(r, c0 + 2).value() if not with_grade else None
            out.append((grade, cl.Condition(uid, op, value, label), points))
        return out

    def _ranges_edited(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        clf = cl.Classifier("ranges", self.win.tset.group, list(self.win.data.grades))
        combine = self.combine.currentData() or "all"
        for grade, cond, _p in self._read_conditions(self.conditions, True):
            if not math.isfinite(cond.value):
                continue
            rule = next((r for r in clf.rules if r.grade == grade), None)
            if rule is None:
                rule = cl.Rule(grade, [], combine)
                clf.rules.append(rule)
            rule.conditions.append(cond)
        self._store("ranges", clf if clf.rules else None)
        self._show_ranges()
        self._validate()

    # ------------------------------------------------------------ score
    def _fit_score(self) -> None:
        uids = [u for u in self.win.tset.built.get("score_parameters", []) if u in self.separations]
        if not uids:
            self._store("score", None)
        else:
            try:
                clf = cl.fit_score(self.fit_data, self.fit_rows, uids, self.win.tset.group,
                                   weigh=self._prep().project_weights)
            except ValueError as exc:
                self.win.statusBar().showMessage(f"⚠️ {exc}", 8000)
                return
            self._store("score", clf)
        self._show_score()
        self._validate()

    def _show_score(self) -> None:
        clf = self._built("score")
        self._filling = True
        self.criteria.setRowCount(0)
        while self.thresholds_row.count():
            w = self.thresholds_row.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.thresholds = []
        if clf is not None:
            for c in clf.criteria:
                self._condition_row(self.criteria, None, c.condition, c.points)
            dropped = [u for u in self.win.tset.built.get("score_parameters", [])
                       if u in self.separations and u not in clf.parameters]
            for uid in dropped:
                s = self.separations[uid][0]
                r = self.criteria.rowCount()
                self.criteria.insertRow(r)
                for c, text in ((0, s.label), (4, "no points: adds nothing beyond the others")):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                    self.criteria.setItem(r, c, item)
            for grade, t in zip(clf.grades[1:], clf.thresholds):
                self.thresholds_row.addWidget(QtWidgets.QLabel(f"{grade} from"))
                spin = QtWidgets.QSpinBox()
                spin.setRange(0, 200)
                spin.setValue(int(t))
                spin.setSuffix(" points")
                spin.valueChanged.connect(self._score_edited)
                self.thresholds_row.addWidget(spin)
                self.thresholds.append(spin)
        self._filling = False

    def _score_edited(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        old = self._built("score")
        criteria = []
        for r in range(self.criteria.rowCount()):
            if self.criteria.cellWidget(r, 1) is None:                   # a parameter given no points
                continue
            uid = self.criteria.item(r, 0).data(UID)[1]
            op = self.criteria.cellWidget(r, 1).currentData()
            try:
                value = float(self.criteria.cellWidget(r, 2).text().replace(",", "."))
            except ValueError:
                continue
            criteria.append(cl.Criterion(cl.Condition(uid, op, value, self.criteria.item(r, 0).text()),
                                         self.criteria.cellWidget(r, 3).value()))
        thresholds = [s.value() for s in self.thresholds]
        clf = cl.Classifier("score", self.win.tset.group, list(self.win.data.grades), criteria=criteria,
                            thresholds=thresholds, note=old.note if old else "")
        self._store("score", clf)
        self._validate()

    # ------------------------------------------------------------ trained classifier
    def _tick_all(self) -> None:
        self._store("forest", self._built("forest"), forest_parameters=list(self.separations))
        if self._kind() == "forest":
            self._fill_params()
        self._show_forest()

    def train_forest(self) -> None:
        """Grow the random forest on the training projects, in the background."""
        self.train_learned("forest")

    def train_learned(self, kind: str) -> None:
        """Grow a random forest or gradient boosting on the training projects, in the background."""
        tset, data, fit_data = self.win.tset, self.data, self.fit_data
        uids = [u for u in tset.built.get(PARAMETERS[kind], []) if u in self.separations]
        if not uids:
            self.win.statusBar().showMessage("ℹ️ Tick the parameters to train on" + (" (or All parameters)" if
                                                                                      kind == "forest" else ""), 8000)
            return
        if kind == "forest":
            settings = {"trees": self.n_trees.value(), "min_leaf": self.min_leaf.value(),
                        "balanced": self.balanced.isChecked()}
            fit = cl.fit_forest
        else:
            settings = {"stages": self.n_stages.value(), "depth": self.depth.value(),
                        "learning_rate": self.rate.value()}
            fit = cl.fit_boosting
        train, group, seed = self.fit_rows, tset.group, tset.seed
        weigh = self._prep().project_weights
        path = self.win.path

        def done(clf) -> None:
            if self.win.path != path or self.data is not data:
                return                                                 # another set or new measurements
            self._store(kind, clf, **{f"{kind}_settings": settings})
            self._show_learned(kind)
            self._validate()
            self.win.statusBar().showMessage(f"✅ Trained: {KINDS[kind]} on {len(uids)} parameters, "
                                             f"{clf.info['objects']} objects", 8000)

        def failed(exc) -> None:
            QtWidgets.QMessageBox.warning(self, "Train", f"Training stopped:\n\n{exc}")

        self.win.tasks.run(f"Training {KINDS[kind].lower()} on {len(uids)} parameters…",
                           lambda _progress, _cancelled: fit(fit_data, train, uids, group, seed=seed, weigh=weigh,
                                                                 **settings),
                           done, failed)

    def _show_learned(self, kind: str) -> None:
        (self._show_forest if kind == "forest" else self._show_boosting)()

    # ------------------------------------------------------------ logistic regression
    def _fit_logistic(self) -> None:
        tset = self.win.tset
        if tset is None or self.win.data is None:
            return
        uids = [u for u in tset.built.get("logistic_parameters", []) if u in self.separations]
        settings = {"strength": self.strength.value(), "balanced": self.log_balanced.isChecked()}
        if not uids:
            self._store("logistic", None, logistic_settings=settings)
        else:
            try:
                clf = cl.fit_logistic(self.fit_data, self.fit_rows, uids, tset.group, **settings,
                                      weigh=self._prep().project_weights)
            except (ValueError, ImportError) as exc:
                self.win.statusBar().showMessage(f"⚠️ {exc}", 8000)
                return
            self._store("logistic", clf, logistic_settings=settings)
        self._show_logistic()
        self._validate()

    def _show_logistic(self) -> None:
        tset = self.win.tset
        st = tset.built.get("logistic_settings", {})
        self.strength.blockSignals(True)
        self.log_balanced.blockSignals(True)
        self.strength.setValue(float(st.get("strength", self.strength.value())))
        self.log_balanced.setChecked(bool(st.get("balanced", self.log_balanced.isChecked())))
        self.strength.blockSignals(False)
        self.log_balanced.blockSignals(False)
        clear_tree(self.log_weights)
        clf = self._built("logistic")
        if clf is None:
            self.log_state.setText("Tick the parameters: the regression is fitted on the training projects at once.")
            return
        two = len(clf.weights) == 1
        self.log_state.setText(f"Fitted on {clf.info.get('objects')} objects and {len(clf.features)} parameters"
                               + (" (one row of weights per grade: the weight shown is the largest)" if not two else
                                  f" — the log-odds of {clf.grades[-1]}") + ".")
        W = np.asarray(clf.weights)
        top = max(float(np.abs(W).max()), 1e-12)
        for j, (uid, label) in enumerate(clf.features):
            w = float(W[0, j]) if two else float(W[np.argmax(np.abs(W[:, j])), j])
            item = SortItem(self.log_weights, [label, f"{w:+.3f}", f"{math.exp(w):.2f}" if two else "—",
                                               "█" * max(1, int(round(25 * abs(w) / top)))])
            item.setData(0, UID, uid)
            for c in (1, 2):
                item.setTextAlignment(c, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def _show_boosting(self) -> None:
        tset = self.win.tset
        st = tset.built.get("boosting_settings", {})
        self.n_stages.setValue(int(st.get("stages", self.n_stages.value())))
        self.depth.setValue(int(st.get("depth", self.depth.value())))
        self.rate.setValue(float(st.get("learning_rate", self.rate.value())))
        clear_tree(self.boost_importance)
        clf = self._built("boosting")
        ticked = [u for u in tset.built.get("boosting_parameters", []) if u in self.separations]
        if clf is None:
            self.boost_state.setText(f"{len(ticked)} parameter(s) ticked. Not trained yet: Train grows the trees on "
                                     "the training projects.")
            return
        info = clf.info
        text = (f"Trained on {info.get('objects')} objects and {len(clf.features)} parameters: {info.get('stages')} "
                f"trees of depth {info.get('depth')}, learning rate {info.get('learning_rate')}.")
        if sorted(ticked) != sorted(u for u, _l in clf.features):
            text += " <b>The ticked parameters changed: Train again.</b>"
        self.boost_state.setText(text)
        top = max(clf.importance) if clf.importance else 1.0
        for (_uid, label), imp in sorted(zip(clf.features, clf.importance), key=lambda x: -x[1]):
            item = SortItem(self.boost_importance, [label, f"{100 * imp:.1f} %",
                                                    "█" * max(1, int(round(30 * imp / max(top, 1e-12))))])
            item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def _show_forest(self) -> None:
        tset = self.win.tset
        st = tset.built.get("forest_settings", {})
        for box, key in ((self.n_trees, "trees"), (self.min_leaf, "min_leaf")):
            box.setValue(int(st.get(key, box.value())))
        self.balanced.setChecked(bool(st.get("balanced", self.balanced.isChecked())))
        clear_tree(self.importance)
        clf = self._built("forest")
        ticked = [u for u in tset.built.get("forest_parameters", []) if u in self.separations]
        if clf is None:
            self.forest_state.setText(f"{len(ticked)} parameter(s) ticked. Not trained yet: Train grows the forest "
                                      "on the training projects.")
            return
        info = clf.info
        text = (f"Trained on {info.get('objects')} objects and {len(clf.features)} parameters: {info.get('trees')} "
                f"trees, at least {info.get('min_leaf')} objects per leaf"
                + (", grades weighing equally" if info.get("balanced") else "")
                + (". Out-of-bag accuracy not shown: augmented copies of an object let the trees that did not see "
                   "it see its copies, so it would flatter (cross-validation by project is the measure)."
                   if pp.Prep.from_dict(clf.prep).augments else
                   f". Out-of-bag accuracy {_pct(info.get('oob_accuracy', math.nan))} (each object judged by the "
                   "trees that did not see it — an internal estimate; the evaluation projects are the real test)."))
        if sorted(ticked) != sorted(u for u, _l in clf.features):
            text += " <b>The ticked parameters changed: Train again.</b>"
        self.forest_state.setText(text)
        top = max(clf.importance) if clf.importance else 1.0
        for (uid, label), imp in sorted(zip(clf.features, clf.importance), key=lambda x: -x[1]):
            item = SortItem(self.importance, [label, f"{100 * imp:.1f} %",
                                                               "█" * max(1, int(round(30 * imp / top)))])
            item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    # ------------------------------------------------------------ validation
    def classifier(self, kind: str | None = None) -> cl.Classifier | None:
        """The classifier of a kind (default: the tab shown), as built."""
        return self._built(kind or self._kind()) if (kind or self._kind()) in KINDS else None

    def _cv_of(self, kind: str) -> va.CrossValidation | None:
        """The cross-validation of a kind, if it checked the classifier as it is now."""
        cv, clf = self.cv.get(kind), self._built(kind)
        return cv if cv is not None and clf is not None and cv.signature == clf.to_dict() else None

    def _validate(self) -> None:
        clear_tree(self.compare)
        data = self.data
        top = len(data.grades) - 1
        shown = self._kind()
        text = ""
        for kind, name in KINDS.items():
            clf = self._built(kind)
            if clf is None:
                SortItem(self.compare, [name, "not built yet" if data.ordered or kind not in SCALE
                                        else "for ordered grades only"])
                continue
            tr = cl.evaluate(clf, data, self.train, intervals=False)
            ev = cl.evaluate(clf, data, self.held)
            cv = self._cv_of(kind)
            cva = cv.agreement if cv else None
            lo, _mean, hi = cv.spread() if cv else (math.nan, math.nan, math.nan)
            item = SortItem(self.compare, [
                name, _pct(tr.accuracy), _ci(ev.accuracy, ev.accuracy_ci, ev) if ev.n else "—", _f(ev.kappa),
                _ci(ev.sensitivity[top], ev.sensitivity_ci[top], ev) if ev.n else "—",
                _ci(ev.specificity[top], ev.specificity_ci[top], ev) if ev.n else "—",
                _ci(cva.accuracy, cva.accuracy_ci, cva) if cva else ("out of date" if kind in self.cv else "—"),
                _f(cva.kappa) if cva else "", f"{_pct(lo)} – {_pct(hi)}" if cv else ""])
            for c in (2, 4, 5):
                item.setToolTip(c, f"{ev.n} objects of {ev.projects} evaluation project(s)\n{_how(ev)}")
            if cv:
                item.setToolTip(6, f"{cva.n} objects, {len(cv.folds)} folds; {_how(cva)}:\n" + "\n".join(
                    f"{names}: {_pct(acc)} of {n}" for names, n, acc in cv.folds)
                    + ("\nNot made: " + "; ".join(cv.skipped) if cv.skipped else ""))
            for c in range(1, 9):
                item.setTextAlignment(c, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if kind == shown:
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
                second = (_matrix(f"Cross-validation ({len(cv.folds)} folds)", cva) if cv
                          else _matrix("Training projects", tr))
                text = (f"<b>{name}</b> — rows: hand grade, columns: predicted grade."
                        f"<table><tr><td>"
                        + (_matrix("Evaluation projects", ev) if ev.n else "<i>No evaluation objects.</i>")
                        + f"</td><td width='40'></td><td>{second}</td></tr></table>")
        self.matrices.setText(text)
        self.save_btn.setEnabled(self.classifier() is not None)
        self.report_btn.setEnabled(any(self._built(k) is not None for k in KINDS))
        self._show_coverage()
        n_proj = len(set(data.project[self.train].tolist()))
        self.cv_note.setText(
            f"By project on the {n_proj} training projects: "
            + ("each project left out once (leave-one-project-out)." if n_proj <= va.MAX_FOLDS
               else f"{va.MAX_FOLDS} folds of whole projects.")
            + " Hand-typed cut-offs are found again in each fold: it judges how the classifier is made."
            + self._tried_note())
        self.cv_btn.setEnabled(n_proj >= 2 and any(self._built(k) for k in KINDS))
        self._show_mistakes()

    def _show_coverage(self) -> None:
        """Say which classifiers leave training objects out (an incomplete shape, a parameter that needs the whole
        object) or give every training object one grade."""
        from modalytics.parameters.registry import ParameterLibrary
        data, n_all = self.data, int(self.train.sum())
        specs = ParameterLibrary().specs
        labels = {c[0]: c[1] for c in data.columns}
        lines, left_out, needs = [], [], set()
        for kind in KINDS:
            clf = self._built(kind)
            if clf is None:
                continue
            used = cl.usable(data, self.train, clf.parameters)
            n = int(used.sum())
            if n < 0.9 * n_all:
                left_out.append((KINDS[kind], n))
                needs |= {u for u in clf.parameters if (s := specs.get(u)) is not None and s.whole_object}
            given = cl.predict(clf, data, used)
            if n and len(set(given.tolist())) == 1 and len(set(data.grade[used].tolist())) > 1:
                lines.append(f"⚠ {KINDS[kind]} gives every training object one grade ({data.grades[int(given[0])]}): "
                             f"it does not separate them.")
        if left_out:
            lines.insert(0, f"⚠ {', '.join(f'{name} grades {n} of {n_all}' for name, n in left_out)} training "
                            f"objects: those with an incomplete shape (cut by the image edge, or marked partial) are "
                            f"left out, since it reads parameters that need the whole object "
                            f"({', '.join(sorted(labels.get(u, u) for u in needs))}). They are left ungraded in "
                            f"projects too. Parameters of the pixels only (phase, texture) grade every object.")
        self.coverage.setText("<br>".join(lines))
        self.coverage.setVisible(bool(lines))

    def _remember_selection(self, clf: cl.Classifier | None) -> None:
        """Keep the parameters a classifier was made from, among those tried on this set (``_tried_note``)."""
        if clf is None or not clf.parameters:
            return
        tried = self.win.tset.built.setdefault("tried", [])
        key = ",".join(sorted(clf.parameters))
        if key not in tried:
            tried.append(key)
            del tried[:-500]

    def _tried_note(self) -> str:
        """A warning once several parameter selections were tried on these projects."""
        n = len(self.win.tset.built.get("tried", [])) if self.win.tset is not None else 0
        if n < 3:
            return ""
        return (f" <b>⚠ {n} different parameter selections were tried on these projects:</b> each was judged on them, "
                f"so the one kept looks better here than it will on new projects (choosing on them is part of the "
                f"fit). The final checks of Write report — projects not used to choose — are the honest figure.")

    def cross_validate(self) -> None:
        """Cross-validate every classifier built, by project, in the background."""
        tset, data, acquired = self.win.tset, self.data, self._acquired_n
        made = {k: self._built(k) for k in KINDS if self._built(k) is not None}
        if not made:
            return
        train, seed = self.train, tset.seed
        uids = list(tset.built.get("score_parameters", []))

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            out = {}
            for i, (kind, clf) in enumerate(made.items()):
                try:
                    out[kind] = va.cross_validate(
                        clf, data, train, uids if kind == "score" else None, seed,
                        lambda f, i=i: progress((i + f) / len(made)), cancelled, acquired)
                except va.Cancelled:
                    raise Cancelled() from None
            return out

        def done(found) -> None:
            if self.data is not data:
                return
            self.cv.update(found)
            self._validate()
            self.win.statusBar().showMessage("✅ Cross-validated: " + ", ".join(
                f"{KINDS[k]} {_pct(v.agreement.accuracy)}" for k, v in found.items()), 8000)

        self.win.tasks.run("Cross-validating by project…", job, done,
                           lambda exc: QtWidgets.QMessageBox.warning(self, "Cross-validate", str(exc)))

    # ------------------------------------------------------------ mistakes
    def _show_mistakes(self, *_args) -> None:
        clear_tree(self.wrong)
        clf, data = self.classifier(), self.data
        if clf is None or data is None:
            self.wrong_note.setText("")
            return
        if self.wrong_source.currentData() == "cv":
            cv = self._cv_of(clf.kind)
            if cv is None:
                self.wrong_note.setText("Cross-validate first.")
                return
            rows, found = cv.rows, cv.details
        else:
            rows = self.held
            found = va.details(clf, data, rows)
        wrong = va.mistakes(data, rows, found)
        self.wrong_note.setText(f"{len(wrong)} of {int(rows.sum())} objects.")
        tset = self.win.tset
        for r, hand, (grade, score, why) in wrong:
            image, klass, number, point = data.where[r]
            item = SortItem(self.wrong, [
                data.projects[data.project[r]], image, f"{klass} #{number}", hand, grade,
                f"{score:.2f}" if clf.learned else f"{score:g}", why])
            item.setToolTip(6, why)
            item.setData(0, UID, (tset.projects[data.project[r]].path, image, tset.group, *point))

    def _open_mistake(self, item, _col=None) -> None:
        path, image, group, x, y = item.data(0, UID)
        self.show_object.emit(path, image, group, float(x), float(y))

    # ------------------------------------------------------------ saving to the profile
    def _show_in_profile(self) -> None:
        tset = self.win.tset
        try:
            prof = self.win.profiles.get(tset.profile_id)
        except Exception:                                            # noqa: BLE001 — removed or unreadable
            self.in_profile.setText(f"The profile '{tset.profile.name}' is not in your library: Save to profile "
                                    "is not possible.")
            self.save_btn.setEnabled(False)
            return
        c = prof.classifier_for(tset.group)
        self.in_profile.setText(f"{prof.name} › {tset.group} is classified by: "
                                + (cl.describe(c) if c else "nothing yet"))
        self.in_profile.setToolTip(c.note if c else "")

    # ------------------------------------------------------------ preparation
    def _prep(self) -> pp.Prep:
        """The set's preparation (``training.prep``), its seed the set's."""
        tset = self.win.tset
        prep = pp.Prep.from_dict(tset.built.get("prep") if tset is not None else None)
        prep.seed = tset.seed if tset is not None else 1
        return prep

    def _prepare(self) -> None:
        """The data as set: normalised (judged on), and with the augmented rows (made from)."""
        tset, raw, prep = self.win.tset, self.win.data, self._prep()
        if getattr(self, "_norm_key", None) != (id(raw), prep.normalize):     # kept while unchanged: background
            self.data, self._norm_key = pp.normalized(raw, prep.normalize), (id(raw), prep.normalize)   # jobs
        acq = aq.Acquisition.from_dict(prep.acquisition)                       # check it is still the same
        acq.seed = prep.seed
        key = (self.win.path, id(raw), acq.key(), tuple(sorted(ds.training_projects(tset, raw))))
        if not acq.copies:
            self._acquired_n = None
        elif self._acquired_key == key and self.acquired is not None:
            if getattr(self, "_acq_norm_key", None) != (id(self.acquired), prep.normalize):
                self._acquired_n = pp.normalized(self.acquired, prep.normalize)
                self._acq_norm_key = (id(self.acquired), prep.normalize)
        else:
            self._acquired_n = None
            self._measure_acquired(acq, key)                     # in the background; then shown again
        fit_key = (id(self.data), id(self._acquired_n), self.train.tobytes(), str(prep.to_dict()))
        if getattr(self, "_fit_key", None) != fit_key:
            self.fit_data, self.fit_rows = pp.augmented(self.data, self.train, prep, self._acquired_n)
            self._fit_key = fit_key
        self._filling = True
        self.normalize.setCurrentIndex(max(0, self.normalize.findData(prep.normalize)))
        self.weigh.setCurrentIndex(max(0, self.weigh.findData(prep.project_weights)))
        self._filling = False
        waiting = acq.copies and self._acquired_n is None
        self.prep_note.setText(("Prepared: " + prep.describe()) if prep.describe() != "none" else
                               "No normalisation, no augmentation.")
        self.prep_note.setToolTip("Recorded with every classifier and in the report's Methods; augmented rows only "
                                  "make the classifiers, which are always judged on real objects."
                                  + ("\nAcquisition copies: being measured…" if waiting else ""))

    def _measure_acquired(self, acq: aq.Acquisition, key) -> None:
        """Measure the training projects' acquisition copies in the background (cached), then rebuild."""
        tset, raw = self.win.tset, self.win.data
        if getattr(self, "_measuring", None) == key or self.win.tasks.busy:
            return
        self._measuring = key
        projects = ds.training_projects(tset, raw)

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            try:
                return aq.measure(tset, projects, acq, progress, cancelled)
            except ds.Cancelled:
                raise Cancelled() from None

        def done(got) -> None:
            self._measuring = None
            if self.win.data is not raw:
                return
            self.acquired, self._acquired_key = got, key
            self.cv = {}
            self._rebuild("acquisition copies measured")

        def failed(exc) -> None:
            self._measuring = None
            QtWidgets.QMessageBox.warning(self, "Acquisition copies", f"Measuring the copies stopped:\n\n{exc}")

        def cancelled() -> None:
            self._measuring = None

        self.win.tasks.run(f"Measuring {acq.copies} acquisition cop{'y' if acq.copies == 1 else 'ies'} of "
                           f"{len(projects)} training project(s)…", job, done, failed, on_cancel=cancelled)

    def _set_prep(self, prep: pp.Prep, why: str) -> None:
        """Keep a new preparation and make the classifiers built again with it (asked first if any)."""
        built = [k for k in KINDS if self._built(k) is not None]
        if built and QtWidgets.QMessageBox.question(
                self, "Preparation", f"{why}: the classifiers built ({', '.join(KINDS[k] for k in built)}) are made "
                "again the same way on the prepared objects — hand-typed cut-offs are found again. Go on?"
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            self._prepare()                                       # the boxes back as they were
            return
        try:                                                      # prepared first: kept only if it can be
            pp.normalized(self.win.data, prep.normalize)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Preparation", f"The objects could not be prepared so ({exc}); "
                                                               f"nothing was changed.")
            self._prepare()
            return
        self.win.tset.built["prep"] = {k: v for k, v in prep.to_dict().items() if k != "seed"}
        self.win._save()
        self.cv = {}
        self._rebuild(why)

    def _stale(self) -> list[str]:
        """The classifiers built that were not made with the set's preparation (e.g. kept from before it changed):
        they are made again."""
        prep = self._prep()
        want = {k: v for k, v in (prep.to_dict() if prep != pp.Prep(seed=prep.seed) else {}).items() if k != "seed"}
        return [k for k in KINDS if (c := self._built(k)) is not None
                and {k2: v for k2, v in (c.prep or {}).items() if k2 != "seed"} != want]

    def _rebuild(self, why: str) -> None:
        """Prepare the data again and make every classifier built again the same way on it."""
        self._prepare()
        if getattr(self, "_measuring", None) is not None:
            return                                                # made again once the copies are measured
        prep, seed = self._prep(), self.win.tset.seed
        built = {k: c for k in KINDS if (c := self._built(k)) is not None}
        if not built:
            self.refresh()
            return
        data, train, acquired = self.data, self.train, self._acquired_n
        uids = list(self.win.tset.built.get("score_parameters", []))

        def job(progress, _cancelled):
            out = {}
            for i, (kind, clf) in enumerate(built.items()):
                clf.prep = prep.to_dict()
                try:
                    out[kind] = va.refit(clf, data, train, uids if kind == "score" else None, seed, acquired)
                except ValueError as exc:
                    out[kind] = exc
                progress((i + 1) / len(built))
            return out

        def done(made) -> None:
            if self.data is not data:
                return
            problems = []
            for kind, clf in made.items():
                if isinstance(clf, Exception):
                    problems.append(f"{KINDS[kind]}: {clf}")
                else:
                    self._store(kind, clf)
            self.refresh()
            self.win.statusBar().showMessage(f"✅ {why[:1].upper() + why[1:]}: {len(made) - len(problems)} "
                                             f"classifier(s) made again" + (f" (⚠️ {'; '.join(problems)})"
                                                                           if problems else ""), 10000)

        self.win.tasks.run(f"Making {len(built)} classifier(s) again on the prepared objects…", job, done,
                           lambda exc: QtWidgets.QMessageBox.warning(self, "Preparation", str(exc)))

    def _normalize_changed(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        prep = self._prep()
        prep.normalize = self.normalize.currentData()
        if prep.normalize != "none":
            single, n = self._one_grade_projects()
            if single and len(single) * 3 >= n and QtWidgets.QMessageBox.question(
                    self, "Normalise per project",
                    f"{len(single)} of the {n} training projects hold objects of one grade only "
                    f"({', '.join(single[:8])}{' …' if len(single) > 8 else ''}). Read relative to their own median, "
                    f"their objects lose exactly that grade's difference — normalisation per project suits projects "
                    f"that hold several grades.\n\nNormalise anyway?") != QtWidgets.QMessageBox.StandardButton.Yes:
                self._prepare()                                       # the box back as it was
                return
        self._set_prep(prep, f"normalisation: {pp.NORMALIZE[prep.normalize].lower()}")

    def _one_grade_projects(self) -> tuple[list[str], int]:
        """(the training projects — with enough complete objects to be a reference — whose objects are all of one
        grade, how many training projects have enough)."""
        data = self.win.data
        complete = ~data.partial_rows if data.partial_rows is not None and len(data.partial_rows) == len(data.grade) \
            else np.ones(len(data.grade), bool)
        single, n = [], 0
        for p in np.unique(data.project[self.train]):
            mine = self.train & complete & (data.project == p)
            if mine.sum() < pp.MIN_REFERENCE:
                continue
            n += 1
            if len(set(data.grade[mine].tolist())) == 1:
                single.append(data.projects[p])
        return single, n

    def _weigh_changed(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        prep = self._prep()
        prep.project_weights = self.weigh.currentData()
        self._set_prep(prep, f"projects weighted: {pp.WEIGHTS[prep.project_weights].lower()}")

    def _edit_augmentation(self) -> None:
        """The augmentation settings, in a dialog."""
        prep = self._prep()
        acq = aq.Acquisition.from_dict(prep.acquisition)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Augmentation")
        form = QtWidgets.QFormLayout(dlg)
        form.addRow(muted("More rows to make the classifiers from, from the training projects only (never the "
                          "evaluation or test projects; in cross-validation, from the folds trained on). The "
                          "classifiers are always judged on real objects."))

        def spin(lo, hi, value, step, suffix="", decimals=2, tip=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(value)
            s.setSuffix(suffix)
            s.setToolTip(tip)
            return s
        head = QtWidgets.QLabel("<b>Acquisition copies</b> — each training project measured again as if acquired "
                                "a little differently (one draw per project and copy)")
        head.setWordWrap(True)
        form.addRow(head)
        copies = QtWidgets.QSpinBox()
        copies.setRange(0, 10)
        copies.setValue(acq.copies)
        copies.setToolTip("Copies of each training project (0: off). Each is measured: takes about as long as "
                          "Measure, per copy (cached)")
        form.addRow("Copies", copies)
        size = spin(0, 20, 100 * acq.pixel_size, 1, " %", 1, "Calibration off by up to ±this")
        form.addRow("Pixel size ±", size)
        gain = spin(0, 50, 100 * acq.gain, 1, " %", 1, "Every pixel value × a factor within ±this")
        form.addRow("Gain ±", gain)
        offset = spin(0, 2, acq.offset, 0.05, " SD", 2, "A shift of the pixel values within ±this × the image's "
                                                        "pixel SD")
        form.addRow("Offset ±", offset)
        noise = spin(0, 1, acq.noise, 0.01, " SD", 2, "Gaussian noise of SD this × the image's pixel SD")
        form.addRow("Noise", noise)
        blur = spin(0, 5, acq.blur_um, 0.1, " µm", 1, "Defocus: a Gaussian blur of σ up to this")
        form.addRow("Defocus up to", blur)
        head2 = QtWidgets.QLabel("<b>From the parameters</b>")
        form.addRow(head2)
        n_copies = QtWidgets.QSpinBox()
        n_copies.setRange(0, 10)
        n_copies.setValue(prep.noise_copies)
        n_copies.setToolTip("Copies of each training object with Gaussian noise added to every parameter (0: off)")
        form.addRow("Noisy copies", n_copies)
        n_sd = spin(0.01, 1, prep.noise_sd, 0.05, " × SD", 2, "Their noise: this × each parameter's SD over the "
                                                              "training objects")
        form.addRow("Noise", n_sd)
        smote = QtWidgets.QCheckBox("SMOTE: top up the smaller classes")
        smote.setChecked(prep.smote)
        smote.setToolTip("Synthetic objects between an object and one of its 5 nearest neighbours of the same class, "
                         "until each class has as many objects as the largest")
        form.addRow("", smote)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        self._augment_dialog = dlg                               # for scripted checks
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        acq = aq.Acquisition(copies.value(), size.value() / 100, gain.value() / 100, offset.value(), noise.value(),
                             blur.value())
        new = pp.Prep(prep.normalize, n_copies.value(), n_sd.value(), smote.isChecked(),
                      {k: v for k, v in acq.to_dict().items() if k != "seed"} if acq.copies else {}, prep.seed,
                      prep.project_weights)
        if new != prep:
            self._set_prep(new, "preparation: " + new.describe())

    def _choose_by_changed(self, *_args) -> None:
        if self.win.tset is not None:
            self.win.tset.choose_by = self.choose_by.currentData()
            self.win._save()

    def write_report(self) -> None:
        """Assess every classifier built (in the background), write the report and its results, and open it.

        1. Choosing by cross-validation: the classifiers not cross-validated as they are now are cross-validated.
        2. The test projects are measured (apart: nothing that builds or chooses sees them), one final check per
           test group.
        3. ``metrics.assess`` chooses and checks; the report and its results are written in the set's runs folder.
        """
        import json

        from modalytics import fileio
        from modalytics.report import classifiers as rep
        from modalytics.services.segmentation import Cancelled
        from modalytics.training import metrics as mt
        from modalytics.training import runs as rn
        tset, data, acquired, prep = self.win.tset, self.data, self._acquired_n, self._prep()
        built = {k: c for k in KINDS if (c := self._built(k)) is not None}
        if not built:
            QtWidgets.QMessageBox.information(self, "Write report", "Build a classifier first.")
            return
        cvs = {k: cv for k in built if (cv := self._cv_of(k)) is not None}
        missing = [k for k in built if k not in cvs] if tset.choose_by == "cross-validation" and \
            len(set(data.project[self.train].tolist())) >= 2 else []
        groups = ds.test_groups(tset)
        train, held, seed = self.train.copy(), self.held.copy(), tset.seed
        tried = len(tset.built.get("tried", []))
        uids = list(tset.built.get("score_parameters", []))
        run = rn.Run.new(rn.set_folder(tset, create=True), "classifiers", f"{tset.group} classifiers")
        self.win._save()

        def job(progress, cancelled):
            # 1. Cross-validated where it is missing
            found = {}
            for i, kind in enumerate(missing):
                try:
                    found[kind] = va.cross_validate(built[kind], data, train, uids if kind == "score" else None, seed,
                                                    lambda f, i=i: progress(0.6 * (i + f) / len(missing)), cancelled,
                                                    acquired)
                except va.Cancelled:
                    raise Cancelled() from None
            # 2. The test groups, measured apart
            finals = {}
            if groups:
                try:
                    tdata = pp.normalized(ds.gather(tset, lambda f: progress(0.6 + 0.3 * f), cancelled, test=True),
                                          prep.normalize)                  # each test project on its own
                except ds.Cancelled:
                    raise Cancelled() from None
                finals = {g: (tdata, np.isin(tdata.project, sorted(idx))) for g, idx in groups.items()}
            # 3. Chosen, checked, written
            a = mt.assess(data, built, train, held, {**cvs, **found}, choose_by=tset.choose_by, finals=finals)
            run.folder.mkdir(parents=True, exist_ok=True)
            results = dict(a, made=dt.datetime.now().isoformat(timespec="seconds"), training_set=tset.name,
                           profile=tset.profile.name, group=tset.group, grades=list(data.grades),
                           ordered=bool(data.ordered), choose_by=tset.choose_by,
                           classifiers={k: c.to_dict() for k, c in built.items()})
            fileio.write_text(run.results, json.dumps(results, indent=1, default=float))
            fileio.write_text(run.report, rep.build(a, built, tset, data, train, held, finals, tried))
            return a, found

        def done(got) -> None:
            from modalytics.ui.settings.widgets import open_in_system
            a, found = got
            if found and self.data is data:
                self.cv.update(found)
                self._validate()
            self.last_report = run
            open_in_system(str(run.report))
            self.win.statusBar().showMessage(f"✅ Report written: recommended {mt.NAMES.get(a.get('recommended'), '—')}"
                                             f" · {run.report}", 10000)

        self.win.tasks.run(f"Comparing {len(built)} classifier(s)…", job, done,
                           lambda exc: QtWidgets.QMessageBox.warning(self, "Write report", str(exc)))

    def _latest_report(self):
        """The latest classifier report of this set (written now, else the newest in its folder), or None."""
        from modalytics.training import runs as rn
        if getattr(self, "last_report", None) is not None and self.last_report.report.exists():
            return self.last_report
        tset = self.win.tset
        runs = [r for r in rn.list_runs(tset, "classifiers") if r.report.exists()] if tset is not None else []
        return runs[0] if runs else None                        # the newest first

    def _export_xlsx(self) -> None:
        import json

        from modalytics.report import xlsx
        from modalytics.ui.training.export import save_xlsx
        run = self._latest_report()
        if run is None or not run.results.exists():
            QtWidgets.QMessageBox.information(self, "XLSX", "Write report first.")
            return
        with open(run.results, encoding="utf-8") as fh:
            results = json.load(fh)
        save_xlsx(self, run.report, lambda path: xlsx.write_classifiers(results, path), "the classifiers")

    def _report_pdf(self) -> None:
        from modalytics.ui.pdf_export import report_pdf
        run = self._latest_report()
        if run is None:
            QtWidgets.QMessageBox.information(self, "PDF", "Write report first.")
            return
        report_pdf(self, run.report, status=self.win.statusBar())

    def _open_reports(self) -> None:
        from modalytics.training import runs as rn
        from modalytics.ui.settings.widgets import open_in_system
        tset = self.win.tset
        for folder in (rn.set_folder(tset) / "classifiers" if tset is not None else None, rn.root()):
            if folder is not None and folder.is_dir():
                open_in_system(folder)
                return

    def _note(self, clf: cl.Classifier) -> str:
        """Where a classifier comes from and how good it is, with the report's measures: balanced accuracy (its 95 %
        interval resampling whole projects; * over objects, with fewer than 5) and ROC AUC, on the evaluation
        projects and cross-validated."""
        from modalytics.training import metrics as mt
        data, parts = self.data, [f"training set '{self.win.tset.name}', {dt.date.today().strftime('%d %b %Y')}"]

        def said(where, m) -> str:
            lo, hi = m.get("balanced_accuracy_ci", (math.nan, math.nan))
            star = " *" if m.get("resampled") == "objects" else ""
            return (f"{where} ({m['projects']} project{'s' if m['projects'] != 1 else ''}, {m['n']} objects): "
                    f"balanced accuracy {_pct(m['balanced_accuracy'])}"
                    + (f" ({100 * lo:.0f}–{100 * hi:.0f}){star}" if math.isfinite(lo) else "")
                    + (f", AUC {m['auc']:.2f}" if math.isfinite(m.get("auc", math.nan)) else ""))
        rows = cl.usable(data, self.held, clf.parameters)
        if rows.any():
            parts.append(said("evaluation", mt.measures(data.grade[rows], cl.predict(clf, data, rows), data.grades,
                                                         data.ordered, mt.scores_of(clf, data, rows), n_boot=500,
                                                         projects=data.project[rows])))
        else:
            parts.append("not evaluated")
        cv = self._cv_of(clf.kind)
        if cv is not None and cv.rows.any():
            scores = cv.scores[cv.rows] if cv.scores is not None else None
            parts.append(said("cross-validation", mt.measures(data.grade[cv.rows], cv.predicted[cv.rows], data.grades,
                                                               data.ordered, scores, n_boot=500,
                                                               projects=data.project[cv.rows])))
        return "; ".join(parts)

    def save_to_profile(self) -> None:
        """Make the classifier shown the profile's classifier for the training set's group."""
        tset, clf = self.win.tset, self.classifier()
        if clf is None:
            return
        try:
            prof = self.win.profiles.get(tset.profile_id)
        except Exception as exc:                                     # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Save to profile", f"The profile could not be read:\n\n{exc}")
            return
        # 1. The library's profile must still have this group, these grades and these parameters
        group = next((g for g in prof.classes.groups if g.name == clf.group), None)
        problems = []
        if group is None or [s.name for s in group.subgroups] != clf.grades:
            problems.append(f"its {clf.group} grades are not {', '.join(clf.grades)} any more")
        missing = [u for u in clf.parameters if u not in prof.parameters]
        if missing:
            problems.append(f"it no longer measures {len(missing)} of the parameters used")
        if problems:
            QtWidgets.QMessageBox.warning(self, "Save to profile", f"The profile '{prof.name}' changed since this "
                                          f"training set was made: {'; '.join(problems)}.")
            return
        clf.note = self._note(clf)
        old = prof.classifier_for(clf.group)
        ask = (f"Make this {KINDS[clf.kind]} classifier the profile '{prof.name}' classifier for {clf.group}?\n\n"
               f"New: {cl.describe(clf)}\n({clf.note})"
               + (f"\n\nIt replaces: {cl.describe(old)}\n({old.note})" if old else ""))
        if QtWidgets.QMessageBox.question(self, "Save to profile", ask) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        prof.classification[clf.group] = clf.to_dict()
        self.win.profiles.save(prof, tset.profile_id)
        self._store(clf.kind, clf)
        self._show_in_profile()
        self.win.statusBar().showMessage(f"✅ Saved to {prof.name}: {clf.group} is classified by "
                                         f"{KINDS[clf.kind]}", 8000)
        self.saved.emit(tset.profile_id)


def _number_validator(parent) -> QtGui.QDoubleValidator:
    """A validator for numbers in any notation (e.g. 4.98e-05)."""
    v = QtGui.QDoubleValidator(parent)
    v.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
    v.setLocale(QtCore.QLocale.c())
    return v


def _matrix(title: str, a: cl.Agreement) -> str:
    """A confusion matrix as a small HTML table."""
    head = "".join(f"<th style='padding:0 8px'>{g}</th>" for g in a.grades)
    body = ""
    for i, g in enumerate(a.grades):
        cells = "".join(f"<td align='right' style='padding:0 8px'>{'<b>' if i == j else ''}{a.confusion[i, j]}"
                        f"{'</b>' if i == j else ''}</td>" for j in range(len(a.grades)))
        body += f"<tr><th align='left' style='padding:0 8px'>{g}</th>{cells}</tr>"
    return (f"<table><tr><td colspan='{len(a.grades) + 1}'><i>{title}</i> — accuracy "
            f"{_pct(a.accuracy)}, κ {_f(a.kappa)}</td></tr><tr><th></th>{head}</tr>{body}</table>")


def _ci(v: float, ci, a: cl.Agreement | None = None) -> str:
    """A proportion with its interval, e.g. "90 % (60–98 %)"; marked * when the interval resampled objects only
    (fewer projects than ``classifiers.MIN_PROJECTS``: likely too narrow)."""
    if v is None or not math.isfinite(v):
        return "—"
    lo, hi = ci
    star = " *" if a is not None and a.resampled == "objects" else ""
    return f"{100 * v:.0f} % ({100 * lo:.0f}–{100 * hi:.0f}){star}" if math.isfinite(lo) else _pct(v)


def _how(a: cl.Agreement) -> str:
    """How an agreement's intervals were made, in words."""
    if a.resampled == "projects":
        return f"95 % intervals resampling whole projects ({a.projects})"
    return (f"* 95 % intervals by objects (Wilson): {a.projects} project(s), fewer than {cl.MIN_PROJECTS} to resample "
            f"whole projects — likely too narrow, since objects of one project resemble each other")


def _f(v: float, digits: int = 2) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.0f} %"
