"""
Training › Parameters: which of the profile's parameters carry information about the grades.

    Measured: 512 graded objects from 16 project(s) …                                         [Measure]
    On [the training projects ▾]  Compare [objects ▾]  Name [Parameters on the training projects, …]  [Analyse]
    Analysis [2026-10-27 10:14 — … ▾]                                                [Open report] [Folder]
    Proposed: Bimodality, Solidity — …
    ┌ ☑ Parameter ┬ Verdict ┬ AUC [95 %] ┬ q ┬ Low → High … ┬ Projects ┬ Group ┬ Notes ┐
    └────────────────────────────────────────────────────────────────────────────────┘
    [Tick the proposal]  [Make the ticked the profile's key parameters]
    (the selected parameter's values per grade)

The measured objects are shared with Classifiers (Measure is here). **Analyse** runs the
relevance analysis (``training.relevance``) in the background — comparing the objects (each a
case: the default), the images or the projects (each one's class a case) — on the training projects
(for choosing: never the evaluation ones) or on the training and evaluation ones (to describe the
data), writes its report and results in the set's runs folder
(``training_runs/<set>/parameters/``) and lists it. The proposal — the best parameter of
each redundancy group that separates the grades — comes ticked; the ticked ones can be
made the profile's **key parameters** (what the project report puts forward). Explore
shows each parameter's verdict and group from the analysis shown here (on the training
projects).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.training import dataset as ds
from modalytics.training import relevance as rv
from modalytics.ui.settings.widgets import clear_tree, header_tips, list_tree, muted, open_in_system
from modalytics.ui.training.plot import DistributionPlot
from modalytics.ui.training.sorting import SortItem, sortable

UID = QtCore.Qt.ItemDataRole.UserRole
COLUMNS = ["Parameter", "Verdict", "AUC [95 %]", "Stands out", "q", "Projects", "Group", "Notes"]
COMPARE = {"objects": "objects", "images": "images", "projects": "projects"}   # what is compared (each a case)
ON = {"training": "the training projects", "all": "the training and evaluation projects"}   # test: never measured
KIND = "parameters"                                 # the analyses' folder in the set's runs folder


class ParametersPanel(QtWidgets.QWidget):
    """Parameter relevance (see the module docstring).

    Args:
        window: The ``TrainingWindow`` (its ``tset``, ``data``, ``profiles``, ``tasks``, ``measure_note``,
            ``measure_btn``).
    """

    def __init__(self, window) -> None:
        super().__init__()
        self.win = window
        self.analysis: rv.Analysis | None = None
        self.run = None
        self._runs_key = None
        self._auto_name = ""
        col = QtWidgets.QVBoxLayout(self)
        # 1. The measured objects (shared with Classifiers)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(window.measure_note, 1)
        bar.addWidget(window.measure_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        col.addLayout(bar)
        # 2. What to analyse
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("On"))
        self.on = QtWidgets.QComboBox()
        for key, text in ON.items():
            self.on.addItem(text, key)
        self.on.setToolTip("The training projects: for choosing parameters (a classifier's, the report's) — the "
                           "evaluation projects stay untouched for judging. Training and evaluation: to describe "
                           "the data (not for choosing). Test projects are never measured for classifiers.")
        self.on.currentIndexChanged.connect(self._on_changed)
        row.addWidget(self.on)
        row.addWidget(QtWidgets.QLabel("Compare"))
        self.unit = QtWidgets.QComboBox()
        for key, text in COMPARE.items():
            self.unit.addItem(text, key)
        self.unit.setToolTip("Objects: each object (gland, cell) a case — do the classes differ, in this material? "
                             "Objects of one sample may resemble each other, so the p-values may be optimistic about "
                             "new samples.\nImages: each image's class a case (the median of its objects) — for "
                             "projects holding images of several samples.\nProjects: each project's class a case — "
                             "will it hold for new samples? Needs many projects per class.")
        self.unit.currentIndexChanged.connect(self._on_changed)
        row.addWidget(self.unit)
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit()
        self.name.setToolTip("The analysis' name: its folder and its report are named after it")
        row.addWidget(self.name, 1)
        self.run_btn = QtWidgets.QPushButton("Analyse")
        self.run_btn.setToolTip("Find which parameters separate the grades, which repeat each other, and propose a "
                                "selection (in the background), then write the report")
        self.run_btn.clicked.connect(self.analyse)
        row.addWidget(self.run_btn)
        col.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Analysis"))
        self.runs = QtWidgets.QComboBox()
        self.runs.setToolTip("This training set's analyses, the newest first")
        self.runs.activated.connect(self._run_picked)
        row.addWidget(self.runs, 1)
        self.report_btn = QtWidgets.QPushButton("Open report")
        self.report_btn.clicked.connect(lambda: self.run and self.run.report.exists()
                                        and open_in_system(str(self.run.report)))
        self.pdf_btn = QtWidgets.QPushButton("PDF")
        self.pdf_btn.setToolTip("The analysis' report as a PDF (A4 across), beside it in its folder, and open it")
        self.pdf_btn.clicked.connect(self._report_pdf)
        self.folder_btn = QtWidgets.QPushButton("Folder")
        self.folder_btn.clicked.connect(self._open_folder)
        self.xlsx_btn = QtWidgets.QPushButton("XLSX…")
        self.xlsx_btn.setToolTip("The analysis as an Excel workbook: every table, each object's value of the "
                                 "parameters shown, and native charts (the AUCs, the values per class)")
        self.xlsx_btn.clicked.connect(self._export_xlsx)
        row.addWidget(self.report_btn)
        row.addWidget(self.pdf_btn)
        row.addWidget(self.xlsx_btn)
        row.addWidget(self.folder_btn)
        col.addLayout(row)
        self.note = muted("")
        self.note.setWordWrap(True)
        self.note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        col.addWidget(self.note)
        # 3. The parameters, and the selected one's values per grade
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.table = sortable(list_tree(COLUMNS))
        for c, w in enumerate((300, 170, 150, 120, 60, 70, 55)):
            self.table.setColumnWidth(c, w)
        header_tips(self.table, [
            "The profile's parameter and its unit\n★ proposed · tick the ones to keep",
            "Separates: q < 0.05 and AUC ≥ 0.7\nWeak: q < 0.05, a smaller AUC\nPromising, not confirmed: AUC ≥ 0.7, "
            "p < 0.05\n  but not after the correction (too few cases)\nNot distinguishable from chance\n"
            "Too few objects to tell",
            "Grades: of two objects of different grades, the chance\nthat the higher-grade one has the larger "
            "value\nCategories: the class that stands out, against the others\n0.5 = no relation · 1 = perfect\n"
            "[95 % interval, resampling the cases]",
            "Grades: larger or smaller with the higher grade\nCategories: the class that stands out most",
            "The p-value (grades: trend test; categories: does any\nclass differ), corrected for the number of "
            "parameters\n(Benjamini–Hochberg) · below 0.05: not chance",
            "Projects where it goes the same way\n(of those with objects of two classes)",
            "Parameters that measure the same thing\n(|Spearman ρ| ≥ 0.8): keep one per group",
            "What to look at before trusting it"])
        self.table.currentItemChanged.connect(self._show_parameter)
        self.table.itemChanged.connect(lambda *_a: self._update_buttons())     # a tick
        split.addWidget(self.table)
        below = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.plot = DistributionPlot()
        below.addWidget(self.plot)
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenLinks(False)
        self.details.setToolTip("The selected parameter per class (median and interquartile range, mean ± SD, the "
                                "Shapiro–Wilk test of normality) and every pair of classes compared")
        below.addWidget(self.details)
        below.setSizes([620, 420])
        split.addWidget(below)
        split.setSizes([420, 260])
        col.addWidget(split, 1)
        row = QtWidgets.QHBoxLayout()
        self.tick_btn = QtWidgets.QPushButton("Tick the proposal")
        self.tick_btn.setToolTip("Tick only the proposed parameters (the best of each group that separates)")
        self.tick_btn.clicked.connect(self._tick_proposal)
        self.relevant_btn = QtWidgets.QPushButton("Tick the relevant")
        self.relevant_btn.setToolTip("Tick the best of each group that separates or weakly separates (a wider set, "
                                     "e.g. for the learned classifiers)")
        self.relevant_btn.clicked.connect(self._tick_relevant)
        self.send_btn = QtWidgets.QPushButton("Send to Build")
        self.send_btn.setToolTip("Make a classifier from the ticked parameters (Training › Classifiers › Build), "
                                 "replacing its selection: Ranges and Score for ordered grades, the learned ones "
                                 "(logistic regression, random forest, gradient boosting) for any classes")
        send = QtWidgets.QMenu(self.send_btn)
        from modalytics.ui.training.build import KINDS
        for text, kinds in [(name, [kind]) for kind, name in KINDS.items()] + [("All of them", list(KINDS))]:
            act = send.addAction(text, lambda k=kinds: self.to_build(k))
            act.setData(kinds)
        self.send_btn.setMenu(send)                         # a click opens it
        self.key_btn = QtWidgets.QPushButton("Make the ticked the profile's key parameters")
        self.key_btn.setToolTip("The profile's key parameters are what a project's report puts forward (summary per "
                                "class, objects table, distributions, methods); everything stays measured")
        self.key_btn.clicked.connect(self.to_profile)
        row.addWidget(self.tick_btn)
        row.addWidget(self.relevant_btn)
        row.addWidget(self.send_btn)
        row.addWidget(self.key_btn)
        row.addStretch(1)
        self.key_note = muted("")
        row.addWidget(self.key_note)
        col.addLayout(row)
        self._update_buttons()

    # ------------------------------------------------------------ showing a set
    def refresh(self) -> None:
        """Show the set's analyses (after it is opened, measured or changed)."""
        tset = self.win.tset
        if tset is None:
            return
        if not self.name.text() or self.name.text() == self._auto_name:
            self._auto_name = self._proposed_name()
            self.name.setText(self._auto_name)
        self._list_runs()
        self._update_buttons()

    def _proposed_name(self) -> str:
        return f"Parameters on {self.on.currentText()}, {self.unit.currentText()} compared"

    def _on_changed(self, *_args) -> None:
        if not self.name.text() or self.name.text() == self._auto_name:
            self._auto_name = self._proposed_name()
            self.name.setText(self._auto_name)

    def rows(self) -> np.ndarray:
        """The objects analysed: the training projects', or all."""
        data = self.win.data
        if self.on.currentData() == "all":
            return np.ones(len(data.grade), bool)
        return ds.training_rows(self.win.tset, data)

    def _update_buttons(self) -> None:
        data = self.win.data
        busy = self.win.tasks.busy
        self.run_btn.setEnabled(self.win.tset is not None and data is not None and len(data.grade) > 0 and not busy)
        self.report_btn.setEnabled(self.run is not None and self.run.report.exists())
        self.pdf_btn.setEnabled(self.report_btn.isEnabled())
        self.xlsx_btn.setEnabled(self.run is not None and self.run.results.exists() and not busy)
        self.runs.setEnabled(bool(self.runs.currentData()) and not busy)
        ticked = bool(self.ticked())
        self.key_btn.setEnabled(ticked and self.win.tset is not None)
        self.tick_btn.setEnabled(self.analysis is not None and bool(self.analysis.proposed()))
        self.relevant_btn.setEnabled(self.analysis is not None and bool(self.relevant()))
        self.send_btn.setEnabled(ticked and data is not None and len(data.grade) > 0 and not busy)
        ordered = data.ordered if data is not None else True
        from modalytics.ui.training.build import SCALE
        for act in self.send_btn.menu().actions():          # categories: no scale for Ranges and Score to climb
            act.setEnabled(ordered or not set(act.data()) <= set(SCALE))

    # ------------------------------------------------------------ analysing
    def analyse(self) -> None:
        """Run the analysis in the background, then write its report and list it."""
        from modalytics.report import relevance as rep
        from modalytics.training import runs as rn
        tset, data = self.win.tset, self.win.data
        if tset is None or data is None or not len(data.grade):
            return
        rows = self.rows()
        if not rows.any():
            QtWidgets.QMessageBox.information(self, "Analyse", f"No graded objects on {self.on.currentText()}.")
            return
        on, unit, name = self.on.currentData(), self.unit.currentData(), self.name.text().strip() or "Parameters"
        run = rn.Run.new(rn.set_folder(tset, create=True), KIND, name)
        run.folder.mkdir(parents=True, exist_ok=True)
        self.win._save()
        set_name, profile, group, path = tset.name, tset.profile.name, tset.group, self.win.path

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            try:
                a = rv.analyse(data, rows, on, group, name, progress=progress, cancelled=cancelled,
                               training_set=set_name, profile=profile, unit=unit)
            except rv.Cancelled:
                raise Cancelled() from None
            a.save(run.results)
            rep.write(a, str(run.report), data, rows)
            return a

        def done(a) -> None:
            if self.win.path != path:
                return
            self.analysis, self.run = a, run
            self._list_runs()
            self._show()
            self.win.analysis_changed()
            self.win.statusBar().showMessage(f"✅ {len(a.proposed())} parameter(s) proposed of {len(a.rows)}: {name}",
                                             8000)

        self.win.tasks.run(f"Analysing {len(data.columns)} parameters on {int(rows.sum())} objects…", job, done,
                           lambda exc: QtWidgets.QMessageBox.warning(self, "Analyse", str(exc)),
                           on_cancel=self._update_buttons)
        self._update_buttons()

    # ------------------------------------------------------------ the set's analyses
    def _list_runs(self) -> None:
        from modalytics.training import runs as rn
        tset = self.win.tset
        found = rn.list_runs(tset, KIND) if tset is not None else []
        key = (tset.runs_folder, tset.name, tuple(str(r.folder) for r in found)) if tset is not None else None
        if key == self._runs_key:
            return
        self._runs_key = key
        current = self.run.folder if self.run is not None else None
        self.runs.blockSignals(True)
        self.runs.clear()
        for r in found:
            self.runs.addItem(r.label, str(r.folder))
        if not found:
            self.runs.addItem("No analyses yet (Analyse makes one)", "")
        kept = next((r for r in found if r.folder == current), None)
        shown = kept or (found[0] if found else None)
        if shown is not None:
            self.runs.setCurrentIndex(self.runs.findData(str(shown.folder)))
        self.runs.blockSignals(False)
        if kept is None:
            self._open_run(shown)
        self._update_buttons()

    def _run_picked(self, index: int) -> None:
        from modalytics.training import runs as rn
        folder = self.runs.itemData(index)
        if folder and (self.run is None or Path(folder) != self.run.folder):
            self._open_run(rn.Run.of(Path(folder)))

    def _open_run(self, run) -> None:
        self.analysis, self.run = None, run
        if run is not None:
            try:
                self.analysis = rv.Analysis.load(run.results)
            except Exception as exc:                                     # noqa: BLE001 (said, not fatal)
                self.note.setText(f"⚠ The analysis could not be read: {exc}")
        self._show()
        self.win.analysis_changed()

    # ------------------------------------------------------------ showing an analysis
    def _show(self) -> None:
        clear_tree(self.table)
        a = self.analysis
        if a is None:
            self.note.setText("Analyse finds which parameters separate the grades on the measured objects — "
                              "with intervals over projects, corrected for the many parameters — and which repeat "
                              "each other, and proposes a selection.")
            self.plot.show_values("", [])
            self._update_buttons()
            return
        proposed = a.proposed()
        tally = " · ".join(f"{rv.VERDICTS[v].lower()} {n}" for v in rv.VERDICTS
                           if (n := sum(1 for r in a.rows if r.verdict == v)))
        kind = ("grades in order: " + " < ".join(a.grades)) if a.ordered else \
            ("categories: " + ", ".join(a.grades))
        self.note.setText(
            (f"<b>Proposed</b>: {', '.join(r.title for r in proposed)} — the best of each group that separates the "
             f"classes. " if proposed else "<b>No parameter separates the classes clearly</b> on these projects. ")
            + f"On {ON[a.on]} ({len(a.projects)} projects; "
            + ", ".join(f"{g} {n}" for g, n in a.counts.items()) + f"; {kind}): {tally}."
            + f" Compared: <b>{rv.UNITS.get(a.unit, a.unit)}</b>"
            + (" (each a case; objects of one sample may resemble each other)." if a.unit == "objects" else
               " (each one's class a case: the median of its objects)." if a.unit in ("images", "projects") else ".")
            + (" <b>Training and evaluation</b>: a description — choose on the training projects."
               if a.on == "all" else ""))
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        colours = {"separates": QtGui.QColor("#2fb3a3"), "weak": QtGui.QColor("#e0a33a"),
                   "promising": QtGui.QColor("#56B4E9")}
        for r in a.rows:
            ci = f" [{r.ci[0]:.2f}, {r.ci[1]:.2f}]" if all(map(math.isfinite, r.ci)) else ""
            stands = (f"{r.best} ({'higher' if r.direction == 'higher' else 'lower'})" if not a.ordered and r.best
                      else ("larger with grade" if r.direction == "higher" else "smaller with grade")
                      if math.isfinite(r.auc) else "")
            item = SortItem(self.table, [
                ("★ " if r.proposed else "") + r.title, rv.VERDICTS[r.verdict],
                (f"{r.auc:.3f}" if math.isfinite(r.auc) else "—") + ci, stands,
                "—" if not math.isfinite(r.q) else ("< 0.001" if r.q < 0.001 else f"{r.q:.3f}"),
                f"{r.agree[0]} / {r.agree[1]}" if r.agree[1] else "—", str(r.group or ""), "; ".join(r.notes())])
            item.setData(0, UID, r.uid)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if r.proposed else QtCore.Qt.CheckState.Unchecked)
            item.setToolTip(0, r.help)
            if r.verdict in colours:
                item.setForeground(1, colours[r.verdict])
            for c in (2, 4, 5, 6):
                item.setTextAlignment(c, right)
        if self.table.topLevelItemCount():
            self.table.setCurrentItem(self.table.topLevelItem(0))
        self._update_buttons()

    def _show_parameter(self, item, _prev=None) -> None:
        a, data, tset = self.analysis, self.win.data, self.win.tset
        if item is None or a is None or data is None or tset is None:
            return
        r = a.row(item.data(0, UID))
        columns = [c[0] for c in data.columns]
        if r is None or r.uid not in columns:
            self.plot.show_values(r.title if r else "", [], None, "Not measured in the objects now held (Measure)")
            return
        j = columns.index(r.uid)
        rows = self.rows() if a.on == self.on.currentData() else (
            np.ones(len(data.grade), bool) if a.on == "all" else ds.training_rows(tset, data))
        series = [(g, data.values[rows & (data.grade == k), j],
                   QtGui.QColor.fromRgbF(*tset.profile.classes.color(tset.group, g))) for k, g in enumerate(data.grades)]
        ci = f" [{r.ci[0]:.3f}, {r.ci[1]:.3f}]" if all(map(math.isfinite, r.ci)) else ""
        how = (f"{r.best} {'higher' if r.direction == 'higher' else 'lower'} than the others" if not a.ordered
               else f"{'larger' if r.direction == 'higher' else 'smaller'} with the higher grade")
        self.plot.show_values(r.title, series, None, f"{rv.VERDICTS[r.verdict]} · AUC {r.auc:.3f}{ci} · {how}")
        self.details.setHtml(details_html(r, a))

    # ------------------------------------------------------------ choosing
    def ticked(self) -> list[str]:
        return [self.table.topLevelItem(i).data(0, UID) for i in range(self.table.topLevelItemCount())
                if self.table.topLevelItem(i).checkState(0) == QtCore.Qt.CheckState.Checked]

    def _tick_proposal(self) -> None:
        if self.analysis is None:
            return
        proposed = {r.uid for r in self.analysis.proposed()}
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if item.data(0, UID) in proposed
                               else QtCore.Qt.CheckState.Unchecked)
        self._update_buttons()

    def relevant(self) -> list[str]:
        """The best of each redundancy group that separates or weakly separates."""
        a = self.analysis
        if a is None:
            return []
        found = [r for r in a.rows if r.verdict in ("separates", "weak")]
        best = {}
        for r in found:
            key = r.group or f"alone:{r.uid}"
            if key not in best or r.auc > best[key].auc:
                best[key] = r
        return [r.uid for r in found if best.get(r.group or f"alone:{r.uid}") is r]

    def _tick_relevant(self) -> None:
        chosen = set(self.relevant())
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if item.data(0, UID) in chosen
                               else QtCore.Qt.CheckState.Unchecked)
        self._update_buttons()

    def to_build(self, kinds: list[str]) -> None:
        """Make these classifiers from the ticked parameters in Build (asked first where one has a selection)."""
        build, uids = self.win.build, self.ticked()
        if not uids or self.win.data is None:
            return
        from modalytics.ui.training.build import KINDS, SCALE
        names = KINDS
        kinds = [k for k in kinds if self.win.data.ordered or k not in SCALE]
        if self.analysis is not None and self.analysis.on == "all" and QtWidgets.QMessageBox.question(
                self, "Send to Build", "This analysis was made on the training and evaluation projects: parameters "
                "chosen from it would flatter the classifier on the evaluation projects. Analyse the training "
                "projects for choosing.\n\nSend them all the same?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        busy = [k for k in kinds if build.selection(k) and set(build.selection(k)) != set(uids)]
        if busy and QtWidgets.QMessageBox.question(
                self, "Send to Build", "Replace the parameters of " + ", ".join(names[k] for k in busy)
                + f" with the {len(uids)} ticked?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        left = []
        for k in kinds:
            left = build.use_parameters(k, uids)
        self.win.kinds.setCurrentWidget(self.win.classifiers_page)
        self.win.classifiers.setCurrentWidget(build)
        titles = {r.uid: r.title for r in self.analysis.rows} if self.analysis else {}
        self.win.statusBar().showMessage(
            f"✅ {len(uids) - len(left)} parameter(s) sent to " + ", ".join(names[k] for k in kinds)
            + (f" — left out (no values in these projects): {', '.join(titles.get(u, u) for u in left)}"
               if left else "") + (" · Random forest, gradient boosting: click Train to grow them"
                                   if {"forest", "boosting"} & set(kinds) else ""),
            10000)

    def to_profile(self) -> None:
        """Make the ticked parameters the profile's key parameters (asked first; everything stays measured)."""
        tset = self.win.tset
        uids = self.ticked()
        if tset is None or not uids:
            return
        try:
            prof = self.win.profiles.get(tset.profile_id)
        except Exception as exc:                                         # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Key parameters", f"The profile could not be read:\n\n{exc}")
            return
        titles = {r.uid: r.title for r in self.analysis.rows} if self.analysis else {}
        missing = [u for u in uids if u not in prof.parameters]
        uids = [u for u in uids if u in prof.parameters]
        old = prof.report_parameters
        ask = (f"Make these the key parameters of the profile '{prof.name}' (what a project's report puts forward)?"
               f"\n\n" + "\n".join(f"  • {titles.get(u, u)}" for u in uids)
               + (f"\n\nThey replace: {', '.join(titles.get(u, u) for u in old)}." if old else "")
               + (f"\n\nLeft out (the profile no longer measures them): {', '.join(missing)}." if missing else "")
               + "\n\nEvery parameter stays measured; projects made from the profile from now on use them.")
        if not uids or QtWidgets.QMessageBox.question(self, "Key parameters", ask) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        prof.report = dict(prof.report, parameters=uids)
        self.win.profiles.save(prof, tset.profile_id)
        tset.profile.report = dict(tset.profile.report, parameters=uids)       # the set's copy says the same
        self.win._save()
        self.key_note.setText(f"✅ {len(uids)} key parameter(s) in {prof.name}")
        self.win.statusBar().showMessage(f"✅ {prof.name}: {len(uids)} key parameter(s)", 8000)

    def _export_xlsx(self) -> None:
        """The analysis shown as a workbook (with the measured objects it was made on, for the values)."""
        from modalytics.report import xlsx
        from modalytics.ui.training.export import save_xlsx
        if self.run is None or not self.run.results.exists():
            return
        a = rv.Analysis.load(self.run.results)
        data = self.win.data
        rows = None
        if data is not None and len(data.grade):
            rows = np.ones(len(data.grade), bool) if a.on == "all" else ds.training_rows(self.win.tset, data)
        save_xlsx(self, self.run.report, lambda path: xlsx.write_parameters(a, path, data, rows), "the analysis")

    def _report_pdf(self) -> None:
        from modalytics.ui.pdf_export import report_pdf
        if self.run is not None:
            report_pdf(self, self.run.report, status=self.win.statusBar())

    def _open_folder(self) -> None:
        from modalytics.training import runs as rn
        tset = self.win.tset
        for folder in (self.run.folder if self.run is not None else None,
                       rn.set_folder(tset) / KIND if tset is not None else None, rn.root()):
            if folder is not None and Path(folder).is_dir():
                open_in_system(folder)
                return


def details_html(r: rv.Relevance, a: rv.Analysis) -> str:
    """One parameter per class (n, median (IQR), mean ± SD, normality) and every pair of classes compared."""
    def num(v):
        return "—" if v is None or not math.isfinite(v) else f"{v:.4g}"
    rows = []
    for c in a.grades:
        st = r.stats.get(c) or []
        if len(st) < 7 or not st[0]:
            rows.append(f"<tr><td>{c}</td><td colspan='4'>no values</td></tr>")
            continue
        n, med, q1, q3, mean, sd, sw = st
        normal = "—" if not math.isfinite(sw) else (f"yes (p {sw:.2f})" if sw >= 0.05 else f"no (p {sw:.3f})")
        rows.append(f"<tr><td>{c}</td><td align='right'>{int(n)}</td><td align='right'>{num(med)} "
                    f"({num(q1)}–{num(q3)})</td><td align='right'>{num(mean)} ± {num(sd)}</td><td>{normal}</td></tr>")
    pairs = "".join(
        f"<tr><td>{x} vs {y}</td><td align='right'>{num(auc)}</td><td align='right'>"
        f"{'—' if not math.isfinite(p) else ('&lt; 0.001' if p < 0.001 else f'{p:.3f}')}</td></tr>"
        for x, y, auc, p in r.pairs)
    case = {"images": "images", "projects": "projects"}.get(a.unit, "objects")
    return (f"<b>{r.title}</b><table cellspacing='0' cellpadding='3'><tr><th align='left'>Class</th><th>n</th>"
            f"<th>Median (IQR)</th><th>Mean ± SD</th><th align='left'>Normal?</th></tr>{''.join(rows)}</table>"
            + (f"<p><b>Pairs</b> (AUC: the chance the second's {case[:-1]} has the larger value; p corrected for "
               f"the pairs, Holm)</p><table cellspacing='0' cellpadding='3'><tr><th align='left'>Pair</th>"
               f"<th>AUC</th><th>p</th></tr>{pairs}</table>" if r.pairs else "")
            + "<p style='color:gray'>Normality is for information: the tests compare ranks and do not assume it.</p>")
