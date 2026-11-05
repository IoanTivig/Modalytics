"""
Training › Segmentation models › Evaluate: existing models judged and compared on the set's projects.

    Objects Gland (Data)  On [the test projects ▾]  Where [as marked ▾] margin [50 µm]  IoU ≥ [0.50]  Choose by [F1 ▾]
    Models to judge
    ┌ ☑ Model ┬ Image type ┬ Works at ┬ Architecture ┬ Trained ┬ From ┐      (installed, or a file added)
    └─────────────────────────────────────────────────────────────────┘
    [Add model file…] [Remove file]
    Name [Gland models on the test projects]                                                     [Evaluate]
    Evaluation [2026-10-26 14:02 — … ▾]                                              [Open report] [Folder]
    ┌ Model ┬ Images ┬ By hand ┬ Found ┬ Missed ┬ Invented ┬ F1 ┬ GlaS F1 ┬ PQ ┬ Obj. Dice ┬ Dice ┬ ms ┐
    └─────────────────────────────────────────────────────────────────────────────────────────────────┘
    To look at [all ▾] (the selected model)   Project  Image  What  IoU  Area  (double-click: open it there)

Nothing is trained here: the models already exist — kept from a training run, released,
or another person's file (*Add model file…*, used where it is, not installed). Any
architecture: each runs at its own resolution on the images of its type. They are judged
as Train judges the models it trains (``training.compare.evaluate_models``): the same
measures with their intervals, compared with the best one, and the same report, without
the training parts but with each model's origin (and a warning when it learned from the
projects it is judged on). Each evaluation is kept in the set's runs folder
(``training_runs/<set>/segmentation/evaluations/``) and listed, the newest first. The
models' outlines are cached, so judging again with other settings is quick.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics.training import dataset as ds
from modalytics.training import segmentation as se
from modalytics.ui.settings.widgets import clear_tree, header_tips, list_tree, muted, open_in_system
from modalytics.ui.training.sorting import SortItem, sortable

KIND_NAMES = {"missed": "Missed (drawn by hand, not found)", "invented": "Invented (found, not drawn by hand)",
              "merge": "Merged (one outline over several)", "split": "Split (one object cut in several)",
              "poor": f"Outlined poorly (IoU < {se.POOR_IOU})"}
ON = {"test": "the test projects", "evaluation": "the evaluation projects", "all": "all the projects"}
UID = QtCore.Qt.ItemDataRole.UserRole
MODEL_COLUMNS = ["Model", "Image type", "Works at", "Architecture", "Trained", "From"]
RESULT_COLUMNS = ["Model", "Images", "By hand", "Found", "Missed", "Invented", "F1", "GlaS F1", "PQ", "Obj. Dice",
                  "Dice", "ms / tile"]
SHOWN = ("f1", "glas_f1", "pq", "object_dice", "dice")          # measures in the results table


class EvaluatePanel(QtWidgets.QWidget):
    """Judge existing segmentation models on a training set's projects (see the module docstring).

    Args:
        window: The ``TrainingWindow`` (its ``tset``, ``path``, ``data``, ``tasks``, ``_save``).
    """

    show_object = QtCore.Signal(str, str, str, float, float, object)   # path, image, group, x, y, outline

    def __init__(self, window) -> None:
        super().__init__()
        self.win = window
        self.comparison = None                              # the evaluation shown (``training.compare.Comparison``)
        self.run = None                                     # its place (``training.runs.Run``)
        self._runs_key = None
        self._auto_name = ""
        self._filling = False
        col = QtWidgets.QVBoxLayout(self)

        # 1. How the models are judged
        from modalytics.training import compare as cmp
        form = QtWidgets.QHBoxLayout()
        self.group_label = QtWidgets.QLabel()
        self.group_label.setToolTip("The training set's group (chosen in Data); all its subgroups count")
        self.on = QtWidgets.QComboBox()
        self._fill_on()
        self.on.setToolTip("Test projects: never used for training or choosing — the fair comparison; with several "
                           "test groups (Data: e.g. one per microscope setup or lab), each can be judged apart. "
                           "Evaluation projects: models trained with this set chose their epoch on them (somewhat "
                           "optimistic). All: a model trained on some of them is judged partly on its own training "
                           "images.")
        self.area = QtWidgets.QComboBox()
        for key, text in se.AREAS.items():
            self.area.addItem(text, key)
        self.area.setToolTip("Whole image: every object of the group was outlined by hand, so what the model finds "
                             "elsewhere is invented (and in training, all of it is \"not the group\"). Around the "
                             "hand outlines: their convex hull widened by the margin (where only some objects were "
                             "outlined). Marked: per image, as set in Properties › Annotation (\"Annotation complete\").\n"
                             "Unreviewed suggestions are always left out: what is under them is not known yet. "
                             "Train judges its models the same way.")
        self.margin = QtWidgets.QDoubleSpinBox()
        self.margin.setRange(0, 5000)
        self.margin.setValue(50)
        self.margin.setSuffix(" µm")
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.5)
        self.iou.setToolTip("A model object matches a hand object when their IoU (overlap / union) is at least this")
        self.criterion = QtWidgets.QComboBox()
        for key in cmp.CRITERIA:
            self.criterion.addItem(cmp.METRICS[key][0], key)
        self.criterion.setToolTip("The measure the best model is proposed by")
        for label, w in (("Objects", self.group_label), ("On", self.on), ("Where", self.area),
                         ("Margin", self.margin), ("Match at IoU ≥", self.iou), ("Choose by", self.criterion)):
            form.addWidget(QtWidgets.QLabel(label))
            form.addWidget(w)
        form.addStretch(1)
        col.addLayout(form)

        # 2. The models to judge
        col.addWidget(QtWidgets.QLabel("<b>Models to judge</b>"))
        self.models = sortable(list_tree(MODEL_COLUMNS))
        for c, w in enumerate((420, 90, 100, 190, 90)):
            self.models.setColumnWidth(c, w)
        header_tips(self.models, [
            "Tick the models to judge\nGreyed: it outlines another group", "The images it segments",
            "The resolution it works at\n(each image is resampled to it)", "Its network and starting weights",
            "When it was trained", "Your installed models, or the folder of a file added\n(used where it is)"])
        self.models.itemChanged.connect(self._model_ticked)
        self.models.currentItemChanged.connect(lambda *_a: self._update_buttons())
        col.addWidget(self.models, 1)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add model file…")
        self.add_btn.setToolTip("A model file (.mmodel) from elsewhere — another person's, or not installed — judged "
                                "where it is")
        self.add_btn.clicked.connect(self._add_files)
        self.remove_btn = QtWidgets.QPushButton("Remove file")
        self.remove_btn.setToolTip("Take the selected added file off the list (the file itself is not touched)")
        self.remove_btn.clicked.connect(self._remove_file)
        row.addWidget(self.add_btn)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        col.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit()
        self.name.setToolTip("The evaluation's name: its folder and its report are named after it")
        row.addWidget(self.name, 1)
        self.run_btn = QtWidgets.QPushButton("Evaluate")
        self.run_btn.setToolTip("Run the ticked models on the projects' images and compare them with the hand "
                                "outlines (in the background), then write the report")
        self.run_btn.clicked.connect(self.evaluate)
        row.addWidget(self.run_btn)
        col.addLayout(row)
        self.note = muted("")
        self.note.setWordWrap(True)
        col.addWidget(self.note)

        # 3. The evaluations, and the results of the one shown
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Evaluation"))
        self.runs = QtWidgets.QComboBox()
        self.runs.setToolTip("This training set's evaluations, the newest first: picking one shows its results")
        self.runs.activated.connect(self._run_picked)
        row.addWidget(self.runs, 1)
        self.report_btn = QtWidgets.QPushButton("Open report")
        self.report_btn.setToolTip("The evaluation's report (in its folder)")
        self.report_btn.clicked.connect(self._open_report)
        self.pdf_btn = QtWidgets.QPushButton("PDF")
        self.pdf_btn.setToolTip("The evaluation's report as a PDF (A4 across), beside it in its folder, and open it")
        self.pdf_btn.clicked.connect(self._report_pdf)
        self.folder_btn = QtWidgets.QPushButton("Folder")
        self.folder_btn.setToolTip("The evaluation's folder: its report and results file")
        self.folder_btn.clicked.connect(self._open_folder)
        self.xlsx_btn = QtWidgets.QPushButton("XLSX…")
        self.xlsx_btn.setToolTip("The evaluation as an Excel workbook: every measure with its interval, per project, "
                                 "per image, and native charts")
        self.xlsx_btn.clicked.connect(self._export_xlsx)
        row.addWidget(self.report_btn)
        row.addWidget(self.pdf_btn)
        row.addWidget(self.xlsx_btn)
        row.addWidget(self.folder_btn)
        col.addLayout(row)
        self.table = sortable(list_tree(RESULT_COLUMNS))
        for c, w in enumerate((420, 60, 65, 55, 60, 65, 55, 60, 55, 70, 55, 65)):
            self.table.setColumnWidth(c, w)
        self.table.header().setStretchLastSection(False)            # the numbers stay beside their titles
        header_tips(self.table, [
            "★ the proposal: the best by the measure chosen", "Images judged", "Objects drawn by hand",
            "Model objects matching a hand object\n(IoU at least the threshold)", "Drawn by hand, not found",
            "Found, not drawn by hand", "Found against missed and invented\n(a match: IoU ≥ the threshold)",
            "The GlaS challenge's F1: a model object counts\nwhen it covers ≥ 50 % of its hand object\n"
            "More lenient; to set beside published results", "Panoptic quality: found and well outlined",
            "The GlaS object Dice: each object against\nthe one overlapping it most",
            "Overlap of all the outlines (area)", "Time per 512 × 512 tile (ONNX Runtime)"])
        self.table.currentItemChanged.connect(lambda *_a: self._show_items())
        col.addWidget(self.table, 1)
        self.result_note = muted("")
        self.result_note.setWordWrap(True)
        col.addWidget(self.result_note)

        # 4. What to look at (the selected model's)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("To look at"))
        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("everything", "")
        for k, name in KIND_NAMES.items():
            self.kind.addItem(name, k)
        self.kind.currentIndexChanged.connect(self._show_items)
        row.addWidget(self.kind)
        self.items_note = muted("")
        row.addWidget(self.items_note, 1)
        col.addLayout(row)
        self.items = sortable(list_tree(["Project", "Image", "What", "IoU", "Area [µm²]"]))
        for c, w in enumerate((140, 220, 300, 60, 90)):
            self.items.setColumnWidth(c, w)
        self.items.header().setStretchLastSection(False)
        header_tips(self.items, ["The selected model's objects\nDouble-click: open the project there", "",
                                 "Missed, invented, merged, split or outlined poorly",
                                 "Its best overlap with the other side", "Its area"])
        self.items.itemDoubleClicked.connect(self._open_item)
        col.addWidget(self.items, 1)

        for w in (self.on, self.area, self.criterion):
            w.currentIndexChanged.connect(self._settings_changed)
        for w in (self.margin, self.iou):
            w.valueChanged.connect(self._settings_changed)
        self._update_buttons()

    def _fill_on(self) -> None:
        """The On choices: the test projects (and each test group, when there are several), the evaluation
        projects, all of them."""
        tset = self.win.tset if self.win is not None else None
        groups = list(ds.test_groups(tset)) if tset is not None else []
        keep = self.on.currentData()
        self.on.blockSignals(True)
        self.on.clear()
        self.on.addItem("all the test projects" if len(groups) > 1 else ON["test"], "test")
        if len(groups) > 1:
            for g in groups:
                self.on.addItem(f"the test group {g}", f"test:{g}")
        self.on.addItem(ON["evaluation"], "evaluation")
        self.on.addItem(ON["all"], "all")
        self.on.setCurrentIndex(max(0, self.on.findData(keep)))
        self.on.blockSignals(False)

    def group_name(self) -> str:
        """The training set's group (Data): the objects judged."""
        return self.win.tset.group if self.win.tset is not None else ""

    def _saved(self) -> dict:
        """Evaluate's own settings in the set: {"checked": [model files], "files": [added files], "criterion"}."""
        return self.win.tset.segmentation.setdefault("evaluate", {})

    # ------------------------------------------------------------ showing a set
    def refresh(self) -> None:
        """Show the training set's settings, models and evaluations (after it is opened or changed)."""
        tset = self.win.tset
        self.setEnabled(tset is not None and bool(tset.projects))
        if tset is None:
            return
        st = tset.segmentation
        self._filling = True
        self.group_label.setText(f"<b>{tset.group}</b> (Data)")
        on = st.get("projects") or ""
        if not on or ("evaluate" not in st and on == "evaluation"):     # unset, or saved before "test" was offered
            on = "test" if ds.test_projects(tset) else "evaluation"
        self._fill_on()
        self.on.setCurrentIndex(max(0, self.on.findData(on)))           # a group gone: the first choice
        self.area.setCurrentIndex(max(0, self.area.findData(st.get("area", "marked"))))
        self.margin.setValue(float(st.get("margin", 50.0)))
        self.margin.setEnabled(self.area.currentData() != "image")
        self.iou.setValue(float(st.get("iou", 0.5)))
        self.criterion.setCurrentIndex(max(0, self.criterion.findData(st.get("evaluate", {}).get("criterion",
                                                                                                    "f1"))))
        self._filling = False
        self.refresh_models()
        self._propose_name()
        self._list_runs()
        self._update_note()

    def _propose_name(self) -> None:
        if not self.name.text() or self.name.text() == self._auto_name:
            self._auto_name = f"{self.group_name()} models on {self.on.currentText()}"
            self.name.setText(self._auto_name)

    def refresh_models(self, check: list[str] | None = None) -> None:
        """The models list again: installed models and the files added; the ticks kept (``check``: ticked too)."""
        from modalytics.image_import.registry import TYPE_NAMES
        from modalytics.services.segmentation import find_models, load_package
        if self.win.tset is None:
            return
        saved = self._saved()
        ticked = set(os.path.normcase(p) for p in saved.get("checked", [])) | \
            set(os.path.normcase(p) for p in (check or []))
        if not saved.get("checked") and not check:                  # nothing chosen yet: the profile's models
            ticked |= {os.path.normcase(m.path) for m in self._profile_models()}
        rows = [(m, "installed") for m in find_models()]
        known = {os.path.normcase(m.path) for m, _f in rows}
        for path in saved.get("files", []):
            if os.path.normcase(path) in known:
                continue
            try:
                rows.append((load_package(path), "file"))
            except Exception as exc:                                    # noqa: BLE001 (said in the list)
                rows.append((path, f"file — not readable: {exc}"))
        self._filling = True
        self.models.clear()
        group = self.group_name()
        for m, origin in rows:
            if isinstance(m, str):                                       # an added file that cannot be read
                item = SortItem(self.models, [os.path.basename(m), "", "", "", "", origin])
                item.setData(0, UID, m)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setDisabled(True)
                continue
            src = m.manifest.get("source", {}) or {}
            arch = (src.get("architecture") or {}).get("name") or src.get("architecture_id", "")
            weights = (src.get("encoder_weights") or {}).get("name", "")
            item = SortItem(self.models, [
                m.name, TYPE_NAMES.get(m.modality, m.modality), f"{m.um_per_px:.4g} µm/px",
                arch + (f" · {weights}" if weights else ""), (src.get("trained") or "")[:10],
                "your models" if origin == "installed" else os.path.dirname(m.path)])
            item.setData(0, UID, m.path)
            item.setToolTip(0, m.path)
            outlines = m.output_class[0] if m.output_class else ""
            if outlines and group and outlines != group:
                item.setDisabled(True)
                item.setToolTip(0, f"It outlines {outlines}, not {group}")
                continue
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if os.path.normcase(m.path) in ticked
                               else QtCore.Qt.CheckState.Unchecked)
        self._filling = False
        self._remember_ticks()
        self._update_note()

    def _profile_models(self) -> list:
        """The profile's models for the set's group (what is ticked before anything is chosen)."""
        from modalytics.services.segmentation import find_models, pick_model
        tset, models = self.win.tset, find_models()
        out = []
        for t in tset.profile.images:
            m = pick_model(models, t, tset.profile, self.group_name())
            if m is not None:
                out.append(m)
        return out

    def select_models(self, paths: list[str]) -> None:
        """Tick exactly these models (e.g. the one Train sends here)."""
        if self.win.tset is None:
            return
        self._saved()["checked"] = list(paths)
        self.refresh_models()

    def checked(self) -> list[str]:
        """The ticked models' files."""
        out = []
        for i in range(self.models.topLevelItemCount()):
            item = self.models.topLevelItem(i)
            if not item.isDisabled() and item.checkState(0) == QtCore.Qt.CheckState.Checked:
                out.append(item.data(0, UID))
        return out

    def _remember_ticks(self) -> None:
        if self.win.tset is not None:
            self._saved()["checked"] = self.checked()

    def _model_ticked(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        self._remember_ticks()
        self.win._save()
        self._update_note()

    def _add_files(self) -> None:
        paths, _f = QtWidgets.QFileDialog.getOpenFileNames(self, "Add model files", "",
                                                           "Modalytics models (*.mmodel)")
        if not paths or self.win.tset is None:
            return
        saved = self._saved()
        files = saved.setdefault("files", [])
        for p in paths:
            if os.path.normcase(p) not in {os.path.normcase(f) for f in files}:
                files.append(p)
        self.refresh_models(check=paths)
        self.win._save()

    def _remove_file(self) -> None:
        item = self.models.currentItem()
        if item is None or self.win.tset is None:
            return
        path = item.data(0, UID)
        saved = self._saved()
        saved["files"] = [f for f in saved.get("files", []) if os.path.normcase(f) != os.path.normcase(path)]
        saved["checked"] = [f for f in saved.get("checked", []) if os.path.normcase(f) != os.path.normcase(path)]
        self.refresh_models()
        self.win._save()

    def _settings_changed(self, *_args) -> None:
        if self._filling or self.win.tset is None:
            return
        self.margin.setEnabled(self.area.currentData() != "image")
        self.win.tset.segmentation.update({"projects": self.on.currentData(),     # its own keys only (Train
                                           "area": self.area.currentData(),       # keeps its settings there too;
                                           "margin": self.margin.value(),         # it judges "where" as set here)
                                           "iou": self.iou.value()})
        self._saved()["criterion"] = self.criterion.currentData()
        self.win._save()
        self._propose_name()
        self._update_note()

    def _paths(self) -> list[str]:
        """The projects judged (as chosen in On)."""
        tset = self.win.tset
        on = self.on.currentData() or ""
        if on == "all":
            return [p.path for p in tset.projects]
        if on.startswith("test:"):
            held = ds.test_groups(tset).get(on[len("test:"):], set())
        else:
            held = ds.test_projects(tset) if on == "test" else ds.evaluation_projects(tset, self.win.data)
        return [p.path for i, p in enumerate(tset.projects) if i in held]

    def _update_note(self) -> None:
        if self.win.tset is None:
            return
        n, projects = len(self.checked()), len(self._paths())
        on = (self.on.currentData() or "").split(":")[0]
        why = {"evaluation": " (models trained with this set chose their epoch on them: somewhat optimistic)",
               "all": " (a model trained on some of them looks better than it is)"}.get(on, "")
        self.note.setText(
            f"{n} model(s) ticked, judged on {self.on.currentText()}: {projects} project(s){why}"
            + ("" if projects else " — <b>none marked</b> (Data tab)")
            + ". Nothing is trained; each model runs at its own resolution. A second evaluation of the same models "
              "is quick (their outlines are kept).")
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = self.win.tasks.busy if self.win is not None else False
        item = self.models.currentItem()
        added = self.win.tset is not None and item is not None and os.path.normcase(item.data(0, UID) or "") in {
            os.path.normcase(f) for f in self._saved().get("files", [])}
        self.remove_btn.setEnabled(added)
        self.run_btn.setEnabled(self.win.tset is not None and bool(self.checked()) and not busy)
        self.report_btn.setEnabled(self.run is not None and self.run.report.exists())
        self.pdf_btn.setEnabled(self.report_btn.isEnabled())
        self.xlsx_btn.setEnabled(self.run is not None and self.run.results.exists() and not busy)
        self.runs.setEnabled(bool(self.runs.currentData()) and not busy)

    # ------------------------------------------------------------ evaluating
    def evaluate(self) -> None:
        """Judge the ticked models on the chosen projects in the background, then write the report."""
        from modalytics.report import comparison as rep
        from modalytics.services.segmentation import load_package
        from modalytics.training import compare as cmp
        from modalytics.training import runs as rn
        tset = self.win.tset
        paths, models = self._paths(), self.checked()
        if not paths:
            QtWidgets.QMessageBox.information(self, "Evaluate", f"No {self.on.currentText()[4:]}: mark some (Data "
                                                                f"tab), or judge on others.")
            return
        if not models:
            QtWidgets.QMessageBox.information(self, "Evaluate", "Tick the models to judge (or add a model file).")
            return
        types = {load_package(p).modality for p in models}
        if len(types) > 1:
            QtWidgets.QMessageBox.information(self, "Evaluate", "The ticked models are for different image types ("
                                              + ", ".join(sorted(types)) + "): judge one type at a time, so they are "
                                              "compared on the same images.")
            return
        group, area, margin, iou = self.group_name(), self.area.currentData(), self.margin.value(), self.iou.value()
        on, _, test_group = (self.on.currentData() or "test").partition(":")         # "test:<group>": one group
        groups = {g: [tset.projects[i].name for i in sorted(idx)] for g, idx in ds.test_groups(tset).items()}
        criterion = self.criterion.currentData()
        name = self.name.text().strip() or "Evaluation"
        run = rn.Run.new(rn.set_folder(tset, create=True), rn.EVALUATIONS, name)     # the set claims its folder
        run.folder.mkdir(parents=True, exist_ok=True)
        self.win._save()
        set_name, path = tset.name, self.win.path

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            try:
                comp = cmp.evaluate_models(models, paths, on, group, name, area, margin, iou, criterion,
                                           progress=lambda d: progress(d.get("fraction", 0.0)), cancelled=cancelled,
                                           results=run.results, training_set=set_name, test_group=test_group,
                                           test_groups=groups if on == "test" and not test_group else None)
            except se.Cancelled:
                raise Cancelled() from None
            rep.write(comp, str(run.report))
            return comp

        def done(comp) -> None:
            if self.win.path != path:
                return
            self.comparison, self.run = comp, run
            self._list_runs()
            self._show_results()
            self.win.statusBar().showMessage(f"✅ Judged {len(comp.done())} model(s) on {len(paths)} project(s): "
                                             f"{name}", 8000)

        def ended() -> None:
            self._list_runs()
            self._update_buttons()

        def failed(exc) -> None:
            ended()
            QtWidgets.QMessageBox.warning(self, "Evaluate", f"The evaluation stopped:\n\n{exc}")

        if self.win.tasks.run(f"Judging {len(models)} model(s) on {len(paths)} project(s)…", job, done, failed,
                              on_cancel=ended):
            self._update_buttons()

    # ------------------------------------------------------------ the set's evaluations
    def _list_runs(self) -> None:
        """The set's evaluations, the newest first; the last one shown (the one shown kept on a refresh)."""
        from modalytics.training import runs as rn
        tset = self.win.tset
        found = rn.list_runs(tset, rn.EVALUATIONS) if tset is not None else []
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
            self.runs.addItem("No evaluations yet (Evaluate makes one)", "")
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
        """Show an evaluation's results (None: nothing shown)."""
        from modalytics.training import runs as rn
        self.comparison, self.run = None, run
        self.result_note.setText("")
        if run is not None:
            try:
                self.comparison = rn.load(run)
            except Exception as exc:                                     # noqa: BLE001 (said, not fatal)
                self.result_note.setText(f"⚠ The evaluation's results could not be read: {exc}")
        self._show_results()

    def _show_results(self) -> None:
        """Each model's measures on the judged projects; the proposal starred and selected."""
        from modalytics.training import compare as cmp
        clear_tree(self.table)
        comp = self.comparison
        if comp is None:
            self._show_items()
            self._update_buttons()
            return
        on = comp.judged_on
        best = comp.best() if len(comp.done()) > 1 else None
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        chosen = None
        for e in comp.entries:
            ev = e.judged(on)
            if ev is None:
                item = SortItem(self.table, [e.label, "—", "", "", "", "", "", "", "", "", "", ""])
                item.setToolTip(0, e.error or "not judged")
                item.setText(1, "✖")
                item.setToolTip(1, e.error)
                continue
            c = ev.total
            values = [("★ " if e is best else "") + e.label, str(c.images), str(c.hand), str(c.tp), str(c.fn),
                      str(c.fp)]
            values += [_f(cmp.METRICS[k][1](c), cmp.METRICS[k][3]) for k in SHOWN]
            values.append(_f(e.ms_per_tile, "{:.0f}"))
            item = SortItem(self.table, values)
            item.setData(0, UID, e.run_id)
            item.setToolTip(0, e.path)
            for k in range(1, len(RESULT_COLUMNS)):
                item.setTextAlignment(k, right)
            if chosen is None or e is best:                          # the proposal selected (else the first)
                chosen = item
        if chosen is not None:
            self.table.setCurrentItem(chosen)
        learned = [e.label for e in comp.entries if (e.origin or {}).get("learned_from_judged")]
        self.result_note.setText(
            (f"★ proposed: {best.label} (best {cmp.METRICS[comp.criterion][0]} on {ON.get(on, on)}). " if best else "")
            + (f"⚠ {', '.join(learned)}: learned from some of the judged projects (flattered). " if learned else "")
            + "The report has every measure with its interval, the differences, figures and examples.")
        self._show_items()
        self._update_buttons()

    def _selected(self):
        item = self.table.currentItem()
        if item is None or self.comparison is None:
            return None
        return next((e for e in self.comparison.entries if e.run_id == item.data(0, UID)), None)

    def _show_items(self, *_args) -> None:
        clear_tree(self.items)
        e = self._selected()
        ev = e.judged(self.comparison.judged_on) if e is not None else None
        if ev is None:
            self.items_note.setText("")
            return
        want = self.kind.currentData()
        shown = [i for i in ev.items if not want or i.kind == want]
        self.items_note.setText(f"{len(shown)} object(s) of {e.label}.")
        for it in shown:
            item = SortItem(self.items, [ev.projects[it.project] if 0 <= it.project < len(ev.projects) else "",
                                         it.image, KIND_NAMES.get(it.kind, it.kind), _f(it.iou, "{:.2f}"),
                                         f"{it.area:.0f}"])
            for k in (3, 4):
                item.setTextAlignment(k, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item.setData(0, UID, it)

    def _open_item(self, item, _col=None) -> None:
        e = self._selected()
        it = item.data(0, UID)
        ev = e.judged(self.comparison.judged_on) if e is not None else None
        if ev is None or not (0 <= it.project < len(ev.paths)):
            QtWidgets.QMessageBox.information(self, "Evaluate", "This evaluation does not say where its projects are "
                                              "(made by an older version): evaluate again to open them from here.")
            return
        self.show_object.emit(ev.paths[it.project], it.image, self.group_name(),
                              float(it.point[0]), float(it.point[1]), it.outline or None)

    def _export_xlsx(self) -> None:
        from modalytics.report import xlsx
        from modalytics.training import compare as cmp
        from modalytics.ui.training.export import save_xlsx
        if self.run is not None and self.run.results.exists():
            comp = cmp.Comparison.load(self.run.results)
            save_xlsx(self, self.run.report, lambda path: xlsx.write_segmentation(comp, path), "the evaluation")

    def _report_pdf(self) -> None:
        from modalytics.ui.pdf_export import report_pdf
        if self.run is not None:
            report_pdf(self, self.run.report, status=self.win.statusBar())

    def _open_report(self) -> None:
        if self.run is not None and self.run.report.exists():
            open_in_system(str(self.run.report))

    def _open_folder(self) -> None:
        """The evaluation's folder; without one, the set's evaluations (or all the sets' runs)."""
        from modalytics.training import runs as rn
        tset = self.win.tset
        for folder in (self.run.folder if self.run is not None else None,
                       rn.set_folder(tset) / rn.EVALUATIONS if tset is not None else None, rn.root()):
            if folder is not None and Path(folder).is_dir():
                open_in_system(folder)
                return


def _f(v: float, fmt: str = "{:.3f}") -> str:
    return "—" if v is None or not math.isfinite(v) else fmt.format(v)
