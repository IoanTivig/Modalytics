"""
Training › Segmentation models › Train: train one or more segmentation models on the training set, compare them,
keep the one you choose.

    for [Phase map ▾] [works at 0.314 µm/px]
    Training images: 0.314 µm/px × 12 · 0.25 µm/px × 3 → resampled to 0.314 µm/px; a tile covers 161 µm  [Use …]
    Models to train
    ┌ ● │ Model                        │ Starting from │ Works at │ State          │ F1   │ PQ   │ … ┐
    │ ● │ U-Net (ResNet-34) ▾          │ MicroNet ▾    │ 0.314    │ ★ done (ep. 27)│ 0.91 │ 0.80 │   │
    │ ● │ Fine-tune: Gland — … (DHM) ▾ │ its weights   │ 0.314    │ done (ep. 12)  │ 0.88 │ …    │   │
    │   │   ↳ … as it was              │               │ 0.314    │ for reference  │ 0.84 │ …    │   │
    └────────────────────────────────────────────────────────────────────────────────────────────┘
    [+ New model] [+ Fine-tune a model] [Remove]
    Epochs [30] Steps [50] Patches [8] Scale [±0 %] Device [A2000 ▾]  Choose by [Object F1 ▾]  Name […]  [Train]
    Run [2026-10-25 01:14 — Gland on phase map ▾]                 (this set's runs, the newest first)
    Training loss per epoch │ Evaluation Dice per epoch      (every model, its colour; the selected one bold)
    [Open report] [Keep the selected model] [Use in profile] [Evaluate…] [Folder]

Each row is a model: a **new** one (an architecture from the library, its encoder starting
from ImageNet, MicroNet, your own weights or nothing) or a **fine-tuned** one (an existing
model trained further: its architecture, image type and µm/px kept), which is then also
judged **as it was**, for reference. The rows are trained one after another the same way
(``training.compare.run``: the settings here, the projects and roles of Data, the area of
Evaluate), each judged on the test projects (else the evaluation ones), timed and sized;
the best by *Choose by* is starred — you decide. Every run has a folder under its training
set (``training.runs``): its report, its results file and its models (``report.comparison``;
with several models it compares them). **Keep** installs the selected model in your models
(a run of one model keeps it by itself).

**Run** lists the set's runs, the newest first; opening the set shows the last one (its
curves, and its results in the rows of the same models). Picking another run puts its
models in the table with their results; change the rows to plan the next run.

The new models work at one resolution: every image is resampled to it from its own pixel
size (``seg_train.resolution_report``). The training images are read in the background and
the line under the settings says what they are; the most common pixel size is proposed
(taken as is when the set has no resolution of its own yet).

Available in the full edition (PyTorch, ``training.seg_train``); otherwise the tab says
what is missing.
"""

from __future__ import annotations

import math
import re
import shutil

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.training import dataset as ds
from modalytics.ui.settings.widgets import muted, open_in_system
from modalytics.ui.training.sorting import SortItem, sortable

SHOWN = {"f1": "F1", "pq": "PQ", "object_dice": "Obj. Dice", "dice": "Dice"}   # results in the table (short)
COLUMNS = ["", "Model", "Starting from", "Works at", "State"] + list(SHOWN.values()) + ["ms/tile", "Size", "Training"]
C_MODEL, C_START, C_UM, C_STATE = 1, 2, 3, 4
RUN = QtCore.Qt.ItemDataRole.UserRole                      # a row's run id (``training.compare.Planned.run_id``)
COLOUR = RUN + 3                                            # a row's colour (its place in the palette: kept when sorted)


def _palette() -> tuple[str, ...]:
    from modalytics.report.comparison import PALETTE
    return PALETTE


class Curves(QtWidgets.QWidget):
    """Training loss and evaluation Dice per epoch, side by side, for every model of a run (its row's colour; the
    selected one drawn bold, the epoch it keeps circled)."""

    def __init__(self) -> None:
        super().__init__()
        self.series: dict[str, tuple[QtGui.QColor, list[dict]]] = {}
        self.epochs = 1
        self.highlight = ""
        self.setMinimumHeight(200)

    def show_series(self, series: dict, epochs: int, highlight: str = "") -> None:
        self.series, self.epochs, self.highlight = dict(series), max(1, epochs), highlight
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.color(QtGui.QPalette.ColorRole.Text)
        muted_c = pal.color(QtGui.QPalette.ColorRole.PlaceholderText)
        rows = [r for _c, rs in self.series.values() for r in rs]
        if not rows:
            p.setPen(muted_c)
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "Training loss and evaluation Dice per epoch appear here while training (every model)")
            return
        half = self.width() // 2
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        centre = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        grid_c = QtGui.QColor(muted_c)
        grid_c.setAlphaF(0.22)
        top_loss = max((r["loss"] for r in rows if math.isfinite(r.get("loss", math.nan))), default=1.0) or 1.0
        loss_step = _nice_step(top_loss)
        loss_top = math.ceil(top_loss / loss_step - 1e-9) * loss_step          # the top on a tick
        x_step = _nice_step(self.epochs, 8, integer=True)
        for i, (key, title, scale, step) in enumerate((("loss", "Training loss", loss_top, loss_step),
                                                       ("dice", "Evaluation Dice", 1.0, 0.2))):
            r = QtCore.QRect(i * half + 46, 22, half - 62, self.height() - 58)
            p.setPen(text)
            p.drawText(QtCore.QRect(r.left(), 2, r.width(), 18), QtCore.Qt.AlignmentFlag.AlignLeft, title)

            def x_of(epoch, r=r):
                return r.left() + r.width() * (epoch - 0.5) / self.epochs

            def y_of(value, r=r, scale=scale):
                return r.bottom() - r.height() * max(0.0, min(1.0, value / scale))

            def xy(epoch, value):
                return QtCore.QPointF(x_of(epoch), y_of(value))
            # 1. The value axis: a grid line and a label at each tick, a short mark halfway between
            decimals = max(0, -math.floor(math.log10(step) + 1e-9)) if step < 1 else 0
            for k in range(int(round(scale / step)) + 1):
                v, y = k * step, y_of(k * step)
                p.setPen(QtGui.QPen(grid_c, 1))
                p.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
                p.setPen(muted_c)
                p.drawLine(QtCore.QPointF(r.left() - 5, y), QtCore.QPointF(r.left(), y))
                p.drawText(QtCore.QRectF(r.left() - 46, y - 8, 39, 16), right, f"{v:.{decimals}f}")
                if k < round(scale / step):
                    ym = y_of(v + step / 2)
                    p.drawLine(QtCore.QPointF(r.left() - 3, ym), QtCore.QPointF(r.left(), ym))
            # 2. The epoch axis: a grid line and a label every few epochs, a short mark at each epoch
            for e in range(1, self.epochs + 1):
                x = x_of(e)
                labelled = e == 1 or e % x_step == 0
                if labelled:
                    p.setPen(QtGui.QPen(grid_c, 1))
                    p.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom()))
                    p.setPen(muted_c)
                    p.drawText(QtCore.QRectF(x - 20, r.bottom() + 6, 40, 14), centre, str(e))
                p.setPen(muted_c)
                p.drawLine(QtCore.QPointF(x, r.bottom()), QtCore.QPointF(x, r.bottom() + (5 if labelled else 3)))
            p.setPen(QtGui.QPen(muted_c, 1))
            p.drawRect(r)
            p.drawText(QtCore.QRectF(r.left(), r.bottom() + 21, r.width(), 16), centre, "epoch · ○ the epoch kept")
            order = sorted(self.series.items(), key=lambda kv: kv[0] == self.highlight)   # the selected on top
            for run_id, (colour, rs) in order:
                strong = not self.highlight or run_id == self.highlight
                c = QtGui.QColor(colour)
                c.setAlphaF(1.0 if strong else 0.35)
                pts = [xy(row["epoch"], row[key]) for row in rs if math.isfinite(row.get(key, math.nan))]
                p.setPen(QtGui.QPen(c, 3 if run_id == self.highlight else 1.6))
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
                best = rs[-1].get("best", 0) if rs else 0
                hit = next((row for row in rs if row["epoch"] == best and math.isfinite(row.get(key, math.nan))), None)
                if hit is not None:
                    p.setBrush(pal.color(QtGui.QPalette.ColorRole.Base))
                    p.drawEllipse(xy(hit["epoch"], hit[key]), 4, 4)
                    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)


def _nice_step(span: float, ticks: int = 5, integer: bool = False) -> float:
    """The tick step of 1, 2 or 5 × a power of ten closest to ``ticks`` steps over ``span`` (whole numbers when
    ``integer``)."""
    raw = max(span, 1e-9) / ticks
    power = 10 ** math.floor(math.log10(raw))
    step = min((m * power for m in (1, 2, 5, 10)), key=lambda v: abs(v - raw))
    return max(1, int(round(step))) if integer else step


class _Bridge(QtCore.QObject):
    """Carries the run's reports from its thread to the window."""

    report = QtCore.Signal(object)


class TrainPanel(QtWidgets.QWidget):
    """Train one or more segmentation models (see the module docstring).

    Args:
        window: The ``TrainingWindow`` (its ``tset``, ``data``, ``path``, ``tasks``, ``profiles``, ``_save``).
        evaluate: The Evaluate panel (its area and margin; to judge a kept model there).
    """

    catalogue_ready = QtCore.Signal()                      # which encoders have ImageNet weights is known
    scales_ready = QtCore.Signal(object)                   # (request, [ImageScale], [skipped]) from the scan

    def __init__(self, window, evaluate) -> None:
        super().__init__()
        self.win, self.evaluate_panel = window, evaluate
        self.catalogue_ready.connect(lambda: self._refill_rows())
        self.scales_ready.connect(self._scales_found)
        self.scales, self._scan_key, self._um_saved, self._um_touched = [], None, False, False
        self.comparison = None                              # the run shown (``training.compare.Comparison``)
        self.run = None                                     # its place (``training.runs.Run``)
        self.folder = None                                  # its folder
        self._runs_key = None                               # the runs listed (to keep the choice on a refresh)
        self._auto_name = ""                                # the name proposed (replaced when the set changes)
        self.kept: dict[str, str] = {}                      # {run id: installed model id}
        self.curves: dict[str, list[dict]] = {}             # {run id: epoch rows}
        self._filling = False
        self.bridge = _Bridge()
        self.bridge.report.connect(self._report)
        col = QtWidgets.QVBoxLayout(self)
        from modalytics.training import seg_train as st
        from modalytics.ui.training.pack_panel import PackPanel
        ok, why = st.available()
        self.pack = PackPanel(self)                         # the training pack: install it, or switch packs
        self.pack.ready.connect(self._pack_ready)
        col.addWidget(self.pack)
        self.missing = muted(why)                           # from the source without the training extra
        self.missing.setWordWrap(True)
        self.missing.setVisible(not ok and not self.pack.isVisible())
        col.addWidget(self.missing)
        self.body = QtWidgets.QWidget()
        self.body.setEnabled(ok)
        col.addWidget(self.body, 1)
        b = QtWidgets.QVBoxLayout(self.body)
        b.setContentsMargins(0, 0, 0, 0)

        # 1. The image type and the resolution the new models work at
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("For"))
        self.modality = QtWidgets.QComboBox()
        from modalytics.image_import.registry import TYPE_NAMES
        for t in ("phase", "wsi", "od", "opd", "dry_mass", "height", "ri", "stain", "feature"):
            self.modality.addItem(TYPE_NAMES.get(t, t), t)
        row.addWidget(self.modality)
        self.um = QtWidgets.QDoubleSpinBox()
        self.um.setDecimals(4)
        self.um.setRange(0.01, 50)
        self.um.setValue(0.314)
        self.um.setPrefix("new models work at ")
        self.um.setSuffix(" µm/px")
        self.um.setToolTip("The resolution new models work at: every image is resampled to it from its own pixel "
                           "size (read from the image), so images of several systems can be mixed. The line below "
                           "proposes it from the training images. A fine-tuned model keeps its own.")
        row.addWidget(self.um)
        row.addStretch(1)
        b.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.scale_note = muted("")
        self.scale_note.setWordWrap(True)
        self.scale_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        row.addWidget(self.scale_note, 1)
        self.use_scale = QtWidgets.QPushButton("Use")
        self.use_scale.setToolTip("Work at the most common pixel size of the training images")
        self.use_scale.clicked.connect(self._use_proposal)
        self.use_scale.setVisible(False)
        row.addWidget(self.use_scale, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        b.addLayout(row)

        # 2. The models to train
        b.addWidget(QtWidgets.QLabel("<b>Models to train</b>"))
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(COLUMNS)
        sortable(self.table)                                # a click on a title sorts the rows (the training order)
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        for c, w in enumerate((26, 260, 190, 70, 170, 52, 52, 66, 52, 60, 64, 64)):
            self.table.setColumnWidth(c, w)
        head = self.table.headerItem()
        from modalytics.training import compare as cmp
        for c, key in enumerate(SHOWN, start=C_STATE + 1):
            head.setToolTip(c, cmp.METRICS[key][0])
        head.setToolTip(C_STATE + len(SHOWN) + 1, "ONNX Runtime time per 512 × 512 tile, as Segment runs the model")
        self.table.currentItemChanged.connect(lambda *_a: (self._show_curves(), self._update_buttons()))
        self.table.setMinimumHeight(130)
        b.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("+ New model", lambda: self._add_row("new"), "An architecture from the library, "
                                 "its encoder starting from weights learned elsewhere (or nothing)"),
                                ("+ Fine-tune a model", lambda: self._add_row("finetune"), "An existing model trained "
                                 "further on these projects (judged as it was too, for reference)"),
                                ("Remove", self._remove_row, "Remove the selected model from the list")):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch(1)
        b.addLayout(row)

        # 3. How they are trained, and which is proposed
        row = QtWidgets.QHBoxLayout()
        self.epochs = _spin(1, 500, 30, "Epochs", "Rounds of training; the best by evaluation Dice is kept")
        self.steps = _spin(5, 2000, 50, "Steps per epoch", "Optimiser steps per epoch")
        self.batch = _spin(1, 64, 8, "Patches per step", "512-px patches per step (fewer if the GPU runs out of "
                                                         "memory)")
        self.jitter = _spin(0, 30, 0, "Scale", "Each training patch resized by a random factor within ±this, so "
                                                "the model tolerates small calibration differences between systems "
                                                "(0: not resized; 10 % is a usual choice for several systems)")
        self.jitter.setSuffix(" %")
        self.jitter.setPrefix("±")
        for label, w in (("Epochs", self.epochs), ("Steps", self.steps), ("Patches", self.batch)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(w)
        # Augmentation: each training patch changed at random, so the model learns what does not matter
        aug = QtWidgets.QHBoxLayout()
        aug.addWidget(QtWidgets.QLabel("Augmentation:"))
        self.flips = QtWidgets.QCheckBox("Mirror")
        self.flips.setToolTip("Half the training patches mirrored (outlines have no preferred side)")
        self.turns = QtWidgets.QCheckBox("Quarter turns")
        self.turns.setToolTip("Each training patch turned by 0–3 quarter turns (outlines have no preferred "
                              "direction)")
        self.intensity = _spin(0, 50, 15, "Brightness / contrast", "Brightness and contrast of each training patch "
                                                                    "varied by up to ±this, so the model tolerates "
                                                                    "small differences in illumination and staining "
                                                                    "(0: unchanged)")
        self.intensity.setSuffix(" %")
        self.intensity.setPrefix("±")
        for w in (self.flips, self.turns):
            w.setChecked(True)
            aug.addWidget(w)
        for label, w in (("Brightness / contrast", self.intensity), ("Scale", self.jitter)):
            aug.addWidget(QtWidgets.QLabel(label))
            aug.addWidget(w)
        aug.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Device"))
        self.device = QtWidgets.QComboBox()
        self.device.setToolTip("Where PyTorch trains (found at run time)")
        row.addWidget(self.device, 1)
        row.addWidget(QtWidgets.QLabel("Choose by"))
        self.criterion = QtWidgets.QComboBox()
        for key in cmp.CRITERIA:
            self.criterion.addItem(cmp.METRICS[key][0], key)
        self.criterion.setToolTip("The measure the best model is proposed by. Object F1: glands found one by one "
                                  "(counting, grading). PQ: found and well outlined. Object Dice: the GlaS "
                                  "challenge's measure. Pixel Dice: area only.")
        row.addWidget(self.criterion)
        b.addLayout(row)
        b.addLayout(aug)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit()
        self.name.setToolTip("The run's name: its folder, its report and its models are named after it")
        row.addWidget(self.name, 1)
        self.train_btn = QtWidgets.QPushButton("Train")
        self.train_btn.setToolTip("Train the models one after another in the background, judge, time and compare "
                                  "them, and write the report")
        self.train_btn.clicked.connect(self.train)
        row.addWidget(self.train_btn)
        b.addLayout(row)
        self.note = muted("")
        self.note.setWordWrap(True)
        b.addWidget(self.note)

        # 4. The runs, progress (every model) and what to do with the result
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Run"))
        self.runs = QtWidgets.QComboBox()
        self.runs.setToolTip("This training set's runs, the newest first: picking one shows its curves and results, "
                             "and puts its models in the table")
        self.runs.activated.connect(self._run_picked)
        row.addWidget(self.runs, 1)
        b.addLayout(row)
        self.curve = Curves()
        b.addWidget(self.curve, 1)
        row = QtWidgets.QHBoxLayout()
        self.report_btn = QtWidgets.QPushButton("Open report")
        self.report_btn.setToolTip("The run's report (in its folder)")
        self.report_btn.clicked.connect(lambda: open_in_system(str(self.run.report)) if self.run else None)
        self.keep_btn = QtWidgets.QPushButton("Keep the selected model")
        self.keep_btn.setToolTip("Install it in your models (Segment and the profiles see it)")
        self.keep_btn.clicked.connect(lambda: self.keep())
        self.use_btn = QtWidgets.QPushButton("Use in profile")
        self.use_btn.setToolTip("Make the selected model (kept first) the profile's model for this group and image "
                                "type")
        self.use_btn.clicked.connect(self.use_in_profile)
        self.eval_btn = QtWidgets.QPushButton("Evaluate…")
        self.eval_btn.setToolTip("Judge the selected model in Evaluate (kept first), where its objects can be looked "
                                 "at and it can be compared with other models")
        self.eval_btn.clicked.connect(self._to_evaluate)
        self.folder_btn = QtWidgets.QPushButton("Folder")
        self.folder_btn.setToolTip("The run's folder: its report, its results file and its models")
        self.folder_btn.clicked.connect(self._open_folder)
        self.pdf_btn = QtWidgets.QPushButton("PDF")
        self.pdf_btn.setToolTip("The run's report as a PDF (A4 across), beside it in the run's folder, and open it")
        self.pdf_btn.clicked.connect(lambda: self.run and _pdf(self, self.run.report))
        self.xlsx_btn = QtWidgets.QPushButton("XLSX…")
        self.xlsx_btn.setToolTip("The run as an Excel workbook: every measure with its interval, per project, per "
                                 "image, the training history, and native charts")
        self.xlsx_btn.clicked.connect(self._export_xlsx)
        for w in (self.report_btn, self.pdf_btn, self.xlsx_btn, self.keep_btn, self.use_btn, self.eval_btn,
                  self.folder_btn):
            row.addWidget(w)
        self.result_note = muted("")
        self.result_note.setWordWrap(True)
        row.addWidget(self.result_note, 1)
        b.addLayout(row)

        self.modality.currentIndexChanged.connect(self._modality_changed)
        self.um.valueChanged.connect(self._um_changed)
        self.jitter.valueChanged.connect(lambda *_a: self._show_scales())
        for w in (self.epochs, self.steps, self.batch, self.jitter):
            w.valueChanged.connect(lambda *_a: self._update_note())
        self._update_buttons()

    # ------------------------------------------------------------ showing a set
    def refresh(self) -> None:
        """Show the training set's settings and models to train (after it is opened or changed)."""
        tset = self.win.tset
        self.setEnabled(tset is not None and bool(tset.projects))
        if tset is None or not self.body.isEnabled():
            return
        st_ = tset.segmentation.get("train", {})
        self._filling = True
        self.modality.setCurrentIndex(max(0, self.modality.findData(st_.get("modality", "phase"))))
        self._um_saved, self._um_touched = "um_per_px" in st_, False
        self.um.setValue(float(st_.get("um_per_px", 0.314)))
        self.jitter.setValue(int(round(100 * float(st_.get("scale_jitter", 0.0)))))
        self.flips.setChecked(bool(st_.get("flips", True)))
        self.turns.setChecked(bool(st_.get("turns", True)))
        self.intensity.setValue(int(round(100 * float(st_.get("intensity", 0.15)))))
        for w, key in ((self.epochs, "epochs"), (self.steps, "steps"), (self.batch, "batch_size")):
            w.setValue(int(st_.get(key, w.value())))
        self.criterion.setCurrentIndex(max(0, self.criterion.findData(st_.get("criterion", "f1"))))
        from modalytics.training.compare import Planned
        saved = self._saved_rows(st_)
        rebuilt = [Planned.of(r).run_id for r in saved] != [p.run_id for p in self.planned()]
        if rebuilt:                                         # else kept, results too
            self.table.clear()
            for r in saved:
                self._add_row(r.get("kind", "new"), r.get("arch_id"), r.get("weights"), r.get("base"), save=False)
        self._filling = False
        if not self.name.text() or self.name.text() == self._auto_name:   # the set's name is its runs' folder
            self._auto_name = f"{tset.group} on {self.modality.currentText()}"
            self.name.setText(self._auto_name)
        self._list_runs()
        if rebuilt and self.comparison is not None:         # the rows are new: the run's results again
            self._show_results(self.comparison)
        self._update_note()
        self._scan()

    @staticmethod
    def _saved_rows(st_: dict) -> list[dict]:
        """The models to train saved with the set (older sets: their one model)."""
        if "rows" in st_:
            return list(st_["rows"])
        if st_.get("start") == "model" and st_.get("base"):
            return [{"kind": "finetune", "base": st_["base"]}]
        if st_:
            return [{"kind": "new", "arch_id": st_.get("arch_id", "unet_resnet34"),
                     "weights": st_.get("start", "imagenet")}]
        return [{"kind": "new", "arch_id": "unet_resnet34", "weights": "imagenet"}]

    def _pack_ready(self) -> None:
        """A first training pack was installed and loaded: training can start."""
        from modalytics.training import seg_train as st
        ok, _why = st.available()
        self.body.setEnabled(ok)
        self.missing.setVisible(False)
        self._list_devices()
        self.refresh()                                      # what was skipped while training was not available

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        """The devices are listed when the tab is first shown (finding them loads PyTorch, which takes seconds)."""
        super().showEvent(event)
        self._list_devices()

    def _list_devices(self) -> None:
        if self.device.count() == 0 and self.body.isEnabled():
            from modalytics.training import encoders as en
            from modalytics.training import seg_train as st
            for d in st.devices():
                self.device.addItem(d.label, d.key)
            if en.catalogue(wait=False) is None:          # ImageNet is then offered only where it exists
                en.load_catalogue(lambda _cat: self.catalogue_ready.emit())

    # ------------------------------------------------------------ the set's runs
    def _list_runs(self) -> None:
        """The set's runs in the Run list, the newest first; the last one shown (the one shown kept on a refresh)."""
        from modalytics.training import runs as rn
        tset = self.win.tset
        found = rn.list_runs(tset) if tset is not None else []
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
            self.runs.addItem("No runs yet (Train makes one)", "")
        kept = next((r for r in found if r.folder == current), None)
        shown = kept or (found[0] if found else None)
        if shown is not None:
            self.runs.setCurrentIndex(self.runs.findData(str(shown.folder)))
        self.runs.blockSignals(False)
        self._enable_runs()
        if kept is None:
            self._open_run(shown, replace_rows=False)

    def _enable_runs(self) -> None:
        self.runs.setEnabled(bool(self.runs.currentData()) and not self.win.tasks.busy)

    def _run_picked(self, index: int) -> None:
        from pathlib import Path

        from modalytics.training import runs as rn
        folder = self.runs.itemData(index)
        if folder and (self.run is None or Path(folder) != self.run.folder):
            self._open_run(rn.Run.of(Path(folder)), replace_rows=True)

    def _open_run(self, run, replace_rows: bool) -> None:
        """Show a run: its curves, and its results in the rows of the same models.

        Args:
            run: The ``training.runs.Run`` (None: nothing shown).
            replace_rows: Put the run's models in the table first (a run picked from the list).
        """
        from modalytics.training import runs as rn
        self._clear_results()
        self.comparison, self.run, self.kept, self.curves = None, run, {}, {}
        self.folder = run.folder if run is not None else None
        if run is not None:
            try:
                self.comparison = rn.load(run)
            except Exception as exc:                                     # noqa: BLE001 (said, not fatal)
                self.result_note.setText(f"⚠ The run's results could not be read: {exc}")
        comp = self.comparison
        if comp is not None:
            if replace_rows:
                self._rows_of(comp)
            self.curves = {e.run_id: list(e.history) for e in comp.entries if e.history}
            self.kept = self._installed(comp)
            self._show_results(comp)
        self._show_curves()
        self._update_buttons()

    def _rows_of(self, comp) -> None:
        """The table: the run's models (a fine-tuned one whose base is no longer installed is said, left out)."""
        from modalytics.services.segmentation import find_models
        installed = {m.id for m in find_models()}
        self.table.clear()
        missing = []
        for e in comp.entries:
            if e.kind == "reference":
                continue
            if e.kind == "finetune":
                if e.base in installed:
                    self._add_row("finetune", base=e.base, save=False)
                else:
                    missing.append(e.label)
            else:
                self._add_row("new", e.arch_id, e.weights, save=False)
        self._row_changed()
        if missing:
            self.win.statusBar().showMessage("⚠️ Not in the table (the model it started from is no longer "
                                             "installed): " + ", ".join(missing), 10000)

    def _clear_results(self) -> None:
        """Every row's state and measures emptied (its "as it was" row removed)."""
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            item.takeChildren()
            for c in range(C_STATE, len(COLUMNS)):
                item.setText(c, "")
                item.setToolTip(c, "")
        self.result_note.setText("")

    # ------------------------------------------------------------ the models to train
    def _colour(self, item) -> QtGui.QColor:
        """A row's colour: the curves' and its icon's (given when the row is made, kept when the rows are sorted)."""
        pal = _palette()
        return QtGui.QColor(pal[int(item.data(0, COLOUR) or 0) % len(pal)])

    def _add_row(self, kind: str, arch_id: str | None = None, weights: str | None = None, base: str | None = None,
                 save: bool = True):
        """A row: a new model (architecture + starting weights) or a fine-tuned one (an existing model)."""
        from modalytics.training.architectures import ArchitectureLibrary
        taken = {self.table.topLevelItem(i).data(0, COLOUR) for i in range(self.table.topLevelItemCount())}
        item = SortItem(self.table, [""] * len(COLUMNS))
        item.setData(0, RUN + 1, kind)
        item.setData(0, COLOUR, next(k for k in range(len(taken) + 1) if k not in taken))   # the first free
        model = QtWidgets.QComboBox()
        start = None
        if kind == "finetune":
            self._fill_bases(model, base)
            if model.count() == 0:
                self.table.takeTopLevelItem(self.table.indexOfTopLevelItem(item))
                QtWidgets.QMessageBox.information(self, "Fine-tune a model", "No installed model of this image type "
                                                  "carries its training weights (made in Modalytics, or the "
                                                  "released ones).")
                return None
            item.setText(C_START, "its own weights")
            item.setToolTip(C_START, "Every weight starts from the model (encoder and decoder); it keeps its "
                                     "architecture, image type and µm/px")
        else:
            bold = QtGui.QFont(model.font())
            bold.setBold(True)
            for key, spec in ArchitectureLibrary().specs.items():
                model.addItem(spec.name + ("  · yours" if spec.source == "user" else ""), key)
                parts = "a network of its own" if spec.custom else f"{spec.decoder} with a {spec.encoder} encoder"
                model.setItemData(model.count() - 1, f"{spec.description}\n({parts})"
                                  + ("\nRecommended (Settings › Architectures)" if spec.recommended else ""),
                                  QtCore.Qt.ItemDataRole.ToolTipRole)
                if spec.recommended:                        # the recommended ones stand out
                    model.setItemData(model.count() - 1, bold, QtCore.Qt.ItemDataRole.FontRole)
            model.setCurrentIndex(max(0, model.findData(arch_id or "unet_resnet34")))
            start = QtWidgets.QComboBox()
            start.setToolTip("Where the encoder starts: weights learned elsewhere (downloaded once), or nothing")
            self.table.setItemWidget(item, C_START, start)
            self._fill_weights(model.currentData(), start, weights or "imagenet")
            model.currentIndexChanged.connect(lambda *_a, m=model, s=start: (self._fill_weights(m.currentData(), s),
                                                                            self._row_changed()))
            start.currentIndexChanged.connect(lambda *_a: self._row_changed())
        if kind == "finetune":
            model.currentIndexChanged.connect(lambda *_a: self._row_changed())
        self.table.setItemWidget(item, C_MODEL, model)
        self._paint_rows()
        self._row_changed(save)
        return item

    def _fill_bases(self, combo: QtWidgets.QComboBox, chosen: str | None = None) -> None:
        """The installed models of this image type that carry their training weights."""
        from modalytics.services.segmentation import find_models
        from modalytics.training import seg_train as st
        keep = chosen or combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for m in find_models():
            if m.modality == self.modality.currentData() and st.trainable(m):
                combo.addItem(f"Fine-tune: {m.name}", m.id)
                combo.setItemData(combo.count() - 1, m.um_per_px, RUN + 2)
        combo.setCurrentIndex(max(0, combo.findData(keep)))
        combo.blockSignals(False)

    def _fill_weights(self, arch_id: str, combo: QtWidgets.QComboBox, chosen: str | None = None) -> None:
        """The starting points that fit the architecture's encoder and the image type, then "nothing"."""
        from modalytics.training.architectures import ArchitectureLibrary
        from modalytics.training.encoders import EncoderLibrary
        keep = chosen or combo.currentData()
        spec = ArchitectureLibrary().get(arch_id or "")
        combo.blockSignals(True)
        combo.clear()
        if spec is not None:
            for w in EncoderLibrary().for_encoder(spec.encoder, self.modality.currentData()):
                combo.addItem(f"{w.name}" + ("" if w.redistributable else "  (non-commercial)"), w.key)
                combo.setItemData(combo.count() - 1, f"{w.description}\nLicence: {w.licence}",
                                  QtCore.Qt.ItemDataRole.ToolTipRole)
        combo.addItem("nothing (random weights: needs many images)", "scratch")
        combo.setCurrentIndex(max(0, combo.findData(keep)))
        combo.blockSignals(False)

    def _refill_rows(self) -> None:
        """Every row's lists again (the image type changed, or ImageNet's encoders became known)."""
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            model = self.table.itemWidget(item, C_MODEL)
            if item.data(0, RUN + 1) == "finetune":
                self._fill_bases(model)
            else:
                self._fill_weights(model.currentData(), self.table.itemWidget(item, C_START))
        self._row_changed()

    def _remove_row(self) -> None:
        item = self.table.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
        if item is None:
            return
        self.table.takeTopLevelItem(self.table.indexOfTopLevelItem(item))
        self._paint_rows()
        self._row_changed()

    def planned(self) -> list:
        """The rows as ``training.compare.Planned``."""
        from modalytics.training.compare import Planned
        out = []
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            model = self.table.itemWidget(item, C_MODEL)
            if item.data(0, RUN + 1) == "finetune":
                out.append(Planned("finetune", base=model.currentData() or ""))
            else:
                start = self.table.itemWidget(item, C_START)
                out.append(Planned("new", model.currentData() or "", start.currentData() or "imagenet"))
        return out

    def _paint_rows(self) -> None:
        """Each row's colour (the curves', the report's) and run id."""
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            pix = QtGui.QPixmap(12, 12)
            pix.fill(QtCore.Qt.GlobalColor.transparent)
            with QtGui.QPainter(pix) as p:
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                p.setBrush(self._colour(item))
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawEllipse(1, 1, 10, 10)
            item.setIcon(0, QtGui.QIcon(pix))

    def _row_changed(self, save: bool = True) -> None:
        """The rows changed: their run ids, resolutions and results (cleared where they no longer match)."""
        planned = self.planned()
        for i, p in enumerate(planned):
            item = self.table.topLevelItem(i)
            if item.data(0, RUN) != p.run_id:               # another model now: its old results go
                item.setData(0, RUN, p.run_id)
                for c in range(C_STATE, len(COLUMNS)):
                    item.setText(c, "")
                item.takeChildren()
            if p.kind == "finetune":
                um = self.table.itemWidget(item, C_MODEL).currentData(RUN + 2)
                item.setText(C_UM, f"{um:.4g}" if um else "")
            else:
                item.setText(C_UM, f"{self.um.value():.4g}")
        if save and not self._filling and self.win.tset is not None:
            self.win.tset.segmentation.setdefault("train", {})["rows"] = [
                {"kind": p.kind, "arch_id": p.arch_id, "weights": p.weights, "base": p.base} for p in planned]
            self.win._save()
        self._update_note()
        self._show_curves()

    # ------------------------------------------------------------ resolution
    def _modality_changed(self, *_args) -> None:
        if self._filling:
            return
        self._refill_rows()
        self._scan()

    def _scan(self) -> None:
        """Read the training images' pixel sizes in the background (the projects of the set, this image type)."""
        import threading

        from modalytics.training import seg_train as st
        tset = self.win.tset
        if tset is None or self._filling:
            return
        key = (tuple(p.path for p in tset.projects), tset.group, self.modality.currentData())
        self._scan_key = key
        self.scale_note.setText("Reading the training images' pixel sizes…")

        def work():
            try:
                scales, skipped = st.image_scales(list(key[0]), key[1], key[2])
            except Exception as exc:                                     # noqa: BLE001 (said, not fatal)
                scales, skipped = [], [str(exc)]
            self.scales_ready.emit((key, scales, skipped))
        threading.Thread(target=work, daemon=True).start()

    def _scales_found(self, result) -> None:
        key, scales, skipped = result
        if key != self._scan_key:                          # an older request: the settings changed since
            return
        self.scales, self.scales_skipped = scales, skipped
        from modalytics.training import seg_train as st
        proposal = st.resolution_report(scales)["proposal"]
        if proposal and not self._um_saved and not self._um_touched:   # no resolution of its own yet: take it
            self._filling = True
            self.um.setValue(proposal)
            self._filling = False
            self._row_changed(save=False)
        self._show_scales()

    def _um_changed(self, *_args) -> None:
        if not self._filling:
            self._um_touched = True
            self._row_changed()
        self._show_scales()

    def _use_proposal(self) -> None:
        from modalytics.training import seg_train as st
        proposal = st.resolution_report(self.scales)["proposal"]
        if proposal:
            self.um.setValue(proposal)

    def _show_scales(self) -> None:
        """The line under the settings: the images' pixel sizes, what they become, and what to check."""
        import html

        from modalytics.training import seg_train as st
        if self._scan_key is None or self.win.tset is None:
            return
        if not self.scales:
            self.scale_note.setText(f"No {html.escape(self.modality.currentText())} image with "
                                    f"{html.escape(self._scan_key[1])} outlined by hand in the projects yet.")
            self.use_scale.setVisible(False)
            return
        works = self.um.value()
        r = st.resolution_report(self.scales, works)
        found = " · ".join(f"{u:g} µm/px × {n}" for u, n, _names in r["groups"])
        same = len(r["groups"]) == 1 and abs(r["groups"][0][0] - works) / works < 1e-3
        lines = [f"Training images: {found} → {'used as they are' if same else f'resampled to {works:.4g} µm/px'} "
                 f"for new models; a {st.PATCH} px tile covers {r['tile_um']:.0f} µm."]
        if r["outliers"]:
            n, (name, um, ratio) = len(r["outliers"]), r["outliers"][0]
            lines.append(f"⚠ {html.escape(name)}{f' and {n - 1} more' if n > 1 else ''}: {um:g} µm/px, "
                         f"{ratio:.3g}× the {um / ratio:g} µm/px of more images — a power of ten: is the pixel size "
                         f"stored right (a decimal point or unit slip)?")
        if r["enlarged"]:
            worst = max(r["enlarged"], key=lambda e: e[2])
            lines.append(f"ℹ {len(r['enlarged'])} image(s) coarser than {works:.4g} µm/px are enlarged (up to "
                         f"{worst[2]:.2g}×, {html.escape(worst[0])}): blurred, no detail gained. For their own "
                         f"detail, train them as a separate model working at their resolution, or work at a "
                         f"coarser one.")
        if len(r["groups"]) > 1 and self.jitter.value() == 0:
            lines.append("Images of several resolutions: Scale ±10 % makes the model tolerant to small calibration "
                         "differences between systems.")
        self.scale_note.setText("<br>".join(lines))
        self.scale_note.setToolTip("\n".join(f"{u:g} µm/px: " + ", ".join(names[:12])
                                              + (" …" if len(names) > 12 else "") for u, _n, names in r["groups"]))
        proposal = r["proposal"]
        self.use_scale.setText(f"Use {proposal:g} µm/px" if proposal else "Use")
        self.use_scale.setVisible(bool(proposal and abs(proposal - works) / proposal > 1e-3))

    def _update_note(self) -> None:
        tset = self.win.tset
        if tset is None:
            return
        held = ds.evaluation_projects(tset, self.win.data)
        n_ev, n_test = len(held), len(ds.test_projects(tset))
        n = self.table.topLevelItemCount()
        judged = (f"judged on the <b>{n_test} test project(s)</b> (never trained on, never used for choosing)"
                  if n_test else f"judged on the {n_ev} evaluation project(s) (no test projects: somewhat "
                                 f"optimistic — mark a few as Test in Data)")
        self.note.setText(
            f"{n} model(s) to outline <b>{tset.group}</b> (all its subgroups), trained one after another from the "
            f"{len(ds.training_projects(tset, self.win.data))} training project(s); the {n_ev} evaluation "
            f"project(s) choose each one's best epoch"
            + ("" if n_ev else " — <b>none marked</b>: the last epoch is kept (Data tab)")
            + f"; {judged}. Where the outlines are complete: as set in Evaluate "
              f"({self.evaluate_panel.area.currentText()})."
            + (f" Takes about {n}× one training." if n > 1 else ""))
        self.train_btn.setEnabled(n > 0 and not self.win.tasks.busy)
        self._enable_runs()

    def _settings(self):
        from modalytics.training import seg_train as st
        ev = self.evaluate_panel
        return st.Settings(start="imagenet", modality=self.modality.currentData(), um_per_px=self.um.value(),
                           epochs=self.epochs.value(), steps=self.steps.value(), batch_size=self.batch.value(),
                           scale_jitter=self.jitter.value() / 100, flips=self.flips.isChecked(),
                           turns=self.turns.isChecked(), intensity=self.intensity.value() / 100,
                           area=ev.area.currentData(),
                           margin_um=ev.margin.value(), device=self.device.currentData() or "",
                           seed=self.win.tset.seed)

    # ------------------------------------------------------------ training
    def train(self) -> None:
        """Train the models one after another in the background; the curves follow each epoch of each."""
        from modalytics.report import comparison as rep
        from modalytics.training import compare as cmp
        from modalytics.training import runs as rn
        from modalytics.training import segmentation as se
        tset = self.win.tset
        runs, left = cmp.plan(self.planned())
        if left and QtWidgets.QMessageBox.question(
                self, "Train", "These cannot be trained and are left out:\n\n" + "\n".join(f"  {x}" for x in left)
                + ("\n\nTrain the others?" if runs else "")) != QtWidgets.QMessageBox.StandardButton.Yes or not runs:
            return
        settings = self._settings()
        rows = lambda idx: [x.path for i, x in enumerate(tset.projects) if i in idx]
        train_paths = rows(ds.training_projects(tset, self.win.data))        # test projects: never trained on
        eval_paths = rows(ds.evaluation_projects(tset, self.win.data))
        test_paths = rows(ds.test_projects(tset))
        groups = {g: [tset.projects[i].name for i in sorted(idx)] for g, idx in ds.test_groups(tset).items()}
        group, criterion = tset.group, self.criterion.currentData()
        waiting = se.unreviewed(train_paths + eval_paths + test_paths, group)
        if waiting and QtWidgets.QMessageBox.question(
                self, "Unreviewed suggestions",
                f"These projects still have unreviewed {group} suggestions:\n\n"
                + "\n".join(f"  {n}: {k}" for n, k in waiting.items())
                + "\n\nTheir places are left out of training and judging (neither object nor background). Review "
                  "them first (Annotate › Review) for the best models.\n\nTrain anyway?"
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        name = self.name.text().strip() or "Training run"
        run = rn.Run.new(rn.set_folder(tset, create=True), "segmentation", name)    # the set claims its folder
        self.run, self.folder = run, run.folder
        results, report_path = run.results, run.report
        tset.segmentation["train"] = dict(tset.segmentation.get("train", {}), modality=settings.modality,
                                          um_per_px=settings.um_per_px, epochs=settings.epochs, steps=settings.steps,
                                          batch_size=settings.batch_size, scale_jitter=settings.scale_jitter,
                                          flips=settings.flips, turns=settings.turns,
                                          intensity=settings.intensity, criterion=criterion)
        self.win._save()
        self.comparison, self.kept, self.curves = None, {}, {}
        self._clear_results()
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            item.setText(C_STATE, "waiting" if any(p.run_id == item.data(0, RUN) for p in runs) else "left out")
        self._show_curves()
        self._update_buttons()
        self.train_btn.setEnabled(False)
        self.runs.setEnabled(False)
        bridge, folder, set_name = self.bridge, self.folder, tset.name

        def job(progress, cancelled):
            from modalytics.services.segmentation import Cancelled
            from modalytics.training import seg_train as st

            def report(d):
                progress(d.get("fraction", 0.0))
                if "run" in d:
                    bridge.report.emit(dict(d))
            try:
                comp = cmp.run(runs, settings, train_paths, eval_paths, test_paths, group, folder, name,
                               settings.area, settings.margin_um, criterion=criterion, progress=report,
                               cancelled=cancelled, results=results, models_dir=run.models,
                               training_set=set_name, test_groups=groups)
            except st.Cancelled:
                raise Cancelled() from None
            rep.write(comp, str(report_path))
            return comp

        def done(comp) -> None:
            self.comparison = comp
            self._show_results(comp)
            self._update_note()
            trained = [e for e in comp.done() if e.kind != "reference"]
            if len(trained) == 1 and len([e for e in comp.entries if e.kind != "reference"]) == 1:
                self.keep(trained[0], quiet=True)           # a run of one model: kept by itself
            self.win.statusBar().showMessage(f"✅ Trained {len(trained)} model(s): {name}", 10000)
            self._list_runs()                                # the new run, at the top of the list
            self._update_note()

        def stopped() -> None:
            self._list_runs()                                # the models finished before it are in its results
            self._update_note()
            self._update_buttons()

        def failed(exc) -> None:
            stopped()
            QtWidgets.QMessageBox.warning(self, "Train", f"Training stopped:\n\n{exc}")

        if not self.win.tasks.run(f"Training {len(runs)} model(s)…", job, done, failed, on_cancel=stopped):
            self._update_note()

    def _row_of(self, run_id: str):
        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            if item.data(0, RUN) == run_id:
                return i, item
        return -1, None

    def _report(self, d: dict) -> None:
        """A step of the run: the model's state (epoch, judging), its curves."""
        _i, item = self._row_of(d.get("run", ""))
        if item is None:
            return
        if "epoch" in d and "loss" in d:
            rows = [r for r in self.curves.get(d["run"], []) if r["epoch"] != d["epoch"]] + [d]
            self.curves[d["run"]] = rows
            item.setText(C_STATE, f"epoch {d['epoch']} of {self.epochs.value()}")
            self._show_curves()
        else:
            stage = d.get("stage", "").split(": ", 1)[-1].rstrip("…")
            if stage and not stage.startswith("Epoch"):
                item.setText(C_STATE, stage)

    def _show_curves(self) -> None:
        series = {}
        for run_id, rows in self.curves.items():
            _i, item = self._row_of(run_id)
            if item is not None:
                series[run_id] = (self._colour(item), rows)
        current = self.table.currentItem()
        epochs = self.epochs.value()                      # while training; a run shown: its own
        if self.comparison is not None:
            epochs = max((r["epoch"] for rs in self.curves.values() for r in rs),
                         default=int(self.comparison.settings.get("epochs", epochs)))
        self.curve.show_series(series, epochs, current.data(0, RUN) if current else "")

    def _show_results(self, comp) -> None:
        """Each model's measures on the judged projects; the proposal starred; a fine-tuned model's start below it."""
        from modalytics.training import compare as cmp
        best = comp.best() if len(comp.done()) > 1 else None
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter

        def fill(item, e):
            ev = e.judged(comp.judged_on)
            state = e.error or ("for reference (not retrained)" if e.kind == "reference" else
                                f"done (epoch {e.best_epoch} kept)" if ev is not None else "not judged")
            item.setText(C_STATE, ("★ " if e is best else "") + state)
            item.setToolTip(C_STATE, e.error or ("proposed: the best by " + cmp.METRICS[comp.criterion][0]
                                                 if e is best else ""))
            for c, key in enumerate(SHOWN, start=C_STATE + 1):
                v = cmp.METRICS[key][1](ev.total) if ev is not None else math.nan
                item.setText(c, "—" if not math.isfinite(v) else cmp.METRICS[key][3].format(v))
            tail = ["—" if not math.isfinite(e.ms_per_tile) else f"{e.ms_per_tile:.0f}",
                    "—" if not math.isfinite(e.size_mb) else f"{e.size_mb:.0f} MB",
                    "—" if e.kind == "reference" else _duration(e.seconds)]
            for c, v in enumerate(tail, start=C_STATE + len(SHOWN) + 1):
                item.setText(c, v)
            for c in range(C_STATE + 1, len(COLUMNS)):
                item.setTextAlignment(c, right)
        for e in comp.entries:
            if e.kind == "reference":
                _i, parent = self._row_of(f"ft:{e.base}")
                if parent is None:
                    continue
                child = QtWidgets.QTreeWidgetItem(parent, [""] * len(COLUMNS))
                child.setText(C_MODEL, f"↳ {e.label}")
                child.setToolTip(C_MODEL, f"{e.label}: the model it started from, not retrained, judged the same way")
                child.setData(0, RUN, e.run_id)
                if math.isfinite(e.um_per_px):
                    child.setText(C_UM, f"{e.um_per_px:.4g}")
                fill(child, e)
                parent.setExpanded(True)
                continue
            _i, item = self._row_of(e.run_id)
            if item is not None:
                fill(item, e)
            if e is best and item is not None:
                self.table.setCurrentItem(item)
        on = "test" if comp.judged_on == "test" else "evaluation (no test projects: optimistic)"
        self.result_note.setText(
            (f"★ proposed: {best.label} (best {cmp.METRICS[comp.criterion][0]} on the {on} projects). " if best else "")
            + "The report has the measures with their intervals"
            + (", the differences, " if best else ", ") + "the curves and examples; you choose what to keep.")
        self._update_buttons()

    # ------------------------------------------------------------ after training
    def _selected(self):
        """The run's entry of the selected row (a trained model, or a fine-tuned model's start as it was)."""
        item = self.table.currentItem()
        if item is None or self.comparison is None:
            return None
        run_id = item.data(0, RUN)
        return next((e for e in self.comparison.entries if e.run_id == run_id and e.path and not e.error), None)

    def _export_xlsx(self) -> None:
        if self.run is not None and self.run.results.exists():
            _xlsx(self, self.run)

    def _update_buttons(self) -> None:
        e = self._selected()
        done = self.comparison is not None
        self.report_btn.setEnabled(done and self.run is not None and self.run.report.exists())
        self.pdf_btn.setEnabled(self.report_btn.isEnabled())
        self.xlsx_btn.setEnabled(done and self.run is not None and self.run.results.exists())
        self.folder_btn.setEnabled(True)
        self.keep_btn.setEnabled(e is not None and e.kind != "reference" and e.run_id not in self.kept)
        for w in (self.use_btn, self.eval_btn):
            w.setEnabled(e is not None)

    @staticmethod
    def _installed(comp) -> dict:
        """{run id: model id} of the run's models already in your models (kept before: Keep copies the file as it
        is, under another name, so a copy has the same size and the same manifest)."""
        import os
        import zipfile

        from modalytics.services.segmentation import find_models

        def fingerprint(path: str):
            try:
                with zipfile.ZipFile(path) as z:
                    return os.path.getsize(path), z.getinfo("manifest.json").CRC
            except (OSError, KeyError, zipfile.BadZipFile):
                return None
        mine = [e for e in comp.entries if e.kind != "reference" and e.path and os.path.exists(e.path)]
        if not mine:
            return {}
        installed = {}
        for m in find_models():
            f = fingerprint(m.path)
            if f is not None:
                installed[f] = m.id
        out = {}
        for e in mine:
            pid = installed.get(fingerprint(e.path))
            if pid is not None:
                out[e.run_id] = pid
        return out

    def _kept_as(self, entry):
        """Where a run's model is installed when kept (in your models; named after the run and the model)."""
        from modalytics import paths
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{self.folder.name}_{entry.run_id}").strip("_")
        return paths.user_dir("models") / f"{safe}.mmodel"

    def keep(self, entry=None, quiet: bool = False):
        """Install a trained model (the selected one by default) in your models; returns its package."""
        from modalytics.services.segmentation import find_models, load_package
        entry = entry or self._selected()
        if entry is None:
            return None
        if entry.kind == "reference":                      # already installed: the model it started from
            return next((m for m in find_models() if m.id == entry.base), None)
        if entry.run_id in self.kept:
            return next((m for m in find_models() if m.id == self.kept[entry.run_id]), None)
        target = self._kept_as(entry)
        if not target.exists():
            shutil.copy2(entry.path, target)
        pkg = load_package(str(target))
        self.kept[entry.run_id] = pkg.id
        self.evaluate_panel.refresh_models()
        self.result_note.setText(f"✅ Kept as {pkg.name} (in your models)." + ("" if quiet else
                                 " Use in profile makes it the profile's model."))
        self._update_buttons()
        return pkg

    def use_in_profile(self) -> None:
        """Make the selected model (kept first) the profile's model for its group and image type."""
        pkg = self.keep()
        if pkg is not None:
            self.offer_to_profile(pkg)

    def offer_to_profile(self, pkg) -> None:
        """Ask, then make ``pkg`` the profile's model for its group and image type."""
        tset = self.win.tset
        group = pkg.output_class[0]
        try:
            prof = self.win.profiles.get(tset.profile_id)
        except Exception as exc:                                         # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Use in profile", f"The profile could not be read:\n\n{exc}")
            return
        old = prof.model_for(group, pkg.modality)
        ask = (f"Make '{pkg.name}' the profile '{prof.name}' model for {group} on {pkg.modality} images?"
               + (f"\n\nIt replaces: {old}." if old else ""))
        if QtWidgets.QMessageBox.question(self, "Use in profile", ask) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        prof.segmentation.setdefault(group, {})[pkg.modality] = pkg.id
        self.win.profiles.save(prof, tset.profile_id)
        self.win.statusBar().showMessage(f"✅ {prof.name}: {group} on {pkg.modality} images is segmented by "
                                         f"{pkg.name}", 8000)

    def _to_evaluate(self) -> None:
        pkg = self.keep()
        if pkg is None:
            return
        ev = self.evaluate_panel
        ev.select_models([pkg.path])
        self.win.segmentation.setCurrentWidget(ev)
        ev.evaluate()

    def _open_folder(self) -> None:
        """The run's folder; without one, the set's runs (or all the sets' runs)."""
        from modalytics.training import runs as rn
        tset = self.win.tset
        for folder in (self.folder, rn.set_folder(tset) / "segmentation" if tset is not None else None, rn.root()):
            if folder is not None and folder.is_dir():
                open_in_system(folder)
                return


def _xlsx(panel, run) -> None:
    from modalytics.report import xlsx
    from modalytics.training import compare as cmp
    from modalytics.ui.training.export import save_xlsx
    comp = cmp.Comparison.load(run.results)
    save_xlsx(panel, run.report, lambda path: xlsx.write_segmentation(comp, path), "the run")


def _pdf(panel, html_path) -> None:
    from modalytics.ui.pdf_export import report_pdf
    report_pdf(panel, html_path, status=panel.win.statusBar())


def _spin(lo: int, hi: int, value: int, _label: str, tip: str) -> QtWidgets.QSpinBox:
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setToolTip(tip)
    return s


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"
