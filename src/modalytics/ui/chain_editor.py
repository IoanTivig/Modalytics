"""
The processing chain editor (Process › Processing chains…; Settings › Profiles › Processing at import).

    ┌ Processing chain — sample01 ────────────────────────────────────────────┐
    │ Name [Denoised OD           ]  ☑ Measure with parameters   makes: OD map  │
    │ ┌ Steps ───────────────┐ ┌ Options of the selected step ┐  before  after  │
    │ │ 1  Median filter     │ │ Neighbourhood  [1.0 µm]      │  [img]   [img]  │
    │ │ 2  OD map            │ └──────────────────────────────┘                 │
    │ └──────────────────────┘                                                  │
    │ [Add step ▾] [Remove] [Up] [Down]                                          │
    │ My chains [Denoised OD ▾] [Load] [Delete]          [Save to my chains]    │
    │ [Apply to this image] [Apply to all whole-slide images]          [Close]  │
    └────────────────────────────────────────────────────────────────────────────┘

Each step works on the previous one's result; Add step offers only the steps
that accept what the chain makes at that point. The preview shows the selected
step's input and result on the image itself, on a coarse level (a whole slide
too), so changing an option shows its effect at once. *Measure with parameters*
decides whether the result feeds the parameters of its type (else it is only for
looking).

Used two ways: on an image of the project (apply it, or update the view it was
opened from: one undo step each), or for a profile (``for_type``: the preview
uses a sample image of that type, and OK hands the chain back).
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from modalytics.core.model import IMAGE_TYPES
from modalytics.processing.chains import Chain, ChainLibrary
from modalytics.processing.registry import ProcessingLibrary
from modalytics.ui.settings.importer_editor import preview
from modalytics.ui.settings.widgets import muted, setting_widget

PREVIEW = 230                 # px: side of the before / after pictures
PREVIEW_LEVEL_MAX = 1400      # px: the preview reads the finest level at most this large


PLURAL = {"wsi": "whole-slide images", "phase": "phase maps", "brightfield": "brightfield images",
          "od": "OD maps", "opd": "OPD maps", "dry_mass": "dry-mass maps", "height": "height maps",
          "ri": "refractive-index maps", "stain": "stain maps", "feature": "feature maps"}


def _type_name(t: str) -> str:
    return IMAGE_TYPES.get(t, t).split(" (")[0]


class ChainEditor(QtWidgets.QDialog):
    """Edit a processing chain, with a step-by-step preview.

    Args:
        image: The image it is shown on (None with ``for_type``: a sample image of that type).
        chain: The chain to start from (default: an empty one).
        project, undo: For Apply (None in profile mode).
        view: The view being edited (Apply then updates it).
        for_type: Profile mode: the image type the chain is for.
        library: The processing steps (default: all here).
    """

    def __init__(self, image=None, chain: Chain | None = None, project=None, undo=None, view=None,
                 for_type: str | None = None, library: ProcessingLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        from modalytics.ui.settings.parameter_editor import sample
        self.library = library or ProcessingLibrary()
        self.chains = ChainLibrary()
        self.project, self.undo, self.view = project, undo, view
        self.profile_mode = image is None
        self.image = image if image is not None else sample("image", for_type or "phase")[0]
        if image is None and for_type and self.image.modality != for_type:
            # A map type imported directly (OPD, OD…): the sample's pixels standing in for one of that type
            from modalytics.core.model import ImageNode
            s = self.image
            self.image = ImageNode(f"sample ({for_type})", s.source, s.pixel_size_um, for_type, s.placement,
                                   dict(s.metadata))
        self.chain = (chain or Chain("New chain", [], measure=True)).copy()
        self._loading = False
        self.setWindowTitle(f"Processing chain — {'profile, ' + _type_name(for_type) if self.profile_mode else image.name}")
        self.resize(980, 600)
        col = QtWidgets.QVBoxLayout(self)

        # 1. Name, measure, what it makes
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit(self.chain.name)
        self.name.textEdited.connect(self._changed)
        top.addWidget(self.name, 1)
        self.measure = QtWidgets.QCheckBox("Measure with parameters")
        self.measure.setChecked(self.chain.measure)
        self.measure.setToolTip("The result feeds the parameters of its image type (in a profile: the standard "
                                "results; applied by hand: extra, labelled results). Off: only for looking.")
        self.measure.toggled.connect(self._changed)
        top.addWidget(self.measure)
        self.makes = muted("", wrap=False)
        top.addWidget(self.makes)
        col.addLayout(top)

        # 2. Steps | options of the selected one | before / after
        mid = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(muted("Steps (each works on the previous one's result)"))
        self.steps = QtWidgets.QListWidget()
        self.steps.currentRowChanged.connect(self._step_selected)
        left.addWidget(self.steps, 1)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add step ▾")
        self.add_btn.clicked.connect(self._add_menu)
        row.addWidget(self.add_btn)
        for text, slot in (("Remove", self._remove), ("Up", lambda: self._move(-1)), ("Down", lambda: self._move(1))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        left.addLayout(row)
        mid.addLayout(left, 2)
        self.options_box = QtWidgets.QGroupBox("Options of the selected step")
        self.options_form = QtWidgets.QFormLayout(self.options_box)
        mid.addWidget(self.options_box, 2)
        pics = QtWidgets.QGridLayout()
        self.before, self.after = QtWidgets.QLabel(), QtWidgets.QLabel()
        for i, (title, w) in enumerate((("Before the step", self.before), ("After it", self.after))):
            pics.addWidget(muted(title, wrap=False), 0, i)
            w.setFixedSize(PREVIEW, PREVIEW)
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            w.setObjectName("Panel")
            pics.addWidget(w, 1, i)
        mid.addLayout(pics)
        col.addLayout(mid, 1)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        # 3. Your chains
        mine = QtWidgets.QHBoxLayout()
        mine.addWidget(QtWidgets.QLabel("My chains"))
        self.saved = QtWidgets.QComboBox()
        mine.addWidget(self.saved, 1)
        for text, slot in (("Load", self._load_saved), ("Delete", self._delete_saved)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            mine.addWidget(b)
        save = QtWidgets.QPushButton("Save to my chains")
        save.setToolTip("Keep it in your library, to apply it to other images or add it to profiles")
        save.clicked.connect(self._save_mine)
        mine.addWidget(save)
        col.addLayout(mine)

        # 4. Apply (project) or OK (profile)
        buttons = QtWidgets.QHBoxLayout()
        if self.profile_mode:
            buttons.addStretch(1)
            self.ok_btn = QtWidgets.QPushButton("OK")
            self.ok_btn.clicked.connect(self._ok)
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            buttons.addWidget(self.ok_btn)
            buttons.addWidget(cancel)
        else:
            self.apply_btn = QtWidgets.QPushButton("Update this view" if view is not None else "Apply to this image")
            self.apply_btn.clicked.connect(self._apply_here)
            buttons.addWidget(self.apply_btn)
            self.apply_all_btn = QtWidgets.QPushButton(
                f"Apply to all {PLURAL.get(self.image.modality, self.image.modality + ' images')}")
            self.apply_all_btn.clicked.connect(self._apply_all)
            buttons.addWidget(self.apply_all_btn)
            buttons.addStretch(1)
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(self.reject)
            buttons.addWidget(close)
        col.addLayout(buttons)

        self._fill_saved()
        self._fill_steps(0)

    # ------------------------------------------------------------ the steps
    def _fill_steps(self, select: int = 0) -> None:
        self.steps.blockSignals(True)
        self.steps.clear()
        for i, s in enumerate(self.chain.steps, start=1):
            spec = self.library.get(s["step"])
            self.steps.addItem(f"{i}  {spec.name if spec else s['step'] + ' (not available here)'}")
        self.steps.blockSignals(False)
        if self.chain.steps:
            self.steps.setCurrentRow(max(0, min(select, len(self.chain.steps) - 1)))
        else:
            self._step_selected(-1)
        self._update_state()

    def _type_before(self, index: int) -> str:
        """The image type entering step ``index`` (the image's, or what the steps before it make)."""
        part = Chain("part", self.chain.steps[:index])
        try:
            return part.output_type(self.image.modality, self.library)
        except ValueError:
            return self.image.modality

    def _add_menu(self) -> None:
        """Steps that accept what the chain makes after the selected step (or at its end)."""
        at = self.steps.currentRow() + 1 if self.chain.steps else 0
        current = self._type_before(at)
        menu = QtWidgets.QMenu(self)
        from modalytics.ui.processing import offered
        fits = self.library.for_type(current)
        for title, group in self.library.grouped(current):      # one submenu per category
            group = [s for s in group if offered(self.project, s.uid)]   # not the steps the profile hides
            if not group:
                continue
            sub = menu.addMenu(title)
            for spec in group:
                act = sub.addAction(spec.name, lambda s=spec, i=at: self._insert(s, i))
                act.setToolTip(spec.description)
            sub.setToolTipsVisible(True)
        if not fits:
            menu.addAction(f"No step processes {_type_name(current).lower()} images").setEnabled(False)
        menu.setToolTipsVisible(True)
        menu.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft()))

    def _insert(self, spec, index: int) -> None:
        self.chain.steps.insert(index, {"step": spec.uid, "options": spec.resolve()})
        if self.chain.name in ("New chain", "") or (len(self.chain.steps) == 1 and self.name.text() == "New chain"):
            self.chain.name = spec.name
            self.name.setText(spec.name)
        self._fill_steps(index)

    def _remove(self) -> None:
        i = self.steps.currentRow()
        if 0 <= i < len(self.chain.steps):
            del self.chain.steps[i]
            self._fill_steps(i - 1)

    def _move(self, step: int) -> None:
        i = self.steps.currentRow()
        j = i + step
        if 0 <= i < len(self.chain.steps) and 0 <= j < len(self.chain.steps):
            self.chain.steps[i], self.chain.steps[j] = self.chain.steps[j], self.chain.steps[i]
            self._fill_steps(j)

    def _step_selected(self, index: int) -> None:
        """Show the selected step's options and its before / after."""
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        if not 0 <= index < len(self.chain.steps):
            self.options_box.setTitle("Options of the selected step")
            self._preview(-1)
            return
        entry = self.chain.steps[index]
        spec = self.library.get(entry["step"])
        self.options_box.setTitle(f"Options of {spec.name if spec else entry['step']}")
        if spec is not None:
            if spec.description:
                self.options_form.addRow(muted(spec.description))
            values = spec.resolve(entry.get("options"))
            entry["options"] = values
            for o in spec.options:
                def changed(v, key=o.key, e=entry):
                    e["options"][key] = v
                    self._update_state()
                    self._preview(self.steps.currentRow())
                self.options_form.addRow(o.label, setting_widget(o, values[o.key], changed))
            if not spec.options:
                self.options_form.addRow(muted("This step has no options."))
        self._preview(index)

    # ------------------------------------------------------------ state, preview
    def _changed(self, *_args) -> None:
        self.chain.name = self.name.text().strip() or "Chain"
        self.chain.measure = self.measure.isChecked()
        self._update_state()

    def problems(self) -> list[str]:
        return self.chain.problems(self.image.modality, self.library)

    def _update_state(self) -> None:
        found = self.problems()
        if found:
            self.makes.setText("")
            self.status.setText("⚠ " + " · ".join(found))
        else:
            out = self.chain.output_type(self.image.modality, self.library)
            self.makes.setText(f"makes: {_type_name(out)}")
            self.status.setText("" if self.chain.measure else "ℹ Only for looking: its parameters are not measured.")
        ok = not found
        for b in ("ok_btn", "apply_btn", "apply_all_btn"):
            if hasattr(self, b):
                getattr(self, b).setEnabled(ok)

    def _level_pixels(self, node) -> np.ndarray:
        """The whole of ``node``'s finest level that is at most ``PREVIEW_LEVEL_MAX`` px across."""
        src = node.source
        lv = next((i for i, l in enumerate(src.levels) if max(l.shape) <= PREVIEW_LEVEL_MAX), len(src.levels) - 1)
        gy, gx = src.levels[lv].grid
        return np.concatenate([np.concatenate([src.read_tile(lv, ty, tx) for tx in range(gx)], 1)
                               for ty in range(gy)], 0)

    def _picture(self, data: np.ndarray):
        if data.ndim == 3 or data.dtype == np.uint8:
            return preview(data, PREVIEW)
        return preview(np.nan_to_num(data.astype(np.float32)), PREVIEW)

    def _preview(self, index: int) -> None:
        """Before / after the selected step, on the image itself (a coarse level of it)."""
        self.before.clear()
        self.after.clear()
        if index < 0:
            self.before.setPixmap(self._picture(self._level_pixels(self.image)))
            return
        from modalytics.processing.run import chain_sources
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            stages = chain_sources(self.image, Chain("part", self.chain.steps[:index + 1]), self.library)
            before = self.image if index == 0 else stages[index - 1]
            self.before.setPixmap(self._picture(self._level_pixels(before)))
            self.after.setPixmap(self._picture(self._level_pixels(stages[index])))
        except Exception as exc:
            self.status.setText(f"⚠ Step {index + 1}: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------ your chains
    def _fill_saved(self) -> None:
        self.saved.clear()
        for c in self.chains.chains():
            self.saved.addItem(f"{c.name}  ({c.summary(self.library)})", c.name)

    def _load_saved(self) -> None:
        chain = self.chains.get(self.saved.currentData() or "")
        if chain is not None:
            self.chain = chain.copy()
            self.name.setText(chain.name)
            self.measure.setChecked(chain.measure)
            self._fill_steps(0)

    def _delete_saved(self) -> None:
        name = self.saved.currentData()
        if name and QtWidgets.QMessageBox.question(self, "My chains", f"Delete “{name}” from your chains?") == \
                QtWidgets.QMessageBox.StandardButton.Yes:
            self.chains.delete(name)
            self._fill_saved()

    def _save_mine(self) -> None:
        self._changed()
        if not self.chain.steps:
            self.status.setText("⚠ Add a step first.")
            return
        path = self.chains.save(self.chain)
        self._fill_saved()
        self.saved.setCurrentIndex(max(0, self.saved.findData(self.chain.name)))
        self.status.setText(f"✔ Saved to your chains ({path.name})")

    # ------------------------------------------------------------ apply / OK
    def _ok(self) -> None:
        self._changed()
        self.accept()

    def _apply_here(self) -> None:
        from modalytics.ui.processing import apply_chain
        self._changed()
        if apply_chain(self, self.project, self.undo, self.image, self.chain, replace=self.view):
            self.accept()

    def _apply_all(self) -> None:
        from modalytics.ui.processing import apply_chain_to_all
        self._changed()
        n = apply_chain_to_all(self, self.project, self.undo, self.image.modality, self.chain)
        if n:
            self.accept()
