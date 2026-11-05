"""
Editing one importer: its properties, its options and its code.

    ┌ Phasics PhaseStudio (.txt, .csv) ─────── built in · Save makes your own copy ┐
    │ [Properties] [Options] [Code]                                              │
    │ YOUR CHOICES AT IMPORT (this computer): pixel size, file's value first,   │
    │   option values, ask at every import                                       │
    │  Properties  name, id, image type, file extensions, requires, description │
    │  Options     values asked at import when the file does not say them       │
    │  Code        editor | helper list (Image, options, readers)               │
    │ result of Test                                              [preview]      │
    │ [Test] [Open in external editor] [Sample file…]           [Revert] [Save] │
    └────────────────────────────────────────────────────────────────────────────┘

Test reads a sample file of your choice with the importer as edited (unsaved)
and checks the result against the image type's contract, as File › Import
does: it shows how sure ``can_read`` is, the size, the pixel size, the value
range in the contract's unit (phase: radians) and a preview. Options take your
defaults.

*Your choices at import* are not part of the script: they are what was
confirmed at the first import on this computer (``image_import.defaults``): the
pixel size used when a file does not give it, whether a file's own value comes
first, option values, and whether import asks. They are changed here and saved
at once, e.g. a new pixel size after changing objective. Modalytics' own
whole-slide reader (locked) shows them too.

The shared parts (saving, built-in copies, the external editor, the table)
are in ``script_editor``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import paths
from modalytics.image_import import defaults, run
from modalytics.image_import.contracts import ContractError, Image, check, describe_unit
from modalytics.image_import.registry import HELP, TYPE_NAMES, ImporterSpec, load_script
from modalytics.parameters.script_file import ImporterScript
from modalytics.ui.settings.script_editor import ScriptEditor
from modalytics.ui.settings.widgets import muted, setting_widget


def preview(pixels: np.ndarray, size: int = 150) -> QtGui.QPixmap:
    """A small picture of imported pixels (grey: 1–99 % range; colour as it is)."""
    a = np.asarray(pixels)
    if a.ndim == 2:
        lo, hi = np.nanpercentile(a, [1, 99]) if a.size else (0, 1)
        g = np.clip((np.nan_to_num(a.astype(np.float32)) - lo) / max(hi - lo, 1e-12), 0, 1)
        rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    else:
        rgb = (a[..., :3] >> 8).astype(np.uint8) if a.dtype == np.uint16 else a[..., :3].astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    img = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(img.copy()).scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                      QtCore.Qt.TransformationMode.SmoothTransformation)


class ImporterEditor(ScriptEditor):
    """Edits the selected importer (see the module docstring)."""

    SCRIPT = ImporterScript
    WHAT = "importer"
    VALUES_TAB = "Options"
    VALUES_ROW = "Option"
    VALUES_PROPERTY = "OPTIONS"
    VALUES_HINT = ("Values the importer needs but some files do not say (e.g. the pixel size in the sample). "
                   "They are asked at import, prefilled with your defaults (Properties tab); the code reads them as "
                   "options[\"key\"], and needs(path) says which to ask for a given file.")
    TEST_TIP = "Read a sample file with it (Ctrl+T)"

    def __init__(self, library, parent=None) -> None:
        super().__init__(library, parent)
        self.samples: dict[str, str] = {}               # importer id → sample file chosen for Test
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(150, 150)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setVisible(False)
        self.result_row.addWidget(self.preview)
        # Your choices at import, above the tabs (this computer's, saved at once; not part of the script)
        self.defaults_box = QtWidgets.QGroupBox("Your choices at import (this computer)")
        self.defaults_form = QtWidgets.QFormLayout(self.defaults_box)
        self.layout().insertWidget(1, self.defaults_box)
        self.defaults_box.setVisible(False)

    # ------------------------------------------------------------ the form
    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.name = QtWidgets.QLineEdit()
        self.key = QtWidgets.QLineEdit()
        self.image_type = QtWidgets.QComboBox()
        for k, v in TYPE_NAMES.items():
            self.image_type.addItem(v, k)
        self.image_type.setToolTip("What it produces. Each image type has a contract (phase: values in radians, "
                                   "pixel size in µm; brightfield: 8/16-bit grey or RGB) that Modalytics checks.")
        self.extensions = QtWidgets.QLineEdit()
        self.extensions.setPlaceholderText(".txt, .csv")
        self.extensions.setToolTip("The file extensions it reads (comma-separated). Several importers may read "
                                   "one extension: the one whose can_read is most sure wins.")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones, e.g. h5py")
        self.requires.setToolTip("Comma-separated PyPI names. Another computer is told to install them "
                                 "(Libraries…) instead of failing on import.")
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(80)
        form.addRow("Name", self.name)
        form.addRow("Id", self.key)
        form.addRow("Image type", self.image_type)
        form.addRow("File extensions", self.extensions)
        form.addRow("Requires", self.requires)
        form.addRow("Description", self.description)
        for sig in (self.name.textEdited, self.key.textEdited, self.extensions.textEdited,
                    self.requires.textEdited, self.description.textChanged):
            sig.connect(self._changed)
        self.image_type.currentIndexChanged.connect(self._changed)
        self.name.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))

    def _extra_buttons(self) -> list[QtWidgets.QPushButton]:
        self.sample_btn = QtWidgets.QPushButton("Sample file…")
        self.sample_btn.setToolTip("The file Test reads")
        self.sample_btn.clicked.connect(lambda: self.choose_sample() and self.test())
        return [self.sample_btn]

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("NAME", p.get("KEY", "")))
        self.name.setText(str(p.get("NAME", "")))
        self.key.setText(str(p.get("KEY", "")))
        self.image_type.setCurrentIndex(max(0, self.image_type.findData(p.get("IMAGE_TYPE", "phase"))))
        self.extensions.setText(", ".join(p.get("EXTENSIONS", []) or []))
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        self.helper.set_entries("importer", HELP)

    def load(self, spec: ImporterSpec | None, path: str | None = None) -> None:
        # A locked reader shown before (show_locked) turned the tabs and buttons off: a script has them
        self.tabs.setEnabled(True)
        for b in (self.test_btn, self.external_btn, self.sample_btn):
            b.setEnabled(True)
        super().load(spec, path)
        self.preview.setVisible(False)
        self._fill_defaults(spec.uid if spec else None, spec)
        self.sample_btn.setToolTip(f"The file Test reads: {self.samples[self._sample_key()]}"
                                   if self._sample_key() in self.samples else "The file Test reads")

    def _fill_defaults(self, uid: str | None, spec: ImporterSpec | None = None) -> None:
        """Show your choices at import for an importer, editable (saved as soon as changed).

        Args:
            uid: The importer id (None: hide the box).
            spec: The importer, for its options (None for Modalytics' own readers).
        """
        while self.defaults_form.rowCount():
            self.defaults_form.removeRow(0)
        self.defaults_box.setVisible(uid is not None)
        if uid is None:
            return
        mine = defaults.pixel(uid)
        if mine is None:
            self.defaults_form.addRow(muted("Not confirmed yet: the first import asks, prefilled with the file's "
                                            "pixel size."))
        mine = mine or {"pixel_size_um": None, "prefer_file": True}
        self.default_pixel = QtWidgets.QDoubleSpinBox()
        self.default_pixel.setDecimals(6)
        self.default_pixel.setRange(0.000001, 10000.0)
        self.default_pixel.setSingleStep(0.001)
        self.default_pixel.setSuffix(" µm")
        self.default_pixel.setValue(mine["pixel_size_um"] or 1.0)
        self.default_pixel.setToolTip("Used for files that do not give their pixel size (and for all files "
                                      "when the box below is off)")
        self.default_pixel.valueChanged.connect(lambda v: self._choice_changed(uid, pixel_size_um=v))
        self.defaults_form.addRow("Pixel size in the sample", self.default_pixel)
        self.prefer_file = QtWidgets.QCheckBox("Use the file's pixel size when it gives one")
        self.prefer_file.setChecked(mine["prefer_file"])
        self.prefer_file.toggled.connect(lambda on: self._choice_changed(uid, prefer_file=on))
        self.defaults_form.addRow(self.prefer_file)
        if spec is not None and spec.options:
            current = defaults.values(spec)
            for o in spec.options:
                self.defaults_form.addRow(o.label, setting_widget(
                    o, current[o.key], lambda v, k=o.key, s=spec: (defaults.save(s, {k: v}),
                                                                   self._choice_changed(uid))))
        self.ask = QtWidgets.QCheckBox("Ask at every import")
        self.ask.setChecked(defaults.asks(uid))
        self.ask.setToolTip("Off: import uses these choices without asking (the status bar says what was used).")
        self.ask.toggled.connect(lambda on: self._choice_changed(uid, ask=on))
        self.defaults_form.addRow(self.ask)

    def _choice_changed(self, uid: str, pixel_size_um: float | None = None, prefer_file: bool | None = None,
                        ask: bool | None = None) -> None:
        if pixel_size_um is not None or prefer_file is not None:
            defaults.save_pixel(uid, pixel_size_um, prefer_file)
        if ask is not None:
            if defaults.pixel(uid) is None:          # switching asking off confirms the value shown
                defaults.save_pixel(uid, self.default_pixel.value(), self.prefer_file.isChecked())
            defaults.set_asks(uid, ask)
        p = defaults.pixel(uid) or {}
        how = "the file's value when it gives one, else " if p.get("prefer_file", True) else ""
        self.result.setText(f"✔ Pixel size: {how}{p.get('pixel_size_um') or 1.0:g} µm"
                            + ("" if defaults.asks(uid) else " · used without asking"))

    def show_locked(self, title: str, text: str, uid: str | None = None) -> None:
        """Show one of Modalytics' own readers: nothing to edit but your choices at import."""
        self.load(None)
        self.title.setText(title)
        self.origin.setText("included · locked")
        self.result.setText(text)
        if uid is not None:
            self.setEnabled(True)
            self.tabs.setEnabled(False)
            for b in (self.test_btn, self.external_btn, self.sample_btn, self.revert_btn, self.save_btn):
                b.setEnabled(False)
            self._fill_defaults(uid)

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.image_type.setEnabled(not builtin)
        self.key.setToolTip("Built-in importers keep their id (your copy replaces the built-in)." if builtin else
                            "Changing the id renames the file.")

    # ------------------------------------------------------------ the script text
    def properties(self) -> dict:
        p = dict(self._known)
        exts = [e.strip().lower() for e in self.extensions.text().replace(";", ",").split(",") if e.strip()]
        p.update({"KEY": self.key.text().strip(), "NAME": self.name.text().strip(),
                  "IMAGE_TYPE": self.image_type.currentData(),
                  "EXTENSIONS": [e if e.startswith(".") else f".{e}" for e in exts],
                  "DESCRIPTION": " ".join(self.description.toPlainText().split())})
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["NAME"]:
            raise ValueError("The importer needs a name")
        if not p["EXTENSIONS"]:
            raise ValueError("List at least one file extension, e.g. .txt")

    def target(self, p: dict) -> Path:
        return paths.user_dir("importers") / p.get("IMAGE_TYPE", "phase") / f"{p['KEY']}.py"

    def uid_of(self, p: dict) -> str:
        return f"{p['IMAGE_TYPE']}:{p['KEY']}"

    # ------------------------------------------------------------ Test
    def _sample_key(self) -> str:
        return self.spec.uid if self.spec is not None else (self.path or "")

    def choose_sample(self) -> bool:
        """Ask for the sample file Test reads. Returns False if cancelled."""
        exts = " ".join(f"*{e}" for e in self.properties().get("EXTENSIONS", [])) if not self.raw else ""
        start = os.path.dirname(self.samples.get(self._sample_key(), "")) or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Sample file for Test", start, (f"Its files ({exts});;" if exts else "") + "All files (*)")
        if path:
            self.samples[self._sample_key()] = path
            self.sample_btn.setToolTip(f"The file Test reads: {path}")
        return bool(path)

    def run_test(self, f: Path) -> None:
        """Read the sample file with the importer as edited; check the contract; show the result."""
        self.preview.setVisible(False)
        sample = self.samples.get(self._sample_key())
        if not sample and not self.choose_sample():
            self.result.setText("ℹ Choose a sample file (Sample file…) for Test to read.")
            return
        sample = self.samples[self._sample_key()]
        # 1. Load the script
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        # 2. How sure it is, what it would ask, then read
        name = os.path.basename(sample)
        lines = []
        if not spec.handles(sample):
            lines.append(f"⚠ {Path(sample).suffix or 'no extension'} is not one of its file extensions "
                         f"(File › Import would not offer it this file).")
        try:
            sure = float(spec.can_read(sample))
        except Exception as exc:
            self.report_error(exc, f)
            return
        asks = [o.label for o in spec.to_ask(sample)]
        values = defaults.values(spec)
        uid = self.spec.uid if self.spec is not None else spec.uid
        t0 = time.perf_counter()
        try:
            result = spec.read(sample, values)
        except Exception as exc:
            self.report_error(exc, f)
            return
        ms = (time.perf_counter() - t0) * 1e3
        # 3. The contract
        images = result if isinstance(result, (list, tuple)) else [result]
        if not images or not all(isinstance(i, Image) for i in images):
            self.result.setText(f"✖ read() must return an Image (or a list of them), not "
                                f"{type(result).__name__}")
            return
        # The pixel size: the file's, else your default (else 1 µm, just for the test)
        in_file = [run.file_pixel_size(i) for i in images]
        px, source = defaults.resolve_pixel(uid, in_file[0])
        if px is None:
            px, source = 1.0, "not in the file; 1 µm used for this test"
        try:
            ok = [check(i, spec.image_type, None if v else px, "file" if v else source)
                  for i, v in zip(images, in_file)]
        except ContractError as exc:
            self.result.setText(f"✖ The result does not meet the {TYPE_NAMES[spec.image_type].lower()} "
                                f"contract: {exc}")
            return
        first = ok[0]
        h, w = first.pixels.shape[:2]
        px = first.pixel_size_um
        finite = first.pixels[np.isfinite(first.pixels)] if first.pixels.dtype.kind == "f" else first.pixels
        rng = f"{finite.min():.4g} … {finite.max():.4g}" if finite.size else "no finite values"
        unit = describe_unit(first.metadata) or "grey levels"
        said = "read from the file" if in_file[0] else source
        lines.insert(0, f"✔ {name}: can_read {sure:.2f} · {len(ok)} image(s) · {w}×{h} px · "
                        f"{px[0]:.6g} µm/px ({said}) · {ms:.0f} ms")
        lines.append(f"  values {rng} ({unit})")
        if spec.options:
            shown = ", ".join(f"{o.label} = {values[o.key]}" for o in spec.options)
            lines.append(f"  options: {shown}" + (f" · asked at import: {', '.join(asks)}" if asks else
                                                   " · not asked for this file"))
        self.result.setText("\n".join(lines))
        self.preview.setPixmap(preview(first.pixels))
        self.preview.setVisible(True)
