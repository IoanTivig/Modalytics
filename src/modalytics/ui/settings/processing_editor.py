"""
Editing one processing step: its properties, its options and its code.

    ┌ OD map ───────────────────────────────── built in · Save makes your own copy ┐
    │ [Properties] [Options] [Code]                                                 │
    │  Properties  name, button label and icon, id, input types, output type,      │
    │              unit, kind, requires, description                                 │
    │  Options     its values (asked when it is applied, stored in profiles)        │
    │  Code        editor | helper list                                              │
    │ result of Test                                         [before] [after]        │
    │ [Test] [Open in external editor]                            [Revert] [Save]   │
    └────────────────────────────────────────────────────────────────────────────────┘

Test applies the step as edited (unsaved) to a sample image of its first input
type (the synthetic phase map or colour image the parameter editor tests on),
checks the result against its image type's contract, and shows the image before
and after, the value range and the time. The shared parts (saving, built-in
copies, the external editor, the options table) are in ``script_editor``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from modalytics import paths
from modalytics.core.model import ACQUIRED_TYPES, IMAGE_TYPES
from modalytics.parameters.script_file import ProcessingScript
from modalytics.processing.registry import CATEGORIES, KINDS, SAME, StepSpec, load_script
from modalytics.ui.settings.importer_editor import preview
from modalytics.ui.settings.script_editor import ScriptEditor

KIND_NAMES = {"pixel": "Each pixel from itself", "neighbourhood": "From the pixels around it",
              "whole": "Needs the whole image"}

HELP = [
    ("Arguments", "pixels", "pixels", "The input: a tile (or the whole image for KIND 'whole'), (H, W) or (H, W, 3)."),
    ("Arguments", 'options["key"]', 'options[""]', "An option's value (Options tab)."),
    ("Arguments", "state", "state", "What prepare(image, options) returned (once per image)."),
    ("Arguments", "um_per_px", "um_per_px", "Size of one pixel at the level computed: convert µm sizes with it."),
    ("Image (prepare)", "image.metadata", "image.metadata", "What the importer recorded (wavelength_nm, unit…)."),
    ("Image (prepare)", "image.pixel_size_um", "image.pixel_size_um", "(x, y) pixel size at full resolution, µm."),
    ("Image (prepare)", "image.modality", "image.modality", "The input's image type (phase, wsi, brightfield…)."),
    ("Helpers", "ndimage.gaussian_filter(x, sigma)", "ndimage.gaussian_filter()",
     "Gaussian smoothing.\nNeeds: from scipy import ndimage"),
    ("Helpers", "ndimage.median_filter(x, size)", "ndimage.median_filter()",
     "Median filter.\nNeeds: from scipy import ndimage"),
    ("Helpers", "op.optical_density(pixels, white, channels)", "op.optical_density()",
     "Optical density against a background level.\nNeeds: from modalytics.parameters import optical as op"),
    ("Helpers", "np.fft.fft2(x)", "np.fft.fft2()", "2D Fourier transform (KIND 'whole').\nNeeds: import numpy as np"),
]


class ProcessingEditor(ScriptEditor):
    """Edits the selected processing step (see the module docstring)."""

    SCRIPT = ProcessingScript
    WHAT = "processing step"
    VALUES_TAB = "Options"
    VALUES_ROW = "Option"
    VALUES_PROPERTY = "OPTIONS"
    VALUES_HINT = ("The step's values (e.g. a background level, a filter width in µm): asked when it is applied, and "
                   "stored with the profiles that apply it at import. The code reads them as options[\"key\"].")
    TEST_TIP = "Apply it to a sample image (Ctrl+T)"

    def __init__(self, library, parent=None) -> None:
        super().__init__(library, parent)
        self.before, self.after = QtWidgets.QLabel(), QtWidgets.QLabel()
        for w in (self.before, self.after):
            w.setFixedSize(140, 140)
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            w.setVisible(False)
            self.result_row.addWidget(w)

    # ------------------------------------------------------------ the form
    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.name = QtWidgets.QLineEdit()
        self.short = QtWidgets.QLineEdit()
        self.short.setPlaceholderText("Empty: the name")
        self.short.setToolTip("The label of its button on the Process tab (when a profile puts it in the toolbar)")
        from modalytics.ui.icon_picker import IconField
        self.icon = IconField()
        self.icon.setToolTip("The icon of its button on the Process tab (empty: its category's icon)")
        self.key = QtWidgets.QLineEdit()
        self.inputs = QtWidgets.QLineEdit()
        self.inputs.setToolTip("Image types it accepts, comma-separated (" + ", ".join(IMAGE_TYPES) + "; * = any)")
        self.output = QtWidgets.QComboBox()
        self.output.addItem("Same type (a filter)", SAME)
        for k, v in IMAGE_TYPES.items():
            if k not in ACQUIRED_TYPES:
                self.output.addItem(v, k)
        self.unit = QtWidgets.QLineEdit()
        self.unit.setPlaceholderText("of the result (empty for filters: the input's)")
        self.kind = QtWidgets.QComboBox()
        for k in KINDS:
            self.kind.addItem(KIND_NAMES[k], k)
        self.kind.setToolTip("Pixel and neighbourhood steps run tile by tile, on whole slides too; whole-image "
                             "steps (e.g. Fourier transforms) only on images that fit in memory")
        self.category = QtWidgets.QComboBox()
        self.category.setEditable(True)
        self.category.addItems(list(CATEGORIES))
        self.category.setToolTip("Where menus list the step (pick one, or type your own)")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones")
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(80)
        for label, w in (("Name", self.name), ("Button label", self.short), ("Icon", self.icon),
                         ("Id", self.key), ("Input types", self.inputs),
                         ("Makes", self.output), ("Unit", self.unit), ("Works on", self.kind),
                         ("Category", self.category),
                         ("Requires", self.requires), ("Description", self.description)):
            form.addRow(label, w)
        for sig in (self.name.textEdited, self.short.textEdited, self.icon.changed, self.key.textEdited,
                    self.inputs.textEdited, self.unit.textEdited, self.requires.textEdited,
                    self.description.textChanged):
            sig.connect(self._changed)
        self.output.currentIndexChanged.connect(self._changed)
        self.kind.currentIndexChanged.connect(self._changed)
        self.category.currentTextChanged.connect(self._changed)
        self.name.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("NAME", p.get("KEY", "")))
        self.name.setText(str(p.get("NAME", "")))
        self.short.setText(str(p.get("SHORT", "") or ""))
        from modalytics.ui.icon_picker import CATEGORY_HINTS
        self.icon.hint = CATEGORY_HINTS.get(str(p.get("CATEGORY", "") or ""), "")
        self.icon.set_icon(str(p.get("ICON", "") or ""))
        self.key.setText(str(p.get("KEY", "")))
        inputs = p.get("INPUT_TYPES", ["*"])
        self.inputs.setText(", ".join([inputs] if isinstance(inputs, str) else inputs))
        self.output.setCurrentIndex(max(0, self.output.findData(p.get("OUTPUT_TYPE", SAME))))
        self.unit.setText(str(p.get("UNIT", "")))
        self.kind.setCurrentIndex(max(0, self.kind.findData(p.get("KIND", "pixel"))))
        self.category.blockSignals(True)
        self.category.setCurrentText(str(p.get("CATEGORY", "") or "Other"))
        self.category.blockSignals(False)
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        self.helper.set_entries("processing", HELP)

    def load(self, spec: StepSpec | None, path: str | None = None) -> None:
        super().load(spec, path)
        self.before.setVisible(False)
        self.after.setVisible(False)

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.output.setEnabled(not builtin)
        self.key.setToolTip("Built-in steps keep their id (your copy replaces the built-in)." if builtin else
                            "Changing the id renames the file; profiles that use the old id will report it missing.")

    # ------------------------------------------------------------ the script text
    def properties(self) -> dict:
        p = dict(self._known)
        inputs = [t.strip() for t in self.inputs.text().split(",") if t.strip()] or ["*"]
        p.update({"KEY": self.key.text().strip(), "NAME": self.name.text().strip(), "INPUT_TYPES": inputs,
                  "OUTPUT_TYPE": self.output.currentData(), "UNIT": self.unit.text().strip(),
                  "KIND": self.kind.currentData(), "DESCRIPTION": " ".join(self.description.toPlainText().split())})
        for key, value in (("SHORT", self.short.text().strip()), ("ICON", self.icon.icon_name())):
            p.pop(key, None)
            if value:
                p[key] = value
        p.pop("CATEGORY", None)
        if self.category.currentText().strip() not in ("", "Other"):
            p["CATEGORY"] = self.category.currentText().strip()
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["NAME"]:
            raise ValueError("The step needs a name")
        from modalytics import icons
        if p.get("ICON") and not icons.allowed(p["ICON"]):
            raise ValueError(f"'{p['ICON']}' is not an icon of the set: Choose… picks one")
        bad = [t for t in p["INPUT_TYPES"] if t != "*" and t not in IMAGE_TYPES]
        if bad:
            raise ValueError(f"Unknown input type(s): {', '.join(bad)} (known: {', '.join(IMAGE_TYPES)}, *)")

    def target(self, p: dict) -> Path:
        folder = "filters" if p.get("OUTPUT_TYPE", SAME) == SAME else p["OUTPUT_TYPE"]
        return paths.user_dir("processing") / folder / f"{p['KEY']}.py"

    def uid_of(self, p: dict) -> str:
        return f"{'filters' if p['OUTPUT_TYPE'] == SAME else p['OUTPUT_TYPE']}:{p['KEY']}"

    # ------------------------------------------------------------ Test
    def run_test(self, f: Path) -> None:
        """Apply the step as edited to a sample image of its first input type; show before / after."""
        from modalytics.processing.run import make_view
        from modalytics.ui.settings.parameter_editor import sample
        self.before.setVisible(False)
        self.after.setVisible(False)
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        kind = next((t for t in spec.input_types if t in ACQUIRED_TYPES), "phase")
        image = sample("image", kind)[0]
        t0 = time.perf_counter()
        try:
            view = make_view(image, spec)
            gy, gx = view.source.levels[0].grid
            out = np.concatenate([np.concatenate([view.source.read_tile(0, ty, tx) for tx in range(gx)], 1)
                                  for ty in range(gy)], 0)
        except Exception as exc:
            self.report_error(exc, f)
            return
        ms = (time.perf_counter() - t0) * 1e3
        finite = out[np.isfinite(out)] if out.dtype.kind == "f" else out
        rng = f"{finite.min():.4g} … {finite.max():.4g}" if finite.size else "no finite values"
        made = "same type (filter)" if spec.is_filter else IMAGE_TYPES[view.image_type].split(" (")[0]
        self.result.setText(f"✔ on a sample {IMAGE_TYPES[kind].split(' (')[0].lower()} "
                            f"({image.source.shape[1]}×{image.source.shape[0]} px): {made}, values {rng}"
                            f"{' ' + view.unit if view.unit else ''} · {ms:.0f} ms")
        gy, gx = image.source.levels[0].grid
        src = np.concatenate([np.concatenate([image.source.read_tile(0, ty, tx) for tx in range(gx)], 1)
                              for ty in range(gy)], 0)
        self.before.setPixmap(preview(src, 140))
        self.after.setPixmap(preview(out if out.ndim == 2 or out.dtype == np.uint8 else
                                     np.clip(out, 0, 255).astype(np.uint8), 140))
        self.before.setToolTip("Before")
        self.after.setToolTip("After")
        self.before.setVisible(True)
        self.after.setVisible(True)
