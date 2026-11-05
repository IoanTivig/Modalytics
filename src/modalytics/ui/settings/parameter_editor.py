"""
Editing one parameter: its properties, its constants and its code.

    ┌ Disorder strength (µm) ───────────── built in: Save makes your own copy ┐
    │ [Properties] [Constants] [Code]                                         │
    │  Properties  name, id, scope, image type, category, unit, description,  │
    │              on by default, order                                       │
    │  Constants   one row per constant: key, label, kind, default, range…    │
    │  Code        editor | helper list (what ctx offers, double-click)       │
    │ result of Test                                                           │
    │ [Test] [Open in external editor]                       [Revert] [Save]  │
    └─────────────────────────────────────────────────────────────────────────┘

Saving, the built-in copy rule, the external editor and the Constants table
are shared with the importer editor (``script_editor.ScriptEditor``).

Test runs the parameter on a sample: an irregular object on a synthetic phase
map (or RGB image, for parameters that do not take phase maps).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PySide6 import QtWidgets

from modalytics import paths
from modalytics.parameters.registry import (SCOPES, TYPE_NAMES, ParameterLibrary, ParameterSpec,
                                             load_script, type_word)
from modalytics.parameters.script_file import ParameterScript
from modalytics.ui.settings.script_editor import ScriptEditor

UNITS = ("", "µm", "µm²", "µm³", "1/µm", "px", "rad", "rad²", "nm", "%")


## Sample used by Test ##

_SAMPLES: dict = {}


def sample(scope: str, image_type: str):
    """(image, polygon, images) to try a parameter on: a phase map, or an RGB image for slide and
    brightfield parameters; ``images`` holds both (same 96 × 96 µm area), so ``ctx.on`` can be tried too.

    The images carry annotations of the default classes, so parameters relating classes can be tried: three
    glands (the sample object is one of them) and lamina propria (stroma) filling the rest of a central region.
    """
    from shapely import affinity
    from shapely.geometry import Polygon

    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    rgb = image_type in ("wsi", "brightfield")
    key = (image_type if rgb else "phase", scope)
    if key not in _SAMPLES:
        from scipy import ndimage
        rng = np.random.default_rng(7)
        smooth = ndimage.gaussian_filter(rng.normal(size=(320, 320)), 6)
        phase = (1.2 + 6 * smooth + 0.05 * rng.normal(size=smooth.shape)).astype(np.float32)
        import cv2
        g = np.clip((phase - phase.min()) / np.ptp(phase), 0, 1)
        g = cv2.resize(g, (384, 384), interpolation=cv2.INTER_LINEAR)       # same 96 µm at 0.25 µm/px
        data = (np.stack([200 - 90 * g, 120 - 60 * g, 190 - 40 * g], axis=2)).astype(np.uint8)
        colour = ImageNode("sample (RGB)", ArraySource(data), (0.25, 0.25), image_type if rgb else "brightfield")
        phase_map = ImageNode("sample (phase)", ArraySource(phase), (0.3, 0.3), "phase",
                              metadata={"phase_to_height_nm": 211.5})
        image = colour if rgb else phase_map
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        r = 20 * (1 + 0.12 * np.sin(5 * t))                      # a wavy, gland-like outline
        poly = Polygon(np.column_stack([48 + 1.4 * r * np.cos(t), 48 + r * np.sin(t)]))
        obj = affinity.rotate(poly, 20)
        from shapely.geometry import Point, box

        from modalytics.core.model import MaskNode
        others = [Point(14, 14).buffer(9, 64), Point(84, 82).buffer(8, 64)]
        stroma = box(4, 4, 92, 92).difference(obj)
        for g in others:
            stroma = stroma.difference(g)
        for img in (phase_map, colour):
            for group, geom in (("Gland", [obj] + others), ("Lamina propria", stroma)):
                mask = MaskNode(geometry=geom, group=group, subgroup="Normal")
                mask.parent = img
                img.children.append(mask)
        _SAMPLES[key] = (image, obj, [phase_map, colour])
    return _SAMPLES[key]


## Duplicating for another image type ##

class DuplicateDialog(QtWidgets.QDialog):
    """Image type, unit and name of a copy of a parameter."""

    def __init__(self, spec: ParameterSpec, library: ParameterLibrary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Duplicate “{spec.label}” for another image type")
        self.spec, self.library = spec, library
        form = QtWidgets.QFormLayout(self)
        self.image_type = QtWidgets.QComboBox()
        for k, v in TYPE_NAMES.items():
            if k != spec.image_type:
                taken = f"{spec.scope}:{type_word(k)}:{spec.key}" in library.specs
                self.image_type.addItem(v + ("  (exists already)" if taken else ""), k)
        self.unit = QtWidgets.QComboBox()
        self.unit.setEditable(True)
        self.unit.addItems(UNITS)
        self.unit.setCurrentText(spec.unit)
        self.unit.setToolTip("Units differ between image types: phase in rad, brightfield in grey levels or "
                             "optical density, …")
        self.label = QtWidgets.QLineEdit(spec.label)
        form.addRow("Image type", self.image_type)
        form.addRow("Unit", self.unit)
        form.addRow("Name", self.label)
        form.addRow(muted_label("The code is copied as it is; check with Test that it suits the new image type."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self) -> None:
        if "exists already" in self.image_type.currentText():
            QtWidgets.QMessageBox.warning(self, "Duplicate", "That image type has this parameter already.")
            return
        self.accept()

    def values(self) -> tuple[str, str, str]:
        return self.image_type.currentData(), self.unit.currentText().strip(), self.label.text().strip()


def muted_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


## The editor ##

class ParameterEditor(ScriptEditor):
    """Edits the selected parameter (see the module docstring)."""

    SCRIPT = ParameterScript
    WHAT = "parameter"
    VALUES_TAB = "Constants"
    VALUES_ROW = "Constant"
    VALUES_PROPERTY = "CONSTANTS"
    VALUES_HINT = ("The fixed values of the formula, kept out of the code: the code reads them as "
                   "ctx.constants[\"key\"], and each profile may choose its own value within Min–Max (Step = the "
                   "increment). Choices (comma-separated) are for the choice kind; Test uses the defaults.")
    TEST_TIP = "Run it on a sample object (Ctrl+T)"

    # ------------------------------------------------------------ the form
    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.label = QtWidgets.QLineEdit()
        self.key = QtWidgets.QLineEdit()
        self.scope = QtWidgets.QComboBox()
        for k, v in SCOPES.items():
            self.scope.addItem(v, k)
        self.category = QtWidgets.QComboBox()
        self.category.setEditable(True)
        self.unit = QtWidgets.QComboBox()
        self.unit.setEditable(True)
        self.unit.addItems(UNITS)
        self.image_type = QtWidgets.QComboBox()
        for k, v in TYPE_NAMES.items():
            self.image_type.addItem(v, k)
        self.image_type.setToolTip("Any image: it uses only the outline (shape). Otherwise the one image type "
                                   "whose pixels it measures; Duplicate for… makes a version for another type.")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones, e.g. PyWavelets")
        self.requires.setToolTip("Comma-separated PyPI names. Another computer is told to install them "
                                 "(Settings › Parameters › Libraries) instead of failing on import.")
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(170)
        self.references = QtWidgets.QPlainTextEdit()
        self.references.setFixedHeight(110)
        self.references.setPlaceholderText("One publication per line: authors (year). Title. Journal "
                                           "volume:pages. doi")
        self.references.setToolTip("Publications that define or use this parameter, one per line")
        self.default_on = QtWidgets.QCheckBox("Ticked in new profiles")
        self.whole = QtWidgets.QCheckBox("Needs the whole object")
        self.whole.setToolTip("On: it needs the whole object — shape, totals over the object, what follows its "
                              "outline — so it is left empty on an object cut by the image edge or marked partial "
                              "(and such an object is not graded by a classifier that reads it).\nOff: the visible "
                              "part is measured — pixel statistics, texture — and partial objects count too where "
                              "the training set includes them.")
        self.order = QtWidgets.QDoubleSpinBox()
        self.order.setRange(0, 100000)
        self.order.setDecimals(0)
        self.order.setToolTip("Where it sits in tables and lists (lower first)")
        form.addRow("Name", self.label)
        form.addRow("Id", self.key)
        form.addRow("Scope", self.scope)
        form.addRow("Image type", self.image_type)
        form.addRow("Category", self.category)
        form.addRow("Unit", self.unit)
        form.addRow("Requires", self.requires)
        form.addRow("Description", self.description)
        form.addRow("References", self.references)
        form.addRow("", self.default_on)
        form.addRow("", self.whole)
        form.addRow("Order", self.order)
        for sig in (self.label.textEdited, self.key.textEdited, self.category.currentTextChanged,
                    self.unit.currentTextChanged, self.description.textChanged, self.references.textChanged,
                    self.requires.textEdited):
            sig.connect(self._changed)
        self.scope.currentIndexChanged.connect(self._scope_changed)
        self.image_type.currentIndexChanged.connect(self._scope_changed)
        self.default_on.toggled.connect(self._changed)
        self.whole.toggled.connect(self._changed)
        self.order.valueChanged.connect(self._changed)
        self.label.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))

    def _extra_buttons(self) -> list[QtWidgets.QPushButton]:
        self.duplicate_btn = QtWidgets.QPushButton("Duplicate for…")
        self.duplicate_btn.setToolTip("Make a copy of this parameter for another image type (e.g. the phase "
                                      "mean for brightfield images), with its own unit")
        self.duplicate_btn.clicked.connect(self.duplicate)
        return [self.duplicate_btn]

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("LABEL", p.get("KEY", "")))
        self.label.setText(str(p.get("LABEL", "")))
        self.key.setText(str(p.get("KEY", "")))
        self.scope.setCurrentIndex(max(0, self.scope.findData(p.get("SCOPE", "object"))))
        self._fill_categories()
        self.category.setCurrentText(str(p.get("CATEGORY", "other")))
        self.unit.setCurrentText(str(p.get("UNIT", "")))
        self.image_type.setCurrentIndex(max(0, self.image_type.findData(p.get("IMAGE_TYPE", "phase"))))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        self.references.setPlainText("\n".join(str(r) for r in (p.get("REFERENCES") or [])))
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.default_on.setChecked(bool(p.get("DEFAULT_ON", True)))
        from modalytics.parameters.registry import whole_object_default
        self.whole.setChecked(bool(p.get("WHOLE_OBJECT", whole_object_default(str(p.get("CATEGORY", "other")),
                                                                              str(p.get("KEY", ""))))))
        self.whole.setEnabled(p.get("SCOPE", "object") == "object")
        self.order.setValue(float(p.get("ORDER", 1000)))
        self.helper.set_scope(self.scope.currentData())

    def _fill_categories(self) -> None:
        current = self.category.currentText()
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItems(self.library.categories(self.scope.currentData(), self.image_type.currentData())
                               or self.library.categories(self.scope.currentData()) or ["other"])
        self.category.setCurrentText(current)
        self.category.blockSignals(False)

    def _scope_changed(self, _i) -> None:
        self._fill_categories()
        self.whole.setEnabled(self.scope.currentData() == "object")
        self.helper.set_scope(self.scope.currentData())
        self._changed()

    def load(self, spec: ParameterSpec | None, path: str | None = None) -> None:
        """Show a parameter (or, with ``spec`` None and a ``path``, a script that failed to load)."""
        super().load(spec, path)
        if self.path is None:
            return
        self.duplicate_btn.setEnabled(spec is not None and not self.raw)
        if spec is not None and spec.note:
            self.result.setText(f"ℹ {spec.note}")

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.scope.setEnabled(not builtin)
        self.image_type.setEnabled(not builtin)
        self.key.setToolTip("Built-in parameters keep their id and scope (your copy replaces the built-in)."
                            if builtin else
                            "Changing the id renames the file; profiles that use the old id will report it missing.")

    # ------------------------------------------------------------ the script text
    def properties(self) -> dict:
        p = dict(self._known)
        p.update({"KEY": self.key.text().strip(), "LABEL": self.label.text().strip(),
                  "UNIT": self.unit.currentText().strip(), "CATEGORY": self.category.currentText().strip() or "other",
                  "SCOPE": self.scope.currentData(), "DEFAULT_ON": self.default_on.isChecked(),
                  **({"WHOLE_OBJECT": self.whole.isChecked()} if self.scope.currentData() == "object" else {}),
                  "ORDER": int(self.order.value()), "IMAGE_TYPE": self.image_type.currentData(),
                  "DESCRIPTION": " ".join(self.description.toPlainText().split())})
        p.pop("REFERENCES", None)
        references = [" ".join(r.split()) for r in self.references.toPlainText().splitlines() if r.strip()]
        if references:
            p["REFERENCES"] = references
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["LABEL"]:
            raise ValueError("The parameter needs a name")

    def target(self, p: dict) -> Path:
        return self._file_for(p.get("SCOPE", "object"), p.get("IMAGE_TYPE", "phase"), p["KEY"])

    def uid_of(self, p: dict) -> str:
        return f"{p['SCOPE']}:{type_word(p['IMAGE_TYPE'])}:{p['KEY']}"

    @staticmethod
    def _file_for(scope: str, image_type: str, key: str) -> Path:
        """Where your parameter of this scope, image type and key lives."""
        return paths.user_dir("parameters") / scope / type_word(image_type) / f"{key}.py"

    # ------------------------------------------------------------ actions
    def run_test(self, f: Path) -> None:
        """Run the parameter (as edited, unsaved) on the sample and show the value or the error."""
        scope = self.scope.currentData() if not self.raw else "object"
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        image, poly, images = sample(scope, spec.image_type)
        from modalytics.parameters.context import ClassContext, ImageContext, ObjectContext
        if scope == "image":
            ctx = ImageContext(image, spec.resolve(), images)
        elif scope == "class":
            ctx = ClassContext(image, ("Gland", ""), spec.resolve(), images)       # the sample's glands
        else:
            ctx = ObjectContext(poly, image, spec.resolve(), images)
        t0 = time.perf_counter()
        try:
            value = spec.compute(ctx)
        except Exception as exc:
            self.report_error(exc, f)
            return
        ms_ = (time.perf_counter() - t0) * 1e3
        try:
            shown = f"{float(value):.6g}" if not (isinstance(value, float) and math.isnan(value)) else "NaN"
        except (TypeError, ValueError):
            self.result.setText(f"✖ compute returned {type(value).__name__}, not a number: {value!r:.80}")
            return
        if scope == "image":
            what = "the whole sample image"
        elif scope == "class":
            what = f"the sample's {len(ctx.objects)} glands (class Gland)"
        else:
            what = f"a sample object ({poly.area:.0f} µm²)"
        unit = f" {spec.unit}" if spec.unit else ""
        self.result.setText(f"= {shown}{unit}   on {what}, {image.modality} "
                            f"{image.size_um[0]:.0f}×{image.size_um[1]:.0f} µm, {ms_:.1f} ms")

    def duplicate(self) -> None:
        """Copy this parameter for another image type (same code; you choose the type, unit and name)."""
        if self.spec is None or self.raw or not self.confirm_leave():
            return
        dlg = DuplicateDialog(self.spec, self.library, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        image_type, unit, label = dlg.values()
        script = ParameterScript.parse(Path(self.spec.path).read_text(encoding="utf-8"))
        script.properties.update(IMAGE_TYPE=image_type, UNIT=unit, LABEL=label)
        script.properties.pop("IMAGE_TYPES", None)
        target = self._file_for(self.spec.scope, image_type, self.spec.key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script.render(), encoding="utf-8")
        self.saved.emit(f"{self.spec.scope}:{type_word(image_type)}:{self.spec.key}")
        self.result.setText(f"✔ Made “{label}” for {TYPE_NAMES[image_type].lower()} images. Check that its code "
                            f"suits that image type (Test), then Save.")
