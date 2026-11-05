"""
Settings › Registration: the methods that place one image on another (browse, add, edit, test, remove).

    ┌ Method                         ┬ (RegistrationEditor) ──────────────────────────┐
    │ Gland-map correlation          │ [Properties] [Options] [Code]                   │
    │ Experimental (6)               │                                                 │
    │   Edge correlation             │ result of Test: 5/5 correct · median 0.4 µm …   │
    │   Mutual information           │                                                 │
    │ [New…] [Remove] [Reload] [Folder]│ [Test] [Reference ▾] [External] [Revert] [Save]│
    └────────────────────────────────┴─────────────────────────────────────────────────┘

Methods are scripts, like processing steps (``registration.registry``): built-in ones
ship with Modalytics and are never changed in place; new ones start from a commented
template. **Test** checks a method against a known answer: patches cut from a reference
(synthetic tissue, or a slide or image file of yours) at known positions, rotations and
mirroring must be found again (``registration.synthetic``).
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics.core.model import IMAGE_TYPES
from modalytics.parameters.script_file import RegistrationScript
from modalytics.registration.registry import FINDS, MethodSpec, RegistrationLibrary, load_script
from modalytics.ui.settings.script_editor import ScriptEditor
from modalytics.ui.settings.widgets import clear_tree, group_item, is_group, keep_on, list_tree, muted, open_in_system

TEMPLATE = '''"""{name}: {description}

Your registration method (made in Settings › Registration). It finds where the moving
image belongs on the reference; see modalytics/registration/registry.py for the format
of this script and modalytics/registration/context.py for what ``ctx`` offers.
"""

KEY = {key!r}
NAME = {name!r}
DESCRIPTION = {description!r}
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = []
REQUIRES = []
OPTIONS = [
    {{'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 2.0, 'minimum': 0.5,
     'maximum': 10.0, 'step': 0.5, 'suffix': ' µm/px', 'tooltip': 'The images are compared at this resolution'}},
]


def register(ctx, options):
    # The two images as grey maps at the same resolution; the reference only around the search circle
    px = options['resolution_um']
    fixed, frame = ctx.read(ctx.reference, px)
    moving, _ = ctx.read(ctx.moving, px)
    fixed, moving = ctx.grey(fixed), ctx.grey(moving)
    # Turn them into whatever makes the two look alike (here: as they are, normalised) …
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-6)
    moving = (moving - moving.mean()) / (moving.std() + 1e-6)
    # … and let the shared search find the best positions, rotations, scales and mirroring
    return ctx.match(fixed, frame, moving)
'''

HELP = [
    ("Arguments", "ctx", "ctx", "The two images and the search: ctx.moving, ctx.reference, ctx.search (centre_world, "
                                "radius_um, angle_range, scale_range, mirror, top_k)."),
    ("Arguments", "options", "options", "This method's option values: options['key']."),
    ("Pixels", "ctx.read", "ctx.read(ctx.reference, px)", "(pixels, frame): the moving image whole, or the reference "
                                                          "around the search circle, at px µm/px."),
    ("Pixels", "ctx.grey", "ctx.grey(pixels)", "One channel (colour → luminance)."),
    ("Pixels", "ctx.feature_map", "ctx.feature_map(image, px, group='Gland')", "A segmentation model's probability "
                                                                            "map (0–1), with its frame."),
    ("Matching", "ctx.match", "ctx.match(fixed, frame, moving)", "The shared search: normalised cross-correlation over "
                                                                 "positions and the asked rotations, scales and "
                                                                 "mirroring, refined; returns candidates."),
    ("Matching", "ctx.refine_ecc", "ctx.refine_ecc(placement, fixed, frame, moving)", "One placement polished by ECC."),
    ("Matching", "ctx.pose_grid", "ctx.pose_grid(angle_step)", "(angles, scales, mirrors) the search asks for."),
    ("Results", "ctx.candidate", "ctx.candidate(matrix=m, score=s)", "A candidate from a 3×3 moving-local-µm → "
                                                                     "world-µm matrix (or placement=…)."),
    ("Results", "ctx.to_world", "ctx.to_world(frame)", "3×3: a reference map pixel → world µm."),
    ("Results", "ctx.moving_grid", "ctx.moving_grid(moving.shape)", "3×3: a moving map pixel → moving local µm."),
    ("Progress", "ctx.progress", "ctx.progress(0.5)", "Report progress (0–1)."),
    ("Progress", "ctx.check", "ctx.check()", "Stop here if the user cancelled."),
]
SYNTHETIC = "Synthetic tissue"


def _types(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()] or ["*"]


class RegistrationEditor(ScriptEditor):
    """Edits the selected registration method (see the module docstring)."""

    SCRIPT = RegistrationScript
    WHAT = "registration method"
    VALUES_TAB = "Options"
    VALUES_ROW = "Option"
    VALUES_PROPERTY = "OPTIONS"
    VALUES_HINT = ("The method's own settings (shown by the Register tool's Method options), read by the code as "
                   "options[\"key\"]. The search radius, rotations, scales, mirroring and number of candidates are "
                   "the tool's and come in ctx.search.")
    TEST_TIP = "Find patches cut from a reference at known places again (Ctrl+T)"

    def __init__(self, library, parent=None) -> None:
        self.reference_path = ""                          # the Test's reference: "" = synthetic tissue
        super().__init__(library, parent)

    def _extra_buttons(self) -> list[QtWidgets.QPushButton]:
        self.reference_btn = QtWidgets.QPushButton(f"Test on: {SYNTHETIC} ▾")
        self.reference_btn.setToolTip("The image the Test cuts its patches from")
        menu = QtWidgets.QMenu(self.reference_btn)
        menu.addAction(SYNTHETIC, lambda: self._set_reference(""))
        menu.addAction("A slide or image file…", self._choose_reference)
        self.reference_btn.setMenu(menu)
        return [self.reference_btn]

    def _set_reference(self, path: str) -> None:
        self.reference_path = path
        self.reference_btn.setText(f"Test on: {Path(path).name if path else SYNTHETIC} ▾")

    def _choose_reference(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reference for the Test", "",
                                                        "Slides and images (*.svs *.tif *.tiff *.ndpi *.scn *.png "
                                                        "*.jpg *.txt *.csv);;All files (*)")
        if path:
            self._set_reference(path)

    # ------------------------------------------------------------ the form
    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.name = QtWidgets.QLineEdit()
        self.key = QtWidgets.QLineEdit()
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(70)
        self.moving = QtWidgets.QLineEdit()
        self.moving.setToolTip("Image types it can place, comma-separated (" + ", ".join(IMAGE_TYPES) + "; * = any). "
                               "A slide never moves.")
        self.reference = QtWidgets.QLineEdit()
        self.reference.setToolTip("Image types it places them on")
        finds = QtWidgets.QWidget()
        frow = QtWidgets.QHBoxLayout(finds)
        frow.setContentsMargins(0, 0, 0, 0)
        self.finds = {}
        for f in FINDS:
            box = QtWidgets.QCheckBox(f)
            box.setEnabled(f != "translation")
            self.finds[f] = box
            frow.addWidget(box)
        frow.addStretch(1)
        self.experimental = QtWidgets.QCheckBox("Experimental (not validated on your kind of images)")
        self.refines = QtWidgets.QCheckBox("Refines only (no search: starts from the current position)")
        self.references = QtWidgets.QPlainTextEdit()
        self.references.setFixedHeight(80)
        self.references.setPlaceholderText("One per line: Author (year). Title. Journal … | doi:10.…")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones")
        for label, w in (("Name", self.name), ("Id", self.key), ("Description", self.description),
                         ("Moves", self.moving), ("Onto", self.reference), ("Finds", finds), ("", self.experimental),
                         ("", self.refines), ("References", self.references), ("Requires", self.requires)):
            form.addRow(label, w)
        for sig in (self.name.textEdited, self.key.textEdited, self.description.textChanged, self.moving.textEdited,
                    self.reference.textEdited, self.experimental.toggled, self.refines.toggled,
                    self.references.textChanged, self.requires.textEdited, *(b.toggled for b in self.finds.values())):
            sig.connect(self._changed)
        self.name.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("NAME", p.get("KEY", "")))
        self.name.setText(str(p.get("NAME", "")))
        self.key.setText(str(p.get("KEY", "")))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        for widget, key in ((self.moving, "MOVING_TYPES"), (self.reference, "REFERENCE_TYPES")):
            v = p.get(key, ["*"])
            widget.setText(", ".join([v] if isinstance(v, str) else v))
        finds = p.get("FINDS", list(FINDS))
        for f, box in self.finds.items():
            box.setChecked(f == "translation" or f in finds)
        self.experimental.setChecked(bool(p.get("EXPERIMENTAL", False)))
        self.refines.setChecked(bool(p.get("REFINES_ONLY", False)))
        self.references.setPlainText("\n".join(str(r.get("text", "")) + (f" | doi:{r['doi']}" if r.get("doi") else "")
                                               for r in p.get("REFERENCES", []) or [] if isinstance(r, dict)))
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.helper.set_entries("registration", HELP)

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.key.setToolTip("Built-in methods keep their id (your copy replaces the built-in)." if builtin else
                            "Changing the id renames the file; profiles naming the old id fall back to the default.")

    # ------------------------------------------------------------ the script text
    def properties(self) -> dict:
        p = dict(self._known)
        refs = []
        for line in self.references.toPlainText().splitlines():
            if not line.strip():
                continue
            text, _sep, doi = line.partition("| doi:")
            refs.append({"text": text.strip(), **({"doi": doi.strip()} if doi.strip() else {})})
        p.update({"KEY": self.key.text().strip(), "NAME": self.name.text().strip(),
                  "DESCRIPTION": " ".join(self.description.toPlainText().split()),
                  "MOVING_TYPES": _types(self.moving.text()), "REFERENCE_TYPES": _types(self.reference.text()),
                  "FINDS": [f for f, b in self.finds.items() if b.isChecked()],
                  "EXPERIMENTAL": self.experimental.isChecked(), "REFERENCES": refs})
        p.pop("REFINES_ONLY", None)
        if self.refines.isChecked():
            p["REFINES_ONLY"] = True
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["NAME"]:
            raise ValueError("The method needs a name")
        for key in ("MOVING_TYPES", "REFERENCE_TYPES"):
            bad = [t for t in p[key] if t != "*" and t not in IMAGE_TYPES]
            if bad:
                raise ValueError(f"Unknown image type(s): {', '.join(bad)} (known: {', '.join(IMAGE_TYPES)}, *)")

    def target(self, p: dict) -> Path:
        return self.library.user / f"{p['KEY']}.py"

    def uid_of(self, p: dict) -> str:
        return p["KEY"]

    # ------------------------------------------------------------ Test
    def _reference_image(self):
        from modalytics.registration.synthetic import synthetic_tissue
        if not self.reference_path:
            return synthetic_tissue()
        if Path(self.reference_path).suffix.lower() in (".svs", ".tif", ".tiff", ".ndpi", ".scn"):
            from modalytics.io.importers import import_tiff_wsi
            return import_tiff_wsi(self.reference_path)
        from modalytics.image_import.run import import_file
        return import_file(self.reference_path)[0]

    def run_test(self, f: Path) -> None:
        """Cut patches from the reference at known places and let the method (as edited) find them again."""
        from modalytics.registration.context import Cancelled
        from modalytics.registration.synthetic import evaluate_method
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        try:
            reference = self._reference_image()
        except Exception as exc:                                     # not readable: say so
            self.result.setText(f"✖ The reference could not be read: {exc}")
            return
        dlg = QtWidgets.QProgressDialog(f"Testing {spec.name} on {reference.name}…", "Cancel", 0, 100, self)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)

        def progress(fr):
            dlg.setValue(int(100 * fr))
            QtWidgets.QApplication.processEvents()
        try:
            report = evaluate_method(spec, reference, trials=5, radius_um=150.0, rotation=True,
                                     mirror="mirror" in spec.finds, progress=progress, cancelled=dlg.wasCanceled)
        except Cancelled:
            self.result.setText("Test cancelled.")
            return
        except Exception as exc:
            self.report_error(exc, f)
            return
        finally:
            dlg.close()
        rows = [f"  {i + 1}. " + (f"{t.error}" if t.found is None else
                                  f"{t.position_um:.1f} µm, {t.rotation_deg:.1f}°"
                                  + ("" if t.mirror_ok else ", mirroring wrong")
                                  + f" · score {t.score:.2f} · {t.seconds:.1f} s"
                                  + (" ✔" if t.success() else " ✖"))
                for i, t in enumerate(report.trials)]
        self.result.setText(f"{'✔' if all(t.success() for t in report.trials) else '◐'} on {reference.name}: "
                            f"{report.summary()}\n" + "\n".join(rows))


class RegistrationPage(QtWidgets.QWidget):
    """The registration methods page.

    Signals:
        library_changed(): methods were added, changed or removed (the Register tool reloads them).
    """

    library_changed = QtCore.Signal()

    def __init__(self, library: RegistrationLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        self.library = library or RegistrationLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(muted("Methods that place one image on another (the Register tool, Arrange tab). Experimental: "
                            "a standard method not validated on your kind of images — Test shows how it does on "
                            "patches with a known answer."))
        col.addWidget(split, 1)
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Method", "Moves → onto"])
        self.tree.setColumnWidth(0, 230)
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your method (a copy of a built-in: the built-in comes back)")
        for b, slot in ((QtWidgets.QPushButton("New…"), self._new), (self.remove_btn, self._remove),
                        (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        split.addWidget(left)
        self.editor = RegistrationEditor(self.library)
        self.editor.saved.connect(self._saved)
        split.addWidget(self.editor)
        split.setSizes([340, 760])
        self.reload()

    def reload(self, select: str | None = None) -> None:
        """Rescan the scripts and rebuild the list (keeps the selection, or selects ``select``)."""
        select = select or (self._spec().key if self._spec() else None)
        self.library.reload()
        self._built = getattr(self, "_built", 0) + 1
        self.tree.blockSignals(True)
        clear_tree(self.tree)
        role = QtCore.Qt.ItemDataRole.UserRole
        chosen = None
        for title, specs in (("Validated", [s for s in self.library.specs.values() if not s.experimental]),
                             ("Experimental", [s for s in self.library.specs.values() if s.experimental])):
            if not specs:
                continue
            top = group_item(self.tree, f"{title} ({len(specs)})")
            for spec in specs:
                mark = "  · yours" if spec.source == "user" else ""
                item = QtWidgets.QTreeWidgetItem(top, [spec.name + mark, f"{', '.join(spec.moving_types)} → "
                                                                         f"{', '.join(spec.reference_types)}"])
                item.setData(0, role, spec.key)
                item.setToolTip(0, spec.description + ("\n\n" + spec.references_text() if spec.references else ""))
                if spec.key == select:
                    chosen = item
            top.setExpanded(True)
        if self.library.errors:
            bad = group_item(self.tree, f"Could not load ({len(self.library.errors)})")
            for path, why in self.library.errors.items():
                item = QtWidgets.QTreeWidgetItem(bad, [Path(path).name])
                item.setToolTip(0, why)
                item.setData(0, role + 1, path)
                if path == select:
                    chosen = item
            bad.setExpanded(True)
        self.tree.blockSignals(False)
        if chosen is None:
            it = QtWidgets.QTreeWidgetItemIterator(self.tree)
            while it.value() and not (it.value().data(0, role) or it.value().data(0, role + 1)):
                it += 1
            chosen = it.value()
        self.tree.setCurrentItem(chosen)
        self._show(chosen)

    def _spec(self) -> MethodSpec | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(key) if key else None

    def _selected(self, cur, prev) -> None:
        if cur is prev:
            return
        if is_group(cur):
            keep_on(self.tree, prev)
            return
        role = QtCore.Qt.ItemDataRole.UserRole
        wanted = [cur.data(0, role + k) for k in range(2)]
        built = self._built
        if not self.editor.confirm_leave():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(prev)
            self.tree.blockSignals(False)
            return
        if self._built != built:
            self.select(wanted[0] or wanted[1])
            return
        self._show(cur)

    def _show(self, item) -> None:
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        self.editor.load(spec, broken if broken else None)
        self.editor.set_missing(self.library.missing_libraries().get(broken or (spec.key if spec else ""), []))
        self.remove_btn.setEnabled(bool(broken) or (spec is not None and spec.source == "user"))

    def select(self, key: str) -> None:
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            v = it.value()
            if key in (v.data(0, QtCore.Qt.ItemDataRole.UserRole), v.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)):
                self.tree.setCurrentItem(v)
                return
            it += 1

    def confirm_leave(self) -> bool:
        return self.editor.confirm_leave()

    def _saved(self, key: str) -> None:
        path = self.editor.path
        self.reload(key or path)
        if not key:
            self.select(path)
        self._changed()

    def _changed(self) -> None:
        from modalytics.tools.arrange import reload_methods
        reload_methods()                                  # the Register tool reads the methods again
        self.library_changed.emit()

    def _new(self) -> None:
        if not self.editor.confirm_leave():
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New registration method", "Name:")
        if not ok or not name.strip():
            return
        key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "method"
        if key[0].isdigit():
            key = "m_" + key
        n, base = 2, key
        while self.library.get(key) or (self.library.user / f"{key}.py").exists():
            key, n = f"{base}_{n}", n + 1
        target = self.library.user / f"{key}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(TEMPLATE.format(key=key, name=name.strip(), description="What it compares and when it "
                                                                                  "works."), encoding="utf-8")
        self.reload(key)
        self.editor.tabs.setCurrentIndex(2)
        self._changed()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        path = broken or (spec.path if spec and spec.source == "user" else None)
        if path is None:
            return
        replaces = spec is not None and spec.key in self.library.overridden
        msg = f"Delete “{spec.name if spec else Path(path).name}”?\n\n{path}" + \
            ("\n\nThe built-in version comes back." if replaces else "")
        if QtWidgets.QMessageBox.question(self, "Remove registration method", msg) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.key if replaces else None)
        self._changed()
