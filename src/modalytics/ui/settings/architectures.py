"""
Settings › Architectures: the networks segmentation models are trained as, and the weights their encoders start from.

    [Architectures] [Encoder weights]
    ┌ Architecture            ┬ Decoder · encoder ┬ Check ┐ ┌ (ArchitectureEditor) ─────────────────────┐
    │ Recommended (4)         │                   │       │ │ [Properties] [Code]                       │
    │   U-Net (ResNet-34)     │ Unet · resnet34   │ ✔     │ │ Name, Id, Description, Decoder, Encoder,  │
    │ Others (2)              │                   │       │ │ Decoder options, Recommended, References  │
    │ [New…] [Remove] [Check all] [Reload] [Folder]       │ │ [Check] [External]        [Revert] [Save] │
    └─────────────────────────────────────────────────────┘ └───────────────────────────────────────────┘

    ┌ Weights                 ┬ Fits                       ┬ For   ┬ Licence ┐  details: what, licence, references,
    │ ImageNet                │ any encoder                │ any   │ …       │  downloaded or not
    │ MicroNet (microscopy)   │ resnet18, resnet34, …      │ any   │ MIT     │
    │ [Import weights…] [Download] [Remove] [Reload] [Folder]                │

Architectures are scripts (``training.architectures``): built-in ones ship with
Modalytics and are never changed in place; Save makes your own copy. **Check** builds
the network, exports it to ONNX and compares ONNX Runtime with PyTorch — the contract
every architecture must meet before it is trained (Segment runs ONNX). Encoder weights
(``training.encoders``) are data, not code: Import weights… makes a package from a
PyTorch or safetensors file after checking it fits the encoder named.
"""

from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics.parameters.script_file import ArchitectureScript
from modalytics.training.architectures import DECODERS, ArchitectureLibrary, ArchSpec, load_script
from modalytics.training.encoders import EncoderLibrary
from modalytics.ui.settings.script_editor import ScriptEditor
from modalytics.ui.settings.widgets import clear_tree, group_item, is_group, keep_on, list_tree, muted, open_in_system

OWN = "(a network of its own: build() in the Code tab)"
TEMPLATE = '''"""{name}: {description}

Your segmentation architecture (made in Settings › Architectures). See
modalytics/training/architectures.py for the format of this file. Check proves that
it exports to ONNX (Segment runs ONNX) before it is trained.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = {key!r}
NAME = {name!r}
DESCRIPTION = {description!r}
DECODER = 'Unet'
ENCODER = 'resnet34'
DECODER_OPTIONS = {{}}
RECOMMENDED = False
REFERENCES = []

# A network of your own instead of DECODER + ENCODER: define build() (DECODER is then ignored).
# ENCODER says what model.encoder is, so encoder weights (ImageNet, MicroNet, yours) can start it.
#
# def build(in_channels, classes, encoder_weights):
#     import segmentation_models_pytorch as smp
#     return smp.Unet('resnet34', encoder_weights=encoder_weights, in_channels=in_channels, classes=classes,
#                     decoder_attention_type='scse')
'''

HELP = [
    ("Properties", "DECODER", "DECODER = 'Unet'", "A segmentation-models-pytorch decoder: " + ", ".join(DECODERS) + "."),
    ("Properties", "ENCODER", "ENCODER = 'resnet34'", "A segmentation-models-pytorch encoder ('tu-…' for a timm one, "
                                                      "e.g. 'tu-convnext_tiny')."),
    ("Properties", "DECODER_OPTIONS", "DECODER_OPTIONS = {'decoder_attention_type': 'scse'}",
     "Passed to the decoder (its own settings)."),
    ("Your own network", "build", "def build(in_channels, classes, encoder_weights):\n    return model",
     "Returns a torch.nn.Module: (N, in_channels, H, W) → (N, classes, H, W) logits, H and W any multiple of 32; "
     "encoder_weights is 'imagenet' or None."),
]


def _run_in_thread(parent: QtWidgets.QWidget, label: str, fn):
    """Run ``fn()`` off the window's thread behind a busy dialog; return its result (or raise its error)."""
    box = {}

    def work():
        try:
            box["result"] = fn()
        except BaseException as exc:                       # handed back to the caller
            box["error"] = exc
    t = threading.Thread(target=work, daemon=True)
    dlg = QtWidgets.QProgressDialog(label, None, 0, 0, parent)
    dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(200)
    t.start()
    while t.is_alive():
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 100)
        t.join(0.05)
    dlg.close()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def check_text(result: dict) -> str:
    """One line for a check's result."""
    if result.get("ok"):
        return (f"✔ Passes: {result['params_m']:.1f} M parameters, ONNX {result['onnx_mb']:.0f} MB, ONNX = PyTorch "
                f"within {result['max_diff']:.1g} ({result['seconds']:.0f} s)")
    return f"✖ {result.get('why', 'failed')}"


class ArchitectureEditor(ScriptEditor):
    """Edits the selected architecture (see the module docstring)."""

    SCRIPT = ArchitectureScript
    WHAT = "architecture"
    TEST_TIP = "Build it, export it to ONNX and compare with PyTorch (Ctrl+T)"

    checked = QtCore.Signal()                              # a saved file was checked (the list shows it)
    catalogue_ready = QtCore.Signal(object)                # the encoders the libraries know (read in the background)

    def __init__(self, library, parent=None) -> None:
        super().__init__(library, parent)
        self.tabs.setTabVisible(1, False)                  # no option rows: the decoder's options are a property
        self.test_btn.setText("Check")
        self.catalogue_ready.connect(self._fill_encoders)
        self._catalogue_asked = False                      # read when the page is first shown (it loads PyTorch)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        if self._catalogue_asked:
            return
        self._catalogue_asked = True
        from modalytics.training import encoders as en
        if en.catalogue(wait=False) is not None:
            self._fill_encoders(en.catalogue())
        else:
            en.load_catalogue(lambda cat: self.catalogue_ready.emit(cat))

    def all_properties(self) -> dict:
        return self.properties()

    def _build_form(self, form: QtWidgets.QFormLayout) -> None:
        self.name = QtWidgets.QLineEdit()
        self.key = QtWidgets.QLineEdit()
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(60)
        self.decoder = QtWidgets.QComboBox()
        self.decoder.addItems([*DECODERS, OWN])
        self.decoder.setToolTip("A segmentation-models-pytorch decoder, or a network of your own (build() in Code)")
        self.encoder = QtWidgets.QComboBox()
        self.encoder.setEditable(True)
        self.encoder.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.encoder.setMaxVisibleItems(20)
        self.encoder.setToolTip("Pick or type an encoder: segmentation-models-pytorch's, or timm's as tu-<model>. "
                                "Those with starting weights come first; grey: random weights only.")
        completer = QtWidgets.QCompleter(self.encoder.model(), self.encoder)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.encoder.setCompleter(completer)
        self.encoder_hint = QtWidgets.QLabel("Loading the list of encoders…")
        self.encoder_hint.setObjectName("Muted")
        self.encoder_hint.setWordWrap(True)
        self.options = QtWidgets.QLineEdit()
        self.options.setPlaceholderText("{}  — e.g. {'decoder_attention_type': 'scse'}")
        self.recommended = QtWidgets.QCheckBox("Recommended (shown in bold in Train's list of models)")
        self.references = QtWidgets.QPlainTextEdit()
        self.references.setFixedHeight(80)
        self.references.setPlaceholderText("One per line: Author (year). Title. Journal … | doi:10.…")
        self.requires = QtWidgets.QLineEdit()
        self.requires.setPlaceholderText("Libraries it imports beyond the included ones")
        for label, w in (("Name", self.name), ("Id", self.key), ("Description", self.description),
                         ("Decoder", self.decoder), ("Encoder", self.encoder), ("", self.encoder_hint),
                         ("Decoder options", self.options),
                         ("", self.recommended), ("References", self.references), ("Requires", self.requires)):
            form.addRow(label, w)
        for sig in (self.name.textEdited, self.key.textEdited, self.description.textChanged,
                    self.decoder.currentIndexChanged, self.encoder.editTextChanged, self.options.textEdited,
                    self.recommended.toggled, self.references.textChanged, self.requires.textEdited):
            sig.connect(self._changed)
        self.name.textEdited.connect(lambda t: self.title.setText(t or "(no name)"))
        self.encoder.editTextChanged.connect(self._hint)
        self.decoder.currentIndexChanged.connect(self._hint)

    def _fill_encoders(self, cat) -> None:
        """The encoders to pick from (with starting weights first; grey: random only), the typed name kept."""
        from PySide6 import QtGui

        from modalytics.training import encoders as en
        keep, loading = self.encoder.currentText(), self._loading
        self._loading = True
        self.encoder.blockSignals(True)
        self.encoder.clear()
        grey = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        for name, starts in en.choices():
            self.encoder.addItem(name)
            i = self.encoder.count() - 1
            self.encoder.setItemData(i, self._weights_text(name, starts), QtCore.Qt.ItemDataRole.ToolTipRole)
            if not starts:
                self.encoder.setItemData(i, grey, QtCore.Qt.ItemDataRole.ForegroundRole)
        self.encoder.setEditText(keep)
        self.encoder.blockSignals(False)
        self._loading = loading
        self._hint()

    @staticmethod
    def _weights_text(name: str, starts: list[str]) -> str:
        if not starts:
            return "No starting weights: trained from random weights only"
        if name.startswith("tu-"):                          # timm's default weights are not always ImageNet
            starts = ["timm's pretrained weights (usually ImageNet)" if s == "ImageNet" else s for s in starts]
        return "Starting weights: " + ", ".join(starts)

    def _hint(self, *_args) -> None:
        """Under the encoder: which starting weights fit it, or that the name is unknown (with close names)."""
        from modalytics.training import encoders as en
        name = self.encoder.currentText().strip()
        cat = en.catalogue(wait=False)
        if self.decoder.currentText() == OWN:
            text = ("The name of your network's encoder (model.encoder): starting weights made for it fit it; "
                    "ImageNet only if build() loads it.")
        elif cat is None:
            text = "Loading the list of encoders…"
        elif not cat:
            text = "The training libraries are not installed: the name is checked when they are."
        elif name in cat:
            text = self._weights_text(name, [w.name for w in en.EncoderLibrary().for_encoder(name)])
        else:
            close = en.known_encoder(name)[1] if name else []
            text = ("✖ segmentation-models-pytorch and timm know no encoder called '" + name + "'"
                    + (f" — did you mean {', '.join(close)}?" if close else "")) if name else "Pick an encoder."
        self.encoder_hint.setText(text)

    def _fill(self, p: dict) -> None:
        self.title.setText(p.get("NAME", p.get("KEY", "")))
        self.name.setText(str(p.get("NAME", "")))
        self.key.setText(str(p.get("KEY", "")))
        self.description.setPlainText(str(p.get("DESCRIPTION", "")))
        self.decoder.setCurrentText(p.get("DECODER") if p.get("DECODER") in DECODERS else OWN)
        self.encoder.setEditText(str(p.get("ENCODER", "")))
        self.options.setText(repr(p.get("DECODER_OPTIONS", {}) or {}))
        self.recommended.setChecked(bool(p.get("RECOMMENDED", False)))
        self.references.setPlainText("\n".join(str(r.get("text", "")) + (f" | doi:{r['doi']}" if r.get("doi") else "")
                                               for r in p.get("REFERENCES", []) or [] if isinstance(r, dict)))
        self.requires.setText(", ".join(p.get("REQUIRES", []) or []))
        self.helper.set_entries("architecture", HELP)

    def load(self, spec, path: str | None = None) -> None:
        super().load(spec, path)
        if spec is not None and spec.custom:                 # build() is in the code: shown as "of its own"
            self._loading = True
            self.decoder.setCurrentText(OWN)
            self._loading = False

    def lock_builtin(self, builtin: bool) -> None:
        self.key.setReadOnly(builtin)
        self.key.setToolTip("Built-in architectures keep their id (your copy replaces the built-in)." if builtin else
                            "Changing the id renames the file; models trained before keep their own record.")

    def properties(self) -> dict:
        p = dict(self._known)
        refs = []
        for line in self.references.toPlainText().splitlines():
            if not line.strip():
                continue
            text, _sep, doi = line.partition("| doi:")
            refs.append({"text": text.strip(), **({"doi": doi.strip()} if doi.strip() else {})})
        try:
            options = ast.literal_eval(self.options.text().strip() or "{}")
        except (ValueError, SyntaxError):
            options = self.options.text()                    # refused by validate
        p.update({"KEY": self.key.text().strip(), "NAME": self.name.text().strip(),
                  "DESCRIPTION": " ".join(self.description.toPlainText().split()),
                  "DECODER": "" if self.decoder.currentText() == OWN else self.decoder.currentText(),
                  "ENCODER": self.encoder.currentText().strip(), "DECODER_OPTIONS": options,
                  "RECOMMENDED": self.recommended.isChecked(), "REFERENCES": refs})
        p.pop("REQUIRES", None)
        requires = [r.strip() for r in self.requires.text().split(",") if r.strip()]
        if requires:
            p["REQUIRES"] = requires
        return p

    def validate(self, p: dict) -> None:
        super().validate(p)
        if not p["NAME"]:
            raise ValueError("The architecture needs a name")
        if not isinstance(p["DECODER_OPTIONS"], dict):
            raise ValueError("Decoder options must be a Python dict, e.g. {'decoder_attention_type': 'scse'}")
        if not p["DECODER"] and not re.search(r"^def build\(", self.code.toPlainText(), re.MULTILINE):
            raise ValueError("A network of its own needs build(in_channels, classes, encoder_weights) in the Code tab")
        if p["DECODER"] and not p["ENCODER"]:
            raise ValueError("The encoder must be named, e.g. resnet34")
        if p["DECODER"]:
            from modalytics.training import encoders as en
            known, close = en.known_encoder(p["ENCODER"])
            if not known:
                raise ValueError(f"segmentation-models-pytorch and timm know no encoder called '{p['ENCODER']}'"
                                 + (f". Did you mean {', '.join(close)}?" if close else ""))

    def target(self, p: dict) -> Path:
        return self.library.user / f"{p['KEY']}.py"

    def run_test(self, f: Path) -> None:
        """Check the architecture as edited: build, export to ONNX, compare with PyTorch."""
        from modalytics.training import architectures as ar
        try:
            spec = load_script(f, "user")
        except ValueError as exc:
            self.report_error(exc, f, loading=True)
            return
        try:
            result = _run_in_thread(self, f"Checking {spec.name}: building, exporting to ONNX, comparing…",
                                    lambda: ar.check(spec))
        except Exception as exc:
            self.report_error(exc, f)
            return
        self.result.setText(check_text(result) + ("" if not self.dirty else "\n(as edited: Save keeps it)"))
        if not self.dirty and self.spec is not None:            # the saved file: remembered for the list
            self.library.remember(self.spec, result)
            self.checked.emit()


class ArchitecturesTab(QtWidgets.QWidget):
    """The architectures list and editor."""

    library_changed = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.library = ArchitectureLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split, 1)
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Architecture", "Decoder · encoder", "Check"])
        self.tree.setColumnWidth(0, 210)
        self.tree.setColumnWidth(1, 170)
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your architecture (a copy of a built-in: the built-in comes back)")
        check_all = QtWidgets.QPushButton("Check all")
        check_all.setToolTip("Check every architecture not checked yet (ONNX export; once per file and version)")
        for b, slot in ((QtWidgets.QPushButton("New…"), self._new), (self.remove_btn, self._remove),
                        (check_all, self.check_all), (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        split.addWidget(left)
        self.editor = ArchitectureEditor(self.library)
        self.editor.saved.connect(self._saved)
        self.editor.checked.connect(lambda: self.reload())
        split.addWidget(self.editor)
        split.setSizes([460, 680])
        self.reload()

    def reload(self, select: str | None = None) -> None:
        select = select or (self._spec().key if self._spec() else None)
        self.library.reload()
        self._built = getattr(self, "_built", 0) + 1
        self.tree.blockSignals(True)
        clear_tree(self.tree)
        role = QtCore.Qt.ItemDataRole.UserRole
        chosen = None
        for title, specs in (("Recommended", [s for s in self.library.specs.values() if s.recommended]),
                             ("Others", [s for s in self.library.specs.values() if not s.recommended])):
            if not specs:
                continue
            top = group_item(self.tree, f"{title} ({len(specs)})")
            for spec in specs:
                done = self.library.checked(spec)
                mark = "—" if done is None else ("✔" if done.get("ok") else "✖")
                parts = "own network" if spec.custom else f"{spec.decoder} · {spec.encoder}"
                item = QtWidgets.QTreeWidgetItem(top, [spec.name + ("  · yours" if spec.source == "user" else ""),
                                                       parts, mark])
                item.setData(0, role, spec.key)
                item.setToolTip(0, spec.description + ("\n\n" + spec.references_text() if spec.references else ""))
                item.setToolTip(2, "Not checked yet" if done is None else check_text(done))
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

    def _spec(self) -> ArchSpec | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(key) if key else None

    def _selected(self, cur, prev) -> None:
        if cur is prev:
            return
        if is_group(cur):
            keep_on(self.tree, prev)
            return
        if not self.editor.confirm_leave():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(prev)
            self.tree.blockSignals(False)
            return
        self._show(cur)

    def _show(self, item) -> None:
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        self.editor.load(spec, broken if broken else None)
        self.editor.set_missing(self.library.missing_libraries().get(broken or (spec.key if spec else ""), []))
        if spec is not None and (done := self.library.checked(spec)) is not None:
            self.editor.result.setText(check_text(done))
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
        self.library_changed.emit()

    def check_all(self) -> None:
        """Check every architecture whose current file has not been checked."""
        todo = [s for s in self.library.specs.values() if self.library.checked(s) is None]
        if not todo:
            self.editor.result.setText("ℹ Every architecture is checked already.")
            return
        try:
            _run_in_thread(self, f"Checking {len(todo)} architecture(s): building, exporting to ONNX, comparing…",
                           lambda: [self.library.check(s) for s in todo])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Check all", f"The check stopped:\n\n{exc}")
        self.reload()

    def _new(self) -> None:
        if not self.editor.confirm_leave():
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New architecture", "Name:")
        if not ok or not name.strip():
            return
        key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "architecture"
        if key[0].isdigit():
            key = "a_" + key
        n, base = 2, key
        while self.library.get(key) or (self.library.user / f"{key}.py").exists():
            key, n = f"{base}_{n}", n + 1
        target = self.library.user / f"{key}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(TEMPLATE.format(key=key, name=name.strip(), description="What it is, in a sentence."),
                          encoding="utf-8")
        self.reload(key)
        self.library_changed.emit()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        path = broken or (spec.path if spec and spec.source == "user" else None)
        if path is None:
            return
        replaces = spec is not None and spec.key in self.library.overridden
        msg = f"Delete “{spec.name if spec else Path(path).name}”?\n\n{path}" + \
            ("\n\nThe built-in version comes back." if replaces else
             "\n\nModels trained as it keep working (they hold their network); fine-tuning them needs it back.")
        if QtWidgets.QMessageBox.question(self, "Remove architecture", msg) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.key if replaces else None)
        self.library_changed.emit()


class ImportWeightsDialog(QtWidgets.QDialog):
    """What a weights file is: name, the encoder it fits, image types, licence, references."""

    def __init__(self, path: str, encoders: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import encoder weights")
        form = QtWidgets.QFormLayout(self)
        form.addRow(muted(f"From {Path(path).name}. The weights are checked against the encoder before they are "
                          f"kept; only numbers are read (no code)."))
        self.name = QtWidgets.QLineEdit(Path(path).stem.split(".")[0])
        self.encoder = QtWidgets.QComboBox()
        self.encoder.setEditable(True)
        self.encoder.addItems(encoders)
        self.encoder.setToolTip("The segmentation-models-pytorch encoder these weights are for")
        self.types = QtWidgets.QLineEdit("*")
        self.types.setToolTip("Image types it is meant for, comma-separated (* = any), e.g. phase")
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setFixedHeight(60)
        self.description.setPlaceholderText("What it was trained on")
        self.licence = QtWidgets.QLineEdit()
        self.licence.setPlaceholderText("e.g. MIT, CC-BY-4.0, research only")
        self.free = QtWidgets.QCheckBox("Models trained from it may be shared freely")
        self.references = QtWidgets.QPlainTextEdit()
        self.references.setFixedHeight(60)
        self.references.setPlaceholderText("One per line: Author (year). Title. Journal … | doi:10.…")
        for label, w in (("Name", self.name), ("Encoder", self.encoder), ("Image types", self.types),
                         ("Description", self.description), ("Licence", self.licence), ("", self.free),
                         ("References", self.references)):
            form.addRow(label, w)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        refs = []
        for line in self.references.toPlainText().splitlines():
            if line.strip():
                text, _sep, doi = line.partition("| doi:")
                refs.append({"text": text.strip(), **({"doi": doi.strip()} if doi.strip() else {})})
        return {"name": self.name.text().strip(), "encoder": self.encoder.currentText().strip(),
                "image_types": [t.strip() for t in self.types.text().split(",") if t.strip()] or ["*"],
                "description": " ".join(self.description.toPlainText().split()), "licence": self.licence.text().strip(),
                "redistributable": self.free.isChecked(), "references": refs}


class EncodersTab(QtWidgets.QWidget):
    """The encoder-weights list, its details, and importing your own."""

    library_changed = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.library = EncoderLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split, 1)
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Weights", "Fits", "For", "Licence"])
        self.tree.setColumnWidth(0, 190)
        self.tree.setColumnWidth(1, 220)
        self.tree.setColumnWidth(2, 60)
        self.tree.currentItemChanged.connect(lambda *_a: self._show())
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.download_btn = QtWidgets.QPushButton("Download")
        self.download_btn.setToolTip("Fetch all its files now (else each is fetched when first used)")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        for b, slot in ((QtWidgets.QPushButton("Import weights…"), self._import), (self.download_btn, self._download),
                        (self.remove_btn, self._remove), (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        split.addWidget(left)
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        self.details.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        self.details.setOpenExternalLinks(True)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)   # the text wraps
        scroll.setWidget(self.details)
        split.addWidget(scroll)
        split.setSizes([620, 520])
        self.reload()

    def reload(self, select: str | None = None) -> None:
        cur = self._spec()
        select = select or (cur.key if cur else None)
        self.library.reload()
        clear_tree(self.tree)
        chosen = None
        for spec in self.library.specs.values():
            fits = "any encoder" if "*" in spec.encoders else ", ".join(spec.encoders)
            item = QtWidgets.QTreeWidgetItem(self.tree, [spec.name + ("  · yours" if spec.source == "user" else ""),
                                                         fits, ", ".join(spec.image_types).replace("*", "any"),
                                                         spec.licence])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, spec.key)
            item.setToolTip(1, fits)
            item.setToolTip(3, spec.licence)
            if spec.key == select or chosen is None:
                chosen = item
        for path, why in self.library.errors.items():
            item = QtWidgets.QTreeWidgetItem(self.tree, [Path(path).name, "could not be read"])
            item.setToolTip(0, why)
        self.tree.setCurrentItem(chosen)
        self._show()

    def _spec(self):
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(key) if key else None

    def _downloaded(self, spec) -> list[str]:
        folder = self.library.cache_dir() / spec.key
        return [enc for enc, f in spec.files.items() if (folder / Path(f["url"]).name).exists()]

    def _show(self) -> None:
        spec = self._spec()
        self.remove_btn.setEnabled(spec is not None and spec.source == "user")
        self.download_btn.setEnabled(spec is not None and spec.kind == "download")
        if spec is None:
            self.details.setText("")
            return
        import html
        e = html.escape
        refs = "".join(f"<li>{e(r.get('text', ''))}" + (f" doi:{e(r['doi'])}" if r.get("doi") else "") + "</li>"
                       for r in spec.references)
        where = {"imagenet": "downloaded once by segmentation-models-pytorch for the encoder used",
                 "download": f"fetched once per encoder (downloaded: {', '.join(self._downloaded(spec)) or 'none yet'})",
                 "package": f"your package, {e(Path(spec.path).name)} (Folder)"}[spec.kind]
        self.details.setText(
            f"<h3>{e(spec.name)}</h3><p>{e(spec.description)}</p>"
            f"<p><b>Fits:</b> {e('any encoder with ImageNet weights' if '*' in spec.encoders else ', '.join(spec.encoders))}"
            f"<br><b>For:</b> {e(', '.join(spec.image_types).replace('*', 'any image type'))}"
            f"<br><b>Licence:</b> {e(spec.licence or 'not stated')} — "
            f"{'models trained from it may be shared' if spec.redistributable else '<b>non-commercial: models trained from it carry that restriction</b>'}"
            f"<br><b>Weights:</b> {where}"
            + (f"<br><b>From:</b> <a href='{e(spec.homepage)}'>{e(spec.homepage)}</a>" if spec.homepage else "")
            + "</p>" + (f"<p><b>References</b></p><ol>{refs}</ol>" if refs else ""))

    def _encoders(self) -> list[str]:
        """Encoder names to offer: the architectures' first, then the ones weights exist for."""
        names = [s.encoder for s in ArchitectureLibrary().specs.values() if s.encoder]
        names += [n for s in self.library.specs.values() for n in s.encoders if n != "*"]
        return list(dict.fromkeys(names))

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import encoder weights", "",
                                                        "Weights (*.pth *.pt *.tar *.safetensors);;All files (*)")
        if not path:
            return
        dlg = ImportWeightsDialog(path, self._encoders(), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        from modalytics.training import encoders as en
        try:
            spec = _run_in_thread(self, "Reading and checking the weights…", lambda: en.import_weights(
                path, v["encoder"], v["name"], description=v["description"], image_types=v["image_types"],
                licence=v["licence"], redistributable=v["redistributable"], references=v["references"],
                library=self.library))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import encoder weights", f"Not imported:\n\n{exc}")
            return
        self.reload(spec.key)
        self.library_changed.emit()

    def _download(self) -> None:
        spec = self._spec()
        if spec is None or spec.kind != "download":
            return
        from modalytics.training import encoders as en
        size = sum(int(f.get("size", 0)) for f in spec.files.values()) / 1e6
        if QtWidgets.QMessageBox.question(self, "Download", f"Fetch {spec.name} for {len(spec.files)} encoders "
                                          f"(about {size:.0f} MB) from {spec.homepage or 'its source'}?") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            _run_in_thread(self, f"Downloading {spec.name}…",
                           lambda: [en.state_dict(spec, enc, self.library) for enc in spec.files])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Download", f"The download stopped:\n\n{exc}")
        self._show()

    def _remove(self) -> None:
        spec = self._spec()
        if spec is None or spec.source != "user":
            return
        if QtWidgets.QMessageBox.question(self, "Remove encoder weights", f"Delete “{spec.name}”?\n\n{spec.path}\n\n"
                                          "Models trained from it keep working.") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        Path(spec.path).unlink(missing_ok=True)
        self.reload()
        self.library_changed.emit()


class ArchitecturesPage(QtWidgets.QWidget):
    """Settings › Architectures: two tabs, Architectures and Encoder weights.

    Signals:
        library_changed(): an architecture or encoder weights were added, changed or removed.
    """

    library_changed = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(muted("The networks segmentation models are trained as (Training › Segmentation models), and "
                            "the weights their encoders can start from. Every architecture must pass its Check — it "
                            "exports to ONNX, which Segment runs — before it is trained. The training libraries "
                            "(the full edition) are needed to check, import or train."))
        self.tabs = QtWidgets.QTabWidget()
        self.architectures = ArchitecturesTab()
        self.encoders = EncodersTab()
        self.tabs.addTab(self.architectures, "Architectures")
        self.tabs.addTab(self.encoders, "Encoder weights")
        col.addWidget(self.tabs, 1)
        self.architectures.library_changed.connect(self.library_changed)
        self.encoders.library_changed.connect(self.library_changed)

    def reload(self) -> None:
        self.architectures.reload()
        self.encoders.reload()

    def confirm_leave(self) -> bool:
        return self.architectures.confirm_leave()
