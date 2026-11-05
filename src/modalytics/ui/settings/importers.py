"""
Settings › Importers: the formats Modalytics reads (browse, add, edit, remove).

    ┌ Importer                  Files ┬ (ImporterEditor) ─────────────────────┐
    │ Phase map (2)                   │ [Properties] [Options] [Code]          │
    │   Lyncée Tec DHM (Phase.txt) .txt│                                        │
    │   Phasics PhaseStudio  .txt .csv│                                        │
    │ Brightfield (0)                 │                                        │
    │ Whole-slide image · locked      │                                        │
    │   Zarr pyramid (folder)         │                                        │
    │ [New…] [Remove] [Reload] [Folder]│ [Test] [External] [Sample…] [Save]    │
    └─────────────────────────────────┴────────────────────────────────────────┘

Importers are scripts, like parameters: built-in ones ship with Modalytics
(our instruments) and are never changed in place; saving one writes your own
copy with the same id, which replaces it until you remove it. New importers
start from a commented template. What an importer must hand back for each
image type is fixed by Modalytics (``image_import.contracts``): phase maps in
radians with their pixel size in µm, brightfield as 8/16-bit grey or RGB.
Whole-slide images need tiled readers, so they are read by Modalytics' own
code only (listed here, locked).
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from modalytics.image_import.registry import TYPE_NAMES, ImporterLibrary, ImporterSpec
from modalytics.ui.settings.importer_editor import ImporterEditor
from modalytics.ui.settings.widgets import (clear_tree, group_item, is_group, keep_on, list_tree, muted,
                                            open_in_system)

# Modalytics' own readers (not scripts), shown but not editable: (importer id for your choices at
# import, or None; name; files; description)
LOCKED = [
    ("wsi:tiff", "Slide file, read in place", ".svs .tif .ndpi .scn",
     "Whole-slide images that are already tiled pyramids (Aperio SVS, pyramidal TIFF and OME-TIFF such as LI-COR "
     "Odyssey scans, Hamamatsu NDPI, Leica SCN) are read where they are, tile by tile: no conversion, no copy. "
     "Their pixel size and scanner details come from the file. Large images that are not pyramids are converted "
     "once into a pyramidal OME-TIFF next to them. Slides are tens of gigabytes, so their readers are part of "
     "Modalytics (File › Import › Images); more slide formats will come with Modalytics updates."),
    (None, "HoloPath project (.hpro)", ".hpro",
     "A HoloPath project: its images (phase maps already in radians, slides) and their masks "
     "(File › Import › HoloPath project). Their pixel sizes are the ones set in HoloPath."),
]

TEMPLATES = {
    "phase": '''
def read(path, options):
    # Read the values (here: a plain table of numbers; skip header lines with skiprows=).
    values = np.loadtxt(path, dtype=np.float32)
    # Pixel size in the sample, if the file gives it (None: asked at import, then your default).
    # unit: the unit of the values as read (rad, mrad, deg or wv); Modalytics converts to radians.
    # height_factor_nm (optional): optical height in nm per unit of the values as read.
    return Image(values, None, unit="rad")
''',
    "brightfield": '''
def read(path, options):
    # Read the pixels (cv2 reads PNG, JPEG, BMP, TIFF…; it gives BGR, reversed here to RGB).
    import cv2
    pixels = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("not an image file")
    if pixels.ndim == 3:
        pixels = pixels[..., 2::-1]
    # Pixel size in the sample, if the file gives it (None: asked at import, then your default)
    return Image(pixels, None)
''',
}

def map_template(image_type: str) -> str:
    """The read() of a new importer for a single-value map (OD, OPD, height…)."""
    from modalytics.image_import.contracts import MAP_UNITS
    units = MAP_UNITS[image_type]
    standard = next(iter(units))
    listed = ", ".join(repr(u) for u in units)
    return f'''
def read(path, options):
    # Read the values (here: a plain table of numbers; skip header lines with skiprows=).
    values = np.loadtxt(path, dtype=np.float32)
    # Pixel size in the sample, if the file gives it (None: asked at import, then your default).
    # unit: the unit of the values as read ({listed}); Modalytics converts to {standard!r}.
    return Image(values, None, unit={standard!r})
'''


TEMPLATE = '''"""{name}: {description}

Your importer (made in Settings › Importers). It reads one file format and
hands back an Image; Modalytics checks it against the {type_name} contract
(modalytics/image_import/contracts.py) and does the rest. See
modalytics/image_import/registry.py for the format of this script.
"""

import numpy as np

from modalytics.image_import import Image

KEY = {key!r}
NAME = {name!r}
IMAGE_TYPE = {image_type!r}
EXTENSIONS = {extensions!r}
DESCRIPTION = {description!r}
OPTIONS = []             # values to ask at import, e.g. [{{"key": "channel", "label": "Channel", "kind": "int", "default": 0}}]


def can_read(path):
    # How sure (0–1) that this file is your format. Check its first bytes or header line, not only
    # the extension: several importers may read the same extension, and the surest one is used.
    return 0.5


def hint(path):
    # Optional: what the file says that helps with what is asked (shown in the import dialog).
    return ""
{read}'''


def _key_from(name: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return key if key and not key[0].isdigit() else f"i_{key}"


class NewImporterDialog(QtWidgets.QDialog):
    """Name, image type and file extensions of a new importer."""

    def __init__(self, library: ImporterLibrary, image_type: str = "phase", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New importer")
        self.library = library
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. Tomocube RI map (.tcf)")
        self.image_type = QtWidgets.QComboBox()
        for k, v in TYPE_NAMES.items():
            self.image_type.addItem(v, k)
        self.image_type.setCurrentIndex(max(0, self.image_type.findData(image_type)))
        self.extensions = QtWidgets.QLineEdit()
        self.extensions.setPlaceholderText(".txt, .csv")
        self.description = QtWidgets.QLineEdit()
        form.addRow("Name", self.name)
        form.addRow("Image type", self.image_type)
        form.addRow("File extensions", self.extensions)
        form.addRow("Description", self.description)
        form.addRow(muted("It starts from a template that reads a table of numbers (phase) or a common image "
                          "file (brightfield); change read() for your format."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept(self) -> None:
        if not self.name.text().strip() or not self.exts():
            QtWidgets.QMessageBox.warning(self, "New importer", "Give it a name and at least one file extension.")
            return
        if self.uid() in self.library.specs:
            QtWidgets.QMessageBox.warning(self, "New importer", "An importer with this name exists already.")
            return
        self.accept()

    def exts(self) -> list[str]:
        raw = [e.strip().lower() for e in self.extensions.text().replace(";", ",").split(",") if e.strip()]
        return [e if e.startswith(".") else f".{e}" for e in raw]

    def uid(self) -> str:
        return f"{self.image_type.currentData()}:{_key_from(self.name.text())}"

    def script(self) -> tuple[str, str]:
        """(file path relative to your importers folder, script text)."""
        image_type, name = self.image_type.currentData(), self.name.text().strip()
        key = _key_from(name)
        text = TEMPLATE.format(name=name, key=key, image_type=image_type, extensions=self.exts(),
                               description=self.description.text().strip() or "reads …",
                               type_name=TYPE_NAMES[image_type].lower(),
                               read=TEMPLATES.get(image_type) or map_template(image_type))
        return f"{image_type}/{key}.py", text


class ImportersPage(QtWidgets.QWidget):
    """The importer library page.

    Signals:
        library_changed(): importers were added, changed or removed.
    """

    library_changed = QtCore.Signal()

    def __init__(self, library: ImporterLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        self.library = library or ImporterLibrary()
        split = QtWidgets.QSplitter()
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(split)

        # 1. Left: the tree and the library actions
        left = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        self.tree = list_tree(["Importer", "Files"])
        self.tree.setColumnWidth(0, 230)
        self.tree.currentItemChanged.connect(self._selected)
        lc.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("Delete your importer (a copy of a built-in: the built-in comes back)")
        for b, slot in ((QtWidgets.QPushButton("New…"), self._new), (self.remove_btn, self._remove),
                        (QtWidgets.QPushButton("Reload"), self.reload),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.library.user))):
            b.clicked.connect(slot)
            row.addWidget(b)
        lc.addLayout(row)
        libraries_btn = QtWidgets.QPushButton("Libraries…")
        libraries_btn.setToolTip("What scripts can import: the included libraries, and yours")
        libraries_btn.clicked.connect(lambda: self.open_libraries())
        lc.addWidget(libraries_btn)
        split.addWidget(left)

        # 2. Right: the editor of the selected importer
        self.editor = ImporterEditor(self.library)
        self.editor.saved.connect(self._saved)
        self.editor.install_requested.connect(self.open_libraries)
        split.addWidget(self.editor)
        split.setSizes([320, 740])
        self.reload()

    # ------------------------------------------------------------ the tree
    def reload(self, select: str | None = None) -> None:
        """Rescan the scripts and rebuild the tree (keeps the selection, or selects ``select``)."""
        select = select or (self._spec().uid if self._spec() else None)
        self.library.reload()
        self._built = getattr(self, "_built", 0) + 1     # rows from an earlier build are gone
        self.tree.blockSignals(True)
        clear_tree(self.tree)
        chosen = None
        role = QtCore.Qt.ItemDataRole.UserRole
        for image_type, type_name in TYPE_NAMES.items():
            of_type = [s for s in self.library.specs.values() if s.image_type == image_type]
            if not of_type and image_type not in ("phase", "brightfield"):
                continue                                 # maps usually come from processing: listed once used
            top = group_item(self.tree, f"{type_name} ({len(of_type)})")
            if not of_type:
                top.setToolTip(0, "None yet: New… adds one")
            for spec in of_type:
                mark = "  · yours" if spec.source == "user" else ""
                item = QtWidgets.QTreeWidgetItem(top, [spec.name + mark, " ".join(spec.extensions)])
                item.setData(0, role, spec.uid)
                item.setToolTip(0, spec.description)
                if spec.uid == select:
                    chosen = item
            top.setExpanded(True)
        locked = group_item(self.tree, "Included · locked")
        locked.setToolTip(0, "Read by Modalytics itself (whole-slide images need tiled readers)")
        for i, (_uid, name, files, _text) in enumerate(LOCKED):
            item = QtWidgets.QTreeWidgetItem(locked, [f"🔒 {name}", files])
            item.setData(0, role + 2, i)
        locked.setExpanded(True)
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
        if chosen is None:                              # first importer
            it = QtWidgets.QTreeWidgetItemIterator(self.tree)
            while it.value() and not it.value().data(0, role):
                it += 1
            chosen = it.value()
        self.tree.setCurrentItem(chosen)
        self._show(chosen)

    def _spec(self) -> ImporterSpec | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        return self.library.get(uid) if uid else None

    def _selected(self, cur, prev) -> None:
        if cur is prev:
            return
        if is_group(cur):                                # a group row only opens / closes
            keep_on(self.tree, prev)
            return
        role = QtCore.Qt.ItemDataRole.UserRole
        wanted = [cur.data(0, role + k) for k in range(3)]      # the row, by its ids
        built = self._built
        if not self.editor.confirm_leave():              # stay on the importer with unsaved edits
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(prev)
            self.tree.blockSignals(False)
            return
        if self._built != built:                         # saving rebuilt the tree: ``cur`` is gone
            self._select_row(wanted)
            return
        self._show(cur)

    def _select_row(self, wanted: list) -> None:
        """Select the row whose ids (UserRole, UserRole + 1…) are ``wanted``."""
        role = QtCore.Qt.ItemDataRole.UserRole
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            if [it.value().data(0, role + k) for k in range(len(wanted))] == wanted:
                self.tree.setCurrentItem(it.value())
                return
            it += 1

    def _show(self, item) -> None:
        role = QtCore.Qt.ItemDataRole.UserRole
        locked = item.data(0, role + 2) if item else None
        if locked is not None:
            uid, name, _files, text = LOCKED[locked]
            self.editor.show_locked(name, text, uid)
            self.remove_btn.setEnabled(False)
            return
        broken = item.data(0, role + 1) if item else None
        spec = self._spec()
        self.editor.load(spec, broken if broken else None)
        self.editor.set_missing(self.library.missing_libraries().get(broken or (spec.uid if spec else ""), []))
        self.remove_btn.setEnabled(bool(broken) or (spec is not None and spec.source == "user"))

    def select(self, uid: str) -> None:
        """Select an importer by id (or a broken script by path)."""
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            v = it.value()
            if uid in (v.data(0, QtCore.Qt.ItemDataRole.UserRole), v.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)):
                self.tree.setCurrentItem(v)
                return
            it += 1

    def confirm_leave(self) -> bool:
        return self.editor.confirm_leave()

    # ------------------------------------------------------------ actions
    def open_libraries(self, install: str = "") -> None:
        """The Libraries window (``install``: a package to put in its install field)."""
        from modalytics.ui.settings.libraries import LibrariesDialog
        dlg = LibrariesDialog(install, self)
        dlg.changed.connect(lambda: (self.reload(), self.library_changed.emit()))
        dlg.exec()

    def _saved(self, uid: str) -> None:
        """The editor wrote a script: rescan and keep it selected."""
        path = self.editor.path
        self.reload(uid or path)
        if not uid:                                      # a broken script: select it again by path
            self.select(path)
        self.library_changed.emit()

    def _new(self) -> None:
        if not self.editor.confirm_leave():
            return
        spec = self._spec()
        dlg = NewImporterDialog(self.library, spec.image_type if spec else "phase", self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, text = dlg.script()
        target = self.library.user / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.reload(dlg.uid())
        self.editor.tabs.setCurrentIndex(2)              # straight to the code
        self.library_changed.emit()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        broken = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) if item else None
        spec = self._spec()
        path = broken or (spec.path if spec and spec.source == "user" else None)
        if path is None:
            return
        replaces = spec is not None and spec.uid in self.library.overridden
        msg = (f"Delete “{spec.name if spec else Path(path).name}”?\n\n{path}"
               + ("\n\nThe built-in version comes back." if replaces else ""))
        if QtWidgets.QMessageBox.question(self, "Remove importer", msg) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.editor.dirty = False
        Path(path).unlink(missing_ok=True)
        self.reload(spec.uid if replaces else None)
        self.library_changed.emit()
