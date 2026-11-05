"""
File › Import › Images: import files with the importer library, confirming the pixel size.

    0. Slide files (SVS, TIFF pyramids, large tiled TIFFs: ``io.tiff_source.is_slide``)
       are opened in place by Modalytics' own reader, never loaded into memory.
    1. Each other file goes to the importer most sure it reads it (``ImporterSpec.score``);
       when two are equally sure, you choose (once for files that tie the same way).
    2. The files are read.
    3. The first time an importer is used, one dialog per importer shows the pixel
       size in the sample: prefilled with the value read from the file(s), with
       "✔ Read from the file", or, when the file does not give it, with your
       default (or a placeholder) and "⚠ Not in the file" plus what the file does
       say (e.g. the camera pitch). The importer's options, if any, are asked
       there too. What you confirm is kept (``image_import.defaults``); with
       "don't ask again" (ticked) later imports use it without asking and the
       status bar says what was used. Settings › Importers changes it.
    4. The images are checked against their image type's contract (phase →
       radians) and returned as ImageNodes; failures are collected, not fatal.
    5. An image larger than ``LARGE_PIXELS`` that is not a pyramid (a stitched
       mosaic saved as one plain file) is converted once into a pyramidal
       OME-TIFF next to the original (``io.convert``) and read in place from it,
       instead of being kept in memory and inside the project file.

Slides go through the same confirmation (``confirm_pixel_size``) as importers
"wsi:tiff" (files read in place). Zarr pyramid folders (HoloPath's conversions) are no longer imported;
those referenced by older projects still open (``io.zarr_source``).
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.image_import import defaults, run
from modalytics.image_import.contracts import describe_unit
from modalytics.image_import.registry import TYPE_NAMES, ImporterLibrary, ImporterSpec
from modalytics.image_import.run import ImportFailed, not_readable
from modalytics.ui.settings.script_editor import current_tokens
from modalytics.ui.settings.widgets import muted, setting_widget

SLIDE = ("wsi:tiff", "Slide file (SVS, pyramidal TIFF)")
SAME = 5e-7                                                  # µm: values closer than this are the same


def file_filter(library: ImporterLibrary) -> str:
    """The Open dialog's filter: every readable extension, slides, then one entry per importer."""
    from modalytics.io.tiff_source import SLIDE_EXTENSIONS
    every = " ".join(f"*{e}" for e in sorted(set(library.extensions()) | set(SLIDE_EXTENSIONS)))
    slides = f"Slides ({' '.join('*' + e for e in SLIDE_EXTENSIONS)})"
    per = [f"{s.name} ({' '.join('*' + e for e in s.extensions)})" for s in library.specs.values()]
    return ";;".join([f"Images ({every})", slides, *per, "All files (*)"])


def _um(v: float) -> str:
    return f"{v:.6g} µm"


class ImportDialog(QtWidgets.QDialog):
    """Confirms the pixel size (and asks the options) for a group of files of one importer.

    Args:
        uid: The importer id (for your saved choice).
        name: The importer's name.
        image_type: Its image type ("wsi" for slides).
        files: The files.
        from_file: Pixel size each file gives, one per image (None where it gives none).
        hint: What the file says that helps (e.g. the camera pitch).
        options: Options to ask (``core.settings.Setting``).
        values: Their current values.

    Attributes:
        pixel: The pixel-size spin box.
        values: The option values chosen.
        keep: "Keep this and don't ask again".
    """

    def __init__(self, uid: str, name: str, image_type: str, files: list[str], from_file: list[float | None],
                 hint: str = "", options=(), values: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Import with {name}")
        self.setMinimumWidth(480)
        self.from_file = list(from_file)
        self.values = dict(values or {})
        form = QtWidgets.QFormLayout(self)
        names = ", ".join(os.path.basename(f) for f in files[:3]) + (f" and {len(files) - 3} more"
                                                                     if len(files) > 3 else "")
        kind = TYPE_NAMES.get(image_type, "Whole-slide image" if image_type == "wsi" else image_type)
        form.addRow(muted(f"{len(files)} file(s): {names} · {kind}"))

        # 1. The pixel size: from the file(s), or yours
        known = [v for v in from_file if v]
        mine = (defaults.pixel(uid) or {}).get("pixel_size_um")
        if known and len(known) == len(from_file):
            same = max(known) - min(known) < SAME
            self.status = ("✔ Pixel size read from the file: " + _um(known[0])) if same else \
                (f"✔ Pixel size read from each file: {_um(min(known))} to {_um(max(known))}. Each file keeps "
                 f"its own unless you change the value below.")
            start, ok = known[0], True
        elif known:
            self.status = (f"✔ Pixel size read from {len(known)} of {len(from_file)} images; ⚠ not in the "
                           f"others: they use the value below.")
            start, ok = known[0], True
        else:
            self.status = "⚠ The file does not give the pixel size in the sample: please enter it." + \
                (f" Proposed: your default ({_um(mine)})." if mine else "")
            start, ok = (mine or 1.0), False
        self.start = start
        self.status_label = QtWidgets.QLabel(self.status)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {current_tokens()['accent' if ok else 'warning']}; "
                                        f"font-weight: 600;")
        form.addRow(self.status_label)
        if hint and not ok:
            form.addRow(muted(hint))
        self.pixel = QtWidgets.QDoubleSpinBox()
        self.pixel.setDecimals(6)
        self.pixel.setRange(0.000001, 10000.0)
        self.pixel.setSingleStep(0.001)
        self.pixel.setSuffix(" µm")
        self.pixel.setValue(start)
        self.pixel.setToolTip("Size of one image pixel in the sample. Changing a value read from the file "
                              "uses yours instead, for these files and later ones (if kept).")
        form.addRow("Pixel size in the sample", self.pixel)

        # 2. The importer's own options
        for o in options:
            form.addRow(o.label, setting_widget(o, self.values[o.key],
                                                lambda v, k=o.key: self.values.__setitem__(k, v)))

        # 3. Keep it
        self.keep = QtWidgets.QCheckBox("Keep this and don't ask again")
        self.keep.setChecked(True)
        self.keep.setToolTip("Later imports of this kind use the file's value (or yours, if you changed it "
                             "or the file has none) without asking.")
        form.addRow(self.keep)
        form.addRow(muted("Settings › Importers changes it later."))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def changed(self) -> bool:
        """Whether the pixel size was changed from the value proposed."""
        return abs(self.pixel.value() - self.start) >= SAME

    def pixels(self) -> list[tuple[float, str]]:
        """(pixel size, where it comes from) per image."""
        if self.changed():
            return [(self.pixel.value(), "entered") for _ in self.from_file]
        return [(v, "file") if v else (self.pixel.value(), "entered") for v in self.from_file]

    def save(self, uid: str) -> None:
        """Keep what was confirmed (and whether to ask again)."""
        prefer_file = not (self.changed() and any(self.from_file))
        defaults.save_pixel(uid, self.pixel.value(), prefer_file)
        defaults.set_asks(uid, not self.keep.isChecked())


def confirm_pixel_size(parent, uid: str, name: str, image_type: str, files: list[str],
                       from_file: list[float | None], hint: str = "", options=(), values: dict | None = None):
    """Settle the pixel size of a group of images: ask (the first time, or if you chose to be asked) or not.

    Returns:
        tuple | None: (pixels, option values, note). pixels: (µm, source) per image, or None when the
            size is not known (note then says so); note: what was used without asking, or "".
            None if the dialog was cancelled.
    """
    values = dict(values or {})
    if defaults.asks(uid):
        dlg = ImportDialog(uid, name, image_type, files, from_file, hint, options, values, parent)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        dlg.save(uid)
        return dlg.pixels(), dlg.values, ""
    pixels = [defaults.resolve_pixel(uid, v) for v in from_file]
    if any(px is None for px, _s in pixels):
        return None, values, f"{name}: the pixel size is not known (set it in Settings › Importers)"
    used = sorted({(round(px, 6), s) for px, s in pixels})
    note = f"{name}: pixel size " + "; ".join(
        f"{px:g} µm ({'read from the file' if s == 'file' else s})" for px, s in used)
    return pixels, values, note


def import_images(parent: QtWidgets.QWidget, files: list[str],
                  library: ImporterLibrary | None = None) -> tuple[list, list[str], list[str]]:
    """Import files (see the module docstring).

    Args:
        parent: For the dialogs.
        files: The chosen files.
        library: The importers (default: all here).

    Returns:
        tuple: (ImageNodes, messages for the files that could not be imported, notes on what was used
            without asking). Cancelling a dialog skips that group of files.
    """
    from modalytics.io.tiff_source import is_slide
    library = library or ImporterLibrary()
    errors: list[str] = []
    notes: list[str] = []

    # 0. Slides: opened in place by Modalytics' own reader
    slides = [p for p in files if is_slide(p)]
    nodes = _import_slides(parent, slides, errors, notes) if slides else []

    # 1. An importer for each other file (ask when two are equally sure)
    groups: dict[str, list[str]] = {}
    asked: dict[tuple, ImporterSpec | None] = {}
    for path in (p for p in files if p not in slides):
        found = library.candidates(path)
        if not found:
            errors.append(not_readable(path, library))
            continue
        tied = tuple(s.uid for s, sc in found if sc >= found[0][1] - 1e-6)
        if len(tied) > 1 and tied not in asked:
            names = [library.get(u).name for u in tied]
            name, ok = QtWidgets.QInputDialog.getItem(
                parent, "Which format?", f"{os.path.basename(path)} could be read by several importers:",
                names, 0, False)
            asked[tied] = library.get(tied[names.index(name)]) if ok else None
        spec = asked.get(tied) if len(tied) > 1 else found[0][0]
        if spec is not None:
            groups.setdefault(spec.uid, []).append(path)

    for uid, paths in groups.items():
        spec = library.get(uid)
        values = defaults.values(spec)

        # 2. Read the files (with your option values; read again below if the dialog changes them)
        read = _read_all(spec, paths, values, errors)
        if not read:
            continue
        from_file = [run.file_pixel_size(img) for _p, images in read for img in images]
        options = list({o.key: o for p, _i in read for o in spec.to_ask(p)}.values())

        # 3. The pixel size (and options): confirmed, or as you chose before
        settled = confirm_pixel_size(parent, uid, spec.name, spec.image_type, [p for p, _i in read], from_file,
                                     spec.hint_for(read[0][0]), options, values)
        if settled is None:
            continue
        pixels, chosen, note = settled
        if pixels is None:
            errors.append(note)
            continue
        if note:
            notes.append(note)
        if options and chosen != values:
            defaults.save(spec, chosen)
            read = _read_all(spec, [p for p, _i in read], chosen, errors)

        # 4. Contract, nodes (5. large images become a pyramid read in place)
        i = 0
        for path, images in read:
            try:
                for node in run.finish(images, spec, path, chosen, pixels[i:i + len(images)]):
                    nodes.append(_as_pyramid(parent, node, path, notes) if _too_large(node) else node)
            except (ImportFailed, InterruptedError, OSError) as exc:
                errors.append(str(exc))
            i += len(images)
    for n in nodes:
        print(f"✅ Imported {n.name} ({n.metadata.get('importer')}, {describe_unit(n.metadata) or 'as read'}, "
              f"{n.pixel_size_um[0]:g} µm/px, {n.metadata.get('pixel_size_source')})")
    for n in notes:
        print(f"ℹ️ {n}")
    return nodes, errors, notes


def _import_slides(parent, paths: list[str], errors: list[str], notes: list[str]) -> list:
    """Open slide files in place, confirming their pixel size like any import."""
    from modalytics.io import importers
    from modalytics.io.tiff_source import TiffPyramidSource
    in_file, ok = [], []
    for p in paths:
        try:
            src = TiffPyramidSource(p)
        except Exception as exc:
            errors.append(f"{os.path.basename(p)}: {exc}")
            continue
        in_file.append(src.pixel_size_um)
        ok.append(p)
        src.close()
    if not ok:
        return []
    settled = confirm_pixel_size(parent, *SLIDE, "wsi", ok, in_file)
    if settled is None:
        return []
    pixels, _values, note = settled
    if pixels is None:
        errors.append(note)
        return []
    if note:
        notes.append(note)
    nodes = []
    for p, (px, source) in zip(ok, pixels):
        node = importers.import_tiff_wsi(p, pixel_size_um=px)
        node.metadata.update(pixel_size_source=source, importer=SLIDE[1])
        nodes.append(node)
    return nodes


def _too_large(node) -> bool:
    from modalytics.io.sources import ArraySource
    from modalytics.io.tiff_source import LARGE_PIXELS
    h, w = node.source.shape
    return isinstance(node.source, ArraySource) and h * w > LARGE_PIXELS


def _as_pyramid(parent, node, path: str, notes: list[str]):
    """Convert a large imported image into a pyramidal OME-TIFF next to its file and read it from there.

    An existing conversion of the same size is reused. Returns the node, now reading the pyramid.

    Raises:
        InterruptedError: If cancelled; OSError: if it cannot be written.
    """
    from modalytics.io.convert import converted_path, write_pyramid
    from modalytics.io.tiff_source import TiffPyramidSource, fingerprint
    out = converted_path(path)
    pixels = node.source.data
    reuse = False
    if os.path.exists(out):
        try:
            reuse = TiffPyramidSource(out).shape == pixels.shape[:2]
        except Exception:
            reuse = False
    if not reuse:
        if not os.access(os.path.dirname(os.path.abspath(out)) or ".", os.W_OK):
            out, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Save the pyramid of a large image", out,
                                                           "Pyramidal OME-TIFF (*.ome.tif)")
            if not out:
                raise InterruptedError(f"{os.path.basename(path)}: not imported (no place for its pyramid)")
        h, w = pixels.shape[:2]
        dlg = QtWidgets.QProgressDialog(f"{os.path.basename(path)} is large ({w:,} × {h:,} px): making a "
                                        f"pyramid next to it,\n{os.path.basename(out)}", "Cancel", 0, 100, parent)
        dlg.setWindowTitle("Import images")
        dlg.setMinimumDuration(300)

        def progress(f: float) -> bool:
            dlg.setValue(int(f * 100))
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()
        progress(0.0)
        try:
            write_pyramid(pixels, out, node.pixel_size_um[0], progress)
        finally:
            dlg.close()
    node.source = TiffPyramidSource(out)
    node.metadata.update(path=os.path.abspath(out), converted_from=os.path.abspath(path),
                         file_fingerprint=fingerprint(out))
    notes.append(f"{node.name}: large image, {'reused' if reuse else 'saved'} as a pyramid ({os.path.basename(out)})")
    return node


def _read_all(spec: ImporterSpec, paths: list[str], values: dict, errors: list[str]) -> list[tuple[str, list]]:
    """Read files with one importer: [(path, Images)]; failures go to ``errors``."""
    out = []
    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
    try:
        for path in paths:
            try:
                out.append((path, run.read(path, spec, values)))
            except ImportFailed as exc:
                errors.append(str(exc))
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    return out
