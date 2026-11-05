"""
Relink: point an image whose file was not found at the file again (Properties › Relink…).

A project keeps an image whose file is missing as a placeholder with its size,
placement and masks (``io.sources.MissingSource``). Relink asks where the file
is now (a slide file, or a zarr folder for HoloPath's conversions), checks it
is the same image (same size in pixels; for slide files, the fingerprint kept
at import) and asks before using a file that differs. It is one undo step;
the masks are not touched.
"""

from __future__ import annotations

import os

from PySide6 import QtWidgets

from modalytics.core.commands import Command


def relink(parent: QtWidgets.QWidget, project, undo, image) -> bool:
    """Ask for the image's file and use it. Returns True if relinked."""
    from modalytics.io.tiff_source import SLIDE_EXTENSIONS, TiffPyramidSource, fingerprint
    from modalytics.io.zarr_source import ZarrPyramidSource
    old_source, old_meta = image.source, dict(image.metadata)
    old_path = getattr(old_source, "path", None) or ""
    start = os.path.dirname(old_path) if os.path.isdir(os.path.dirname(old_path) or "\0") else os.path.expanduser("~")

    # 1. Where is it now?
    if getattr(old_source, "kind", "") == "zarr":
        path = QtWidgets.QFileDialog.getExistingDirectory(parent, f"Where is {os.path.basename(old_path)}?", start)
    else:
        exts = " ".join(f"*{e}" for e in SLIDE_EXTENSIONS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent, f"Where is {os.path.basename(old_path)}?", start, f"Slide files ({exts});;All files (*)")
    if not path:
        return False
    try:
        new = ZarrPyramidSource(path) if os.path.isdir(path) else TiffPyramidSource(path)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(parent, "Relink", f"{path}\n\n{exc}")
        return False

    # 2. The same image? (size, and the fingerprint kept at import)
    doubts = []
    if tuple(new.shape) != tuple(old_source.shape):
        doubts.append(f"it is {new.shape[1]:,} × {new.shape[0]:,} px, the image was "
                      f"{old_source.shape[1]:,} × {old_source.shape[0]:,} px")
    known = old_meta.get("file_fingerprint")
    if known and isinstance(new, TiffPyramidSource) and fingerprint(path) != known:
        doubts.append("its contents differ from the file imported")
    if doubts and QtWidgets.QMessageBox.question(
            parent, "Relink", f"{os.path.basename(path)} may not be the same image: {'; '.join(doubts)}.\n\n"
            f"The masks may not fit it. Use it anyway?") != QtWidgets.QMessageBox.StandardButton.Yes:
        return False

    # 3. Use it (one undo step; placement and masks unchanged)
    meta = dict(old_meta, path=os.path.abspath(path))
    if isinstance(new, TiffPyramidSource):
        meta["file_fingerprint"] = fingerprint(path)
    undo.push(Command(f"Relink {image.name}", lambda: project.set_source(image, new, meta),
                      lambda: project.set_source(image, old_source, old_meta)))
    return True
