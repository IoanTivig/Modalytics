"""
Importing one file: choose the importer, read, settle the pixel size, check the contract.

Two steps, so the UI can ask in between:

    read(path, spec, options)        → the importer's Images, as the file gives them
    finish(images, spec, path, …)    → ImageNodes, with the pixel size settled and the
                                       image type's contract checked (phase → radians)

The UI (``ui.image_import``) shows the pixel size read from the file (or asks
for it when the file does not give it) between the two. Scripts and the command
line call ``import_file``, which does both and settles the pixel size without
asking: a value passed in, else what you confirmed earlier (``defaults``), else
the file's.
"""

from __future__ import annotations

import os

from modalytics.image_import import defaults
from modalytics.image_import.contracts import ContractError, Image, check, to_node
from modalytics.image_import.registry import ImporterLibrary, ImporterSpec


class ImportFailed(ValueError):
    """A file could not be imported (the message says why, for the user)."""


def not_readable(path: str, library: ImporterLibrary) -> str:
    """Why no importer takes a file, for the user (which importers know its extension, if any)."""
    name, ext = os.path.basename(path), os.path.splitext(path)[1].lower()
    knowing = [s.name for s in library.specs.values() if s.handles(path)]
    if knowing:
        return (f"{name}: not recognised by the importers for {ext} files ({'; '.join(knowing)}). "
                f"Settings › Importers can add one for its format.")
    return f"{name}: no importer reads {ext or 'files without an extension'} files. Settings › Importers can add one."


def choose(path: str, library: ImporterLibrary) -> ImporterSpec:
    """The importer for a file (the most confident one).

    Raises:
        ImportFailed: If no importer reads it.
    """
    found = library.candidates(path)
    if not found:
        raise ImportFailed(not_readable(path, library))
    return found[0][0]


def file_pixel_size(image: Image) -> float | None:
    """The pixel size an image's file gives (x, if it gives x and y), or None."""
    px = image.pixel_size_um
    return None if px is None else float(px[0] if isinstance(px, (tuple, list)) else px)


def read(path: str, spec: ImporterSpec, values: dict) -> list[Image]:
    """Run the importer on a file.

    Returns:
        list[Image]: As the importer returned them (pixel size as the file gives it, possibly None).

    Raises:
        ImportFailed: If the importer fails or returns something that is not an Image.
    """
    try:
        result = spec.read(path, values)
    except ImportFailed:
        raise
    except Exception as exc:
        raise ImportFailed(f"{spec.name} could not read {os.path.basename(path)}: {exc}") from exc
    images = list(result) if isinstance(result, (list, tuple)) else [result]
    for img in images:
        if not isinstance(img, Image):
            raise ImportFailed(f"{spec.name} returned {type(img).__name__}, not an Image")
    if not images:
        raise ImportFailed(f"{spec.name} found no image in {os.path.basename(path)}")
    return images


def finish(images: list[Image], spec: ImporterSpec, path: str, values: dict,
           pixels: list[tuple[float, str]]) -> list:
    """Check the images against their contract and make ImageNodes.

    Args:
        images: From ``read``.
        spec: The importer.
        path: The file.
        values: The option values used.
        pixels: (pixel size µm, where it comes from) per image; "file" keeps the file's own value.

    Raises:
        ImportFailed: If an image does not meet its contract.
    """
    nodes = []
    for i, (img, (px, source)) in enumerate(zip(images, pixels)):
        override = None if source == "file" and img.pixel_size_um is not None else px
        try:
            ok = check(img, spec.image_type, override, source)
        except ContractError as exc:
            raise ImportFailed(f"{spec.name}, {os.path.basename(path)}: {exc}") from exc
        if len(images) > 1 and not ok.name:
            ok.name = f"{os.path.splitext(os.path.basename(path))[0]} ({i + 1})"
        ok.metadata["importer_options"] = values
        nodes.append(to_node(ok, spec.image_type, path, spec.name))
    return nodes


def import_file(path: str, library: ImporterLibrary | None = None, importer: ImporterSpec | None = None,
                options: dict | None = None, pixel_size_um: float | None = None) -> list:
    """Import a file into ImageNodes without asking.

    Args:
        path: The file.
        library: Importers to choose from (default: all here).
        importer: Use this one instead of choosing.
        options: Option values (missing ones: your saved values, else the importer's defaults).
        pixel_size_um: Use this pixel size (default: as you confirmed earlier, else the file's).

    Returns:
        list[ImageNode]: One per image in the file.

    Raises:
        ImportFailed: With a message for the user (also when the pixel size is not known).
    """
    library = library or ImporterLibrary()
    spec = importer or choose(path, library)
    values = spec.resolve({**defaults.values(spec), **(options or {})})
    images = read(path, spec, values)
    pixels = []
    for img in images:
        px, source = (pixel_size_um, "given") if pixel_size_um else \
            defaults.resolve_pixel(spec.uid, file_pixel_size(img))
        if px is None:
            raise ImportFailed(f"{os.path.basename(path)}: the file does not give the pixel size in the sample. "
                               f"Import it once with File › Import › Images, or set it in Settings › Importers.")
        pixels.append((px, source))
    return finish(images, spec, path, values, pixels)
