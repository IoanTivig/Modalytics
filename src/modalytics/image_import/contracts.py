"""
What an imported image must be, per image type: the *contract*.

Importers (one script per file format, ``image_import.registry``) only read
their format and hand back an ``Image``. This module checks it against the
contract of its image type and turns it into the ``ImageNode`` the rest of
Modalytics works with, so everything downstream (display, registration,
parameters) can rely on one form per type:

    phase        2D float32 values in **radians**. The importer says which unit it
                 read (``unit``: rad, mrad, deg, wv/waves); values are converted here,
                 and an optional phase-to-height factor (nm per *that* unit) with them,
                 so the optical height (phase × factor) stays the same.
    brightfield  2D grey or RGB image, 8- or 16-bit (an alpha channel is dropped).
    od, stain    2D float optical density, ≥ 0 ("OD").
    opd          2D float optical path difference in **nm** (read in nm or µm).
    dry_mass     2D float dry-mass surface density in **pg/µm²**.
    height       2D float thickness in **µm** (read in µm or nm).
    ri           2D float refractive index (dimensionless).
    feature      2D float feature map (dimensionless).
    wsi          not importable by scripts: whole-slide pyramids need tiled readers,
                 which ship with Modalytics (locked).

The types other than phase and brightfield are usually made by processing (a view
of an imported image); an importer can bring one in directly when another program
already computed it. Such an image has no original under it: parameters of its own
type measure it, those needing the original (e.g. phase disorder) do not apply.

The pixel size in the sample is part of every contract. An importer gives the
value its file states, or None when the file does not state it; the import
step (``run``, the File › Import dialog) then fills it from what you confirmed
(``defaults``). The original unit, the conversion and where the pixel size came
from are recorded in the image's metadata (``unit_read``, ``unit_factor``,
``pixel_size_source``), so an imported image says where it came from.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

# Phase units an importer may read → factor to radians
PHASE_UNITS = {"rad": 1.0, "mrad": 1e-3, "deg": math.pi / 180.0, "wv": 2.0 * math.pi, "waves": 2.0 * math.pi,
               "wave": 2.0 * math.pi, "λ": 2.0 * math.pi}
SCRIPTABLE = ("phase", "brightfield", "od", "stain", "opd", "dry_mass", "height", "ri", "feature")
# Single-value maps an importer may produce: units it may read → factor to the standard unit (the first)
MAP_UNITS = {
    "od": {"OD": 1.0, "": 1.0},
    "stain": {"OD": 1.0, "": 1.0},
    "opd": {"nm": 1.0, "µm": 1000.0, "um": 1000.0},
    "dry_mass": {"pg/µm²": 1.0, "pg/um2": 1.0, "pg/um²": 1.0},
    "height": {"µm": 1.0, "um": 1.0, "nm": 1e-3},
    "ri": {"": 1.0, "RIU": 1.0},
    "feature": {"": 1.0},
}


class ContractError(ValueError):
    """The importer's result does not meet its image type's contract (the message says what is wrong)."""


@dataclass
class Image:
    """What an importer hands back for one image.

    Attributes:
        pixels: The values: 2D (H, W), or (H, W, 3/4) for colour.
        pixel_size_um: Size of one pixel in the sample, µm (one number, or (x, y)), as the
            file states it; None when the file does not say (it is then asked at import).
        unit: Unit of the values as read (phase: "rad", "mrad", "deg", "wv").
        name: Display name (default: the file name).
        metadata: Anything worth keeping (instrument, wavelength, settings…).
        height_factor_nm: Phase only: nm of optical height per unit of the values as read.
    """

    pixels: np.ndarray
    pixel_size_um: float | tuple[float, float] | None
    unit: str = ""
    name: str = ""
    metadata: dict = field(default_factory=dict)
    height_factor_nm: float | None = None


def check(image: Image, image_type: str, pixel_size_um: float | None = None, source: str = "file") -> Image:
    """Check an importer's result against its contract and normalise it (phase → radians).

    Args:
        image: What the importer returned.
        image_type: The importer's image type.
        pixel_size_um: The pixel size to use instead of the image's own (confirmed or entered at import).
        source: Where the pixel size comes from ("file", "entered", "your default"…), kept in the metadata.

    Returns:
        Image: The normalised image (a new one; ``image`` is not changed).

    Raises:
        ContractError: With what is wrong.
    """
    if image_type not in SCRIPTABLE:
        raise ContractError(f"{image_type} images are imported by Modalytics' own readers only")
    px = np.asarray(image.pixels)
    if px.ndim not in (2, 3) or px.shape[0] < 2 or px.shape[1] < 2:
        raise ContractError(f"pixels must be a 2D image (got shape {px.shape})")
    size = pixel_size_um if pixel_size_um is not None else image.pixel_size_um
    if size is None:
        raise ContractError("the pixel size in the sample is not known (the file does not give it)")
    sx, sy = (size, size) if np.isscalar(size) else tuple(size)
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        raise ContractError(f"the pixel size must be a positive number of µm (got {size!r})")
    meta = dict(image.metadata, pixel_size_source=source)
    if image.pixel_size_um is not None and pixel_size_um is not None:
        meta["pixel_size_in_file"] = image.pixel_size_um

    if image_type == "phase":
        # 1. One channel of numbers
        if px.ndim != 2:
            raise ContractError(f"a phase map has one value per pixel (got shape {px.shape})")
        if not np.issubdtype(px.dtype, np.number):
            raise ContractError(f"phase values must be numbers (got {px.dtype})")
        # 2. Radians
        unit = (image.unit or "").strip()
        if unit.lower() not in {u.lower() for u in PHASE_UNITS}:
            raise ContractError(f"unknown phase unit '{unit}': the importer must say which of "
                                f"{', '.join(sorted(PHASE_UNITS))} its values are in")
        factor = next(f for u, f in PHASE_UNITS.items() if u.lower() == unit.lower())
        px = (px.astype(np.float64) * factor).astype(np.float32)
        height = image.height_factor_nm / factor if image.height_factor_nm else None
        meta.update(unit="rad", unit_read=unit, unit_factor=factor)
        if height is not None:
            meta["phase_to_height_nm"] = height
        return Image(px, (sx, sy), "rad", image.name, meta, height)

    if image_type in MAP_UNITS:
        # A single-value map: one number per pixel, converted to the type's standard unit, within its range
        if px.ndim != 2:
            raise ContractError(f"a {image_type} map has one value per pixel (got shape {px.shape})")
        if not np.issubdtype(px.dtype, np.number):
            raise ContractError(f"{image_type} values must be numbers (got {px.dtype})")
        units = MAP_UNITS[image_type]
        standard = next(iter(units))
        unit = (image.unit or standard).strip()
        if unit not in units:
            raise ContractError(f"unknown {image_type} unit '{unit}': the importer must say which of "
                                f"{', '.join(repr(u) for u in units)} its values are in")
        factor = units[unit]
        px = (px.astype(np.float64) * factor).astype(np.float32)
        from modalytics.processing.types import MADE
        low = MADE[image_type][1]
        if low is not None and np.nanmin(px) < low - 1e-6:
            raise ContractError(f"{image_type} values must be ≥ {low} (got {float(np.nanmin(px)):.4g})")
        meta.update(unit=standard, unit_read=unit, unit_factor=factor)
        return Image(px, (sx, sy), standard, image.name, meta)

    # brightfield: grey or RGB, 8 / 16 bit (floats 0–1 are scaled to 16 bit)
    if px.ndim == 3 and px.shape[2] not in (3, 4):
        raise ContractError(f"a colour image has 3 (RGB) or 4 (RGBA) channels (got {px.shape[2]})")
    if px.ndim == 3 and px.shape[2] == 4:
        px = px[..., :3]
    if np.issubdtype(px.dtype, np.floating):
        if np.nanmin(px) < 0 or np.nanmax(px) > 1:
            raise ContractError("floating-point brightfield values must lie in 0–1 (or use 8/16-bit integers)")
        px = (np.nan_to_num(px) * 65535).round().astype(np.uint16)
    elif px.dtype not in (np.uint8, np.uint16):
        raise ContractError(f"brightfield pixels must be 8- or 16-bit (got {px.dtype})")
    meta.setdefault("unit", image.unit or "grey level")
    return Image(np.ascontiguousarray(px), (sx, sy), meta["unit"], image.name, meta)


def describe_unit(meta: dict) -> str:
    """How an image's values were converted, e.g. "rad, converted from wv × 6.283" (from its metadata)."""
    unit, read, factor = meta.get("unit", ""), meta.get("unit_read"), meta.get("unit_factor")
    if not read or read == unit:
        return unit
    return f"{unit}, converted from {read} × {factor:.4g}"


def describe_source(meta: dict) -> str:
    """One line on where an image came from: its values' unit, the pixel size's origin, the importer."""
    parts = []
    unit = describe_unit(meta)
    if unit:
        parts.append(f"Values in {unit}")
    source = meta.get("pixel_size_source")
    if source:
        said = {"file": "from the file", "entered": "entered at import", "given": "given at import"}.get(
            source, f"({source})")
        parts.append(f"{'p' if parts else 'P'}ixel size {said}")
    if meta.get("importer"):
        parts.append(f"read by {meta['importer']}" if parts else f"Read by {meta['importer']}")
    return " · ".join(parts)


def to_node(image: Image, image_type: str, path: str, importer: str):
    """The ImageNode for a normalised image (placed at the origin, as every new image).

    Args:
        image: From ``check``.
        image_type: Its image type.
        path: The file it came from.
        importer: The importer's name (kept in the metadata).
    """
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    meta = dict(image.metadata, path=os.path.abspath(path), importer=importer)
    name = image.name or os.path.splitext(os.path.basename(path))[0]
    node = ImageNode(name, ArraySource(image.pixels), image.pixel_size_um, image_type, metadata=meta)
    if image_type in MAP_UNITS:                   # drawn like the same type made by processing
        from modalytics.core.model import Display
        from modalytics.processing.types import MADE
        _unit, _low, colormap, clim = MADE[image_type]
        node.display = Display(clim=clim or node.source.value_range(), colormap=colormap)
    return node
