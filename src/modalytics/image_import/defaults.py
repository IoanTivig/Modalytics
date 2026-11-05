"""
Your choices per importer, on this computer (Qt-free): the pixel size and option values.

The first import with an importer always asks for the pixel size in the sample
(prefilled with the value read from the file, when the file gives one) and for
the importer's options. What you confirm is kept here; with "don't ask again"
later imports use it without asking, until you change it in Settings ›
Importers or switch asking back on there:

    importer_pixel     {importer id: {"pixel_size_um": µm, "prefer_file": bool}}
                       pixel_size_um: for files that do not give it (and the
                       override when prefer_file is False); prefer_file: use
                       the file's value when it has one (False after you
                       changed a value read from a file)
    importer_options   {importer id: {option key: value}}
    importer_silent    [importer ids that do not ask any more]

Importer ids are those of the importer library ("phase:phasestudio_qpi");
Modalytics' own readers use their own ("wsi:tiff", slide files read in place).
"""

from __future__ import annotations

from modalytics import app_settings
from modalytics.image_import.registry import ImporterSpec


# ---------------------------------------------------------------- pixel size
def pixel(uid: str) -> dict | None:
    """Your pixel-size choice for an importer: {"pixel_size_um", "prefer_file"} (None if never confirmed)."""
    p = (app_settings.get("importer_pixel") or {}).get(uid)
    return None if p is None else {"pixel_size_um": p.get("pixel_size_um"), "prefer_file": p.get("prefer_file", True)}


def save_pixel(uid: str, pixel_size_um: float | None = None, prefer_file: bool | None = None) -> None:
    """Keep your pixel size (and whether a file's own value is preferred); None leaves a field unchanged."""
    s = app_settings.load()
    allp = dict(s.get("importer_pixel") or {})
    p = dict(allp.get(uid) or {"pixel_size_um": None, "prefer_file": True})
    if pixel_size_um is not None:
        p["pixel_size_um"] = float(pixel_size_um)
    if prefer_file is not None:
        p["prefer_file"] = bool(prefer_file)
    allp[uid] = p
    s["importer_pixel"] = allp
    app_settings.save(s)


def resolve_pixel(uid: str, from_file: float | None) -> tuple[float | None, str]:
    """The pixel size to use without asking, and where it comes from.

    Args:
        uid: The importer id.
        from_file: The value the file gives (None if it gives none).

    Returns:
        tuple: (µm or None if unknown, "file" / "your value" / "your default" / "").
    """
    p = pixel(uid) or {}
    mine = p.get("pixel_size_um")
    if mine and not p.get("prefer_file", True):
        return mine, "your value"
    if from_file:
        return float(from_file), "file"
    if mine:
        return mine, "your default"
    return None, ""


# ---------------------------------------------------------------- asking
def asks(uid: str) -> bool:
    """Whether import asks (always the first time, then as you chose)."""
    return pixel(uid) is None or uid not in (app_settings.get("importer_silent") or [])


def set_asks(uid: str, ask: bool) -> None:
    s = app_settings.load()
    silent = [u for u in (s.get("importer_silent") or []) if u != uid]
    s["importer_silent"] = silent if ask else silent + [uid]
    app_settings.save(s)


# ---------------------------------------------------------------- options
def saved(spec: ImporterSpec) -> dict | None:
    """Your saved values for this importer's options (None if never saved)."""
    stored = (app_settings.get("importer_options") or {}).get(spec.uid)
    if stored is None:
        return None
    return {k: v for k, v in stored.items() if any(o.key == k for o in spec.options)}


def values(spec: ImporterSpec) -> dict:
    """The option values to use or propose: yours where saved, else the importer's defaults."""
    return spec.resolve(saved(spec) or {})


def save(spec: ImporterSpec, chosen: dict) -> None:
    """Keep your option values ({option key: value}, cleaned against the options)."""
    s = app_settings.load()
    options = dict(s.get("importer_options") or {})
    options[spec.uid] = spec.resolve({**(saved(spec) or {}), **chosen})
    s["importer_options"] = options
    app_settings.save(s)


def describe(spec: ImporterSpec, chosen: dict) -> str:
    """Option values in words, e.g. "Detrend degree = 2"."""
    def shown(o):
        v = chosen.get(o.key, o.default)
        return f"{v:g}{o.suffix}" if isinstance(v, float) else f"{v}{o.suffix}"
    return ", ".join(f"{o.label} = {shown(o)}" for o in spec.options if o.key in chosen)
