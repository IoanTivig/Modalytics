"""
General settings of the program (Settings › General), kept in the user
library as ``settings.json``.

Only per-computer preferences live here: which GPU draws and which runs the
models, the theme, recent projects. Everything about an *analysis* belongs to
its profile, so that sharing a profile shares the analysis.

Environment variables (``MODALYTICS_GPU``, ``MODALYTICS_INFERENCE``) still win
over these settings, for scripts and troubleshooting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modalytics import fileio, paths

DEFAULTS = {
    "render_gpu": "",          # "" = automatic; else a substring of the adapter name, or "software"
    "inference_device": "",    # "" = automatic; else a device key ("dml:1", "cuda:0", "cpu")
    "theme": "dark",
    "recent_projects": [],     # most recent first
    "importer_pixel": {},      # {importer id: {"pixel_size_um", "prefer_file"}} confirmed at the first import
    "importer_options": {},    # {importer id: your default option values} (Settings › Importers changes them)
    "importer_silent": [],     # importer ids whose defaults are used without asking at import
    "diagnostics": False,      # write the diagnostic log (Settings › General; ``diagnostics``)
    "script_options": {},      # {external script id: the option values last used}
    "arrange": {}, "registration_options": {},             # how imported images were last arranged (ui.arrange_dialog)
    "classify_segmenting": True,   # Annotate › Classify when segmenting: suggestions graded by the profile's classifier
}
MAX_RECENT = 10


def _file() -> Path:
    return paths.user_home() / "settings.json"


def load() -> dict:
    """All settings (defaults for anything missing or unreadable)."""
    out = dict(DEFAULTS)
    try:
        stored = json.loads(_file().read_text(encoding="utf-8"))
        out.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return out


def save(settings: dict) -> None:
    """Write settings atomically (unknown keys are dropped)."""
    data = {k: settings.get(k, v) for k, v in DEFAULTS.items()}
    tmp = _file().with_name("settings.json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    fileio.replace(tmp, _file())


def get(key: str):
    return load()[key]


def put(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)


def add_recent(path: str) -> list[str]:
    """Put a project at the top of the recent list (kept to ``MAX_RECENT`` existing files)."""
    path = os.path.abspath(path)
    recent = [path] + [p for p in load()["recent_projects"] if os.path.normcase(p) != os.path.normcase(path)]
    recent = recent[:MAX_RECENT]
    put("recent_projects", recent)
    return recent


def recent_projects() -> list[str]:
    """Recent projects that still exist, most recent first."""
    return [p for p in load()["recent_projects"] if os.path.exists(p)]


def apply_to_environment() -> None:
    """Make the saved GPU choices effective for ``render.gpu`` / ``services.compute``
    (unless the environment already sets them). Call before the GPU is set up."""
    s = load()
    if s["render_gpu"] and not os.environ.get("MODALYTICS_GPU"):
        os.environ["MODALYTICS_GPU"] = s["render_gpu"]
    if s["inference_device"] and not os.environ.get("MODALYTICS_INFERENCE"):
        os.environ["MODALYTICS_INFERENCE"] = s["inference_device"]
