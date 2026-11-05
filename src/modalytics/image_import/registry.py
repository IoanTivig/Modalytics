"""
The importer library: one script per file format, like parameters.

An importer is one ``.py`` file::

    NAME = "PhaseStudio / Phasics QPI (.txt, .csv)"
    KEY = "phasestudio_qpi"
    IMAGE_TYPE = "phase"             # what it produces (phase, brightfield; see contracts.py)
    EXTENSIONS = [".txt", ".csv"]
    DESCRIPTION = "…"
    REQUIRES = []                    # libraries beyond the included ones (as for parameters)
    OPTIONS = [                      # values asked at import when the file does not say them
        {"key": "pixel_size_um", "label": "Pixel size in the sample", "kind": "float", "default": 1.0},
    ]

    def can_read(path) -> float:     # 0–1: how sure it is that the file is its format
    def needs(path) -> list[str]:    # optional: which OPTIONS to ask for this file (default: all)
    def hint(path) -> str:           # optional: what the file says, shown where the options are asked
    def read(path, options) -> Image | list[Image]

``Image`` (``from modalytics.image_import import Image``) is checked against its
image type's contract and normalised (phase → radians) by
``contracts.check``: importers only read their format.

An importer's *id* is ``<image type>:<key>``. Scripts are read from the
built-in folder (``resources/importers``) and the user library
(``paths.user_dir("importers")``), laid out as ``<image type>/<key>.py``; a
user script with the same id replaces the built-in one. Whole-slide images are
imported by Modalytics' own tiled readers only (locked).

WARNING: importer scripts are Python and run with Modalytics' rights: only add
importers you trust.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from modalytics import paths
from modalytics.core.settings import Setting
from modalytics.image_import.contracts import SCRIPTABLE

TYPE_NAMES = {"phase": "Phase map", "brightfield": "Brightfield", "od": "OD map", "stain": "Stain map",
              "opd": "OPD map", "dry_mass": "Dry-mass map", "height": "Height map", "ri": "Refractive-index map",
              "feature": "Feature map"}


@dataclass
class ImporterSpec:
    """One importer of the library (see the module docstring for the fields)."""

    key: str
    name: str
    image_type: str
    extensions: tuple[str, ...]
    description: str = ""
    options: list[Setting] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    can_read: Callable = field(default=None, repr=False)
    needs: Callable | None = field(default=None, repr=False)
    hint: Callable | None = field(default=None, repr=False)
    read: Callable = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"

    @property
    def uid(self) -> str:
        return f"{self.image_type}:{self.key}"

    def handles(self, path: str) -> bool:
        """Whether the file's extension is one of this importer's."""
        return Path(path).suffix.lower() in self.extensions

    def score(self, path: str) -> float:
        """How sure this importer is that it reads the file (0 if not its extension or on any error)."""
        if not self.handles(path):
            return 0.0
        try:
            return max(0.0, min(1.0, float(self.can_read(path))))
        except Exception:
            return 0.0

    def to_ask(self, path: str) -> list[Setting]:
        """The options to ask for this file (all of them unless the importer says otherwise)."""
        if self.needs is None:
            return list(self.options)
        try:
            keys = list(self.needs(path))
        except Exception:
            keys = [o.key for o in self.options]
        return [o for o in self.options if o.key in keys]

    def hint_for(self, path: str) -> str:
        """What the importer says about this file where its options are asked ("" if nothing or on error)."""
        try:
            return str(self.hint(path)) if self.hint is not None else ""
        except Exception:
            return ""

    def resolve(self, chosen: dict | None = None) -> dict:
        """Option values: defaults, overridden (and cleaned) by ``chosen``."""
        chosen = chosen or {}
        return {o.key: (o.clean(chosen[o.key]) if o.key in chosen else o.default) for o in self.options}


def load_script(path: str | os.PathLike, source: str = "user") -> ImporterSpec:
    """Import one importer script and describe it.

    Raises:
        ValueError: If the script is not a usable importer (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()
    path = Path(path)
    name = f"_modalytics_importer_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ValueError(f"it imports '{exc.name}', which is not installed: install it with Libraries… "
                         f"(Settings › Importers)") from exc
    except Exception as exc:
        raise ValueError(f"the script fails to load: {exc}") from exc
    key = getattr(module, "KEY", None)
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        raise ValueError("KEY must be a name of letters, digits and _")
    for fn in ("can_read", "read"):
        if not callable(getattr(module, fn, None)):
            raise ValueError(f"the script must define {fn}(…)")
    image_type = str(getattr(module, "IMAGE_TYPE", ""))
    if image_type not in SCRIPTABLE:
        raise ValueError(f"IMAGE_TYPE must be one of {', '.join(SCRIPTABLE)} (whole-slide images are read by "
                         f"Modalytics itself)")
    exts = getattr(module, "EXTENSIONS", ())
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in ([exts] if isinstance(exts, str) else exts))
    if not exts:
        raise ValueError("EXTENSIONS must list at least one file extension, e.g. ['.txt']")
    try:
        options = [o if isinstance(o, Setting) else Setting.from_dict(dict(o)) for o in getattr(module, "OPTIONS", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPTIONS: {exc}") from exc
    return ImporterSpec(key=key, name=str(getattr(module, "NAME", key)), image_type=image_type, extensions=exts,
                        description=str(getattr(module, "DESCRIPTION", "")), options=options,
                        requires=tuple(getattr(module, "REQUIRES", ()) or ()), can_read=module.can_read,
                        needs=getattr(module, "needs", None) if callable(getattr(module, "needs", None)) else None,
                        hint=getattr(module, "hint", None) if callable(getattr(module, "hint", None)) else None,
                        read=module.read, path=str(path), source=source)


class ImporterLibrary:
    """Every importer available here (built-in + user), loaded on creation.

    Attributes:
        specs: {importer id: spec}, sorted by (image type, name).
        errors: {script path: why it was not loaded}.
        overridden: Ids of built-ins replaced by a user script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin = Path(builtin) if builtin is not None else paths.builtin_dir("importers")
        self.user = Path(user) if user is not None else paths.user_dir("importers")
        self.specs: dict[str, ImporterSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        self.reload()

    def reload(self) -> None:
        found: dict[str, ImporterSpec] = {}
        self.errors, self.overridden = {}, []
        for source, folder in (("builtin", self.builtin), ("user", self.user)):
            for path in sorted(folder.rglob("*.py")) if folder.is_dir() else []:
                if path.name.startswith(("_", "EXAMPLE")) or "__pycache__" in path.parts:
                    continue
                try:
                    spec = load_script(path, source)
                except ValueError as exc:
                    self.errors[str(path)] = str(exc)
                    continue
                if spec.uid in found:
                    if found[spec.uid].source == source:
                        self.errors[str(path)] = f"duplicate id {spec.uid} (also {found[spec.uid].path})"
                        continue
                    self.overridden.append(spec.uid)
                found[spec.uid] = spec
        self.specs = dict(sorted(found.items(), key=lambda kv: (kv[1].image_type, kv[1].name.lower())))

    def get(self, uid: str) -> ImporterSpec | None:
        return self.specs.get(uid)

    def extensions(self) -> list[str]:
        """Every extension some importer reads (for the Open dialog's filter)."""
        return sorted({e for s in self.specs.values() for e in s.extensions})

    def candidates(self, path: str) -> list[tuple[ImporterSpec, float]]:
        """Importers that think they can read the file, most confident first."""
        scored = [(s, s.score(path)) for s in self.specs.values()]
        return sorted([(s, sc) for s, sc in scored if sc > 0], key=lambda t: -t[1])

    def user_files(self) -> dict[str, Path]:
        return {uid: Path(s.path) for uid, s in self.specs.items() if s.source == "user"}

    def missing_libraries(self) -> dict[str, list[str]]:
        """{importer id or script path: libraries it declares but are not available here}."""
        from modalytics.parameters import libraries
        from modalytics.parameters.registry import _declared_requires
        out = {uid: libraries.missing(s.requires) for uid, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}


# The Code tab's helper list for importers: (group, label, text to insert, help)
HELP = [
    ("Result", "Image(pixels, pixel_size_um, unit=…)", "Image(values, pixel_size_um, unit='rad')",
     "What read() returns (one Image, or a list for a file holding several). pixels: 2D array (phase) or "
     "H×W×3 (colour). pixel_size_um: size of one pixel in the sample. unit: phase unit as read "
     "(rad, mrad, deg, wv); Modalytics converts to radians."),
    ("Result", "Image(…, height_factor_nm=…)", "height_factor_nm=",
     "Phase only: optical height in nm per unit of the values as read (converted with the values)."),
    ("Result", "Image(…, name=…)", "name=", "Display name (default: the file name)."),
    ("Result", "Image(…, metadata={…})", "metadata={}",
     "Anything worth keeping: instrument, wavelength, settings… (shown in Properties)."),
    ("Result", "PHASE_UNITS", "PHASE_UNITS",
     "The phase units the contract converts to radians, with their factors.\nNeeds: from modalytics.image_import "
     "import PHASE_UNITS"),
    ("Options", 'options["key"]', 'options[""]',
     "The value of an option (declared on the Options tab); asked at import when needs() lists it."),
    ("Reading text", "np.loadtxt(f, delimiter=…)", "np.loadtxt(f, delimiter=None, dtype=np.float32)",
     "A table of numbers (skip the header first, or pass skiprows=).\nNeeds: import numpy as np"),
    ("Reading text", "open(path, encoding=…)", 'open(path, "r", encoding="utf-8", errors="replace")',
     "Open a text file (read the header line by line, then the values)."),
    ("Reading images", "tifffile.imread(path)", "tifffile.imread(path)",
     "A TIFF image as an array (any bit depth).\nNeeds: import tifffile"),
    ("Reading images", "tifffile.TiffFile(path).pages[0].tags", "tifffile.TiffFile(path).pages[0].tags",
     "TIFF tags (resolution, descriptions, vendor metadata).\nNeeds: import tifffile"),
    ("Reading images", "cv2.imread(path, cv2.IMREAD_UNCHANGED)", "cv2.imread(path, cv2.IMREAD_UNCHANGED)",
     "PNG, JPEG, BMP, TIFF… as stored (cv2 gives BGR: reverse with [..., ::-1]).\nNeeds: import cv2"),
    ("Reading images", "np.load(path)", "np.load(path)", "A .npy array.\nNeeds: import numpy as np"),
    ("Reading images", "h5py.File(path, 'r')", "h5py.File(path, 'r')",
     "HDF5 files (install h5py in Libraries first).\nNeeds: import h5py"),
]
