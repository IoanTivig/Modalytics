"""
The parameter library: finding, loading and describing parameter scripts.

A parameter is one ``.py`` file: module-level metadata plus ``compute(ctx)``
returning one number. The format is HoloPath's, so HoloPath parameter scripts
work unchanged; Modalytics adds three optional fields::

    KEY = "disorder_strength"        # unique within its scope
    LABEL = "Disorder strength"
    UNIT = "µm"
    CATEGORY = "disorder"            # grouping in Settings › Parameters
    SCOPE = "object"                 # object | class | image
    DESCRIPTION = "…"                # what it measures, its formula, how to read it
    REFERENCES = ["…"]               # optional: the publications that define it (one per entry)
    DEFAULT_ON = True                # selected in a new profile
    WHOLE_OBJECT = False             # object scope: it needs the whole object (shape, totals, what follows the
                                     #      outline): left empty on objects cut by the image edge or marked
                                     #      partial; False: the visible part is measured (pixel statistics,
                                     #      texture). Unset: by its category (shape, relations) and key (totals)
    ORDER = 210                      # column order
    IMAGE_TYPE = "phase"             # new: the one image type it measures: "phase", "wsi",
                                     #      "brightfield", or "*" (any image: outline-only measures);
                                     #      default "phase", as HoloPath measured phase maps
    REQUIRES = ["PyWavelets"]        # new: libraries it imports beyond the included ones
                                     #      (Settings › Parameters › Libraries installs them)
    CONSTANTS = [{"key": "degree", "label": "Detrend degree", "kind": "int",
                  "default": 2, "minimum": 0, "maximum": 4}]  # new: values a profile may change
                                     #      (core.settings.Setting; older name: SETTINGS)
    def compute(ctx):                # ctx: parameters.context.ObjectContext / ImageContext
        return ctx.correlation_length_for(ctx.constants["degree"])

A parameter's *id* is ``<scope>:<image type>:<key>`` (e.g. ``object:any:area``,
``object:phase:mean``): the same key may exist once per scope and image type
(a phase map's mean and a brightfield image's mean are different measures,
with different units). ``any`` stands for ``"*"``. Ids from before image types
(``object:mean``) are still understood (``find``).

Scripts are read from the built-in folder (``resources/parameters``) and the
user library (``paths.user_dir("parameters")``), laid out as
``<scope>/<image type>/<key>.py`` (any layout loads; this one is what the
editor writes). A user script with the same id replaces the built-in one.
Files starting with ``_`` or ``EXAMPLE`` are skipped.

Older scripts may list ``IMAGE_TYPES = [...]``: one entry is that type, ``["*"]``
is any image, several load as the first one with a note asking to choose.

WARNING: parameter scripts are Python and run with Modalytics' rights: only
add scripts you trust (the same holds for HoloPath).
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

SCOPES = {"object": "Per object", "class": "Per class in an image", "image": "Whole image"}
ANY = "*"
# Image types a parameter can be for, in display order, with the folder / id word of each
TYPE_NAMES = {"*": "Any image", "phase": "Phase map", "wsi": "Whole-slide image", "brightfield": "Brightfield",
              "od": "OD map", "opd": "OPD map", "dry_mass": "Dry-mass map", "height": "Height map",
              "ri": "Refractive-index map", "stain": "Stain map", "feature": "Feature map"}
TYPE_WORD = {"*": "any", "phase": "phase", "wsi": "wsi", "brightfield": "brightfield", "od": "od", "opd": "opd",
             "dry_mass": "dry_mass", "height": "height", "ri": "ri", "stain": "stain", "feature": "feature"}


def type_word(image_type: str) -> str:
    """The word for an image type in ids and folders ("*" → "any")."""
    return TYPE_WORD.get(image_type, image_type)


# Object parameters that need the whole object when a script does not say (``WHOLE_OBJECT``): its shape, what
# follows its outline, and totals over it; pixel statistics and texture are measured on the visible part.
WHOLE_CATEGORIES = ("shape", "relations")
WHOLE_KEYS = ("phase_volume", "dry_mass", "integrated_od", "optical_volume", "volume", "integrated",
              "band_anisotropy", "band_scattering_coefficient")


def whole_object_default(category: str, key: str) -> bool:
    """Whether an object parameter of this category and key needs the whole object (``WHOLE_OBJECT`` unset)."""
    return category in WHOLE_CATEGORIES or key in WHOLE_KEYS


@dataclass
class ParameterSpec:
    """One parameter of the library.

    Attributes:
        key, label, unit, category, scope, description, default_on, order: The
            script's metadata (see the module docstring).
        image_type: The image type it measures ("*" = any image).
        note: Something to fix in the script (e.g. it listed several image types).
        requires: Libraries it needs beyond those included with Modalytics.
        constants: Its declared constants (values a profile may change).
        references: Publications that define or use it (one citation each).
        compute: ``compute(ctx) -> float``.
        path: The script file.
        source: "builtin" or "user".
        whole_object: (object scope) It needs the whole object — shape, totals, what follows the outline — so it
            is left empty on an object cut by the image edge or marked partial; else (pixel statistics, texture)
            the visible part is measured (``WHOLE_OBJECT``; unset: ``whole_object_default``).
    """

    key: str
    label: str
    unit: str
    category: str
    scope: str
    description: str
    default_on: bool
    order: float
    image_type: str
    constants: list[Setting] = field(default_factory=list)
    references: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    compute: Callable = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"
    note: str = ""
    whole_object: bool = False

    def help_text(self) -> str:
        """Its description followed by its references (for tooltips and reports)."""
        refs = "".join(f"\n• {r}" for r in self.references)
        return self.description + (f"\n\nReferences:{refs}" if refs else "")

    @property
    def uid(self) -> str:
        """``<scope>:<image type>:<key>``: how profiles refer to the parameter."""
        return f"{self.scope}:{type_word(self.image_type)}:{self.key}"

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.image_type, self.image_type)

    @property
    def label_with_unit(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def accepts(self, image_type: str) -> bool:
        """Whether it can measure images of this type."""
        return self.image_type in (ANY, image_type)

    def resolve(self, chosen: dict | None = None) -> dict:
        """Its constants: declared defaults, overridden (and cleaned) by ``chosen``."""
        chosen = chosen or {}
        return {s.key: (s.clean(chosen[s.key]) if s.key in chosen else s.default) for s in self.constants}


def load_script(path: str | os.PathLike, source: str = "user") -> ParameterSpec:
    """Import one parameter script and describe it.

    Raises:
        ValueError: If the script is not a usable parameter (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()                         # the user's libraries are importable
    path = Path(path)
    name = f"_modalytics_param_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        needs = libraries.missing(_declared_requires(path))
        hint = (f"it needs {', '.join(needs)}" if needs else f"it imports '{exc.name}', which is not installed")
        raise ValueError(f"{hint}: install it in Settings › Parameters › Libraries") from exc
    except Exception as exc:
        raise ValueError(f"the script fails to load: {exc}") from exc
    key = getattr(module, "KEY", None)
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        raise ValueError("KEY must be a name of letters, digits and _")
    if not callable(getattr(module, "compute", None)):
        raise ValueError("the script must define compute(ctx)")
    scope = getattr(module, "SCOPE", "object")
    if scope not in SCOPES:
        raise ValueError(f"SCOPE must be one of {', '.join(SCOPES)}")
    image_type, note = image_type_of(module)
    try:
        declared = getattr(module, "CONSTANTS", getattr(module, "SETTINGS", []))    # SETTINGS: older name
        constants = [s if isinstance(s, Setting) else Setting.from_dict(dict(s)) for s in declared]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CONSTANTS: {exc}") from exc
    return ParameterSpec(
        key=key, label=str(getattr(module, "LABEL", key)), unit=str(getattr(module, "UNIT", "")),
        category=str(getattr(module, "CATEGORY", "other")), scope=scope,
        description=str(getattr(module, "DESCRIPTION", "")), default_on=bool(getattr(module, "DEFAULT_ON", True)),
        order=float(getattr(module, "ORDER", 1000)), image_type=image_type, constants=constants,
        requires=tuple(getattr(module, "REQUIRES", ()) or ()), note=note,
        references=tuple(str(r) for r in (getattr(module, "REFERENCES", ()) or ())),
        whole_object=bool(getattr(module, "WHOLE_OBJECT", whole_object_default(
            str(getattr(module, "CATEGORY", "other")), str(key)))) if scope == "object" else False,
        compute=module.compute, path=str(path), source=source)


def distinct_labels(specs) -> dict[str, str]:
    """{parameter id: its label}, with the image type added where two of ``specs`` share a label
    (e.g. "Uniformity Coef. (Phase map)" and "Uniformity Coef. (OD map)")."""
    specs = list(specs)
    counts: dict[str, int] = {}
    for s in specs:
        counts[s.label] = counts.get(s.label, 0) + 1
    return {s.uid: s.label if counts[s.label] == 1 else f"{s.label} ({TYPE_NAMES.get(s.image_type, s.image_type)})"
            for s in specs}


def image_type_of(module) -> tuple[str, str]:
    """(image type, note) of a loaded script: IMAGE_TYPE, else the older IMAGE_TYPES list."""
    if hasattr(module, "IMAGE_TYPE"):
        return str(module.IMAGE_TYPE), ""
    return image_type_from_list(getattr(module, "IMAGE_TYPES", ("phase",)))


def image_type_from_list(types) -> tuple[str, str]:
    """An older IMAGE_TYPES value as one image type (and a note when it listed several)."""
    types = [types] if isinstance(types, str) else list(types or ["phase"])
    if ANY in types:
        return ANY, ""
    if len(types) > 1:
        return types[0], (f"It lists several image types ({', '.join(types)}); it now measures one "
                          f"({types[0]}). Duplicate it for the others.")
    return types[0], ""


def _declared_requires(path: Path) -> list[str]:
    """REQUIRES of a script, read without running it (it may fail to import)."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REQUIRES" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except ValueError:
                return []
    return []


class ParameterLibrary:
    """Every parameter available here (built-in + user), loaded on creation.

    Args:
        builtin: Built-in scripts folder (default: the resources).
        user: User scripts folder (default: the user library).

    Attributes:
        specs: {parameter id: spec}, sorted by (scope, order, label).
        errors: {script path: why it was not loaded}.
        overridden: Ids of built-ins replaced by a user script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin = Path(builtin) if builtin is not None else paths.builtin_dir("parameters")
        self.user = Path(user) if user is not None else paths.user_dir("parameters")
        self.specs: dict[str, ParameterSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        self.reload()

    def reload(self) -> None:
        """Scan both folders again (after adding, editing or removing scripts)."""
        found: dict[str, ParameterSpec] = {}
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
        self.specs = dict(sorted(found.items(), key=lambda kv: (kv[1].scope, kv[1].order, kv[1].label)))

    def get(self, uid: str) -> ParameterSpec | None:
        return self.specs.get(uid) or self.specs.get(self.find(uid) or "")

    def find(self, uid: str) -> str | None:
        """The current id for an id, including ids from before image types (``object:mean`` →
        ``object:phase:mean``; a key that exists for several types prefers phase, then any image)."""
        if uid in self.specs:
            return uid
        parts = uid.split(":")
        if len(parts) != 2:
            return None
        scope, key = parts
        matches = [s for s in self.specs.values() if s.scope == scope and s.key == key]
        matches.sort(key=lambda s: {"phase": 0, ANY: 1}.get(s.image_type, 2))
        return matches[0].uid if matches else None

    def migrate(self, chosen: dict[str, dict]) -> dict[str, dict]:
        """A profile's parameter choices with every id in the current form (unknown ones kept as they are)."""
        return {(self.find(uid) or uid): constants for uid, constants in chosen.items()}

    def of_type(self, scope: str, image_type: str) -> list[ParameterSpec]:
        return [s for s in self.specs.values() if s.scope == scope and s.image_type == image_type]

    def of_scope(self, scope: str) -> list[ParameterSpec]:
        return [s for s in self.specs.values() if s.scope == scope]

    def categories(self, scope: str | None = None, image_type: str | None = None) -> list[str]:
        """Category names in first-appearance order (of a scope / image type, if given)."""
        out = []
        for s in self.specs.values():
            if (scope is None or s.scope == scope) and (image_type is None or s.image_type == image_type) \
                    and s.category not in out:
                out.append(s.category)
        return out

    def user_files(self) -> dict[str, Path]:
        """{id: script} of the user's own parameters (what a profile export must carry)."""
        return {uid: Path(s.path) for uid, s in self.specs.items() if s.source == "user"}

    def missing_libraries(self) -> dict[str, list[str]]:
        """{parameter id or script path: libraries it declares but are not available here}."""
        from modalytics.parameters import libraries
        out = {uid: libraries.missing(s.requires) for uid, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}

    def all_files(self) -> dict[str, Path]:
        return {uid: Path(s.path) for uid, s in self.specs.items()}

    def defaults(self) -> dict[str, dict]:
        """{id: default constants} of the parameters that are on by default (a new profile's list)."""
        return {uid: s.resolve() for uid, s in self.specs.items() if s.default_on}
