"""
The processing library: one script per processing step, like parameters and importers.

A processing step makes a new version of an image: another image type (a
slide's OD map, a phase map's dry-mass map) or the same type filtered (blurred,
denoised). The result is a *view* of the image (``core.model.ImageView``): same
pixel grid, placement and masks. A step is one ``.py`` file::

    KEY = "od_map"
    NAME = "OD map"
    INPUT_TYPES = ["wsi", "brightfield"]      # what it accepts ("*" = any image)
    OUTPUT_TYPE = "od"                        # what it makes; "same" = the input's type (a filter)
    UNIT = "OD"                               # unit of the result ("" = the input's, for filters)
    KIND = "pixel"                            # "pixel": each pixel from the same pixel;
                                              # "neighbourhood": from the pixels around it (MARGIN);
                                              # "whole": needs the whole image at once (small images only)
    CATEGORY = "Quantitative maps"            # optional: where menus list it (see CATEGORIES)
    DESCRIPTION = "…"
    REQUIRES = []
    OPTIONS = [ … ]                           # its values (same form as parameter constants)

    def prepare(image, options) -> dict       # optional: once per image (e.g. a background level)
    def margin(options, um_per_px) -> int     # "neighbourhood": pixels of context needed around a tile
    def process(pixels, options, state, um_per_px) -> np.ndarray
    def process(pixels, options, state, um_per_px, where)       # or: also told where the pixels are

``process`` gets the input's pixels (a tile, or the whole image for "whole"),
the option values, what ``prepare`` returned, and the size of a pixel at the
level being computed (sizes in µm stay right at every zoom level). A step whose
result depends on the position (a flat-field correction, a fitted background)
adds a ``where`` argument: a ``Where`` with the block's first row and column
and the level's size, in that level's pixels (``where.fraction`` gives each
pixel's position as 0–1 of the image, the same at every level). It returns
an array of the same height and width: 2D float for a new image type, or the
input's shape for a filter. Steps whose result is another image type are
checked against that type's contract (``processing.types``).

An id is ``<output type or "filters">:<key>``. Scripts are read from the
built-in folder (``resources/processing``) and the user library
(``paths.user_dir("processing")``), laid out as ``<output type>/<key>.py`` (filters:
``filters/<key>.py``); a user script with the same id replaces the built-in one.

WARNING: processing scripts are Python and run with Modalytics' rights: only add
steps you trust.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from modalytics import paths
from modalytics.core.model import IMAGE_TYPES
from modalytics.core.settings import Setting

KINDS = ("pixel", "neighbourhood", "whole")
SAME = "same"                                      # OUTPUT_TYPE of filters
# How menus and lists group steps (CATEGORY; a step without one is listed under "Other")
CATEGORIES = ("Quantitative maps", "Feature maps", "Correction", "Denoise", "Enhance", "Other")
# The categories in the order of a usual workflow: the Process tab's groups and the profile's step list
WORKFLOW_ORDER = ("Correction", "Denoise", "Enhance", "Feature maps", "Quantitative maps", "Other")
# The icon of a step (or chain) button that names none (Material Design Icons, see ui.icon_picker)
CATEGORY_ICONS = {"Quantitative maps": "mdi6.chart-bell-curve-cumulative", "Feature maps": "mdi6.texture-box",
                  "Correction": "mdi6.tune-variant", "Denoise": "mdi6.blur", "Enhance": "mdi6.contrast-box",
                  "Other": "mdi6.cog-outline"}
CHAIN_ICON = "mdi6.link-variant"
# The Process tab's step buttons when a profile has not chosen (those that fit its image types)
DEFAULT_TOOLBAR = ("filters:gaussian_blur", "filters:median", "filters:background_flatten", "filters:phase_unwrap",
                   "filters:flat_field", "filters:white_balance", "filters:clahe", "filters:unsharp_mask", "opd:opd",
                   "height:height_map", "dry_mass:dry_mass", "ri:ri_map", "od:od_map", "stain:colour_deconvolution")


@dataclass(frozen=True)
class Where:
    """Where the pixels given to ``process`` lie (for steps that depend on the position).

    Attributes:
        row, col: The block's first row and column, in the level's pixels (may be negative: margins).
        shape: (height, width) of the level.
        level: Level index (0 = full resolution).
    """

    row: int
    col: int
    shape: tuple[int, int]
    level: int = 0

    def fraction(self, block_shape: tuple[int, int]):
        """(ys, xs): each pixel's centre as 0–1 of the image's height and width (1D arrays)."""
        import numpy as np
        h, w = self.shape
        ys = (np.arange(block_shape[0]) + self.row + 0.5) / h
        xs = (np.arange(block_shape[1]) + self.col + 0.5) / w
        return ys, xs


@dataclass
class StepSpec:
    """One processing step of the library (see the module docstring for the fields)."""

    key: str
    name: str
    input_types: tuple[str, ...]
    output_type: str
    unit: str = ""
    kind: str = "pixel"
    category: str = ""
    description: str = ""
    options: list[Setting] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    process: Callable = field(default=None, repr=False)
    wants_where: bool = False
    prepare: Callable | None = field(default=None, repr=False)
    margin: Callable | None = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"
    short: str = ""
    icon: str = ""

    @property
    def button_label(self) -> str:
        """The label of its toolbar button (SHORT, else its name)."""
        return self.short or self.name

    @property
    def button_icon(self) -> str:
        """The icon of its toolbar button (ICON, else its category's)."""
        return self.icon or CATEGORY_ICONS.get(self.group, CATEGORY_ICONS["Other"])

    @property
    def folder(self) -> str:
        return "filters" if self.output_type == SAME else self.output_type

    @property
    def uid(self) -> str:
        return f"{self.folder}:{self.key}"

    @property
    def group(self) -> str:
        """The category menus list it under."""
        return self.category or "Other"

    @property
    def is_filter(self) -> bool:
        return self.output_type == SAME

    def accepts(self, image_type: str) -> bool:
        """Whether it can process an image of this type."""
        return "*" in self.input_types or image_type in self.input_types

    def output_for(self, image_type: str) -> str:
        """The image type it makes from one of ``image_type``."""
        return image_type if self.is_filter else self.output_type

    def resolve(self, chosen: dict | None = None) -> dict:
        """Option values: defaults, overridden (and cleaned) by ``chosen``."""
        chosen = chosen or {}
        return {o.key: (o.clean(chosen[o.key]) if o.key in chosen else o.default) for o in self.options}

    def margin_px(self, options: dict, um_per_px: float) -> int:
        """Pixels of context a "neighbourhood" step needs around a tile (0 for the others)."""
        if self.kind != "neighbourhood" or self.margin is None:
            return 0
        return max(0, int(self.margin(options, um_per_px)))


def load_script(path: str | os.PathLike, source: str = "user") -> StepSpec:
    """Import one processing script and describe it.

    Raises:
        ValueError: If the script is not a usable step (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()
    path = Path(path)
    name = f"_modalytics_processing_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ValueError(f"it imports '{exc.name}', which is not installed: install it with Libraries…") from exc
    except Exception as exc:
        raise ValueError(f"the script fails to load: {exc}") from exc
    key = getattr(module, "KEY", None)
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        raise ValueError("KEY must be a name of letters, digits and _")
    if not callable(getattr(module, "process", None)):
        raise ValueError("the script must define process(pixels, options, state, um_per_px)")
    inputs = getattr(module, "INPUT_TYPES", ("*",))
    inputs = tuple([inputs] if isinstance(inputs, str) else inputs)
    unknown = [t for t in inputs if t != "*" and t not in IMAGE_TYPES]
    if unknown or not inputs:
        raise ValueError(f"INPUT_TYPES: unknown image type(s) {unknown}" if unknown else "INPUT_TYPES is empty")
    output = str(getattr(module, "OUTPUT_TYPE", SAME))
    if output != SAME and output not in IMAGE_TYPES:
        raise ValueError(f"OUTPUT_TYPE must be one of {', '.join(IMAGE_TYPES)} or \"same\" (a filter)")
    kind = str(getattr(module, "KIND", "pixel"))
    if kind not in KINDS:
        raise ValueError(f"KIND must be one of {', '.join(KINDS)}")
    try:
        options = [o if isinstance(o, Setting) else Setting.from_dict(dict(o)) for o in getattr(module, "OPTIONS", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPTIONS: {exc}") from exc

    def optional(fn):
        f = getattr(module, fn, None)
        return f if callable(f) else None
    import inspect
    try:
        wants_where = "where" in inspect.signature(module.process).parameters
    except (TypeError, ValueError):
        wants_where = False
    icon = str(getattr(module, "ICON", "") or "")
    if icon and not re.fullmatch(r"mdi6\.[a-z0-9-]+", icon):
        raise ValueError("ICON must be a Material Design icon name such as 'mdi6.blur' (Choose… picks one)")
    return StepSpec(key=key, name=str(getattr(module, "NAME", key)), short=str(getattr(module, "SHORT", "") or ""),
                    icon=icon, input_types=inputs, output_type=output,
                    unit=str(getattr(module, "UNIT", "")), kind=kind,
                    category=str(getattr(module, "CATEGORY", "") or ""),
                    description=str(getattr(module, "DESCRIPTION", "")), options=options,
                    requires=tuple(getattr(module, "REQUIRES", ()) or ()), process=module.process,
                    wants_where=wants_where,
                    prepare=optional("prepare"), margin=optional("margin"), path=str(path), source=source)


class ProcessingLibrary:
    """Every processing step available here (built-in + user), loaded on creation.

    Attributes:
        specs: {step id: spec}, sorted by (output folder, name).
        errors: {script path: why it was not loaded}.
        overridden: Ids of built-ins replaced by a user script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin = Path(builtin) if builtin is not None else paths.builtin_dir("processing")
        self.user = Path(user) if user is not None else paths.user_dir("processing")
        self.specs: dict[str, StepSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        self.reload()

    @classmethod
    def of(cls, specs) -> "ProcessingLibrary":
        """A library holding just these steps (e.g. one loaded from a script being tested)."""
        lib = cls.__new__(cls)
        lib.builtin = lib.user = None
        lib.specs = {s.uid: s for s in specs}
        lib.errors, lib.overridden = {}, []
        return lib

    def reload(self) -> None:
        found: dict[str, StepSpec] = {}
        self.errors, self.overridden = {}, []
        for source, folder in (("builtin", self.builtin), ("user", self.user)):
            for path in sorted(folder.rglob("*.py")) if folder.is_dir() else []:
                if path.name.startswith("_") or "__pycache__" in path.parts:
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
        self.specs = dict(sorted(found.items(), key=lambda kv: (kv[1].folder, kv[1].name.lower())))

    def get(self, uid: str) -> StepSpec | None:
        return self.specs.get(uid)

    def for_type(self, image_type: str) -> list[StepSpec]:
        """The steps that can process an image of this type."""
        return [s for s in self.specs.values() if s.accepts(image_type)]

    def grouped(self, image_type: str | None = None) -> list[tuple[str, list[StepSpec]]]:
        """[(category, its steps)] in ``CATEGORIES`` order (then any others), for menus and lists.

        Args:
            image_type: Only the steps that process this type (default: all).
        """
        specs = self.for_type(image_type) if image_type else list(self.specs.values())
        names = list(CATEGORIES) + sorted({s.group for s in specs} - set(CATEGORIES))
        return [(c, [s for s in specs if s.group == c]) for c in names if any(s.group == c for s in specs)]

    def user_files(self) -> dict[str, Path]:
        return {uid: Path(s.path) for uid, s in self.specs.items() if s.source == "user"}

    def missing_libraries(self) -> dict[str, list[str]]:
        """{step id or script path: libraries it declares but are not available here}."""
        from modalytics.parameters import libraries
        from modalytics.parameters.registry import _declared_requires
        out = {uid: libraries.missing(s.requires) for uid, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}
