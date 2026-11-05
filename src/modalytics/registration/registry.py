"""
Registration method scripts: the library of methods that place one image on another (Qt-free).

A method is one ``.py`` file in ``resources/registration`` (built in) or in the user
library (``paths.user_dir("registration")``; a file with a built-in's KEY replaces it)::

    KEY = 'gland_map_ncc'                    # letters, digits, _
    NAME = 'Gland-map correlation'
    DESCRIPTION = 'What it compares, when it works, what it finds'
    MOVING_TYPES = ['phase']                 # image types it can place ('*' = any)
    REFERENCE_TYPES = ['wsi']                # image types it places them on
    FINDS = ['translation', 'rotation', 'scale', 'mirror']
    EXPERIMENTAL = False                     # True: shown as experimental (not validated on this data)
    REFINES_ONLY = False                     # True: no search, it refines the current placement
    REFERENCES = [{'text': 'Author (year). Title. Journal …', 'doi': '10.…'}]
    REQUIRES = []                            # extra libraries it imports
    OPTIONS = [{'key': …, 'label': …, 'kind': …, 'default': …}]   # its own settings

    def register(ctx, options):
        # ctx: registration.context.RegistrationContext — the two images, the search (rough position,
        #      radius, rotations, scales, mirroring, how many candidates), pixels, feature maps, matching
        return [ctx.candidate(matrix, score), …]          # best first

The Register tool gives every method the same search: the moving image's current
position, the search radius, the rotations, scale corrections and mirroring to try
and the number of candidates; a method's own OPTIONS come on top.
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

BUILTIN = Path(__file__).resolve().parent.parent / "resources" / "registration"
FINDS = ("translation", "rotation", "scale", "mirror")
DEFAULT_METHOD = "gland_map_ncc"


@dataclass
class MethodSpec:
    """One registration method of the library (see the module docstring for the fields)."""

    key: str
    name: str
    moving_types: tuple[str, ...]
    reference_types: tuple[str, ...]
    finds: tuple[str, ...] = FINDS
    description: str = ""
    experimental: bool = False
    refines_only: bool = False
    references: list[dict] = field(default_factory=list)
    options: list[Setting] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    register: Callable = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"

    @property
    def uid(self) -> str:
        return self.key

    def can_move(self, image_type: str) -> bool:
        return "*" in self.moving_types or image_type in self.moving_types

    def can_reference(self, image_type: str) -> bool:
        return "*" in self.reference_types or image_type in self.reference_types

    def fits(self, moving_type: str, reference_type: str) -> bool:
        """Whether it can place an image of ``moving_type`` on one of ``reference_type``."""
        return self.can_move(moving_type) and self.can_reference(reference_type)

    def resolve(self, chosen: dict | None = None) -> dict:
        """Option values: defaults, overridden (and cleaned) by ``chosen``."""
        chosen = chosen or {}
        return {o.key: (o.clean(chosen[o.key]) if o.key in chosen else o.default) for o in self.options}

    def references_text(self) -> str:
        return "\n".join(r.get("text", "") + (f" doi:{r['doi']}" if r.get("doi") else "") for r in self.references)


def _types(value, what: str) -> tuple[str, ...]:
    types = tuple([value] if isinstance(value, str) else value or ())
    unknown = [t for t in types if t != "*" and t not in IMAGE_TYPES]
    if unknown or not types:
        raise ValueError(f"{what}: unknown image type(s) {unknown}" if unknown else f"{what} is empty")
    return types


def load_script(path: str | os.PathLike, source: str = "user") -> MethodSpec:
    """Import one registration script and describe it.

    Raises:
        ValueError: If the script is not a usable method (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()
    path = Path(path)
    name = f"_modalytics_registration_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
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
    if not callable(getattr(module, "register", None)):
        raise ValueError("the script must define register(ctx, options)")
    finds = tuple(getattr(module, "FINDS", FINDS) or FINDS)
    bad = [f for f in finds if f not in FINDS]
    if bad or "translation" not in finds:
        raise ValueError(f"FINDS must hold translation and may hold {', '.join(FINDS[1:])}")
    try:
        options = [o if isinstance(o, Setting) else Setting.from_dict(dict(o)) for o in getattr(module, "OPTIONS", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPTIONS: {exc}") from exc
    refs = [dict(r) if isinstance(r, dict) else {"text": str(r)} for r in getattr(module, "REFERENCES", []) or []]
    return MethodSpec(key=key, name=str(getattr(module, "NAME", key)),
                      moving_types=_types(getattr(module, "MOVING_TYPES", ("*",)), "MOVING_TYPES"),
                      reference_types=_types(getattr(module, "REFERENCE_TYPES", ("*",)), "REFERENCE_TYPES"),
                      finds=finds, description=str(getattr(module, "DESCRIPTION", "")),
                      experimental=bool(getattr(module, "EXPERIMENTAL", False)),
                      refines_only=bool(getattr(module, "REFINES_ONLY", False)), references=refs, options=options,
                      requires=tuple(getattr(module, "REQUIRES", ()) or ()), register=module.register,
                      path=str(path), source=source)


class RegistrationLibrary:
    """Every registration method: built in, replaced or added by the user's own scripts.

    Attributes:
        specs: {key: MethodSpec}, built-in order first.
        errors: {script path: why it could not be loaded}.
        overridden: Keys of built-ins replaced by the user's own script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin_dir = Path(builtin) if builtin else BUILTIN
        self.user = Path(user) if user else paths.user_dir("registration")
        self.reload()

    @property
    def problems(self) -> dict[str, str]:
        """{file name: why it could not be loaded}."""
        return {Path(p).name: why for p, why in self.errors.items()}

    def reload(self) -> None:
        self.specs: dict[str, MethodSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        for folder, source in ((self.builtin_dir, "builtin"), (self.user, "user")):
            for f in sorted(folder.glob("*.py")) if folder.is_dir() else []:
                if f.name.startswith("_"):
                    continue
                try:
                    spec = load_script(f, source)
                except ValueError as exc:
                    self.errors[str(f)] = str(exc)
                    continue
                if source == "user" and spec.key in self.specs:
                    self.overridden.append(spec.key)
                self.specs[spec.key] = spec

    def get(self, key: str) -> MethodSpec | None:
        return self.specs.get(key)

    def for_pair(self, moving_type: str, reference_type: str) -> list[MethodSpec]:
        """The methods that can place a ``moving_type`` image on a ``reference_type`` one (validated first)."""
        out = [s for s in self.specs.values() if s.fits(moving_type, reference_type)]
        return sorted(out, key=lambda s: s.experimental)

    def moving_types(self) -> set[str]:
        """Image types some method can place ('*' when one takes any)."""
        return {t for s in self.specs.values() for t in s.moving_types}

    def user_files(self) -> dict[str, Path]:
        return {f.stem: f for f in sorted(self.user.glob("*.py"))} if self.user.is_dir() else {}

    def missing_libraries(self) -> dict[str, list[str]]:
        """{method key or script path: libraries it declares but are not available here}."""
        from modalytics.parameters import libraries
        from modalytics.parameters.registry import _declared_requires
        out = {k: libraries.missing(s.requires) for k, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}
