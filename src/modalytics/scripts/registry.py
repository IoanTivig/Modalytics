"""
External scripts: tasks done beside the analysis, outside the pathologist's workflow (Qt-free).

For example HoloPath's *DHM Phase Map Extractor*: walk an instrument's folders,
take every ``Phase.txt``, rename it after its slide, area and image, and copy them
all into one folder — which is then imported with the usual importers. A script
does not draw anything: it declares its options, Modalytics asks for them (Scripts
menu, or Settings › Scripts › Run), runs it in the background with a progress bar,
a log and Cancel, and afterwards offers to import the files it hands over.

A script is one ``.py`` file::

    KEY = "dhm_phase_extractor"              # its id: letters, digits, _
    NAME = "DHM phase map extractor"
    CATEGORY = "Data preparation"            # menus group scripts by it
    DESCRIPTION = "Extract every Phase.txt …"
    REQUIRES = []                            # libraries beyond the included ones (as for parameters)
    OPTIONS = [                              # asked before it runs; values as options["key"]
        {"key": "input", "label": "Input folder", "kind": "folder", "default": ""},
    ]

    def run(options, ctx):
        # ctx.log(text) · ctx.progress(0–1) · ctx.cancelled() · ctx.import_files(paths)
        ...
        return "12 files copied"             # optional: a summary shown at the end

Scripts are read from the built-in folder (``resources/scripts``) and the user
library (``paths.user_dir("scripts")``), as ``<key>.py``; a user script with the
same id replaces the built-in one. A profile lists the scripts its projects offer
(Profile.scripts): they appear in the Scripts menu, between Tools and Help.
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


class Cancelled(Exception):
    """Raised in a script (by ``ScriptContext.check``) when the user cancels."""


@dataclass
class ScriptSpec:
    """One external script (see the module docstring)."""

    key: str
    name: str
    category: str = ""
    description: str = ""
    options: list[Setting] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    run: Callable = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"

    @property
    def uid(self) -> str:
        return self.key

    @property
    def group(self) -> str:
        return self.category or "Other"

    def resolve(self, chosen: dict | None = None) -> dict:
        """Option values: defaults, overridden (and cleaned) by ``chosen``."""
        chosen = chosen or {}
        return {o.key: (o.clean(chosen[o.key]) if o.key in chosen else o.default) for o in self.options}


class ScriptContext:
    """What a running script can do besides its work: say what happens, show progress, stop when asked, hand files
    over for import.

    Args:
        log: Called with each line the script writes.
        progress: Called with 0–1.
        cancelled: Returns True once the user asked to stop.
    """

    def __init__(self, log: Callable[[str], None] | None = None, progress: Callable[[float], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> None:
        self._log, self._progress, self._cancelled = log, progress, cancelled
        self.to_import: list[str] = []
        self.lines: list[str] = []

    def log(self, text: str) -> None:
        """Write a line to the run's log (and the diagnostic log, when on)."""
        self.lines.append(str(text))
        if self._log is not None:
            self._log(str(text))

    def progress(self, fraction: float) -> None:
        if self._progress is not None:
            self._progress(max(0.0, min(1.0, float(fraction))))

    def cancelled(self) -> bool:
        """True once the user asked to stop: the script should then return (or call ``check``)."""
        return bool(self._cancelled is not None and self._cancelled())

    def check(self) -> None:
        """Stop here (raise ``Cancelled``) if the user asked to."""
        if self.cancelled():
            raise Cancelled()

    def import_files(self, files) -> None:
        """Files (or a folder: all its files) to offer for import into the open project once the script ends."""
        items = [files] if isinstance(files, (str, os.PathLike)) else list(files)
        for f in items:
            f = Path(f)
            if f.is_dir():
                self.to_import += [str(p) for p in sorted(f.iterdir()) if p.is_file()]
            elif f.is_file():
                self.to_import.append(str(f))


def run(spec: ScriptSpec, options: dict, ctx: ScriptContext):
    """Run a script with these option values; returns its summary (or None)."""
    from modalytics import diagnostics
    values = spec.resolve(options)
    diagnostics.event("script started", script=spec.uid, options=values)
    result = spec.run(values, ctx)
    diagnostics.event("script finished", script=spec.uid, files_to_import=len(ctx.to_import))
    return result


def load_script(path: str | os.PathLike, source: str = "user") -> ScriptSpec:
    """Import one script file and describe it.

    Raises:
        ValueError: If it is not a usable script (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()
    path = Path(path)
    name = f"_modalytics_script_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
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
    if not callable(getattr(module, "run", None)):
        raise ValueError("the script must define run(options, ctx)")
    try:
        options = [o if isinstance(o, Setting) else Setting.from_dict(dict(o)) for o in getattr(module, "OPTIONS", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPTIONS: {exc}") from exc
    return ScriptSpec(key=key, name=str(getattr(module, "NAME", key)),
                      category=str(getattr(module, "CATEGORY", "") or ""),
                      description=str(getattr(module, "DESCRIPTION", "")), options=options,
                      requires=tuple(getattr(module, "REQUIRES", ()) or ()), run=module.run, path=str(path),
                      source=source)


class ScriptLibrary:
    """Every external script available here (built-in + user), loaded on creation.

    Attributes:
        specs: {script id: spec}, sorted by (category, name).
        errors: {script path: why it was not loaded}.
        overridden: Ids of built-ins replaced by a user script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin = Path(builtin) if builtin is not None else paths.builtin_dir("scripts")
        self.user = Path(user) if user is not None else paths.user_dir("scripts")
        self.specs: dict[str, ScriptSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        self.reload()

    def reload(self) -> None:
        found: dict[str, ScriptSpec] = {}
        self.errors, self.overridden = {}, []
        for source, folder in (("builtin", self.builtin), ("user", self.user)):
            for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
                if path.name.startswith("_"):
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
        self.specs = dict(sorted(found.items(), key=lambda kv: (kv[1].group.lower(), kv[1].name.lower())))

    def get(self, uid: str) -> ScriptSpec | None:
        return self.specs.get(uid)

    def grouped(self) -> list[tuple[str, list[ScriptSpec]]]:
        """[(category, its scripts)] in alphabetical order, for menus and lists."""
        out: dict[str, list] = {}
        for s in self.specs.values():
            out.setdefault(s.group, []).append(s)
        return list(out.items())

    def user_files(self) -> dict[str, Path]:
        """{id: script} of the user's own scripts (what a profile export must carry)."""
        return {uid: Path(s.path) for uid, s in self.specs.items() if s.source == "user"}

    def all_files(self) -> dict[str, Path]:
        return {uid: Path(s.path) for uid, s in self.specs.items()}

    def missing_libraries(self) -> dict[str, list[str]]:
        """{script id or path: libraries it declares but are not available here}."""
        from modalytics.parameters import libraries
        from modalytics.parameters.registry import _declared_requires
        out = {uid: libraries.missing(s.requires) for uid, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}
