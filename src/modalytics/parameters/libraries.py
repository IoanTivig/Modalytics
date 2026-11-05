"""
Libraries that parameter and importer scripts may import (Qt-free).

Two kinds:

* **Included** — everything installed with Modalytics (numpy, scipy,
  scikit-image, shapely, OpenCV…). Locked: their versions are the ones
  Modalytics is tested with, and nothing a user installs may change them.
* **Yours** — small helper or statistics / image-processing libraries a user
  adds, installed with pip into the user library's ``libraries`` folder (never
  into Modalytics' own environment) and removable again.

Rules for "yours" (``check_name``, ``install_commands``):

1. Ready-built packages only (``--only-binary=:all:``): nothing that needs a
   compiler on the user's computer. Pure-Python packages are always ready-built.
2. The included libraries are pinned (``--constraint``): an install that would
   need another version of one of them is refused by pip, with its reason.
3. Large machine-learning frameworks are refused: they belong to Training.
4. Dependencies that are already included are not installed again; only what
   is missing goes into the folder (planned with pip's ``--dry-run --report``).
5. Installs go to a staging folder first; only when pip succeeds are older
   versions of the same libraries removed and the new ones moved in
   (``finish_install``). pip's own ``--target --upgrade`` would leave the old
   version's metadata behind, and a failed update must not lose the old one.
   Updating = installing the newest compatible version this way.

A parameter declares what it needs as ``REQUIRES = ["PyWavelets"]``;
``missing`` tells which of those are not available here.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from modalytics import paths

# Too large, or meant for Training (stage 7), not for parameters
BLOCKED = {"torch", "torchvision", "torchaudio", "tensorflow", "tensorflow-cpu", "tensorflow-gpu",
           "tensorflow-intel", "keras", "jax", "jaxlib", "paddlepaddle", "mxnet", "cntk", "theano"}


@dataclass(frozen=True)
class Library:
    """One installed library (a Python distribution)."""

    name: str
    version: str
    summary: str = ""


def canonical(name: str) -> str:
    """PEP 503 normalised name: 'PyWavelets' → 'pywavelets', 'scikit_image' → 'scikit-image'."""
    return re.sub(r"[-_.]+", "-", name).lower().strip()


def folder() -> Path:
    """Where your libraries are installed (created if missing)."""
    return paths.user_dir("libraries")


def activate() -> None:
    """Make your libraries importable (after the included ones, so those always win). Idempotent."""
    path = str(folder())
    if path not in sys.path:
        sys.path.append(path)


## What is here ##

def included() -> list[Library]:
    """The libraries installed with Modalytics (locked), sorted by name."""
    mine = os.path.normcase(os.path.abspath(folder()))
    search = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != mine]
    out = {}
    for d in md.distributions(path=search):
        name = d.metadata.get("Name")
        if name and canonical(name) not in out:
            out[canonical(name)] = Library(name, d.version, d.metadata.get("Summary") or "")
    return sorted(out.values(), key=lambda lib: lib.name.lower())


def yours() -> list[Library]:
    """The libraries installed into your libraries folder, sorted by name."""
    out = []
    for d in md.distributions(path=[str(folder())]):                  # (not the staging folder inside it)
        name = d.metadata.get("Name")
        if name:
            out.append(Library(name, d.version, d.metadata.get("Summary") or ""))
    return sorted(out, key=lambda lib: lib.name.lower())


def available() -> set[str]:
    """Canonical names of every library a script can import (included + yours)."""
    return {canonical(lib.name) for lib in included()} | {canonical(lib.name) for lib in yours()}


def missing(requires) -> list[str]:
    """Which of a parameter's REQUIRES are not available here (names as written).

    Args:
        requires: Names, optionally with a version ("PyWavelets>=1.4"); the version is not checked.
    """
    have = available()
    return [r for r in requires or [] if canonical(re.split(r"[<>=!~ ;\[]", r, 1)[0]) not in have]


## Installing ##

def pip_available() -> bool:
    """Whether pip can run here (always from source; the installer must include it)."""
    return not getattr(sys, "frozen", False) and importlib.util.find_spec("pip") is not None


def check_name(requirement: str) -> str:
    """Why a requirement may not be installed ("" if it may).

    Args:
        requirement: "name" or "name==1.2" etc.
    """
    name = canonical(re.split(r"[<>=!~ ;\[]", requirement.strip(), 1)[0])
    if not name or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name):
        return "That is not a package name."
    if name in BLOCKED:
        return (f"{name} is a large machine-learning framework: Modalytics uses those only for Training, "
                f"not for parameters.")
    if name in {canonical(lib.name) for lib in included()}:
        return f"{name} is included with Modalytics already (locked version)."
    return ""


def constraints_file(target: Path) -> Path:
    """A pip constraints file pinning every included library to its installed version."""
    lines = [f"{lib.name}=={lib.version}" for lib in included()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def pip() -> list[str]:
    """How pip is run: ``python -m pip`` from the source; installed, the app's console twin with ``--pip``."""
    from modalytics import paths
    return [*paths.helper_command(), "--pip"] if paths.installed() else [sys.executable, "-m", "pip"]


def plan_command(requirement: str, report: Path, constraints: Path) -> list[str]:
    """pip command that resolves what installing ``requirement`` needs, without installing anything."""
    return [*pip(), "install", "--dry-run", "--quiet", "--ignore-installed",
            "--only-binary=:all:", "--constraint", str(constraints), "--report", str(report), requirement]


def plan_from_report(report: dict) -> tuple[list[str], list[str]]:
    """What to install from a pip ``--report``.

    Returns:
        tuple: (["name==version" to install into your folder], [problems: included libraries
        the plan would change]).
    """
    inc = {canonical(lib.name): lib.version for lib in included()}
    todo, problems = [], []
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        name, version = meta.get("name", ""), meta.get("version", "")
        c = canonical(name)
        if c in inc:
            if inc[c] != version:
                problems.append(f"it needs {name} {version}, Modalytics includes {inc[c]}")
            continue                                   # already included: not installed again
        todo.append(f"{name}=={version}")
    return todo, problems


def staging() -> Path:
    """An empty staging folder for one install (inside the libraries folder, so moving is instant)."""
    path = folder() / ".staging"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    return path


def install_command(requirements: list[str], target: Path) -> list[str]:
    """pip command that installs exactly these into ``target`` (no dependency resolution: the plan did it)."""
    return [*pip(), "install", "--no-deps", "--only-binary=:all:", "--target",
            str(target), *requirements]


def finish_install(stage: Path) -> list[tuple[str, str, str]]:
    """Move a successful staged install into the libraries folder, replacing older versions.

    Returns:
        list[tuple[str, str, str]]: (name, old version or "", new version) per library.
    """
    root = folder()
    old = {canonical(lib.name): lib.version for lib in yours()}
    changes = []
    for d in md.distributions(path=[str(stage)]):
        name = d.metadata["Name"]
        if canonical(name) in old:
            remove(name)                               # the old version, by its own file list
        changes.append((name, old.get(canonical(name), ""), d.version))
    for item in stage.iterdir():
        dest = root / item.name
        if dest.exists():                              # a file shared with nothing we know: replace it
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(stage, ignore_errors=True)
    return changes


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


## Removing ##

def remove(name: str) -> list[str]:
    """Delete one of your libraries (its files listed in its RECORD, and its metadata folder).

    Returns:
        list[str]: The top-level folders/files removed (for the log).

    Raises:
        KeyError: If no library of that name is in your folder.
    """
    root = folder()
    dist = next((d for d in md.distributions(path=[str(root)]) if canonical(d.metadata["Name"]) == canonical(name)),
                None)
    if dist is None:
        raise KeyError(name)
    tops = set()
    for f in dist.files or []:
        parts = Path(str(f)).parts
        if parts and parts[0] not in ("..",):
            tops.add(parts[0])
    removed = []
    for top in sorted(tops):
        p = root / top
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(top)
        elif p.exists():
            p.unlink()
            removed.append(top)
    for p in root.glob("*.dist-info"):                  # in case RECORD did not list it
        if canonical(p.name.split("-")[0]) == canonical(name):
            shutil.rmtree(p, ignore_errors=True)
    return removed
