"""
Where Modalytics keeps its files.

Two places:

* **Built-in resources** ship with the program (``src/modalytics/resources``):
  the profiles, parameters and importers Modalytics comes with. They are templates,
  never edited in place.
* **The user library** belongs to the person using the computer: profiles,
  parameters, importers and models they made, imported or edited. On first start the
  built-in profiles are copied here, so everyone can edit them like any other
  (see ``profiles.library``).

    Windows   %APPDATA%\\Modalytics\\
    Linux     $XDG_DATA_HOME/modalytics/  (default ~/.local/share/modalytics/)
    macOS     ~/Library/Application Support/Modalytics/

Environment override: ``MODALYTICS_HOME`` puts the whole user library in
another folder (tests use a fresh temporary one).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"


CLI_EXE = "modalytics-cli.exe"        # the installed app's console twin (packaging/modalytics.spec)


def installed() -> bool:
    """True in Modalytics as installed (a frozen build: the installer's), False when run from the source — what a
    message may tell the user to do differently (reinstall, rather than ``pip install``)."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Where the program's own files are: the installed bundle (PyInstaller's ``sys._MEIPASS``), else the
    repository (``models/``, ``docs/`` beside ``src/``)."""
    if installed():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def helper_command() -> list[str]:
    """The command that runs Modalytics without a window, for its child processes (GPU probes, pip): installed,
    the console twin beside the app (its output can be read, and it never flashes a window: started with
    ``CREATE_NO_WINDOW``); from the source, ``python -m modalytics``."""
    if installed():
        return [str(Path(sys.executable).with_name(CLI_EXE))]
    return [sys.executable, "-m", "modalytics"]


def user_home() -> Path:
    """Root of the user library (created if missing)."""
    if os.environ.get("MODALYTICS_HOME"):
        root = Path(os.environ["MODALYTICS_HOME"])
    elif sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Modalytics"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "Modalytics"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "modalytics"
    root.mkdir(parents=True, exist_ok=True)
    return root


def user_dir(kind: str) -> Path:
    """One folder of the user library ("profiles", "parameters", "importers", "processing", "models"), created
    if missing."""
    path = user_home() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def builtin_dir(kind: str) -> Path:
    """One folder of the built-in resources ("profiles", "parameters", "importers", "processing", "scripts")."""
    return RESOURCES / kind
