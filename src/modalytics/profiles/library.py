"""
The profile library: the profiles on this computer, one YAML file each.

Profiles live in the user library (``paths.user_dir("profiles")``). The ones
Modalytics ships with (``resources/profiles``) are copied there on first start
and are then ordinary, editable profiles: nothing is locked. "Reset to
built-in" copies the shipped version back over an edited one.

A profile's *id* is its file name without ``.yaml`` (e.g. ``colonic_tissue``).
It stays the same when the profile is renamed, so projects and other profiles
can refer to it.

Writes are atomic (``.tmp`` + ``os.replace``) like project saves.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from modalytics import fileio, paths
from modalytics.profiles.model import Profile, ProfileError

SUFFIX = ".yaml"
_INSTALLED = ".builtins.json"     # built-in ids already copied once (so a deleted one stays deleted)


@dataclass
class Entry:
    """A profile in the library.

    Attributes:
        id: File name without ``.yaml``.
        path: The file.
        profile: The parsed profile, or None if the file cannot be read.
        error: Why it cannot be read ("" if it can).
    """

    id: str
    path: Path
    profile: Profile | None
    error: str = ""


def slug(name: str) -> str:
    """File-safe id from a name: 'Colonic tissue (v2)' → 'colonic_tissue_v2'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "profile"


class ProfileLibrary:
    """The profiles in one folder (by default the user library).

    Args:
        folder: Where the profile files are (default: the user library).
        builtin: Where the shipped profiles are (default: the resources).
    """

    def __init__(self, folder: str | os.PathLike | None = None, builtin: str | os.PathLike | None = None) -> None:
        self.folder = Path(folder) if folder is not None else paths.user_dir("profiles")
        self.builtin = Path(builtin) if builtin is not None else paths.builtin_dir("profiles")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.install_builtins()

    # 1. Built-ins
    def builtin_ids(self) -> list[str]:
        return sorted(p.stem for p in self.builtin.glob(f"*{SUFFIX}")) if self.builtin.is_dir() else []

    def install_builtins(self) -> list[str]:
        """Copy shipped profiles that were never installed here (first start, or a new release).

        Returns:
            list[str]: Ids installed now.
        """
        marker = self.folder / _INSTALLED
        try:
            done = set(json.loads(marker.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            done = set()
        new = [i for i in self.builtin_ids() if i not in done]
        for pid in new:
            if not self.path(pid).exists():
                self._write_text(self.path(pid), (self.builtin / f"{pid}{SUFFIX}").read_text(encoding="utf-8"))
        if new:
            self._write_text(marker, json.dumps(sorted(done | set(new))))
        return new

    def is_builtin(self, pid: str) -> bool:
        """True if Modalytics ships a profile with this id (it may have been edited here)."""
        return (self.builtin / f"{pid}{SUFFIX}").exists()

    def reset_builtin(self, pid: str) -> Profile:
        """Replace the library copy of a shipped profile by the shipped version.

        Raises:
            KeyError: If no shipped profile has this id.
        """
        if not self.is_builtin(pid):
            raise KeyError(pid)
        text = (self.builtin / f"{pid}{SUFFIX}").read_text(encoding="utf-8")
        self._write_text(self.path(pid), text)
        return Profile.from_yaml(text)

    # 2. Reading
    def path(self, pid: str) -> Path:
        return self.folder / f"{pid}{SUFFIX}"

    def entries(self, kind: str | None = None) -> list[Entry]:
        """Every profile file (unreadable ones too, with the reason), sorted by name.

        Args:
            kind: Only profiles of this kind (unreadable files are then left out).
        """
        out = []
        for p in sorted(self.folder.glob(f"*{SUFFIX}")):
            try:
                out.append(Entry(p.stem, p, Profile.from_yaml(p.read_text(encoding="utf-8"))))
            except (OSError, ProfileError) as exc:
                out.append(Entry(p.stem, p, None, str(exc)))
        if kind is not None:
            out = [e for e in out if e.profile is not None and e.profile.kind == kind]
        return sorted(out, key=lambda e: (e.profile.name.lower() if e.profile else e.id))

    def get(self, pid: str) -> Profile:
        """Read one profile.

        Raises:
            KeyError: If there is no such profile.
            ProfileError: If its file cannot be used.
        """
        p = self.path(pid)
        if not p.exists():
            raise KeyError(pid)
        return Profile.from_yaml(p.read_text(encoding="utf-8"))

    def find(self, name: str) -> str | None:
        """Id of the profile with this name (case-insensitive), or None."""
        return next((e.id for e in self.entries() if e.profile and e.profile.name.lower() == name.lower()), None)

    def unique_name(self, name: str) -> str:
        """``name``, or ``name (2)``, ``name (3)``… whichever no profile uses yet."""
        taken = {e.profile.name.lower() for e in self.entries() if e.profile}
        base, n, out = name, 2, name
        while out.lower() in taken:
            out, n = f"{base} ({n})", n + 1
        return out

    # 3. Writing
    def save(self, profile: Profile, pid: str | None = None) -> str:
        """Write a profile (new or edited); its revision goes up by one when it changed.

        Args:
            profile: The profile.
            pid: Its id; None for a new profile (made from its name, kept unique).

        Returns:
            str: The id.
        """
        if pid is None:
            pid, n = slug(profile.name), 2
            base = pid
            while self.path(pid).exists():
                pid, n = f"{base}_{n}", n + 1
        elif self.path(pid).exists():
            try:
                old = self.get(pid)
                if old.to_dict() != profile.to_dict():
                    profile.revision = old.revision + 1
            except ProfileError:
                pass
        self._write_text(self.path(pid), profile.to_yaml())
        return pid

    def duplicate(self, pid: str, name: str | None = None) -> str:
        """Copy a profile under a new name ("<name> (copy)" by default). Returns the new id."""
        p = self.get(pid)
        p.name = self.unique_name(name or f"{p.name} (copy)")
        p.revision = 1
        return self.save(p)

    def delete(self, pid: str) -> None:
        """Remove a profile file (a shipped one can be brought back with ``reset_builtin``)."""
        self.path(pid).unlink(missing_ok=True)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        fileio.write_text(path, text)


DEFAULT_ID = "colonic_tissue"


def default_profile(library: ProfileLibrary | None = None) -> tuple[Profile, str]:
    """The profile used when none was chosen: the demo, HoloPath imports, project files
    from before profiles existed. It is Colonic tissue, from the library if it is still
    there (as edited here), else as shipped.

    Returns:
        tuple: (a copy of the profile, its library id or "" if it came from the shipped file).
    """
    library = library or ProfileLibrary()
    try:
        return library.get(DEFAULT_ID), DEFAULT_ID
    except (KeyError, ProfileError):
        text = (library.builtin / f"{DEFAULT_ID}{SUFFIX}").read_text(encoding="utf-8")
        return Profile.from_yaml(text), ""
