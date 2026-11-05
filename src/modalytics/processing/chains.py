"""
Processing chains: named lists of processing steps, each working on the previous one's result (Qt-free).

    "Denoised OD" = Median filter (1 µm) → OD map            measure: yes

Applied to an image, a chain gives one view (``core.model.ImageView``): the last
step's result, computed tile by tile through every step (``run.ProcessedSource``
reading the previous step's pixels, margins included). A chain marked
*measure* feeds the parameters of its result's type; otherwise it is only for
looking. A single step applied from the Process menu is a chain of one.

Chains live in three places:

* **Your chains** (``ChainLibrary``): YAML files in the user library
  (``paths.user_dir("chains")``), made in the chain editor and applied from the
  Process tab to one image or to all images of a type.
* **Profiles** (``Profile.processing``): applied at import; their views are the
  *standard* ones, whose measured parameters are the plain results.
* **Projects**: each view keeps its chain (``run.recipes``).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from modalytics import paths
from modalytics.processing.registry import ProcessingLibrary


@dataclass
class Chain:
    """A named list of processing steps.

    Attributes:
        name: Shown in Layers and in results (e.g. "Denoised OD").
        steps: [{"step": step id, "options": {…}}], applied in order.
        measure: Its result feeds the parameters of its type (else: only for looking).
        short: Its toolbar button's label (empty: its name).
        icon: Its toolbar button's icon, a Material Design icon name (empty: a chain icon).
    """

    name: str
    steps: list[dict] = field(default_factory=list)
    measure: bool = True
    short: str = ""
    icon: str = ""

    @property
    def button_label(self) -> str:
        return self.short or self.name

    @property
    def button_icon(self) -> str:
        from modalytics.processing.registry import CHAIN_ICON
        return self.icon or CHAIN_ICON

    def __post_init__(self) -> None:
        # Every step as {"step", "options"} (options may be left out when written by hand)
        self.steps = [{"step": str(s["step"]), "options": dict(s.get("options") or {})} for s in self.steps]

    def same_steps(self, other: "Chain", library: ProcessingLibrary) -> bool:
        """Whether two chains do the same (same steps, options compared with their defaults filled in)."""
        def resolved(chain):
            out = []
            for s in chain.steps:
                spec = library.get(s["step"])
                out.append((s["step"], spec.resolve(s["options"]) if spec else s["options"]))
            return out
        return resolved(self) == resolved(other)

    def to_dict(self) -> dict:
        d = {"name": self.name, "steps": [{"step": s["step"], "options": dict(s.get("options") or {})}
                                          for s in self.steps], "measure": bool(self.measure)}
        if self.short:
            d["short"] = self.short
        if self.icon:
            d["icon"] = self.icon
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        """A chain from its file form; an older single-step entry ({"step", "options"}) is a chain of one.

        Raises:
            ValueError: If the data is not a chain.
        """
        if not isinstance(d, dict):
            raise ValueError("not a processing chain")
        if "steps" not in d and "step" in d:              # one step, as profiles first stored them
            return cls(str(d.get("name") or d["step"].split(":")[-1]),
                       [{"step": str(d["step"]), "options": dict(d.get("options") or {})}],
                       bool(d.get("measure", True)))
        steps = [{"step": str(s["step"]), "options": dict(s.get("options") or {})} for s in d.get("steps") or []]
        return cls(str(d.get("name") or "Chain"), steps, bool(d.get("measure", True)), str(d.get("short") or ""),
                   str(d.get("icon") or ""))

    def copy(self) -> "Chain":
        return copy.deepcopy(self)

    # ------------------------------------------------------------ against the step library
    def output_type(self, input_type: str, library: ProcessingLibrary) -> str:
        """The image type it makes from one of ``input_type``.

        Raises:
            ValueError: If a step is missing or does not accept what the previous one makes.
        """
        current = input_type
        for i, s in enumerate(self.steps, start=1):
            spec = library.get(s["step"])
            if spec is None:
                raise ValueError(f"step {i} ('{s['step']}') is not available here")
            if not spec.accepts(current):
                raise ValueError(f"step {i} ({spec.name}) does not process {current} images")
            current = spec.output_for(current)
        return current

    def problems(self, input_type: str, library: ProcessingLibrary) -> list[str]:
        """What prevents applying it to an image of ``input_type`` (empty = fine)."""
        if not self.steps:
            return [f"“{self.name}” has no steps"]
        try:
            self.output_type(input_type, library)
            return []
        except ValueError as exc:
            return [f"“{self.name}”: {exc}"]

    def summary(self, library: ProcessingLibrary | None = None) -> str:
        """The steps in words, e.g. "Median filter (denoise) → OD map"."""
        names = []
        for s in self.steps:
            spec = library.get(s["step"]) if library is not None else None
            names.append(spec.name if spec else s["step"].split(":")[-1])
        return " → ".join(names) if names else "(no steps)"


class ChainLibrary:
    """Your saved chains: one YAML file each in ``paths.user_dir("chains")``.

    Attributes:
        folder: Where they are stored.
    """

    def __init__(self, folder: str | Path | None = None) -> None:
        self.folder = Path(folder) if folder is not None else paths.user_dir("chains")

    def _file(self, name: str) -> Path:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_") or "chain"
        return self.folder / f"{slug}.yaml"

    def chains(self) -> list[Chain]:
        """Every saved chain, by name (files that cannot be read are skipped)."""
        out = []
        for f in sorted(self.folder.glob("*.yaml")) if self.folder.is_dir() else []:
            try:
                out.append(Chain.from_dict(yaml.safe_load(f.read_text(encoding="utf-8"))))
            except (OSError, ValueError, yaml.YAMLError, KeyError, TypeError):
                continue
        return sorted(out, key=lambda c: c.name.lower())

    def get(self, name: str) -> Chain | None:
        return next((c for c in self.chains() if c.name == name), None)

    def save(self, chain: Chain) -> Path:
        """Write a chain (replacing one with the same name)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._file(chain.name)
        path.write_text("# Modalytics processing chain\n" + yaml.safe_dump(chain.to_dict(), sort_keys=False,
                                                                           allow_unicode=True), encoding="utf-8")
        return path

    def delete(self, name: str) -> None:
        for f in self.folder.glob("*.yaml") if self.folder.is_dir() else []:
            try:
                if Chain.from_dict(yaml.safe_load(f.read_text(encoding="utf-8"))).name == name:
                    f.unlink()
            except (OSError, ValueError, yaml.YAMLError, KeyError, TypeError):
                continue


# ---------------------------------------------------------------- chains and profiles
def input_types(chain: Chain, library: ProcessingLibrary) -> list[str]:
    """The imported image types a chain can be applied to (its steps fit them all the way)."""
    from modalytics.core.model import ACQUIRED_TYPES
    return [t for t in ACQUIRED_TYPES if not chain.problems(t, library)]


def profiles_using(name: str, profiles) -> list[tuple[str, str, str, Chain]]:
    """The profiles holding a chain called ``name`` (their own copy).

    Args:
        name: The chain's name.
        profiles: A ``profiles.library.ProfileLibrary``.

    Returns:
        list: (profile id, profile name, image type, the profile's copy of the chain).
    """
    out = []
    for entry in profiles.entries():
        if entry.profile is None:
            continue
        for image_type in entry.profile.processing:
            for c in entry.profile.chains_for(image_type):
                if c.name == name:
                    out.append((entry.id, entry.profile.name, image_type, c))
    return out


def update_profiles(chain: Chain, targets: list[tuple[str, str]], profiles) -> int:
    """Replace the copies of ``chain`` (same name) in these profiles; each keeps its own *measured* choice.

    Args:
        chain: The chain as it is now.
        targets: (profile id, image type) pairs to update.
        profiles: A ``profiles.library.ProfileLibrary``.

    Returns:
        int: How many copies were replaced.
    """
    n = 0
    for pid in {p for p, _t in targets}:
        profile = profiles.get(pid)
        for image_type in {t for p, t in targets if p == pid}:
            chains = profile.chains_for(image_type)
            for i, c in enumerate(chains):
                if c.name == chain.name:
                    new = chain.copy()
                    new.measure = c.measure
                    chains[i] = new
                    n += 1
            profile.processing[image_type] = [c.to_dict() for c in chains]
        profiles.save(profile, pid)
    return n
