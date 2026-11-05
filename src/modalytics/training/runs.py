"""
Training runs on disk (Qt-free): where a run's report, results and models go, and reading them back.

    training_runs/
      Colonic Glands Simplified/                        the training set (``TrainingSet.runs_folder``)
        segmentation/                                   the kind of models (classification: later)
          2026-10-25 01-14 Gland on phase map/          date and time first: the names sort by date
            Report - Gland on phase map - 2026-10-25 01-14.html
            Results - Gland on phase map - 2026-10-25 01-14.json    (``compare.Comparison``: every count)
            models/
              unet_resnet34__imagenet.mmodel …
          evaluations/                                  existing models judged (Training › Evaluate; no models/:
            2026-10-26 14-02 Released against kept/     the results name the models' files)
              Report - … .html, Results - … .json

The results store the models' paths relative to the run's folder, so a run can be moved or
copied whole. Names are shortened (``NAME_CHARS``) and stripped of the characters Windows
does not allow, so the paths stay well under its 260 characters. Renaming a training set
renames its folder (``rename_set``); the set remembers its folder, so two sets never share one.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from modalytics import paths

NAME_CHARS = 36                       # a set's or run's name in folder and file names (longer: shortened)
STAMP = "%Y-%m-%d %H-%M"              # in folder and file names (no ":" on Windows)
KINDS = ("segmentation", "classification")
EVALUATIONS = "segmentation/evaluations"     # the kind of an evaluation's folder (Training › Evaluate)
LEGACY_RESULTS, LEGACY_REPORT = "comparison.json", "report.html"    # runs written before this layout


def root() -> Path:
    """The folder of every training set's runs (in the user library)."""
    return paths.user_dir("training_runs")


def safe_name(text: str, n: int = NAME_CHARS) -> str:
    """``text`` as a folder or file name: the characters Windows forbids replaced, at most ``n`` characters."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", text)
    s = re.sub(r"\s+", " ", s).strip()[:n].rstrip(" .")
    return s or "Untitled"


def set_folder(tset, create: bool = False) -> Path:
    """The training set's folder of runs: the one it remembers, else one named after it (not yet claimed).

    Args:
        tset: The ``TrainingSet``.
        create: Claim it (``tset.runs_folder``, unique among the sets: " (2)"… if the name is taken) and make it.
            The caller saves the set.
    """
    if tset.runs_folder:
        folder = root() / tset.runs_folder
    else:
        stem = safe_name(tset.name)
        folder, n = root() / stem, 2
        if create:
            while folder.exists():                      # another set's (sets may share a name)
                folder, n = root() / f"{stem} ({n})", n + 1
            tset.runs_folder = folder.name
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def rename_set(tset, new_name: str) -> None:
    """The set is renamed: its folder of runs follows when the new name is free (else it keeps its folder)."""
    if not tset.runs_folder:
        return
    old, new = root() / tset.runs_folder, root() / safe_name(new_name)
    if old == new or not old.exists() or new.exists():
        return
    try:
        old.rename(new)
    except OSError:                                     # a file of it open elsewhere: kept where it is
        return
    tset.runs_folder = new.name


@dataclass
class Run:
    """One training run's folder.

    Attributes:
        folder: Its folder (``<set>/<kind>/<date time> <name>``).
        name: The run's name (from the folder; the results hold the full one).
        when: When it started (from the folder; None for a folder named otherwise).
    """

    folder: Path
    name: str
    when: dt.datetime | None = None

    @classmethod
    def new(cls, set_dir: Path, kind: str, name: str, when: dt.datetime | None = None) -> "Run":
        """A new run's place (not created yet: ``compare.run`` makes it)."""
        when = (when or dt.datetime.now()).replace(second=0, microsecond=0)
        base = set_dir / kind
        folder, n = base / f"{when.strftime(STAMP)} {safe_name(name)}", 2
        while folder.exists():                          # two runs of one name in the same minute
            folder, n = base / f"{when.strftime(STAMP)} {safe_name(name)} ({n})", n + 1
        return cls(folder, safe_name(name), when)

    @classmethod
    def of(cls, folder: Path) -> "Run":
        """The run in ``folder`` (its date and name read from the folder's name)."""
        folder = Path(folder)
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}) (.*)$", folder.name)
        when = None
        if m:
            try:
                when = dt.datetime.strptime(m.group(1), STAMP)
            except ValueError:
                pass
        return cls(folder, m.group(2) if m else folder.name, when)

    def _file(self, prefix: str, suffix: str, legacy: str) -> Path:
        found = sorted(self.folder.glob(f"{prefix} - *{suffix}"))
        if found:
            return found[0]
        if (self.folder / legacy).exists():
            return self.folder / legacy
        stamp = f" - {self.when.strftime(STAMP)}" if self.when else ""
        return self.folder / f"{prefix} - {self.name}{stamp}{suffix}"

    @property
    def report(self) -> Path:
        """The HTML report (``report.comparison``)."""
        return self._file("Report", ".html", LEGACY_REPORT)

    @property
    def results(self) -> Path:
        """The results (``compare.Comparison``): settings, projects, every model's curves and counts."""
        return self._file("Results", ".json", LEGACY_RESULTS)

    @property
    def models(self) -> Path:
        """The folder of the run's models."""
        return self.folder / "models"

    @property
    def label(self) -> str:
        """How the run is listed: its date and time, then its name."""
        return (f"{self.when:%Y-%m-%d %H:%M} — " if self.when else "") + self.name


def list_runs(tset, kind: str = "segmentation") -> list[Run]:
    """The training set's runs of one kind that have results, the newest first."""
    base = set_folder(tset) / kind
    if not base.is_dir():
        return []
    out = [Run.of(d) for d in base.iterdir() if d.is_dir()]
    out = [r for r in out if r.results.exists()]
    # The newest first: by the minute in the name, then (two in one minute) by when their results were written
    return sorted(out, key=lambda r: (r.when or dt.datetime.fromtimestamp(r.folder.stat().st_mtime),
                                      r.results.stat().st_ctime), reverse=True)


def load(run: Run):
    """The run's results (``compare.Comparison``), the models' paths made absolute again."""
    from modalytics.training.compare import Comparison
    return Comparison.load(run.results)
