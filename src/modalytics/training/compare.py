"""
Comparing segmentation architectures on the same data (Qt-free).

Each chosen architecture is trained the same way (the same settings, training projects
and seed), from each chosen starting point of its encoder (ImageNet, MicroNet, your own:
``training.encoders``; a pair that does not fit is left out), its best epoch chosen on the evaluation
projects, and then judged on the **test** projects when there are any (never used for
training or choosing), else on the evaluation projects — said plainly in the report,
since choosing on a set and reporting it flatters the winner. Every model is measured
the same way (``training.segmentation``), timed with ONNX Runtime as Segment runs it,
and sized. The software **proposes** the best by the criterion chosen; you decide.

**Confidence intervals** come from the bootstrap over images (Efron 1979): the images
are drawn with replacement ``BOOTSTRAP`` times and the measure recomputed from their
pooled counts; the 2.5th and 97.5th percentiles bound the 95 % interval. Two models
are compared on the **same** draws (a paired bootstrap): the interval of their
difference says whether one is clearly better (it does not contain 0) or not.

**Evaluations** (``evaluate_models``, Training › Evaluate) judge models that already exist —
kept from a run, released, or another person's file — the same way, without training
them: on the test, evaluation or all the projects, each at its own resolution. Each one's
origin is read from its file (architecture, starting weights, when and on how many images
it was trained, and on which projects when it says: judging a model on projects it learned
from flatters it, and the report says so).

References:
    Efron B. Bootstrap Methods: Another Look at the Jackknife. Ann Stat. 1979;7(1):1-26.
        doi:10.1214/aos/1176344552
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from modalytics.training.segmentation import Counts

BOOTSTRAP = 2000

# key: (label, measure of pooled counts, higher is better, format)
METRICS = {
    "f1": ("Object F1", lambda c: c.f1, True, "{:.3f}"),
    "glas_f1": ("Object F1 (GlaS)", lambda c: c.glas_f1, True, "{:.3f}"),
    "pq": ("Panoptic quality (PQ)", lambda c: c.pq, True, "{:.3f}"),
    "object_dice": ("Object Dice (GlaS)", lambda c: c.object_dice, True, "{:.3f}"),
    "dice": ("Pixel Dice", lambda c: c.dice, True, "{:.3f}"),
    "iou": ("Pixel IoU", lambda c: c.iou, True, "{:.3f}"),
    "precision": ("Object precision", lambda c: c.precision, True, "{:.3f}"),
    "recall": ("Object recall", lambda c: c.recall, True, "{:.3f}"),
    "sq": ("Matched IoU (SQ)", lambda c: c.sq, True, "{:.3f}"),
    "object_hausdorff": ("Object Hausdorff (µm)", lambda c: c.object_hausdorff, False, "{:.1f}"),
    "hd95": ("HD95 of matches (µm)", lambda c: c.hd95, False, "{:.1f}"),
    "nsd": ("NSD of matches", lambda c: c.nsd, True, "{:.3f}"),
}
CRITERIA = ("f1", "pq", "object_dice", "dice")       # what "best" can be judged by


# ---------------------------------------------------------------- statistics
def pooled(counts: list) -> Counts:
    out = Counts()
    for c in counts:
        out.add(c)
    return out


def _array(counts: list) -> tuple[np.ndarray, list[str]]:
    keys = list(Counts.__dataclass_fields__)
    return np.array([[getattr(c, k) for k in keys] for c in counts], dtype=np.float64), keys


def _draws(n_images: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_images, size=(n, n_images))


def bootstrap(counts: list, measure: Callable, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """95 % interval of ``measure`` over images (the images drawn with replacement, their counts pooled)."""
    if len(counts) < 2:
        return math.nan, math.nan
    a, keys = _array(counts)
    values = []
    for row in a[_draws(len(counts), n, seed)].sum(axis=1):
        v = measure(Counts(**dict(zip(keys, row))))
        if v is not None and math.isfinite(v):
            values.append(v)
    if not values:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_difference(a: list, b: list, measure: Callable, n: int = BOOTSTRAP,
                      seed: int = 0) -> tuple[float, float, float]:
    """(difference a − b, its 95 % interval) on the same images drawn for both (a paired bootstrap)."""
    if len(a) != len(b) or len(a) < 2:
        return math.nan, math.nan, math.nan
    xa, keys = _array(a)
    xb, _k = _array(b)
    draws = _draws(len(a), n, seed)
    diffs = []
    for ra, rb in zip(xa[draws].sum(axis=1), xb[draws].sum(axis=1)):
        va, vb = measure(Counts(**dict(zip(keys, ra)))), measure(Counts(**dict(zip(keys, rb))))
        if math.isfinite(va) and math.isfinite(vb):
            diffs.append(va - vb)
    d = measure(pooled(a)) - measure(pooled(b))
    if not diffs:
        return d, math.nan, math.nan
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(d), float(lo), float(hi)


# ---------------------------------------------------------------- the comparison
@dataclass
class Planned:
    """One model of a training run.

    Attributes:
        kind: "new" (an architecture from starting weights) or "finetune" (an existing model trained further).
        arch_id, weights: A new model's architecture and the encoder weights it starts from ("scratch": none).
        base: A fine-tuned model's package id.
    """

    kind: str = "new"
    arch_id: str = ""
    weights: str = "imagenet"
    base: str = ""

    @property
    def run_id(self) -> str:
        return f"ft:{self.base}" if self.kind == "finetune" else f"{self.arch_id}+{self.weights}"

    @classmethod
    def of(cls, item) -> "Planned":
        """From a Planned, a dict ({"kind", "arch_id", "weights", "base"}), an (architecture, weights) pair or an
        architecture key (from ImageNet)."""
        if isinstance(item, Planned):
            return item
        if isinstance(item, dict):
            return cls(**{k: v for k, v in item.items() if k in ("kind", "arch_id", "weights", "base")})
        if isinstance(item, str):
            return cls("new", item, "imagenet")
        return cls("new", item[0], item[1])


@dataclass
class Entry:
    """One model of a run: a new one (an architecture from starting weights), a fine-tuned one, or a fine-tuned
    model's starting point as it was (``kind`` "reference": not retrained, judged the same way).

    Attributes:
        arch_id, label: Which architecture; the model's label.
        weights, weights_name: What it started from: encoder weights (key, name), or "model:<id>" (a model).
        kind, base: "new", "finetune" or "reference"; the starting model's id (fine-tuned and reference).
        um_per_px: The resolution it works at (a fine-tuned model keeps its base's).
        path: Its model package ("" if training failed).
        history: Per epoch: {"epoch", "loss", "dice", "seconds", "best"}.
        best_epoch: The epoch kept (best evaluation Dice).
        seconds: Training time.
        params_m: Parameters (millions).
        size_mb: Package size (MB).
        ms_per_tile: ONNX Runtime time per 512 × 512 tile on ``device``.
        device: Where the speed was measured.
        evaluation, test: ``training.segmentation.Evaluation`` on those projects (None: none, or failed; an
            evaluation's judgement on the evaluation or all the projects is in ``evaluation``).
        error: Why it failed.
        origin: An existing model's (``kind`` "existing", an evaluation): what its file says of it (``origin``).
    """

    arch_id: str
    label: str
    path: str = ""
    history: list = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0
    params_m: float = math.nan
    size_mb: float = math.nan
    ms_per_tile: float = math.nan
    device: str = ""
    evaluation: object = None
    test: object = None
    error: str = ""
    weights: str = "imagenet"
    weights_name: str = "ImageNet"
    kind: str = "new"
    base: str = ""
    um_per_px: float = math.nan
    origin: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Unique in a run: the architecture and its starting weights; a fine-tuned model's base (as it was); an
        existing model's file."""
        if self.kind == "existing":
            return f"file:{self.path}"
        if self.kind == "finetune":
            return f"ft:{self.base}"
        if self.kind == "reference":
            return f"ref:{self.base}"
        return f"{self.arch_id}+{self.weights}"

    def judged(self, on: str):
        return self.test if on == "test" else self.evaluation


@dataclass
class Comparison:
    """A comparison of architectures (see the module docstring)."""

    name: str
    group: str
    modality: str
    um_per_px: float
    settings: dict
    projects: dict                       # {"training": [names], "evaluation": [...], "test": [...]}
    area: str
    margin_um: float
    iou_min: float
    criterion: str
    entries: list = field(default_factory=list)
    started: str = ""
    finished: str = ""
    versions: dict = field(default_factory=dict)
    training_device: str = ""
    skipped: list = field(default_factory=list)
    training_set: str = ""
    data: dict = field(default_factory=dict)  # {role: seg_train.data_counts} — what each role holds, for the report
    purpose: str = "training"                 # "training" (models trained, ``run``) or "evaluation" (``evaluate_models``)
    judged: str = ""                          # an evaluation's projects: "test", "evaluation" or "all"
    test_group: str = ""                      # judged on "test": the test group ("" = every test project)
    test_groups: dict = field(default_factory=dict)   # {test group: [project names]} — each a final check apart

    @property
    def judged_on(self) -> str:
        """The projects judged on: a training run's test projects when there are any, else its evaluation ones; an
        evaluation's chosen ones."""
        if self.judged:
            return self.judged
        return "test" if self.projects.get("test") else "evaluation"

    def done(self) -> list:
        return [e for e in self.entries if not e.error and e.judged(self.judged_on) is not None]

    def score(self, entry: Entry, key: str | None = None) -> float:
        ev = entry.judged(self.judged_on)
        return METRICS[key or self.criterion][1](ev.total) if ev is not None else math.nan

    def best(self) -> Entry | None:
        """The proposal: the best by the criterion (a better tie-breaker: faster)."""
        _label, _fn, higher, _f = METRICS[self.criterion]
        done = [e for e in self.done() if math.isfinite(self.score(e))]
        if not done:
            return None
        return max(done, key=lambda e: ((1 if higher else -1) * self.score(e), -e.ms_per_tile))

    def to_dict(self, base: str | os.PathLike | None = None) -> dict:
        """Plain data (the per-image counts included, the example pictures not).

        Args:
            base: The models' paths inside this folder are stored relative to it (the run's folder: it can then be
                moved whole); others (an installed model, as it was) stay absolute.
        """
        def rel(path: str) -> str:
            if not path or base is None:
                return path
            try:
                return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
            except ValueError:
                return path

        def ev(e):
            if e is None:
                return None
            return {"projects": e.projects, "paths": list(e.paths), "total": asdict(e.total),
                    "per_image": [{"project": e.projects[p], "image": n, "counts": asdict(c)} for p, n, c in e.per_image],
                    "items": [asdict(i) for i in e.items], "skipped": e.skipped}
        d = {k: v for k, v in asdict(self).items() if k != "entries"}
        d["entries"] = [dict({k: v for k, v in asdict(e).items() if k not in ("evaluation", "test")},
                             path=rel(e.path), evaluation=ev(e.evaluation), test=ev(e.test)) for e in self.entries]
        return d

    def save(self, path: str | os.PathLike) -> None:
        """Written to ``path`` (the models' paths relative to its folder)."""
        from modalytics import fileio
        fileio.write_text(path, json.dumps(self.to_dict(Path(path).parent), indent=1, default=float))

    @classmethod
    def from_dict(cls, d: dict, base: str | os.PathLike | None = None) -> "Comparison":
        """Read back from ``to_dict`` (the counts per image; the example pictures are only in the report).

        Args:
            base: The folder relative paths are read from (the run's folder).
        """
        from modalytics.training.segmentation import Evaluation, Item
        keys = set(Counts.__dataclass_fields__)
        counts = lambda c: Counts(**{k: v for k, v in (c or {}).items() if k in keys})
        item_keys = set(Item.__dataclass_fields__)

        def ev(x):
            if not x:
                return None
            names = list(x.get("projects", []))
            per_image = [(names.index(p["project"]) if p.get("project") in names else -1, p.get("image", ""),
                          counts(p.get("counts"))) for p in x.get("per_image", [])]
            per_project = [Counts() for _n in names]               # rebuilt from the images (not stored twice)
            for p, _n, c in per_image:
                if p >= 0:
                    per_project[p].add(c)
            items = [Item(**{k: (tuple(v) if k == "point" else v) for k, v in i.items() if k in item_keys})
                     for i in x.get("items", [])]
            return Evaluation(names, list(x.get("paths", [])), per_project, counts(x.get("total")), items,
                              list(x.get("skipped", [])), {}, per_image)

        def path(p: str) -> str:
            if not p or base is None or Path(p).is_absolute():
                return p
            return str(Path(base) / p)
        fields = set(Entry.__dataclass_fields__) - {"evaluation", "test", "path"}
        entries = [Entry(**{k: v for k, v in e.items() if k in fields}, path=path(e.get("path", "")),
                         evaluation=ev(e.get("evaluation")), test=ev(e.get("test"))) for e in d.get("entries", [])]
        own = set(cls.__dataclass_fields__) - {"entries"}
        return cls(**{k: v for k, v in d.items() if k in own}, entries=entries)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Comparison":
        """Read a run's results file."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), Path(path).parent)


def versions() -> dict:
    """The software that trained and measured the models (for the report's reproducibility section)."""
    out = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}"}
    from modalytics import __version__
    out["modalytics"] = __version__
    for mod in ("torch", "segmentation_models_pytorch", "timm", "onnxruntime", "numpy", "shapely"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def speed(pkg, device: str = "auto", runs: int = 5) -> tuple[float, str]:
    """(ms per 512 × 512 tile, device name) with ONNX Runtime, as Segment runs the model."""
    from modalytics.services import compute
    from modalytics.services.segmentation import session
    sess = session(pkg, device)
    cin = int(pkg.manifest["in_channels"])
    x = np.random.default_rng(0).standard_normal((1, cin, 512, 512)).astype(np.float32)
    name = sess.get_inputs()[0].name
    sess.run(None, {name: x})                                   # warm-up (kernels compiled, memory placed)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / runs * 1000, compute.resolve(device).name


def parameters_m(path: str) -> float:
    """Parameters of a package's network (millions), from its weights."""
    import zipfile
    from safetensors.numpy import load
    from modalytics.training.seg_train import WEIGHTS
    with zipfile.ZipFile(path) as z:
        tensors = load(z.read(WEIGHTS))
    return sum(int(np.prod(t.shape)) for k, t in tensors.items() if not k.endswith("num_batches_tracked")) / 1e6


def plan(items: list) -> tuple[list[Planned], list[str]]:
    """The models a run can train: new ones whose starting weights fit their encoder, fine-tuned ones that carry
    their weights (``Planned.of`` reads each item).

    Returns:
        ([Planned], [what was left out and why]).
    """
    from modalytics.services.segmentation import find_models
    from modalytics.training import architectures as ar
    from modalytics.training import encoders as en
    from modalytics.training import seg_train
    archs, weights = ar.ArchitectureLibrary(), en.EncoderLibrary()
    models = None
    runs, left = [], []
    for item in items:
        p = Planned.of(item)
        if p.kind == "finetune":
            models = models if models is not None else {m.id: m for m in find_models()}
            m = models.get(p.base)
            if m is None:
                left.append(f"{p.base}: not installed")
            elif not seg_train.trainable(m, archs):
                left.append(f"{m.name}: no training weights (or its architecture is not here)")
            else:
                runs.append(p)
            continue
        spec = archs.get(p.arch_id)
        ws = weights.get(p.weights) if p.weights != "scratch" else None
        if spec is None:
            left.append(f"{p.arch_id}: not in the library")
        elif p.weights != "scratch" and ws is None:
            left.append(f"{p.weights}: not in the library")
        elif ws is not None and not ws.fits(spec.encoder):
            left.append(f"{spec.name} from {ws.name}: no weights for its encoder ({spec.encoder})")
        else:
            runs.append(p)
    return runs, left


def origin(pkg, judged: list[str] | None = None) -> dict:
    """What an existing model's file says of it, for an evaluation (``Entry.origin``).

    Args:
        pkg: The model package.
        judged: The projects it is judged on: those it learned from are named (``"learned_from_judged"``).

    Returns:
        {"file", "architecture", "weights", "um_per_px", "modality", "trained", "run", "images", "epochs",
         "best_epoch", "trained_on" (its training and evaluation projects, when its file names them: else None),
         "learned_from_judged"}.
    """
    src = pkg.manifest.get("source", {}) or {}
    tr = src.get("training", {}) or {}
    arch = src.get("architecture", {}) or {}
    named = "projects" in tr
    trained_on = list(tr.get("projects", [])) + list(tr.get("evaluation_projects", [])) if named else None
    return {"file": pkg.path, "architecture": arch.get("name") or src.get("architecture_id", ""),
            "weights": (src.get("encoder_weights") or {}).get("name", ""), "um_per_px": pkg.um_per_px,
            "modality": pkg.modality, "trained": src.get("trained", ""), "run": tr.get("run", ""),
            "images": tr.get("images"), "epochs": tr.get("epochs"), "best_epoch": tr.get("best_epoch"),
            "trained_on": trained_on,
            "learned_from_judged": sorted(set(trained_on) & set(judged or [])) if named else []}


def evaluate_models(model_paths: list[str], paths: list[str], on: str, group: str, name: str,
                    area: str = "marked", margin_um: float = 50.0, iou_min: float = 0.5, criterion: str = "f1",
                    device: str = "auto", progress: Callable[[dict], None] | None = None,
                    cancelled: Callable[[], bool] | None = None, results: str | os.PathLike | None = None,
                    training_set: str = "", test_group: str = "", test_groups: dict | None = None) -> Comparison:
    """Judge existing models on the same projects and compare them (an evaluation: nothing is trained).

    Each model runs at its own resolution on the images of its type; its outlines are cached (judging again with
    other settings is quick). A model that outlines another group than ``group`` is left out (said).

    Args:
        model_paths: The models' files (installed or anywhere).
        paths: The projects judged.
        on: Which they are: "test", "evaluation" or "all" (for the report).
        group: The class group outlined by hand.
        name: The evaluation's name.
        area, margin_um, iou_min: How the models are judged (``training.segmentation.evaluate``).
        criterion: The measure "best" is judged by (``CRITERIA``).
        device: Where the models run ("auto": the best here).
        progress: Called with {"stage", "fraction", "run" (``Entry.run_id``), "k", "n"}.
        results: The results file, saved after each model.
        training_set: The training set's name (for the report).
        test_group: On "test": the test group judged ("" = every test project; for the report).
        test_groups: On every test project: {test group: [project names]}, for the report's per-group results.

    Returns:
        The ``Comparison`` (``purpose`` "evaluation").
    """
    from modalytics.services.segmentation import load_package
    from modalytics.training import seg_train
    from modalytics.training.segmentation import evaluate
    report = progress or (lambda _d: None)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    pkgs = []
    for path in model_paths:
        try:
            pkgs.append((path, load_package(path), ""))
        except Exception as exc:                                  # an unreadable file: said in the report
            pkgs.append((path, None, f"the model could not be read: {exc}"))
    modality = next((p.modality for _f, p, _e in pkgs if p is not None), "")
    named = [p.name for _f, p, _e in pkgs if p is not None]
    label = lambda p: p.name + (f" ({os.path.basename(p.path)})" if named.count(p.name) > 1 else "")   # told apart
    comp = Comparison(name, group, modality, math.nan, {}, {on: names}, area, margin_um, iou_min, criterion,
                      started=dt.datetime.now().isoformat(timespec="seconds"), versions=versions(),
                      training_set=training_set, purpose="evaluation", judged=on, test_group=test_group,
                      test_groups=dict(test_groups or {}))
    report({"stage": "Counting the images and objects judged…", "fraction": 0.0})
    comp.data = {on: seg_train.data_counts(paths, group, modality, area, margin_um)} if modality else {}
    n = len(pkgs)
    for k, (path, pkg, error) in enumerate(pkgs):
        if pkg is None:
            comp.entries.append(Entry("", os.path.basename(path), path=path, kind="existing", error=error))
            continue
        src = pkg.manifest.get("source", {}) or {}
        weights = src.get("encoder_weights") or {}
        entry = Entry(src.get("architecture_id", ""), label(pkg), path=pkg.path, kind="existing", base=pkg.id,
                      weights=weights.get("key", ""), weights_name=weights.get("name", ""),
                      um_per_px=pkg.um_per_px, origin=origin(pkg, names))
        comp.entries.append(entry)
        outlines = pkg.output_class[0] if pkg.output_class else ""
        if outlines and outlines != group:
            entry.error = f"it outlines {outlines}, not {group}"
            continue
        if pkg.modality != modality:
            entry.error = f"it is for {pkg.modality} images, the others for {modality}: judged apart"
            continue
        report({"stage": f"{entry.label} ({k + 1} of {n}): judging…", "run": entry.run_id, "k": k, "n": n,
                "fraction": k / n})
        try:
            entry.evaluation = evaluate(paths, group, {pkg.modality: pkg}, area, margin_um, iou_min,
                                        lambda f, k=k: report({"fraction": (k + 0.9 * f) / n}), cancelled,
                                        device=device, use_cache=True, keep_examples=3)
        except Exception as exc:
            if type(exc).__name__ == "Cancelled":
                raise
            entry.error = f"judging failed: {exc}"
            continue
        if on == "test":                                         # where ``judged`` looks for it
            entry.test, entry.evaluation = entry.evaluation, None
        try:
            entry.ms_per_tile, entry.device = speed(pkg, device)
        except Exception as exc:
            entry.device = f"not timed ({exc})"
        entry.size_mb = os.path.getsize(pkg.path) / 1e6
        entry.params_m = parameters_m(pkg.path)
        report({"stage": f"{entry.label}: done", "run": entry.run_id, "k": k + 1, "n": n, "fraction": (k + 1) / n})
        if results:
            comp.save(results)
    comp.finished = dt.datetime.now().isoformat(timespec="seconds")
    if results:
        comp.save(results)
    report({"stage": "Done", "fraction": 1.0, "k": n, "n": n})
    return comp


def cross(arch_ids: list[str], weights_keys: list[str]) -> list[Planned]:
    """Every architecture with every starting point (``plan`` leaves out the pairs that do not fit)."""
    return [Planned("new", a, w) for a in arch_ids for w in weights_keys]


def run(runs: list, settings, train_paths: list[str], eval_paths: list[str], test_paths: list[str],
        group: str, out_dir: str | os.PathLike, name: str, area: str = "marked", margin_um: float = 50.0,
        iou_min: float = 0.5, criterion: str = "f1", device: str = "auto",
        progress: Callable[[dict], None] | None = None, cancelled: Callable[[], bool] | None = None,
        results: str | os.PathLike | None = None, models_dir: str | os.PathLike | None = None,
        training_set: str = "", test_groups: dict | None = None) -> Comparison:
    """Train each model of a run, judge, time and size it (see the module docstring); writes its results.

    A fine-tuned model is followed by its starting model **as it was** (not retrained, judged the same way), so
    the report says whether fine-tuning helped.

    Args:
        runs: The models (``Planned``, or what ``Planned.of`` reads).
        settings: ``seg_train.Settings``: image type, µm/px, epochs, steps, batch, scale, device, seed (start and
            architecture: per model; a fine-tuned model keeps its base's image type and µm/px).
        train_paths, eval_paths, test_paths: The projects of each role.
        group: The class group outlined.
        out_dir: The run's folder (``training.runs``).
        name: The run's name.
        area, margin_um, iou_min: How the models are judged (``training.segmentation.evaluate``).
        criterion: The measure "best" is judged by (``CRITERIA``).
        device: Where the models run for judging and timing ("auto": the best here).
        progress: Called with {"stage", "fraction", "arch", "run" (``Entry.run_id``), "k", "n"} and the epochs'
            rows.
        results: The results file (default ``comparison.json`` in ``out_dir``), saved after each model.
        models_dir: Where the models go (default ``out_dir``).
        training_set: The training set's name (for the report).
        test_groups: {test group: [project names]} of the test projects, for the report's per-group results.

    Raises:
        Cancelled (of ``seg_train``) if cancelled.
    """
    from modalytics.services.segmentation import find_models, load_package
    from modalytics.training import architectures as ar
    from modalytics.training import encoders as en
    from modalytics.training import seg_train
    from modalytics.training.segmentation import evaluate
    report = progress or (lambda _d: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = Path(results) if results else out_dir / "comparison.json"
    models_dir = Path(models_dir) if models_dir else out_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    names = lambda paths: [os.path.splitext(os.path.basename(p))[0] for p in paths]
    comp = Comparison(name, group, settings.modality, settings.um_per_px,
                      {k: v for k, v in asdict(settings).items() if k not in ("arch_id", "start")},
                      {"training": names(train_paths), "evaluation": names(eval_paths), "test": names(test_paths)},
                      area, margin_um, iou_min, criterion, started=dt.datetime.now().isoformat(timespec="seconds"),
                      versions=versions(), training_set=training_set, test_groups=dict(test_groups or {}))
    runs = [Planned.of(r) for r in runs]
    report({"stage": "Counting the images and objects of each role…", "fraction": 0.0})
    comp.data = {role: seg_train.data_counts(paths, group, settings.modality, area, margin_um)
                 for role, paths in (("training", train_paths), ("evaluation", eval_paths), ("test", test_paths))}
    archs, weights_lib = ar.ArchitectureLibrary(), en.EncoderLibrary()
    models = {m.id: m for m in find_models()} if any(p.kind == "finetune" for p in runs) else {}
    varied = len({p.weights for p in runs if p.kind == "new"}) > 1        # labels name the weights when they differ
    n = len(runs)

    def judge(entry, pkg, k):
        """Judged on the evaluation (and test) projects, timed and sized."""
        for role, paths, attr, (a, b) in (("evaluation", eval_paths, "evaluation", (0.8, 0.9)),
                                          ("test", test_paths, "test", (0.9, 0.97))):
            if not paths:
                continue
            report({"stage": f"{entry.label} ({k + 1} of {n}): judging on the {role} projects…",
                    "arch": entry.arch_id, "run": entry.run_id, "k": k, "n": n, "fraction": (k + a) / n})
            try:
                ev = evaluate(paths, group, {pkg.modality: pkg}, area, margin_um, iou_min,
                              lambda f, k=k, a=a, b=b: report({"fraction": (k + a + (b - a) * f) / n}),
                              cancelled, device=device, use_cache=False, keep_examples=3)
            except Exception as exc:
                if type(exc).__name__ == "Cancelled":
                    raise seg_train.Cancelled() from None
                entry.error = f"judging on the {role} projects failed: {exc}"
                return
            setattr(entry, attr, ev)
        try:
            entry.ms_per_tile, entry.device = speed(pkg, device)
        except Exception as exc:
            entry.device = f"not timed ({exc})"
        entry.size_mb = os.path.getsize(pkg.path) / 1e6
        entry.params_m = parameters_m(pkg.path)
        entry.um_per_px = pkg.um_per_px

    for k, p in enumerate(runs):
        # 1. What is trained, and its label
        base = models.get(p.base) if p.kind == "finetune" else None
        if base is not None:
            arch = seg_train.package_architecture(base, archs)
            label = f"{base.name} — fine-tuned"
            entry = Entry(arch.key, label, weights=f"model:{base.id}", weights_name=base.name, kind="finetune",
                          base=base.id)
            s = replace(settings, start="model", area=area, margin_um=margin_um)
            out = models_dir / f"ft_{base.id}.mmodel"
        else:
            spec, ws = archs.get(p.arch_id), weights_lib.get(p.weights)
            wname = ws.name if ws is not None else ("random weights" if p.weights == "scratch" else p.weights)
            label = (spec.name if spec is not None else p.arch_id) + (f" · {wname}" if varied else "")
            entry = Entry(p.arch_id, label, weights=p.weights, weights_name=wname)
            s = replace(settings, start=p.weights, arch_id=p.arch_id, area=area, margin_um=margin_um)
            out = models_dir / f"{p.arch_id}__{p.weights}.mmodel"
        comp.entries.append(entry)

        def sub(d, k=k, entry=entry, share=0.8):
            report(dict(d, arch=entry.arch_id, run=entry.run_id, k=k, n=n,
                        stage=f"{entry.label} ({k + 1} of {n}): {d.get('stage', '')}",
                        fraction=(k + share * d.get("fraction", 0.0)) / n))
        # 2. Trained
        try:
            res = seg_train.train(s, train_paths, eval_paths, group, str(out), base=base,
                                  name=f"{group} — {label} (run '{name}')", note={"run": name}, progress=sub,
                                  cancelled=cancelled)
        except seg_train.Cancelled:
            raise
        except Exception as exc:                             # one failing does not stop the others
            entry.error = f"{type(exc).__name__}: {exc}"
            comp.save(results)
            continue
        entry.path, entry.history, entry.best_epoch = res.path, res.history, res.best_epoch
        entry.seconds = res.history[-1]["seconds"] if res.history else 0.0
        comp.skipped = sorted(set(comp.skipped) | set(res.skipped))
        pkg = load_package(res.path)
        key = pkg.manifest.get("source", {}).get("training", {}).get("device", "")
        comp.training_device = next((d.label for d in seg_train.devices() if d.key == key), key)
        # 3. Judged; a fine-tuned model's starting point too, as it was
        judge(entry, pkg, k)
        if base is not None:
            ref = Entry(entry.arch_id, f"{base.name} — as it was", path=base.path, weights=entry.weights,
                        weights_name=base.name, kind="reference", base=base.id)
            comp.entries.append(ref)
            judge(ref, base, k)
        comp.save(results)                                    # after each: nothing lost if a later one fails
    comp.finished = dt.datetime.now().isoformat(timespec="seconds")
    comp.save(results)
    report({"stage": "Done", "fraction": 1.0, "k": n, "n": n})
    return comp


