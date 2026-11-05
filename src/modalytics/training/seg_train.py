"""
Training and fine-tuning segmentation models (the full edition: PyTorch; Qt-free).

A segmentation model learns, from images and their hand outlines of a class group,
to outline that group. The result is an ordinary model package (``.mmodel``): the
ONNX network that both editions run, plus the PyTorch weights (``weights.safetensors``:
numbers only, no code) from which it can be fine-tuned again.

**Starting point.**

* *An existing model* (fine-tuning): its weights, e.g. the released colon gland
  model adapted to a lab's images; the same architecture, image type and µm/px.
* *Encoder weights* (new model): the encoder starts from weights learned elsewhere —
  ImageNet (natural images), MicroNet (micrographs) or your own (``training.encoders``);
  usually far better than nothing with few annotated images.
* *Nothing* (new model, random weights): needs many annotated images.

**Data.** For each image of the model's type in the projects: the image read at the
model's µm/px over the evaluated area (the whole image, or around the hand outlines, or
per image: the whole image where its annotation is marked complete — as in
``training.segmentation``), normalised as the model sees it when it runs
(z-score over the region); the hand outlines of the group (all its subgroups) as
the target; pixels outside the area are not used, nor those under unreviewed
suggestions (an object nobody has judged yet: neither object nor background).
The training projects teach,
the evaluation projects choose the best epoch and judge the result.

**Resolution.** A model works at one resolution (µm/px), whatever the images: each
image is read at its own pixel size (stored with it) and resampled to the model's, in
training and when Segment runs it — so images of several systems can be mixed, and the
model learns objects at their size in µm. ``resolution_report`` proposes the working
resolution from the training images (the most common pixel size) and says which images
would be enlarged (coarser than it: blurred, no detail gained) or look wrong (a pixel
size far from the others').

**Training.** Random 512-px patches (most of them on an object), **augmented** as set
(``Settings``; each recorded with the model and in the report): mirrored (``flips``) and
turned by quarter turns (``turns``) — outlines have no preferred direction —, brightness
and contrast varied by up to ±``intensity`` — and, if asked (``scale_jitter``), resized by
a random factor within ±jitter, so the model tolerates small calibration differences
between systems; loss = binary
cross-entropy + soft Dice (1 − 2·Σpg / (Σp + Σg)); AdamW, learning rate falling as
a cosine over the epochs; mixed precision on CUDA. After each epoch the model
outlines the evaluation images (sliding 512-px windows) and their Dice is measured;
the epoch with the best Dice is kept (with no evaluation projects: the last).

Architectures come from the library (``training.architectures``: segmentation-models-pytorch
decoders and encoders, or networks of your own); one must pass its check (ONNX export)
before it is trained.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

WEIGHTS = "weights.safetensors"
PATCH = 512


class Cancelled(Exception):
    """Raised when training is cancelled."""


def available() -> tuple[bool, str]:
    """(True, "") when the training libraries are installed, else (False, what is missing)."""
    from importlib.util import find_spec                          # found without importing (torch takes seconds)
    missing = [pkg for mod, pkg in (("torch", "torch"), ("segmentation_models_pytorch", "segmentation-models-pytorch"),
                                    ("onnx", "onnx"), ("safetensors", "safetensors")) if find_spec(mod) is None]
    if not missing:
        return True, ""
    from modalytics import paths
    if paths.installed():
        return False, "Training needs the training pack (PyTorch and its libraries): install it here."
    return False, f"Training needs {', '.join(missing)}: uv sync --extra training"


# ---------------------------------------------------------------- devices
@dataclass
class Device:
    """A device PyTorch can train on. ``key``: "cuda:0", "xpu:0", "mps" or "cpu"."""

    key: str
    name: str
    memory_gb: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name} ({self.memory_gb:.0f} GB)" if self.memory_gb else self.name


def devices() -> list[Device]:
    """The devices here, best first: CUDA (NVIDIA, or AMD with a ROCm build) by memory, Intel XPU, Apple MPS,
    then the CPU (always; slow)."""
    import torch
    out = []
    if torch.cuda.is_available():
        gpus = [Device(f"cuda:{i}", torch.cuda.get_device_name(i),
                       torch.cuda.get_device_properties(i).total_memory / 2 ** 30)
                for i in range(torch.cuda.device_count())]
        out += sorted(gpus, key=lambda d: -d.memory_gb)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        out += [Device(f"xpu:{i}", torch.xpu.get_device_name(i)) for i in range(torch.xpu.device_count())]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append(Device("mps", "Apple GPU (Metal)"))
    out.append(Device("cpu", "CPU (slow)"))
    return out


# ---------------------------------------------------------------- networks and packages
def build(arch_id: str, in_channels: int, encoder_weights: str | None = None):
    """The network of a library architecture (``encoder_weights``: "imagenet" or None, random)."""
    from modalytics.training import architectures as ar
    spec = ar.ArchitectureLibrary().get(arch_id)
    if spec is None:
        raise ValueError(f"the architecture '{arch_id}' is not in the library")
    return ar.build(spec, in_channels, imagenet=encoder_weights == "imagenet")


def package_architecture(pkg, library=None):
    """The architecture a model package was trained as (see ``ArchitectureLibrary.for_package``)."""
    from modalytics.training import architectures as ar
    return (library or ar.ArchitectureLibrary()).for_package(pkg)


def trainable(pkg, library=None) -> bool:
    """Whether a model package carries its PyTorch weights and its architecture can be built here."""
    with zipfile.ZipFile(pkg.path) as z:
        if WEIGHTS not in z.namelist():
            return False
    try:
        package_architecture(pkg, library)
    except ValueError:
        return False
    return True


def load_weights(pkg) -> dict:
    """The PyTorch weights of a model package."""
    from safetensors.torch import load
    with zipfile.ZipFile(pkg.path) as z:
        return load(z.read(WEIGHTS))


def write_package(model, arch, manifest: dict, out_path: str) -> str:
    """Write a model package: the ONNX network (dynamic batch and size), the weights and the manifest.

    The manifest's ``self_test`` records the logits of a fixed input, as ``scripts/convert_hmodel.py`` does;
    ``source.architecture`` the architecture (``arch``: an ``ArchSpec``, or a library key); a network of your
    own also travels as ``architecture.py`` (for the record: it is never run from the package).
    """
    if isinstance(arch, str):
        from modalytics.training import architectures as ar
        arch = ar.ArchitectureLibrary().get(arch) or ar.ArchSpec(arch, arch)
    import torch
    from safetensors.torch import save
    model = model.float().cpu().eval()
    cin = int(manifest["in_channels"])
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        torch.onnx.export(model, torch.zeros(1, cin, PATCH, PATCH), onnx_path, input_names=["image"],
                          output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False,
                          dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}})
        with torch.no_grad():
            ref = model(torch.linspace(-2, 2, cin * 64 * 64).reshape(1, cin, 64, 64)).numpy()
        manifest = dict(manifest, self_test={"input": "linspace(-2, 2) over (1, C, 64, 64)",
                                             "logits_mean": float(ref.mean()), "logits_std": float(ref.std())})
        manifest.setdefault("source", {})["architecture_id"] = arch.key
        manifest["source"]["architecture"] = arch.definition()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        part = out_path + ".part"
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(onnx_path, "model.onnx")
            z.writestr(WEIGHTS, save({k: v.contiguous() for k, v in model.state_dict().items()}))
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            if arch.custom and arch.path and Path(arch.path).exists():
                z.write(arch.path, "architecture.py")
    from modalytics import fileio
    fileio.replace(part, out_path)
    return out_path


# ---------------------------------------------------------------- data
@dataclass
class Sample:
    """One image prepared for training: ``pixels`` (C, H, W) normalised, ``target`` and ``valid`` (H, W) bool."""

    pixels: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    name: str = ""


def _rasterize(polys: list, frame: tuple, shape: tuple) -> np.ndarray:
    """Pixels whose centres lie inside the polygons (each tested only within its bounding box)."""
    from modalytics.parameters.measures import rasterize
    x0, y0, px, py = frame
    out = np.zeros(shape, bool)
    for p in polys:
        bx0, by0, bx1, by1 = p.bounds
        c0, r0 = max(0, int((bx0 - x0) / px) - 1), max(0, int((by0 - y0) / py) - 1)
        c1, r1 = min(shape[1], int(math.ceil((bx1 - x0) / px)) + 1), min(shape[0], int(math.ceil((by1 - y0) / py)) + 1)
        if c1 > c0 and r1 > r0:
            out[r0:r1, c0:c1] |= rasterize(p, x0 + c0 * px, y0 + r0 * py, px, py, (r1 - r0, c1 - c0))
    return out


def data_counts(paths: list[str], group: str, modality: str, area: str = "image", margin_um: float = 50.0) -> dict:
    """What ``prepare`` uses of the projects, counted (the pixels not read): for the run's report.

    Returns:
        dict: {"projects", "images" (with a place to learn from), "complete" (of them marked fully outlined),
        "objects" (the hand outlines of ``group``)}.
    """
    from modalytics.parameters.engine import objects_of
    from modalytics.training.dataset import _open
    from modalytics.training.segmentation import evaluated_area
    out = {"projects": 0, "images": 0, "complete": 0, "objects": 0}
    for path in paths:
        try:
            images = _open(path)
        except Exception:                                           # unreadable: prepare says so
            continue
        used = False
        for img in images:
            if img.modality != modality:
                continue
            hand = [p for m in img.masks if m.group == group for p in objects_of(m.geometry)]
            w, h = img.size_um
            complete = img.fully_outlined(group)
            region = evaluated_area(hand, w, h, area, margin_um, complete)
            if region is None or region.is_empty:
                continue
            used = True
            out["images"] += 1
            out["complete"] += int(complete)
            out["objects"] += len(hand)
        out["projects"] += int(used)
    return out


def prepare(paths: list[str], group: str, modality: str, um_per_px: float, in_channels: int,
            area: str = "image", margin_um: float = 50.0, progress=None, cancelled=None) -> tuple[list, list[str]]:
    """The images of ``modality`` in the projects, with their hand outlines of ``group`` (see the module docstring).

    Returns:
        tuple: ([Sample], [what was left out, and why]).
    """
    from modalytics.parameters.engine import objects_of
    from modalytics.services.segmentation import read_region
    from modalytics.training.dataset import _open
    from modalytics.training.segmentation import evaluated_area, pending, without_pending
    samples, skipped = [], []
    for pi, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            images = _open(path)
        except Exception as exc:                                    # unreadable: reported, the others go on
            skipped.append(f"{name}: {exc}")
            continue
        for img in images:
            if cancelled is not None and cancelled():
                raise Cancelled()
            if img.modality != modality:
                continue
            hand = [p for m in img.masks if m.group == group for p in objects_of(m.geometry)]
            w, h = img.size_um
            region = evaluated_area(hand, w, h, area, margin_um, img.fully_outlined(group))
            if region is None or region.is_empty:
                skipped.append(f"{name} › {img.name}: no {group} outlined by hand")
                continue
            region, n_pending = without_pending(region, img, group)
            if n_pending:
                skipped.append(f"{name} › {img.name}: {n_pending} unreviewed suggestion(s), their places left out")
            pixels, frame = read_region(img, region.bounds, um_per_px)
            pixels = np.asarray(pixels, np.float32)
            if in_channels == 1 and pixels.ndim == 3:
                pixels = pixels[..., :3].mean(axis=2)
            if in_channels == 3 and pixels.ndim == 2:
                pixels = np.repeat(pixels[..., None], 3, axis=2)
            pixels = pixels[..., :3] if pixels.ndim == 3 else pixels[..., None]
            pixels = (pixels - pixels.mean()) / max(float(pixels.std()), 1e-6)  # z-score over the region, as at use
            shape = pixels.shape[:2]
            target = _rasterize(hand, frame, shape)
            whole = area == "image" or (area == "marked" and img.fully_outlined(group))
            valid = np.ones(shape, bool) if whole else _rasterize([region], frame, shape)
            todo = pending(img, group)
            if todo:                                       # under an unreviewed suggestion: not used
                from shapely.ops import unary_union
                valid &= ~_rasterize([unary_union(todo).buffer(2.0)], frame, shape)
            samples.append(Sample(np.ascontiguousarray(pixels.transpose(2, 0, 1)), target, valid,
                                  f"{name} › {img.name}"))
        if progress is not None:
            progress((pi + 1) / max(len(paths), 1))
    return samples, skipped


# ---------------------------------------------------------------- resolution
@dataclass
class ImageScale:
    """The pixel size of one training image (µm/px, the mean of x and y)."""

    project: str
    image: str
    um_per_px: float

    @property
    def name(self) -> str:
        return f"{self.project} › {self.image}"


_SCALES: dict = {}                                           # {(path, modified, group, modality): [ImageScale]}


def image_scales(paths: list[str], group: str, modality: str,
                 cancelled: Callable[[], bool] | None = None) -> tuple[list[ImageScale], list[str]]:
    """The pixel sizes of the images training would read: of ``modality``, with ``group`` outlined by hand.

    Returns:
        ([ImageScale], [projects that could not be read, and why]). Remembered per file and modification time.
    """
    from modalytics.training.dataset import _open
    out, skipped = [], []
    for path in paths:
        if cancelled is not None and cancelled():
            raise Cancelled()
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            key = (path, os.path.getmtime(path), group, modality)
            if key not in _SCALES:
                _SCALES[key] = [ImageScale(name, img.name, float(np.mean(img.pixel_size_um)))
                                for img in _open(path) if img.modality == modality
                                and any(m.group == group for m in img.masks)]
            out += _SCALES[key]
        except Exception as exc:                                    # unreadable: said, the others go on
            skipped.append(f"{name}: {exc}")
    return out, skipped


def _sig(v: float) -> float:
    return float(f"{v:.4g}")


def resolution_report(scales: list[ImageScale], works_at: float | None = None, tile: int = PATCH) -> dict:
    """What the training images' pixel sizes mean for a working resolution.

    Args:
        scales: From ``image_scales``.
        works_at: The model's resolution (None: the proposal).

    Returns:
        {"proposal": the most common pixel size (ties: the finer) or None, "works_at",
         "groups": [(µm/px, number of images, [names])] most common first,
         "enlarged": [(name, µm/px, factor)] images coarser than 1.25 × the working resolution,
         "outliers": [(name, µm/px, ratio to the size it slipped from)] pixel sizes that look wrongly stored
         (``dataset.unit_slips``: a power of ten from a more common size; other differences are normal),
         "tile_um": the width a tile covers at the working resolution}.
    """
    from modalytics.training.dataset import unit_slips
    groups: dict[float, list[str]] = {}
    for sc in scales:
        groups.setdefault(_sig(sc.um_per_px), []).append(sc.name)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    proposal = ordered[0][0] if ordered else None
    works_at = works_at or proposal
    enlarged = [(sc.name, sc.um_per_px, sc.um_per_px / works_at) for sc in scales
                if works_at and sc.um_per_px > 1.25 * works_at]
    slips = unit_slips({u: len(names) for u, names in groups.items()})
    outliers = [(sc.name, sc.um_per_px, sc.um_per_px / slips[_sig(sc.um_per_px)]) for sc in scales
                if _sig(sc.um_per_px) in slips]
    return {"proposal": proposal, "works_at": works_at, "groups": [(u, len(n), n) for u, n in ordered],
            "enlarged": enlarged, "outliers": outliers, "tile_um": tile * works_at if works_at else None}


def _pad(s: Sample, size: int) -> Sample:
    """A sample at least ``size`` px each way (reflected; the added pixels are not used)."""
    _c, h, w = s.pixels.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if not ph and not pw:
        return s
    mode = "reflect" if h > 1 and w > 1 and ph < h and pw < w else "edge"
    pixels = np.pad(s.pixels, ((0, 0), (0, ph), (0, pw)), mode=mode)
    return Sample(pixels, np.pad(s.target, ((0, ph), (0, pw))), np.pad(s.valid, ((0, ph), (0, pw))), s.name)


def batch(samples: list, n: int, rng: np.random.Generator, size: int = PATCH, on_object: float = 0.7,
          scale_jitter: float = 0.0, flips: bool = True, turns: bool = True, intensity: float = 0.15):
    """``n`` random training patches (pixels, target, valid), augmented (see the module docstring).

    Args:
        scale_jitter: Each patch resized by a random factor in [1 − jitter, 1 + jitter] (0: not resized); the
            samples must be at least ``size × (1 + jitter)`` px each way (``_pad``).
        flips: Half the patches mirrored.
        turns: Each patch turned by 0–3 quarter turns.
        intensity: Contrast × a factor in [1 − intensity, 1 + intensity] and brightness + an offset in
            [−intensity, +intensity] (the normalised pixels; 0: unchanged).
    """
    import cv2
    xs, ys, vs = [], [], []
    for _ in range(n):
        s = samples[int(rng.integers(len(samples)))]
        _c, h, w = s.pixels.shape
        # 0. How much of the image the patch takes (resized to ``size`` below): more, objects look smaller
        out_size, size = size, (int(round(size * rng.uniform(1 - scale_jitter, 1 + scale_jitter)))
                                if scale_jitter else size)
        # 1. Where: most patches centred on a pixel of an object, the others anywhere
        if rng.random() < on_object and s.target.any():
            rr, cc = np.nonzero(s.target)
            k = int(rng.integers(len(rr)))
            r0 = int(np.clip(rr[k] - size // 2 + rng.integers(-size // 4, size // 4 + 1), 0, h - size))
            c0 = int(np.clip(cc[k] - size // 2 + rng.integers(-size // 4, size // 4 + 1), 0, w - size))
        else:
            r0, c0 = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
        x = s.pixels[:, r0:r0 + size, c0:c0 + size]
        y = s.target[r0:r0 + size, c0:c0 + size]
        v = s.valid[r0:r0 + size, c0:c0 + size]
        if size != out_size:                               # resized: pixels smoothly, outlines and area nearest
            x = np.stack([cv2.resize(ch, (out_size, out_size), interpolation=cv2.INTER_LINEAR) for ch in x])
            y = cv2.resize(y.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST).astype(bool)
            v = cv2.resize(v.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        size = out_size
        # 2. Quarter turns and flips (outlines have no preferred direction), brightness and contrast
        if turns:
            k = int(rng.integers(4))
            x, y, v = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(v, k)
        if flips and rng.random() < 0.5:
            x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
        if intensity:
            x = x * rng.uniform(1 - intensity, 1 + intensity) + rng.uniform(-intensity, intensity)
        xs.append(np.ascontiguousarray(x, np.float32))
        ys.append(np.ascontiguousarray(y, np.float32))
        vs.append(np.ascontiguousarray(v, np.float32))
    return np.stack(xs), np.stack(ys)[:, None], np.stack(vs)[:, None]


# ---------------------------------------------------------------- training
@dataclass
class Settings:
    """How a model is trained.

    Attributes:
        start: "model" (fine-tune ``base``), "scratch" (random weights) or the key of encoder weights
            (``training.encoders``: "imagenet", "micronet", your own).
        arch_id: Architecture of a new model (a ``training.architectures`` key; fine-tuning keeps the base's).
        modality, um_per_px: Image type and resolution of a new model (fine-tuning keeps the base's).
        epochs, steps, batch_size: Epochs; optimiser steps per epoch; patches per step.
        lr: Learning rate (default: 1e-4 when fine-tuning, else 1e-3).
        scale_jitter: Patches resized by a random factor within ±this (0.1 = ±10 %; 0: not resized).
        flips, turns, intensity: The other augmentations (``batch``): mirrored, quarter turns, brightness and
            contrast within ±intensity (0: unchanged).
        area, margin_um: Where the hand outlines are complete (see ``training.segmentation``).
        device: A ``Device.key`` ("" = the best here).
        seed: Makes the patches repeatable.
    """

    start: str = "model"
    arch_id: str = "unet_resnet34"
    modality: str = "phase"
    um_per_px: float = 0.314006
    epochs: int = 30
    steps: int = 50
    batch_size: int = 8
    lr: float | None = None
    scale_jitter: float = 0.0
    flips: bool = True
    turns: bool = True
    intensity: float = 0.15
    area: str = "image"
    margin_um: float = 50.0
    device: str = ""
    seed: int = 1


@dataclass
class Result:
    """A training run: the package written, the history per epoch and the epoch kept."""

    path: str
    history: list = field(default_factory=list)
    best_epoch: int = 0
    skipped: list = field(default_factory=list)


def _loss(logits, target, valid):
    """Binary cross-entropy + soft Dice over the valid pixels."""
    import torch
    import torch.nn.functional as F
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    g = target * valid
    dice = 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)
    return bce + dice


def predict(model, pixels: np.ndarray, device, size: int = PATCH, stride: int = 384) -> np.ndarray:
    """Probabilities (H, W) of a (C, H, W) image: sliding windows, averaged where they overlap."""
    import torch
    c, h, w = pixels.shape
    H, W = max(h, size), max(w, size)
    padded = np.pad(pixels, ((0, 0), (0, H - h), (0, W - w)), mode="edge")
    out, count = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    rows = list(range(0, H - size + 1, stride)) + ([H - size] if (H - size) % stride else [])
    cols = list(range(0, W - size + 1, stride)) + ([W - size] if (W - size) % stride else [])
    model.eval()
    with torch.no_grad():
        for r in rows:
            for c0 in cols:
                x = torch.from_numpy(np.ascontiguousarray(padded[None, :, r:r + size, c0:c0 + size])).to(device)
                out[r:r + size, c0:c0 + size] += torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
                count[r:r + size, c0:c0 + size] += 1
    return (out / np.maximum(count, 1))[:h, :w]


def dice_of(model, samples: list, device) -> float:
    """Pixel Dice of the model's outlines (probability ≥ 0.5) with the hand outlines, over all the samples."""
    inter = total = 0.0
    for s in samples:
        found = predict(model, s.pixels, device) >= 0.5
        inter += float((found & s.target & s.valid).sum())
        total += float((found & s.valid).sum() + (s.target & s.valid).sum())
    return 2 * inter / total if total else math.nan


def train(settings: Settings, train_paths: list[str], eval_paths: list[str], group: str, out_path: str,
          base=None, name: str = "", note: dict | None = None,
          progress: Callable[[dict], None] | None = None,
          cancelled: Callable[[], bool] | None = None) -> Result:
    """Train (or fine-tune) a segmentation model and write it as a package (see the module docstring).

    Args:
        settings: How.
        train_paths, eval_paths: The training and the evaluation projects.
        group: The class group outlined.
        out_path: The ``.mmodel`` to write.
        base: The model package to fine-tune (``settings.start == "model"``).
        name: The new model's name.
        note: Recorded in the manifest's ``source.training`` (e.g. the training set).
        progress: Called with {"stage", "fraction", and after each epoch "epoch", "loss", "dice", "best"}.

    Raises:
        ValueError: Nothing to train on, or a base model without weights.
        Cancelled: If ``cancelled`` returned True.
    """
    import torch

    from modalytics.training import architectures as ar
    from modalytics.training import encoders as en
    t0 = time.perf_counter()
    report = progress or (lambda _d: None)

    # 1. What is trained: the base's architecture, image type and resolution, or the settings'
    archs = ar.ArchitectureLibrary()
    weights = None
    if settings.start == "model":
        if base is None or not trainable(base, archs):
            raise ValueError("the model to fine-tune has no PyTorch weights, or its architecture is not here")
        arch = package_architecture(base, archs)
        modality, um, cin = base.modality, base.um_per_px, int(base.manifest["in_channels"])
    else:
        arch = archs.get(settings.arch_id)
        if arch is None:
            raise ValueError(f"the architecture '{settings.arch_id}' is not in the library")
        modality, um = settings.modality, settings.um_per_px
        cin = 3 if modality == "wsi" else 1
        if settings.start != "scratch":
            weights = en.EncoderLibrary().get(settings.start)
            if weights is None:
                raise ValueError(f"the encoder weights '{settings.start}' are not in the library")
            if weights.kind == en.IMAGENET and not arch.custom and not en.imagenet_available(arch.encoder, wait=True):
                raise ValueError(f"{arch.name}'s encoder ({arch.encoder}) has no ImageNet weights: start it from "
                                 f"other weights, or from nothing")
            if not weights.fits(arch.encoder):
                raise ValueError(f"{weights.name} has no weights for {arch.name}'s encoder ({arch.encoder})")
    # The architecture must pass its check (ONNX export) before hours are spent training it
    report({"stage": f"Checking {arch.name}…", "fraction": 0.0})
    checked = archs.check(arch, cin) if arch.source != "package" else {"ok": True}
    if not checked["ok"]:
        raise ValueError(f"{arch.name} does not pass its check: {checked['why']}")
    # 2. The data
    report({"stage": "Reading the images and their outlines…", "fraction": 0.0})
    tr, skipped = prepare(train_paths, group, modality, um, cin, settings.area, settings.margin_um,
                          lambda f: report({"stage": "Reading the training images…", "fraction": 0.05 * f}), cancelled)
    ev, skipped_ev = prepare(eval_paths, group, modality, um, cin, settings.area, settings.margin_um, None, cancelled)
    skipped += skipped_ev
    if not tr:
        raise ValueError(f"no {modality} image with {group} outlined by hand in the training projects")
    tr = [_pad(s, int(math.ceil(PATCH * (1 + settings.scale_jitter)))) for s in tr]
    # 3. The network, on the device
    device = torch.device(settings.device or devices()[0].key)
    model = ar.build(arch, cin, imagenet=weights is not None and weights.kind == en.IMAGENET)
    if settings.start == "model":
        model.load_state_dict(load_weights(base))
    elif weights is not None and weights.kind != en.IMAGENET:
        try:
            state = en.state_dict(weights, arch.encoder, progress=lambda f: report(
                {"stage": f"Downloading {weights.name} ({arch.encoder})…", "fraction": 0.05 * f}),
                cancelled=cancelled)
        except en.Cancelled:
            raise Cancelled() from None
        en.load_into(model, state)
    model.to(device)
    lr = settings.lr or (1e-4 if settings.start == "model" else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=settings.epochs * settings.steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(settings.seed)
    torch.manual_seed(settings.seed)
    # 4. Epochs; the best by Dice on the evaluation images is kept
    history, best, best_state, best_epoch = [], -math.inf, None, 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for step in range(settings.steps):
            if cancelled is not None and cancelled():
                raise Cancelled()
            x, y, v = (torch.from_numpy(a).to(device)
                       for a in batch(tr, settings.batch_size, rng, scale_jitter=settings.scale_jitter,
                                      flips=settings.flips, turns=settings.turns, intensity=settings.intensity))
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = _loss(logits.float(), y, v)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            report({"stage": f"Epoch {epoch} of {settings.epochs}",
                    "fraction": 0.05 + 0.9 * ((epoch - 1) + (step + 1) / settings.steps) / settings.epochs})
        dice = dice_of(model, ev, device) if ev else math.nan
        # Kept: the best Dice on the evaluation images (without any: the last epoch; always some epoch)
        if best_state is None or not ev or (math.isfinite(dice) and dice > best):
            best, best_epoch = (dice if math.isfinite(dice) else best), epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "dice": dice,
               "seconds": time.perf_counter() - t0, "best": best_epoch}
        history.append(row)
        report(dict(row, stage=f"Epoch {epoch} of {settings.epochs}",
                    fraction=0.05 + 0.9 * epoch / settings.epochs))
    # 5. The package: ONNX for running, the weights for fine-tuning again
    report({"stage": "Writing the model…", "fraction": 0.97})
    model.load_state_dict(best_state)
    label = arch.name
    parent = base.manifest if base is not None else {}
    manifest = {
        "format": "modalytics-model", "format_version": 1,
        "name": name or f"{group} — {label} (trained {dt.date.today().isoformat()})",
        "task": "binary-segmentation", "modality": modality, "in_channels": cin, "input_um_per_px": um,
        "tile": PATCH, "overlap": int(parent.get("overlap", 256)), "pad_multiple": 32, "blend": "gaussian",
        "normalization": "zscore-region", "output": "sigmoid", "threshold": 0.5,
        "classes": [{"group": group, "subgroup": ""}],
        "source": {"architecture_id": arch.key, "trained": dt.datetime.now().isoformat(timespec="seconds"),
                   "parent": base.id if base is not None else "",
                   "encoder_weights": weights.record() if weights is not None else
                   parent.get("source", {}).get("encoder_weights", {}),
                   "training": dict(note or {}, start=settings.start, epochs=settings.epochs, steps=settings.steps,
                                    batch_size=settings.batch_size, lr=lr, scale_jitter=settings.scale_jitter,
                                    flips=settings.flips, turns=settings.turns, intensity=settings.intensity,
                                    best_epoch=best_epoch,
                                    evaluation_dice=round(best, 4) if ev and math.isfinite(best) else None,
                                    images=len(tr), evaluation_images=len(ev), device=str(device),
                                    projects=[os.path.splitext(os.path.basename(p))[0] for p in train_paths],
                                    evaluation_projects=[os.path.splitext(os.path.basename(p))[0]
                                                         for p in eval_paths],
                                    seconds=round(time.perf_counter() - t0, 1))},
    }
    write_package(model, arch, manifest, out_path)
    report({"stage": "Done", "fraction": 1.0})
    return Result(out_path, history, best_epoch, skipped)


def add_weights(mmodel: str, hmodel: str) -> None:
    """Put a HoloPath ``.hmodel``'s PyTorch weights into the ``.mmodel`` made from it, so it can be fine-tuned.

    The weights are read in PyTorch's safe mode (tensors only) and stored as safetensors; the ONNX network and the
    manifest stay as they are.
    """
    import torch
    from safetensors.torch import save
    with zipfile.ZipFile(hmodel) as z:
        state = torch.load(io.BytesIO(z.read("weights.pt")), map_location="cpu", weights_only=True)
    with zipfile.ZipFile(mmodel) as z:
        files = {n: z.read(n) for n in z.namelist() if n != WEIGHTS}
    files[WEIGHTS] = save({k: v.contiguous() for k, v in state.items()})
    part = mmodel + ".part"
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)
    from modalytics import fileio
    fileio.replace(part, mmodel)
