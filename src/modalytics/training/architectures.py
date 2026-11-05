"""
Segmentation architectures: the library of networks a segmentation model can be trained as (Qt-free).

An architecture is one ``.py`` file in ``resources/architectures`` (built in) or in the
user library (``paths.user_dir("architectures")``; a file with a built-in's KEY replaces
it). Most are a decoder and an encoder of segmentation-models-pytorch (Iakubovskii,
MIT licence), with no code at all::

    KEY = 'unet_resnet34'                    # letters, digits, _
    NAME = 'U-Net (ResNet-34)'
    DESCRIPTION = 'What it is, in a sentence'
    DECODER = 'Unet'                         # a segmentation-models-pytorch decoder (DECODERS)
    ENCODER = 'resnet34'                     # a segmentation-models-pytorch encoder ('tu-…': a timm one)
    DECODER_OPTIONS = {}                     # passed to the decoder (e.g. {'decoder_attention_type': 'scse'})
    RECOMMENDED = True                       # shown in bold in Train's list of models
    REFERENCES = [{'text': 'Author (year). Title. …', 'doi': '10.…'}]
    REQUIRES = []                            # extra libraries it imports

A network of your own adds ``build`` (DECODER is then ignored)::

    ENCODER = 'resnet34'                     # what ``model.encoder`` is, so encoder weights can be loaded into it
    def build(in_channels, classes, encoder_weights):
        # encoder_weights: 'imagenet' (start the encoder from ImageNet if you can) or None
        return torch_module                  # (N, in_channels, H, W) → (N, classes, H, W) logits

**The contract.** Whatever the network, Segment runs it as ONNX: an image tile in
(any size divisible by 32), one logit map per class out, the same size. ``check``
proves it before anything is trained: the network is built, exported to ONNX (opset
17, dynamic size) and run by ONNX Runtime on two tile sizes; the outputs must match
PyTorch's. The result is remembered per file and library versions
(``architectures/.checks.json``), so a check runs once.

Model packages record the architecture they were trained as (``source.architecture``:
key, name, decoder, encoder, options), so a model built from segmentation-models-pytorch
parts can be fine-tuned again even where its file is missing; a network of your own
needs its file in the library to be fine-tuned (never to run: the package holds ONNX).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from modalytics import paths

BUILTIN = Path(__file__).resolve().parent.parent / "resources" / "architectures"
# segmentation-models-pytorch's decoders (the classes of the package)
DECODERS = ("Unet", "UnetPlusPlus", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "Linknet", "MAnet", "PAN",
            "UPerNet", "Segformer", "DPT")
CHECK_SIZES = ((512, 512), (384, 640))       # tile sizes the ONNX network must take
TOLERANCE = 1e-3                             # largest |ONNX − PyTorch| logit difference accepted (relative to the range)


@dataclass
class ArchSpec:
    """One architecture of the library (see the module docstring for the fields)."""

    key: str
    name: str
    description: str = ""
    decoder: str = ""
    encoder: str = ""
    decoder_options: dict = field(default_factory=dict)
    recommended: bool = False
    references: list[dict] = field(default_factory=list)
    requires: tuple[str, ...] = ()
    build_fn: Callable | None = field(default=None, repr=False)
    path: str = ""
    source: str = "builtin"

    @property
    def uid(self) -> str:
        return self.key

    @property
    def custom(self) -> bool:
        """A network of its own (``build``), not segmentation-models-pytorch parts."""
        return self.build_fn is not None

    def definition(self) -> dict:
        """What a model package records (enough to build it again when it is made of library parts)."""
        return {"key": self.key, "name": self.name, "decoder": "" if self.custom else self.decoder,
                "encoder": self.encoder, "decoder_options": dict(self.decoder_options), "custom": self.custom}

    def references_text(self) -> str:
        return "\n".join(r.get("text", "") + (f" doi:{r['doi']}" if r.get("doi") else "") for r in self.references)

    def file_hash(self) -> str:
        return hashlib.sha1(Path(self.path).read_bytes()).hexdigest() if self.path and Path(self.path).exists() \
            else ""


def load_script(path: str | os.PathLike, source: str = "user") -> ArchSpec:
    """Import one architecture script and describe it (no PyTorch needed).

    Raises:
        ValueError: If the script is not a usable architecture (the message says why).
    """
    from modalytics.parameters import libraries
    libraries.activate()
    path = Path(path)
    name = f"_modalytics_architecture_{source}_{re.sub(r'[^0-9a-zA-Z_]', '_', str(path.with_suffix('')))}"
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
    build_fn = getattr(module, "build", None)
    decoder, encoder = str(getattr(module, "DECODER", "") or ""), str(getattr(module, "ENCODER", "") or "")
    if build_fn is not None and not callable(build_fn):
        raise ValueError("build must be a function build(in_channels, classes, encoder_weights)")
    if build_fn is None:
        if decoder not in DECODERS:
            raise ValueError(f"DECODER must be one of {', '.join(DECODERS)} (or define build(…))")
        if not encoder:
            raise ValueError("ENCODER must name a segmentation-models-pytorch encoder, e.g. 'resnet34'")
    options = getattr(module, "DECODER_OPTIONS", {}) or {}
    if not isinstance(options, dict):
        raise ValueError("DECODER_OPTIONS must be a dict")
    refs = [dict(r) if isinstance(r, dict) else {"text": str(r)} for r in getattr(module, "REFERENCES", []) or []]
    return ArchSpec(key=key, name=str(getattr(module, "NAME", key)), description=str(getattr(module, "DESCRIPTION", "")),
                    decoder=decoder, encoder=encoder, decoder_options=dict(options),
                    recommended=bool(getattr(module, "RECOMMENDED", False)), references=refs,
                    requires=tuple(getattr(module, "REQUIRES", ()) or ()), build_fn=build_fn, path=str(path),
                    source=source)


def from_definition(d: dict) -> ArchSpec:
    """An architecture from a model package's record (library parts only; see the module docstring)."""
    if d.get("custom") or d.get("decoder") not in DECODERS:
        raise ValueError(f"'{d.get('name', d.get('key', '?'))}' is a network of its own: add its file to the "
                         f"architectures library to fine-tune it")
    return ArchSpec(key=d["key"], name=d.get("name", d["key"]), decoder=d["decoder"], encoder=d["encoder"],
                    decoder_options=dict(d.get("decoder_options") or {}), source="package")


class ArchitectureLibrary:
    """Every architecture: built in, replaced or added by the user's own scripts.

    Attributes:
        specs: {key: ArchSpec}, built-in order first.
        errors: {script path: why it could not be loaded}.
        overridden: Keys of built-ins replaced by the user's own script.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin_dir = Path(builtin) if builtin else BUILTIN
        self.user = Path(user) if user else paths.user_dir("architectures")
        self.reload()

    @property
    def problems(self) -> dict[str, str]:
        return {Path(p).name: why for p, why in self.errors.items()}

    def reload(self) -> None:
        self.specs: dict[str, ArchSpec] = {}
        self.errors: dict[str, str] = {}
        self.overridden: list[str] = []
        for folder, source in ((self.builtin_dir, "builtin"), (self.user, "user")):
            files = sorted(folder.glob("*.py")) if folder.is_dir() else []
            if source == "builtin":                       # the shipped order: ORDER.txt, then the rest by name
                order = (folder / "ORDER.txt").read_text(encoding="utf-8").split() \
                    if (folder / "ORDER.txt").exists() else []
                files.sort(key=lambda f: (order.index(f.stem) if f.stem in order else len(order), f.name))
            for f in files:
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

    def get(self, key: str) -> ArchSpec | None:
        return self.specs.get(key)

    def recommended(self) -> list[str]:
        return [k for k, s in self.specs.items() if s.recommended]

    def for_package(self, pkg) -> ArchSpec:
        """The architecture a model package was trained as: the library's (same key), else its own record.

        Raises:
            ValueError: A network of its own whose file is not in the library.
        """
        src = pkg.manifest.get("source", {})
        key = src.get("architecture_id", "")
        if key in self.specs:
            return self.specs[key]
        if src.get("architecture"):
            return from_definition(src["architecture"])
        raise ValueError(f"the architecture '{key}' is not in the library")

    def missing_libraries(self) -> dict[str, list[str]]:
        from modalytics.parameters import libraries
        from modalytics.parameters.registry import _declared_requires
        out = {k: libraries.missing(s.requires) for k, s in self.specs.items() if s.requires}
        out.update({p: libraries.missing(_declared_requires(Path(p))) for p in self.errors})
        return {k: v for k, v in out.items() if v}

    # ------------------------------------------------------------ the check
    def _checks_file(self) -> Path:
        return self.user / ".checks.json"

    def _check_key(self, spec: ArchSpec, in_channels: int) -> str:
        return f"{spec.key}:{in_channels}:{spec.file_hash()}:{_versions()}"

    def checked(self, spec: ArchSpec, in_channels: int = 1) -> dict | None:
        """The remembered check of this architecture as it is now (None: not checked yet)."""
        try:
            data = json.loads(self._checks_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data.get(self._check_key(spec, in_channels))

    def remember(self, spec: ArchSpec, result: dict, in_channels: int = 1) -> None:
        from modalytics import fileio
        try:
            data = json.loads(self._checks_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        prefix = f"{spec.key}:{in_channels}:"
        data = {k: v for k, v in data.items() if not k.startswith(prefix)}      # older versions of this file
        data[self._check_key(spec, in_channels)] = result
        self._checks_file().parent.mkdir(parents=True, exist_ok=True)
        fileio.write_text(self._checks_file(), json.dumps(data, indent=1))

    def check(self, spec: ArchSpec, in_channels: int = 1, use_saved: bool = True) -> dict:
        """Check the contract (see the module docstring), remembering the result.

        Returns:
            {"ok", "why", "params_m", "onnx_mb", "max_diff", "seconds"}.
        """
        if use_saved and (saved := self.checked(spec, in_channels)) is not None:
            return saved
        result = check(spec, in_channels)
        self.remember(spec, result, in_channels)
        return result


def _versions() -> str:
    """The library versions a check depends on (a new PyTorch or ONNX Runtime checks again)."""
    from importlib import metadata
    out = []
    for names in (("torch",), ("segmentation-models-pytorch",), ("timm",),
                  ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu")):   # read without importing them
        found = "?"
        for dist in names:
            try:
                found = metadata.version(dist)
                break
            except metadata.PackageNotFoundError:
                continue
        out.append(found)
    return "/".join(out)


# ---------------------------------------------------------------- building (PyTorch)
def build(spec: ArchSpec, in_channels: int, classes: int = 1, imagenet: bool = False):
    """The PyTorch network of an architecture, random or with its encoder from ImageNet.

    Other encoder weights are loaded afterwards (``training.encoders.load_into``).
    """
    if spec.custom:
        model = spec.build_fn(in_channels, classes, "imagenet" if imagenet else None)
    else:
        import segmentation_models_pytorch as smp
        model = getattr(smp, spec.decoder)(encoder_name=spec.encoder, encoder_weights="imagenet" if imagenet else None,
                                           in_channels=in_channels, classes=classes, activation=None,
                                           **spec.decoder_options)
    enc = getattr(model, "encoder", None)
    if hasattr(enc, "set_swish"):                          # ONNX cannot export EfficientNet's memory-efficient Swish
        enc.set_swish(memory_efficient=False)
    return model


def check(spec: ArchSpec, in_channels: int = 1) -> dict:
    """Build the network (random weights), export it to ONNX and compare ONNX Runtime with PyTorch.

    Returns:
        {"ok", "why", "params_m", "onnx_mb", "max_diff", "seconds"} — ``why`` says what failed.
    """
    import tempfile

    import numpy as np
    t0 = time.perf_counter()
    out = {"ok": False, "why": "", "params_m": None, "onnx_mb": None, "max_diff": None, "seconds": 0.0}
    try:
        import onnxruntime as ort
        import torch
        torch.manual_seed(0)
        # 1. Build
        try:
            model = build(spec, in_channels).float().eval()
        except Exception as exc:
            out["why"] = f"it cannot be built: {type(exc).__name__}: {exc}"
            return out
        out["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "model.onnx")
            # 2. Export, as model packages are written
            try:
                torch.onnx.export(model, torch.zeros(1, in_channels, *CHECK_SIZES[0]), f, input_names=["image"],
                                  output_names=["logits"], opset_version=17, do_constant_folding=True, dynamo=False,
                                  dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}})
            except Exception as exc:
                out["why"] = f"it does not export to ONNX: {type(exc).__name__}: {str(exc)[:300]}"
                return out
            out["onnx_mb"] = round(os.path.getsize(f) / 1e6, 1)
            # 3. Run both on two tile sizes; the outputs must agree
            try:
                sess = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
            except Exception as exc:
                out["why"] = f"ONNX Runtime cannot load it: {str(exc)[:300]}"
                return out
            rng = np.random.default_rng(0)
            worst = 0.0
            for h, w in CHECK_SIZES:
                x = rng.standard_normal((1, in_channels, h, w)).astype(np.float32)
                with torch.no_grad():
                    ref = model(torch.from_numpy(x)).numpy()
                if ref.shape != (1, 1, h, w):
                    out["why"] = f"a {h}×{w} tile gives logits of shape {tuple(ref.shape)}, not (1, 1, {h}, {w})"
                    return out
                try:
                    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                except Exception as exc:
                    out["why"] = f"ONNX Runtime fails on a {h}×{w} tile: {str(exc)[:300]}"
                    return out
                scale = max(float(np.abs(ref).max()), 1.0)
                worst = max(worst, float(np.abs(got - ref).max()) / scale)
            out["max_diff"] = worst
            if worst > TOLERANCE:
                out["why"] = f"ONNX and PyTorch disagree (largest difference {worst:.2g} of the logit range)"
                return out
        out["ok"] = True
        return out
    finally:
        out["seconds"] = round(time.perf_counter() - t0, 1)
