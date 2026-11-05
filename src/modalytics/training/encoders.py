"""
Encoder weights: the library of starting points for a new segmentation model's encoder (Qt-free).

A new model's encoder (the part that turns an image into features) can start from
weights learned elsewhere instead of at random. Which ones fit depends on the
architecture's encoder (ResNet-34 weights fit only a ResNet-34). The library holds:

* **ImageNet** — natural photographs; segmentation-models-pytorch downloads the
  weights of any of its encoders once (``kind: "imagenet"``).
* **Built-in downloads** (``resources/encoders/*.json``, ``kind: "download"``): one
  file per encoder, fetched once from a fixed address into the user library's
  ``encoders/cache``, its size (and, where known, SHA-256) checked. MicroNet (NASA,
  MIT licence) is one: encoders trained on ~100,000 microscopy images.
* **Your own** (``*.mencoder`` in ``paths.user_dir("encoders")``, ``kind: "package"``):
  a zip of ``manifest.json`` and ``weights.safetensors`` (numbers only, no code) — made
  by Import weights… from a PyTorch or safetensors file, or by pretraining.

Every entry says which encoders it fits, for which image types, its licence (and
whether trained models may be shared freely), and its references; a trained model
records the weights it started from.

**The encoders** (``catalogue``): the names segmentation-models-pytorch and timm know
(timm's as ``tu-<model>``) and which have published ImageNet weights (timm: its default
pretrained weights, usually ImageNet). Reading it imports both libraries (~5 s), so it
is read once, in the background (``load_catalogue``); until then ImageNet is assumed
to fit every encoder, and training checks for itself.

Loading (``load_into``): the weights are matched to the encoder by name and shape;
the first convolution is adapted when the channels differ (colour weights summed for
a one-channel phase map, as segmentation-models-pytorch does); a file that leaves
more than 5 % of the encoder unmatched is refused.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from modalytics import paths

BUILTIN = Path(__file__).resolve().parent.parent / "resources" / "encoders"
WEIGHTS = "weights.safetensors"
EXTENSION = ".mencoder"
IMAGENET = "imagenet"
MIN_MATCHED = 0.95                       # share of the encoder's tensors a weights file must fill


class Cancelled(Exception):
    """Raised when a download is cancelled."""


# ---------------------------------------------------------------- the encoders the libraries know
_CATALOGUE: dict | None = None
_CATALOGUE_LOCK = threading.Lock()


def _read_catalogue() -> dict:
    import segmentation_models_pytorch as smp
    import timm
    out = {}
    for name in smp.encoders.get_encoder_names():            # segmentation-models-pytorch's own
        settings = smp.encoders.encoders[name].get("pretrained_settings", {})
        out[name] = {"source": "smp", "imagenet": "imagenet" in settings, "weights": sorted(settings)}
    pretrained = {m.split(".")[0] for m in timm.list_models(pretrained=True)}
    for name in timm.list_models():                           # timm's, as smp names them
        out[f"tu-{name}"] = {"source": "timm", "imagenet": name in pretrained,
                             "weights": ["timm default"] if name in pretrained else []}
    return out


def catalogue(wait: bool = True) -> dict | None:
    """{encoder name: {"source": "smp" | "timm", "imagenet": bool, "weights": [...]}} (see the module docstring).

    Args:
        wait: Read it now if it is not read yet (~5 s); False: None until it is.

    Returns:
        The catalogue; {} when the training libraries are not installed.
    """
    global _CATALOGUE
    if _CATALOGUE is None and wait:
        with _CATALOGUE_LOCK:
            if _CATALOGUE is None:
                try:
                    _CATALOGUE = _read_catalogue()
                except ImportError:
                    _CATALOGUE = {}
    return _CATALOGUE


def load_catalogue(done: Callable[[dict], None] | None = None) -> threading.Thread:
    """Read the catalogue in a background thread; ``done(catalogue)`` is called from that thread."""
    def work():
        try:
            cat = catalogue()
            if done is not None:
                done(cat)
        except BaseException:                             # noqa: BLE001 — e.g. the program closing meanwhile
            pass
    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t


def imagenet_available(encoder: str, wait: bool = False) -> bool:
    """Whether ``encoder`` has ImageNet weights (True when not known yet, or a name of your own)."""
    cat = catalogue(wait)
    if not cat or encoder not in cat:
        return True
    return cat[encoder]["imagenet"]


def choices(library: "EncoderLibrary | None" = None) -> list[tuple[str, list[str]]]:
    """The encoder names to offer, those with starting weights first: [(name, [the weights that fit it])].

    Order: encoders with weights of the library beyond ImageNet (MicroNet, yours); segmentation-models-pytorch's
    with ImageNet; timm's with pretrained weights; then those that start only from random weights.
    """
    cat = catalogue() or {}
    library = library or EncoderLibrary()
    rows = []
    for name, info in cat.items():
        starts = [s.name for s in library.specs.values() if s.fits(name)]
        beyond = any(s.kind != IMAGENET and s.fits(name) for s in library.specs.values())
        rank = 0 if beyond else (1 if info["source"] == "smp" and info["imagenet"] else
                                 2 if info["imagenet"] else 3 if info["source"] == "smp" else 4)
        rows.append((rank, name, starts))
    rows.sort(key=lambda r: r[0])                         # stable: each library's own order inside a rank
    return [(name, starts) for _rank, name, starts in rows]


def known_encoder(encoder: str) -> tuple[bool, list[str]]:
    """(whether the libraries know ``encoder``, close names if not) — reads the catalogue."""
    cat = catalogue() or {}
    if not cat or encoder in cat:
        return True, []
    return False, difflib.get_close_matches(encoder, list(cat), n=3, cutoff=0.6)


@dataclass
class EncoderSpec:
    """One entry of the encoder-weights library (see the module docstring).

    Attributes:
        key, name, description: Which.
        kind: "imagenet", "download" or "package".
        encoders: The encoders it fits ("*": any encoder that has ImageNet weights).
        image_types: Image types it is meant for ("*": any).
        licence, redistributable: Its licence; whether models trained from it may be shared freely.
        references: [{"text", "doi"?}].
        files: ``download``: {encoder: {"url", "size", "sha256"?}}.
        homepage: Where it comes from.
        path: Its file (a package or a description).
    """

    key: str
    name: str
    kind: str
    description: str = ""
    encoders: tuple[str, ...] = ("*",)
    image_types: tuple[str, ...] = ("*",)
    licence: str = ""
    redistributable: bool = True
    references: list[dict] = field(default_factory=list)
    files: dict = field(default_factory=dict)
    homepage: str = ""
    path: str = ""
    source: str = "builtin"

    def fits(self, encoder: str, image_type: str | None = None) -> bool:
        """Whether it can start this encoder (for this image type, when given)."""
        if "*" in self.encoders:                          # ImageNet: where the libraries publish weights
            enc_ok = self.kind != IMAGENET or imagenet_available(encoder)
        else:
            enc_ok = encoder in self.encoders
        return enc_ok and (image_type is None or "*" in self.image_types or image_type in self.image_types)

    def record(self) -> dict:
        """What a trained model records about its starting weights."""
        return {"key": self.key, "name": self.name, "licence": self.licence, "redistributable": self.redistributable,
                "references": list(self.references), "homepage": self.homepage}

    def references_text(self) -> str:
        return "\n".join(r.get("text", "") + (f" doi:{r['doi']}" if r.get("doi") else "") for r in self.references)


def _spec_from_dict(d: dict, path: str, source: str) -> EncoderSpec:
    key = str(d.get("key", ""))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError("key must be a name of letters, digits and _")
    kind = d.get("kind", "package")
    if kind not in ("imagenet", "download", "package"):
        raise ValueError(f"unknown kind '{kind}'")
    encoders = tuple(d.get("encoders") or ())
    if kind == "download":
        files = d.get("files") or {}
        if not files or any("url" not in f for f in files.values()):
            raise ValueError("a download needs files: {encoder: {url, size}}")
        encoders = tuple(files)
    if not encoders:
        raise ValueError("encoders is empty")
    types = d.get("image_types") or ["*"]
    return EncoderSpec(key=key, name=str(d.get("name", key)), kind=kind, description=str(d.get("description", "")),
                       encoders=encoders, image_types=tuple([types] if isinstance(types, str) else types),
                       licence=str(d.get("licence", "")), redistributable=bool(d.get("redistributable", True)),
                       references=list(d.get("references") or []), files=dict(d.get("files") or {}),
                       homepage=str(d.get("homepage", "")), path=path, source=source)


def read_package(path: str | os.PathLike, source: str = "user") -> EncoderSpec:
    """Describe a ``.mencoder`` package (its manifest; the weights are read when used).

    Raises:
        ValueError: Not a usable package.
    """
    try:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            if WEIGHTS not in z.namelist():
                raise ValueError(f"it has no {WEIGHTS}")
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"not an encoder package: {exc}") from exc
    return _spec_from_dict(dict(manifest, kind="package"), str(path), source)


class EncoderLibrary:
    """Every encoder-weights entry: built in, then the user's (a user entry with a built-in's key replaces it).

    Attributes:
        specs: {key: EncoderSpec}, built-in order (ORDER.txt) first.
        errors: {file path: why it could not be read}.
    """

    def __init__(self, builtin: str | os.PathLike | None = None, user: str | os.PathLike | None = None) -> None:
        self.builtin_dir = Path(builtin) if builtin else BUILTIN
        self.user = Path(user) if user else paths.user_dir("encoders")
        self.reload()

    def reload(self) -> None:
        self.specs: dict[str, EncoderSpec] = {}
        self.errors: dict[str, str] = {}
        for folder, source in ((self.builtin_dir, "builtin"), (self.user, "user")):
            if not folder.is_dir():
                continue
            files = sorted(folder.glob("*.json")) + sorted(folder.glob(f"*{EXTENSION}"))
            if source == "builtin":                       # the shipped order: ORDER.txt, then the rest by name
                order = (folder / "ORDER.txt").read_text(encoding="utf-8").split()                     if (folder / "ORDER.txt").exists() else []
                files.sort(key=lambda f: (order.index(f.stem) if f.stem in order else len(order), f.name))
            for f in files:
                if f.name.startswith(("_", ".")):
                    continue
                try:
                    spec = read_package(f, source) if f.suffix == EXTENSION else \
                        _spec_from_dict(json.loads(f.read_text(encoding="utf-8")), str(f), source)
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    self.errors[str(f)] = str(exc)
                    continue
                self.specs[spec.key] = spec

    def get(self, key: str) -> EncoderSpec | None:
        return self.specs.get(key)

    def for_encoder(self, encoder: str, image_type: str | None = None) -> list[EncoderSpec]:
        """The entries that fit an encoder (and image type), ImageNet first."""
        return [s for s in self.specs.values() if s.fits(encoder, image_type)]

    def cache_dir(self) -> Path:
        return self.user / "cache"


# ---------------------------------------------------------------- the weights themselves (PyTorch)
def _download(url: str, target: Path, size: int | None, sha256: str | None,
              progress: Callable[[float], None] | None, cancelled: Callable[[], bool] | None) -> None:
    """Fetch ``url`` into ``target`` (through a ``.part`` file), checking size and SHA-256 when given."""
    import urllib.request
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    got = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as out:
        total = size or int(r.headers.get("Content-Length") or 0)
        while chunk := r.read(1 << 20):
            if cancelled is not None and cancelled():
                out.close()
                part.unlink(missing_ok=True)
                raise Cancelled()
            out.write(chunk)
            digest.update(chunk)
            got += len(chunk)
            if progress and total:
                progress(min(got / total, 1.0))
    if size and got != size:
        part.unlink(missing_ok=True)
        raise ValueError(f"the download of {url} is incomplete ({got} of {size} bytes)")
    if sha256 and digest.hexdigest() != sha256.lower():
        part.unlink(missing_ok=True)
        raise ValueError(f"the file from {url} is not the expected one (SHA-256 differs)")
    os.replace(part, target)


def _clean(state) -> dict:
    """A plain {name: tensor} from what a weights file holds ("state_dict" wrappers and "module." prefixes off)."""
    for key in ("state_dict", "model", "encoder"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    out = {}
    for k, v in state.items():
        if not hasattr(v, "shape"):
            continue
        out[re.sub(r"^(module\.|encoder\.)", "", k)] = v
    return out


def read_weights(path: str | os.PathLike) -> dict:
    """The tensors of a PyTorch (``.pth``, ``.pt``, ``.tar``: tensors only, never code) or safetensors file."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return _clean(load_file(str(path)))
    import torch
    return _clean(torch.load(str(path), map_location="cpu", weights_only=True))


def state_dict(spec: EncoderSpec, encoder: str, library: EncoderLibrary | None = None,
               progress: Callable[[float], None] | None = None, cancelled: Callable[[], bool] | None = None) -> dict:
    """The weights of ``spec`` for ``encoder`` ({name: tensor}); a download is fetched once.

    Raises:
        ValueError: The entry does not fit this encoder, or the file is not right.
    """
    if not spec.fits(encoder):
        raise ValueError(f"{spec.name} has no weights for the {encoder} encoder")
    if spec.kind == "package":
        from safetensors.torch import load
        with zipfile.ZipFile(spec.path) as z:
            return _clean(load(z.read(WEIGHTS)))
    if spec.kind == "download":
        f = spec.files[encoder]
        library = library or EncoderLibrary()
        target = library.cache_dir() / spec.key / Path(f["url"].split("?")[0]).name
        if not target.exists() or (f.get("size") and target.stat().st_size != int(f["size"])):
            _download(f["url"], target, int(f["size"]) if f.get("size") else None, f.get("sha256"), progress,
                      cancelled)
        return read_weights(target)
    raise ValueError("ImageNet weights are loaded by segmentation-models-pytorch itself")


def load_into(model, state: dict, encoder_attr: str = "encoder") -> dict:
    """Load encoder weights into ``model.<encoder_attr>`` (see the module docstring).

    Returns:
        {"matched", "total", "adapted": [names of tensors whose channels were adapted]}.

    Raises:
        ValueError: The weights fill less than ``MIN_MATCHED`` of the encoder.
    """
    import torch
    enc = getattr(model, encoder_attr, None)
    if enc is None:
        raise ValueError(f"the network has no '{encoder_attr}' to load encoder weights into")
    own = enc.state_dict()
    take, adapted = {}, []
    for name, target in own.items():
        src = state.get(name)
        if src is None:
            continue
        src = src.to(target.dtype)
        if tuple(src.shape) == tuple(target.shape):
            take[name] = src
        elif src.ndim == 4 and src.shape[0] == target.shape[0] and src.shape[2:] == target.shape[2:]:
            # The first convolution, other channels: summed into one (phase), or spread over more
            c = target.shape[1]
            if c == 1:
                take[name] = src.sum(dim=1, keepdim=True)
            else:
                reps = -(-c // src.shape[1])
                take[name] = src.repeat(1, reps, 1, 1)[:, :c] * (src.shape[1] / c)
            adapted.append(name)
    counted = [n for n in own if not n.endswith("num_batches_tracked")]
    matched = sum(1 for n in counted if n in take)
    if matched < MIN_MATCHED * len(counted):
        missing = [n for n in counted if n not in take][:5]
        raise ValueError(f"these weights fit only {matched} of the encoder's {len(counted)} tensors (e.g. missing "
                         f"{', '.join(missing)}): they are for another encoder")
    with torch.no_grad():
        enc.load_state_dict({**own, **take})
    return {"matched": matched, "total": len(counted), "adapted": adapted}


def import_weights(src: str | os.PathLike, encoder: str, name: str, *, key: str = "", description: str = "",
                   image_types: list[str] | None = None, licence: str = "", redistributable: bool = False,
                   references: list[dict] | None = None, homepage: str = "",
                   library: EncoderLibrary | None = None) -> EncoderSpec:
    """Make an encoder package from a weights file, after checking it fits ``encoder``.

    Raises:
        ValueError: The file does not fit the encoder, or the key is taken.
    """
    import segmentation_models_pytorch as smp
    from safetensors.torch import save
    library = library or EncoderLibrary()
    state = read_weights(src)
    probe = smp.encoders.get_encoder(encoder, in_channels=3, weights=None)  # the encoder, to match against

    class _Holder:                                        # load_into expects a model with an encoder
        pass
    holder = _Holder()
    holder.encoder = probe
    load_into(holder, state)
    kept = {k: v.contiguous() for k, v in probe.state_dict().items()}       # only what the encoder uses
    key = key or re.sub(r"[^a-z0-9]+", "_", f"{name} {encoder}".lower()).strip("_")
    if key[:1].isdigit():
        key = "w_" + key
    if library.get(key) is not None:
        raise ValueError(f"'{key}' is already in the library")
    manifest = {"format": "modalytics-encoder", "format_version": 1, "key": key, "name": name,
                "description": description, "encoders": [encoder], "image_types": image_types or ["*"],
                "licence": licence, "redistributable": redistributable, "references": references or [],
                "homepage": homepage, "source_file": Path(src).name}
    out = library.user / f"{key}{EXTENSION}"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        z.writestr(WEIGHTS, save(kept))
    os.replace(part, out)
    library.reload()
    return library.get(key)
