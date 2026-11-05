"""
Automatic segmentation with ONNX models (Qt-free, runs in a worker thread).

Pipeline for one region of one image:

1. **Read** the region at the model's training resolution: pick the finest
   pyramid level not coarser than the model's µm/px, read the covering tiles,
   resample to exactly the model's µm/px. (HoloPath's WSI gland model failed at
   pyramid level 1 because it was fed the wrong scale; here scale is explicit.)
2. **Predict** exactly as HoloPath did: z-score over the region, reflect-pad to
   a multiple of 32, 512-px tiles with 256-px overlap, Gaussian-weighted
   blending, sigmoid.
3. **Vectorise**: threshold, trace outlines with holes (OpenCV), convert to the
   image's local µm, drop specks, simplify by half a pixel.

Model packages (``.mmodel``) are zips of ``model.onnx`` + ``manifest.json``
(see ``scripts/convert_hmodel.py``). They run on the best device ONNX Runtime
can use here (``services.compute``: CUDA, DirectML, …, or the CPU).
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from modalytics import paths
from modalytics.core import annotations as ann
from modalytics.render import tiling

MODELS_DIR = Path(os.environ.get("MODALYTICS_MODELS", paths.app_root() / "models"))   # the shipped models


def model_dirs() -> list[Path]:
    """Folders searched for model packages: the models shipped with Modalytics, then the user
    library's (trained or imported ones; a user model with the same id wins)."""
    return [MODELS_DIR, paths.user_dir("models")]


BACKGROUND_LEVEL = 215       # brightness (0–255) above which an unsaturated RGB pixel is empty glass
BACKGROUND_MAX_SAT = 25      # max channel spread (0–255) of empty glass
BACKGROUND_FRACTION = 0.6    # objects more background than this are dropped


class Cancelled(Exception):
    """Raised inside a run when the user cancels."""


## Model packages ##

@dataclass
class ModelPackage:
    """A segmentation model and what it needs to run correctly.

    Attributes:
        path: The ``.mmodel`` file.
        manifest: Parsed ``manifest.json``.
    """

    path: str
    manifest: dict = field(repr=False)

    @property
    def id(self) -> str:
        """The package file name without ``.mmodel``: how profiles refer to the model."""
        return Path(self.path).stem

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def modality(self) -> str:
        return self.manifest["modality"]

    @property
    def um_per_px(self) -> float:
        return float(self.manifest["input_um_per_px"])

    @property
    def output_class(self) -> tuple[str, str]:
        c = self.manifest["classes"][0]
        return c["group"], c.get("subgroup", "")

    def onnx_bytes(self) -> bytes:
        with zipfile.ZipFile(self.path) as z:
            return z.read("model.onnx")


def load_package(path: str) -> ModelPackage:
    """Read a ``.mmodel`` manifest (the network itself is loaded on first use)."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
    if manifest.get("format") != "modalytics-model":
        raise ValueError(f"{path} is not a Modalytics model package")
    return ModelPackage(os.path.abspath(path), manifest)


def find_models(folder: str | os.PathLike | None = None) -> list[ModelPackage]:
    """All model packages (sorted by name).

    Args:
        folder: One folder to scan; None scans every ``model_dirs()`` folder, a later
            folder's package replacing an earlier one with the same id.
    """
    by_id: dict[str, ModelPackage] = {}
    for d in ([Path(folder)] if folder is not None else model_dirs()):
        for p in sorted(d.glob("*.mmodel")) if d.is_dir() else []:
            try:
                pkg = load_package(str(p))
            except Exception as exc:                  # a broken package must not hide the others
                print(f"⚠️ {p.name}: {exc}")
                continue
            by_id[pkg.id] = pkg
    return sorted(by_id.values(), key=lambda m: m.name)


def installed_models() -> dict[str, Path]:
    """{model id: package file} of every installed model."""
    return {m.id: Path(m.path) for m in find_models()}


## Inference ##

_SESSIONS: dict[tuple[str, str], object] = {}


def session(pkg: ModelPackage, device: str = "auto"):
    """A cached ONNX Runtime session on the requested device.

    Args:
        pkg: Model package.
        device: "auto" (the best device on this machine), "cpu", or a device key
            such as "dml:1" (see ``services.compute``).

    Returns:
        onnxruntime.InferenceSession
    """
    import onnxruntime as ort
    from modalytics.services import compute
    key = (pkg.path, compute.resolve(device).key)
    if key not in _SESSIONS:
        sess = ort.InferenceSession(pkg.onnx_bytes(), providers=compute.providers(device))
        compute.check_session(sess, device)          # warns if it silently fell back to the CPU
        _SESSIONS[key] = sess
    return _SESSIONS[key]


def _best_batch(sess) -> int:
    """Tiles per call. DirectML here is fastest one tile at a time (measured: 9.6 ms/tile
    at batch 1 vs 25 ms/tile at batch 4 on an RX 7900 XT); CPU and CUDA like small batches."""
    return 1 if sess.get_providers()[0] == "DmlExecutionProvider" else 4


def _gaussian(tile: int) -> np.ndarray:
    sigma = tile / 4.0
    ax = np.linspace(-(tile - 1) / 2.0, (tile - 1) / 2.0, tile)
    g = np.exp(-0.5 * (ax / sigma) ** 2).astype(np.float32)
    return np.outer(g, g)


MIN_TISSUE = 0.002          # a tile with less tissue than this fraction is pure glass: skipped

from concurrent.futures import ThreadPoolExecutor  # noqa: E402  (kept next to its only users)

_GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx")      # model calls, in order
_POST = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vectorize")  # block post-processing


def predict(pkg: ModelPackage, image: np.ndarray, device: str = "auto", batch: int | None = None,
            progress: Callable[[float], None] | None = None, cancelled: Callable[[], bool] | None = None,
            stats: tuple[float, float] | None = None, overlap: int | None = None,
            tissue: np.ndarray | None = None, tile: int | None = None) -> np.ndarray:
    """Probability map of an image already at the model's resolution (HoloPath-identical by default).

    Args:
        pkg: Model package.
        image: (H, W) or (H, W, C) array (any dtype).
        device: "auto", "cpu" or a device key (``services.compute``).
        batch: Tiles per inference call (default: best for the provider).
        progress: Called with 0–1.
        cancelled: Returns True to abort (raises ``Cancelled``).
        stats: (mean, std) to normalise with; default = this image's own (as HoloPath).
            Block-wise runs pass the statistics of the whole region.
        overlap: Tile overlap in px (default: the model's, 256 = as trained).
        tile: Inference tile size in px (default: the model's 512). The networks are fully
            convolutional, so larger tiles give the same kind of output with more context.
        tissue: Optional (H, W) bool map; tiles without tissue are not run and
            do not take part in the blending (their pixels get probability 0).

    Returns:
        np.ndarray: (H, W) float32 probabilities in [0, 1].
    """
    m = pkg.manifest
    tile, mult = int(m["tile"] if tile is None else tile), int(m["pad_multiple"])
    overlap = int(m["overlap"] if overlap is None else overlap)
    h, w = image.shape[:2]
    # 1. z-score over the whole region (scalar mean/std over all channels, as in training)
    img = image.astype(np.float32)
    mean, std = stats if stats is not None else (float(img.mean()), float(img.std()))
    if std > 1e-8:
        img = (img - mean) / std
    if img.ndim == 2:
        img = img[..., None]
    # 2. Reflect-pad to a multiple of 32 and to at least one tile
    ph = max((mult - h % mult) % mult, tile - h)
    pw = max((mult - w % mult) % mult, tile - w)
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect" if min(h, w) > 1 else "edge")
    H, W = img.shape[:2]
    # 3. Tile starts (always one tile touching the far edge); skip pure-glass tiles
    stride = tile - overlap

    def starts(dim):
        s = list(range(0, dim - tile + 1, stride))
        if not s or s[-1] + tile < dim:
            s.append(max(0, dim - tile))
        return s

    ys, xs = np.array(starts(H)), np.array(starts(W))
    run = np.ones((len(ys), len(xs)), bool)
    if tissue is not None:
        import cv2
        t = np.pad(tissue, ((0, ph), (0, pw)), mode="constant").astype(np.uint8)
        integral = cv2.integral(t)                                         # O(1) tissue sum per tile
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        frac = (integral[Y + tile, X + tile] - integral[Y, X + tile] - integral[Y + tile, X]
                + integral[Y, X]) / (tile * tile)
        has = (frac >= MIN_TISSUE).astype(np.int32)
        # Run a tile if it OR any tile overlapping it has tissue: every pixel near tissue then gets
        # exactly the tiles of a full run (glass tiles vote "no gland" at tissue edges; dropping
        # those votes turned glass next to glands into gland). Only deep glass is skipped.
        ny = (np.abs(ys[:, None] - ys[None, :]) < tile).astype(np.int32)
        nx = (np.abs(xs[:, None] - xs[None, :]) < tile).astype(np.int32)
        run = (ny @ has @ nx.T) > 0
    coords = [(int(ys[i]), int(xs[j])) for i, j in zip(*np.nonzero(run))]
    gauss = _gaussian(tile)
    prob_sum = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    sess = session(pkg, device)
    batch = batch or _best_batch(sess)
    # 4. Inference with Gaussian blending. A one-worker executor runs the model on tile k+1
    #    while this thread blends tile k (ONNX Runtime releases the GIL), so GPU and CPU overlap.
    def submit(k):
        chunk = coords[k:k + batch]
        x = np.ascontiguousarray(np.stack([img[y0:y0 + tile, x0:x0 + tile].transpose(2, 0, 1)
                                           for y0, x0 in chunk]), dtype=np.float32)
        return chunk, _GPU.submit(sess.run, None, {"image": x})

    pending = submit(0) if coords else None
    k = 0
    while pending is not None:
        if cancelled is not None and cancelled():
            pending[1].cancel()
            raise Cancelled()
        chunk, future = pending
        pending = submit(k + batch) if k + batch < len(coords) else None
        logits = future.result()[0][:, 0]
        prob = 1.0 / (1.0 + np.exp(-logits))
        for (y0, x0), p in zip(chunk, prob):
            prob_sum[y0:y0 + tile, x0:x0 + tile] += p * gauss
            weight[y0:y0 + tile, x0:x0 + tile] += gauss
        k += batch
        if progress is not None:
            progress(min(1.0, k / len(coords)))
    out = np.where(weight > 1e-6, prob_sum / np.maximum(weight, 1e-6), 0.0)
    return out[:h, :w].astype(np.float32)


def tissue_map(pixels: np.ndarray) -> np.ndarray | None:
    """True where an RGB pixel is not bare glass (bright and colourless); None for non-RGB data."""
    if pixels.ndim != 3 or pixels.dtype != np.uint8:
        return None
    # Element-wise min/max of the three channels: ~40× faster than .min(axis=2) on large images
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    lo = np.minimum(np.minimum(r, g), b)
    hi = np.maximum(np.maximum(r, g), b)
    glass = (lo >= BACKGROUND_LEVEL) & ((hi - lo) <= BACKGROUND_MAX_SAT)
    return ~glass


## Reading a region at the model's resolution ##

def read_region(image, rect_local: tuple[float, float, float, float], um_per_px: float) -> tuple[np.ndarray, tuple]:
    """Pixels of a local-µm rectangle of an image, resampled to ``um_per_px``.

    Raises ``ValueError`` for an image whose file is not found (its pixels are a placeholder).

    Args:
        image: ImageNode.
        rect_local: (x0, y0, x1, y1) in the image's local µm (clipped to the image).
        um_per_px: Target resolution (the model's).

    Returns:
        tuple: (array (H, W[, C]), (x0, y0, px_x, px_y)) — the local µm of the
        array's top-left corner and the µm size of its pixels.
    """
    import cv2

    src = image.source
    if getattr(src, "missing", False):
        raise ValueError(f"{image.name}: the image file is not found (relink it in Properties)")
    sx, sy = image.pixel_size_um
    w_um, h_um = image.size_um
    x0, y0 = max(0.0, rect_local[0]), max(0.0, rect_local[1])
    x1, y1 = min(w_um, rect_local[2]), min(h_um, rect_local[3])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The region does not overlap the image")
    # 1. Finest level not coarser than the target (so we never upsample unnecessarily)
    sizes = [sx * lv.downsample[1] for lv in src.levels]
    level = max((i for i, s in enumerate(sizes) if s <= um_per_px * 1.001), default=0)
    lv = src.levels[level]
    px_x, px_y = sx * lv.downsample[1], sy * lv.downsample[0]
    c0, r0 = int(math.floor(x0 / px_x)), int(math.floor(y0 / px_y))
    c1, r1 = int(math.ceil(x1 / px_x)), int(math.ceil(y1 / px_y))
    c1, r1 = min(c1, lv.shape[1]), min(r1, lv.shape[0])
    # 2. Read the covering tiles and crop
    th, tw = lv.tile
    rows = []
    for ty in range(r0 // th, (r1 - 1) // th + 1):
        rows.append(np.concatenate([src.read_tile(level, ty, tx) for tx in range(c0 // tw, (c1 - 1) // tw + 1)],
                                   axis=1))
    block = np.concatenate(rows, axis=0)
    block = block[r0 - (r0 // th) * th: r1 - (r0 // th) * th, c0 - (c0 // tw) * tw: c1 - (c0 // tw) * tw]
    # 3. Resample to the model's µm/px
    out_w = max(1, int(round((c1 - c0) * px_x / um_per_px)))
    out_h = max(1, int(round((r1 - r0) * px_y / um_per_px)))
    if (out_w, out_h) != (block.shape[1], block.shape[0]):
        interp = cv2.INTER_AREA if out_w < block.shape[1] else cv2.INTER_LINEAR
        block = cv2.resize(np.ascontiguousarray(block), (out_w, out_h), interpolation=interp)
    real_px_x = (c1 - c0) * px_x / out_w
    real_px_y = (r1 - r0) * px_y / out_h
    return block, (c0 * px_x, r0 * px_y, real_px_x, real_px_y)


## Vectorisation ##

def drop_background_objects(mask: np.ndarray, pixels: np.ndarray, fraction: float = BACKGROUND_FRACTION,
                            tissue: np.ndarray | None = None) -> np.ndarray:
    """Remove connected objects that are mostly empty slide background.

    Models trained on tissue can call large areas of bare glass "gland". A
    per-pixel filter would also cut the (white) lumen out of every real gland,
    so the decision is per object: an object is dropped when more than
    ``fraction`` of its pixels are bright and colourless.

    Args:
        mask: (H, W) bool prediction.
        pixels: (H, W, 3) uint8 RGB at the same resolution.
        fraction: Background share above which an object is dropped.
        tissue: ``tissue_map(pixels)`` if already computed.

    Returns:
        np.ndarray: The filtered mask.
    """
    import cv2

    background = ~(tissue if tissue is not None else tissue_map(pixels))
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    size = np.bincount(labels.ravel(), minlength=n)
    bg = np.bincount(labels.ravel(), weights=background.ravel(), minlength=n)
    keep = (bg / np.maximum(size, 1)) <= fraction
    keep[0] = False
    return keep[labels]


def vectorize(mask: np.ndarray, frame: tuple, min_area_um2: float = 0.0, simplify_px: float = 0.5) -> MultiPolygon:
    """Binary mask → MultiPolygon (with holes) in the image's local µm.

    Args:
        mask: (H, W) bool.
        frame: (x0, y0, px_x, px_y) from ``read_region``.
        min_area_um2: Drop objects smaller than this.
        simplify_px: Douglas–Peucker tolerance in pixels (0.5 removes staircase noise only).

    Returns:
        MultiPolygon: Valid, in local µm.
    """
    import cv2

    x0, y0, px, py = frame
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return ann.EMPTY
    hierarchy = hierarchy[0]
    to_um = lambda c: np.column_stack([x0 + (c[:, 0, 0] + 0.5) * px, y0 + (c[:, 0, 1] + 0.5) * py])
    polys = []
    # RETR_CCOMP: top-level contours are outlines, their children are holes
    for i, (_nxt, _prev, child, parent) in enumerate(hierarchy):
        if parent != -1 or len(contours[i]) < 3:
            continue
        holes = []
        j = child
        while j != -1:
            if len(contours[j]) >= 3:
                holes.append(to_um(contours[j]))
            j = hierarchy[j][0]
        polys.append(Polygon(to_um(contours[i]), holes))
    if not polys:
        return ann.EMPTY
    # Everything below works per object (vectorised shapely calls). Operating on one big
    # MultiPolygon would make GEOS union all objects at every step (~2 s per 4096² block).
    parts = np.array(polys, dtype=object)
    bad = ~shapely.is_valid(parts)
    if bad.any():
        parts[bad] = shapely.make_valid(parts[bad])
    # Contours run through pixel centres: grow by half a pixel so outlines follow pixel edges
    # (areas are then exact for the mask, and objects split between blocks touch and merge).
    # Separate 8-connected objects are ≥ 1 px apart, so the grown pieces never overlap.
    parts = shapely.buffer(parts, 0.5 * min(px, py), join_style="mitre")
    if simplify_px > 0:
        parts = shapely.simplify(parts, simplify_px * min(px, py), preserve_topology=True)
    out = [q for g in parts for q in shapely.get_parts(shapely.get_parts(g))
           if isinstance(q, Polygon) and q.area >= max(min_area_um2, 1e-12)]
    return MultiPolygon(out) if out else ann.EMPTY


## One call: region → geometry ##

def _model_input(pkg: ModelPackage, pixels: np.ndarray) -> np.ndarray:
    """Match the model's channel count (grey ↔ RGB)."""
    cin = pkg.manifest["in_channels"]
    if cin == 1 and pixels.ndim == 3:
        return pixels[..., :3].astype(np.float32).mean(axis=2)
    if cin == 3 and pixels.ndim == 2:
        return np.repeat(pixels[..., None], 3, axis=2)
    return pixels[..., :3] if pixels.ndim == 3 else pixels


CACHE_VERSION = 2   # bump whenever the prediction pipeline changes (v2: tiles next to tissue always run)
CACHE_DIR = Path(os.environ.get("MODALYTICS_CACHE",
                               Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "Modalytics" / "cache"))


def _cache_dir(image, pkg: ModelPackage, overlap: int, stats, block_px: int) -> Path | None:
    """Folder for a whole-image probability cache, or None if the image has no stable identity."""
    import hashlib
    path = getattr(image.source, "path", None)
    if path is None:
        return None
    st = os.stat(pkg.path)
    key = json.dumps([CACHE_VERSION, os.path.abspath(path), list(image.source.shape), image.pixel_size_um,
                      pkg.path, st.st_size, int(st.st_mtime), overlap, block_px,
                      round(stats[0], 4), round(stats[1], 4)])
    return CACHE_DIR / "segmentation" / hashlib.sha1(key.encode()).hexdigest()[:16]


def _coarse(image, rect, pkg: ModelPackage):
    """Region statistics and tissue map from a coarse read (≤ ~4 Mpx)."""
    x0, y0, x1, y1 = rect
    um = pkg.um_per_px
    coarse_um = um * max(1.0, math.sqrt((x1 - x0) * (y1 - y0) / um ** 2 / 4e6))
    coarse, frame = read_region(image, rect, coarse_um)
    c = _model_input(pkg, coarse).astype(np.float32)
    return (float(c.mean()), float(c.std())), tissue_map(coarse), frame


def segment(image, rect_local, pkg: ModelPackage, threshold: float = 0.5, min_area_um2: float = 20.0,
            device: str = "auto", progress=None, cancelled=None, block_px: int = 4096,
            ignore_background: bool = True, overlap: int | None = None, use_cache: bool = True) -> MultiPolygon:
    """Segment a region of an image and return the objects in local µm.

    Regions larger than ``block_px``² model pixels (e.g. a whole slide at full
    resolution, which would need ~35 GB for one probability map) are processed
    block by block: each block is read with a margin of one tile overlap, so
    predictions at block edges see the same context as in a single pass, and
    only its core is kept. All blocks are normalised with the statistics of
    the whole region (computed on a coarse read), like a single pass.

    Speed-ups for large RGB regions (``ignore_background``): blocks and tiles of
    pure glass are not run at all. Whole-image runs keep their probability maps
    in a cache, so re-running (e.g. with another threshold) skips the model.

    Args:
        image: ImageNode.
        rect_local: (x0, y0, x1, y1) local µm.
        pkg: Model package.
        threshold: Probability threshold.
        min_area_um2: Smallest object kept.
        device: "auto", "cpu" or a device key (``services.compute``).
        progress: Called with 0–1.
        cancelled: Returns True to abort.
        block_px: Block size in model pixels.
        ignore_background: For RGB images, skip glass and drop objects that are
            mostly glass (see ``drop_background_objects``).
        overlap: Tile overlap in px (default: the model's 256 = as trained; 128 is ~1.8× faster).
        use_cache: Use / fill the whole-image probability cache.

    Returns:
        MultiPolygon: The segmented objects (local µm).
    """
    um = pkg.um_per_px
    overlap = int(pkg.manifest["overlap"] if overlap is None else overlap)
    w_um, h_um = image.size_um
    x0, y0 = max(0.0, rect_local[0]), max(0.0, rect_local[1])
    x1, y1 = min(w_um, rect_local[2]), min(h_um, rect_local[3])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The region does not overlap the image")
    nx = max(1, math.ceil((x1 - x0) / um / block_px))
    ny = max(1, math.ceil((y1 - y0) / um / block_px))

    def post(prob, pixels, tissue, keep=None):
        found = prob >= threshold
        if keep is not None:
            found &= keep
        if ignore_background and tissue is not None:
            found = drop_background_objects(found, pixels, tissue=tissue)
        return found

    # 1. Small region: one pass, exactly as HoloPath (glass tiles skipped only for RGB)
    if nx * ny == 1:
        pixels, frame = read_region(image, (x0, y0, x1, y1), um)
        tissue = tissue_map(pixels) if ignore_background else None
        prob = predict(pkg, _model_input(pkg, pixels), device, progress=progress, cancelled=cancelled,
                       overlap=overlap, tissue=tissue)
        return vectorize(post(prob, pixels, tissue), frame, min_area_um2)

    # 2. Large region: whole-region statistics and tissue map from a coarse read
    stats, coarse_tissue, cframe = _coarse(image, (x0, y0, x1, y1), pkg)
    whole = (x0, y0, x1, y1) == (0.0, 0.0, w_um, h_um)
    cache = _cache_dir(image, pkg, overlap, stats, block_px) if (use_cache and whole) else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    margin = overlap * um
    bw, bh = (x1 - x0) / nx, (y1 - y0) / ny
    blocks = []
    for by in range(ny):
        for bx in range(nx):
            cx0, cy0 = x0 + bx * bw, y0 + by * bh
            # 3. Skip blocks with no tissue at all (coarse map, block + margin)
            if ignore_background and coarse_tissue is not None:
                fx, fy, fpx, fpy = cframe
                r0, r1 = int((cy0 - margin - fy) // fpy), int((cy0 + bh + margin - fy) // fpy) + 1
                c0, c1 = int((cx0 - margin - fx) // fpx), int((cx0 + bw + margin - fx) // fpx) + 1
                if not coarse_tissue[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)].any():
                    continue
            blocks.append((bx, by, cx0, cy0))
    inner, seam, pending = [], [], []

    def collect(future):
        a, b = future.result()
        inner.extend(a)
        seam.extend(b)

    for k, (bx, by, cx0, cy0) in enumerate(blocks):
        cx1, cy1 = cx0 + bw, cy0 + bh
        pixels, frame = read_region(image, (cx0 - margin, cy0 - margin, cx1 + margin, cy1 + margin), um)
        cached = cache / f"{by}_{bx}.npz" if cache is not None else None
        tissue = tissue_map(pixels) if ignore_background else None
        if cached is not None and cached.exists():
            prob = np.load(cached)["prob"].astype(np.float32) / 255.0          # 4. cache hit: no inference
            if progress is not None:
                progress((k + 1) / len(blocks))
        else:
            prob = predict(pkg, _model_input(pkg, pixels), device, stats=stats, cancelled=cancelled,
                           overlap=overlap, tissue=tissue,
                           progress=None if progress is None else (lambda f, k=k: progress((k + f) / len(blocks))))
            if cached is not None:
                q = np.round(prob * 255).astype(np.uint8)
                q[q < 13] = 0          # p < 0.05 → 0 and p > 0.95 → 1: thresholds are limited to
                q[q > 242] = 255       # 0.05–0.95, so this cannot change a result, but it compresses ~10×
                np.savez_compressed(cached, prob=q)
        # 5. Keep only the core (pixels whose centre lies in it) and vectorise — on a worker,
        #    so the next block's inference starts right away
        def finish(prob=prob, pixels=pixels, tissue=tissue, frame=frame, core_rect=(cx0, cy0, cx1, cy1)):
            fx, fy, px, py = frame
            cols = (np.arange(prob.shape[1]) + 0.5) * px + fx
            rows = (np.arange(prob.shape[0]) + 0.5) * py + fy
            core = np.zeros(prob.shape, bool)
            core[np.ix_((rows >= core_rect[1]) & (rows < core_rect[3]),
                        (cols >= core_rect[0]) & (cols < core_rect[2]))] = True
            geom = vectorize(post(prob, pixels, tissue, core), frame, 0.0)
            # Objects touching the core's edge may continue in the next block: merge those later
            ex0, ey0, ex1, ey1 = core_rect
            tol = 1.5 * max(px, py)
            a, b = [], []
            for g in geom.geoms:
                gx0, gy0, gx1, gy1 = g.bounds
                (b if (gx0 <= ex0 + tol or gy0 <= ey0 + tol or gx1 >= ex1 - tol or gy1 >= ey1 - tol) else a).append(g)
            return a, b
        # At most one finished block waits for vectorisation: memory stays bounded even when
        # outlines are slower than the GPU (an unbounded queue grew to many GB on a whole slide)
        while len(pending) >= 1:
            collect(pending.pop(0))
        pending.append(_POST.submit(finish))
        if cancelled is not None and cancelled():
            raise Cancelled()
    for f in pending:
        collect(f)
    # 6. Merge across blocks (objects cut by a block edge touch and fuse), then drop specks
    # 6. Only objects at block seams are merged (a union over all objects took ~15 min on a slide)
    joined = list(shapely.get_parts(shapely.union_all(seam))) if seam else []
    out = [g for g in inner + joined if isinstance(g, Polygon) and g.area >= min_area_um2]
    return MultiPolygon(out) if out else ann.EMPTY


## Probability maps (for registration) ##

def probability_map(image, rect_local, pkg: ModelPackage, out_um: float, device: str = "auto",
                    progress=None, cancelled=None, block_px: int = 4096,
                    ignore_background: bool = True) -> tuple[np.ndarray, tuple]:
    """The model's probability map of a region, at ``out_um`` µm per pixel.

    Prediction runs at the model's own resolution, block by block (bounded
    memory, region statistics as in ``segment``); each block's probabilities
    are averaged down to ``out_um`` and stitched.

    Args:
        image: ImageNode.
        rect_local: (x0, y0, x1, y1) local µm (clipped to the image).
        pkg: Model package.
        out_um: Output pixel size (µm), usually coarser than the model's.
        device, progress, cancelled, block_px, ignore_background: as in ``segment``.

    Returns:
        tuple: (float32 array, (x0, y0, out_um, out_um)) — the array's top-left
        corner in local µm and its pixel size.
    """
    import cv2

    um = pkg.um_per_px
    w_um, h_um = image.size_um
    x0, y0 = max(0.0, rect_local[0]), max(0.0, rect_local[1])
    x1, y1 = min(w_um, rect_local[2]), min(h_um, rect_local[3])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The region does not overlap the image")
    out_w, out_h = max(1, int(round((x1 - x0) / out_um))), max(1, int(round((y1 - y0) / out_um)))
    out = np.zeros((out_h, out_w), np.float32)
    nx = max(1, math.ceil((x1 - x0) / um / block_px))
    ny = max(1, math.ceil((y1 - y0) / um / block_px))
    stats, coarse_tissue, cframe = _coarse(image, (x0, y0, x1, y1), pkg) if nx * ny > 1 else (None, None, None)
    margin = int(pkg.manifest["overlap"]) * um
    # Blocks aligned to the output grid, so each fills a whole number of output pixels
    bw = math.ceil((x1 - x0) / nx / out_um) * out_um
    bh = math.ceil((y1 - y0) / ny / out_um) * out_um
    for by in range(ny):
        for bx in range(nx):
            if cancelled is not None and cancelled():
                raise Cancelled()
            cx0, cy0 = x0 + bx * bw, y0 + by * bh
            cx1, cy1 = min(x1, cx0 + bw), min(y1, cy0 + bh)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            pixels, frame = read_region(image, (cx0 - margin, cy0 - margin, cx1 + margin, cy1 + margin), um)
            tissue = tissue_map(pixels) if ignore_background else None
            if tissue is not None and not tissue.any():
                continue
            k = by * nx + bx
            prob = predict(pkg, _model_input(pkg, pixels), device, stats=stats, cancelled=cancelled, tissue=tissue,
                           progress=None if progress is None else (lambda f, k=k: progress((k + f) / (nx * ny))))
            # Resample the block (frame = its own local-µm grid) onto the output grid
            fx, fy, px, py = frame
            c0, r0 = int(round((cx0 - x0) / out_um)), int(round((cy0 - y0) / out_um))
            c1, r1 = min(out_w, int(round((cx1 - x0) / out_um))), min(out_h, int(round((cy1 - y0) / out_um)))
            if c1 <= c0 or r1 <= r0:
                continue
            m = np.array([[out_um / px, 0, (x0 + c0 * out_um - fx) / px + 0.5 * out_um / px - 0.5],
                          [0, out_um / py, (y0 + r0 * out_um - fy) / py + 0.5 * out_um / py - 0.5]], np.float32)
            small = cv2.resize(prob, None, fx=px / out_um, fy=py / out_um, interpolation=cv2.INTER_AREA) \
                if out_um > px * 1.5 else prob
            if small is not prob:                        # mapping into the downsampled block
                sx, sy = prob.shape[1] / small.shape[1], prob.shape[0] / small.shape[0]
                m = np.array([[m[0, 0] / sx, 0, (m[0, 2] + 0.5) / sx - 0.5],
                              [0, m[1, 1] / sy, (m[1, 2] + 0.5) / sy - 0.5]], np.float32)
            out[r0:r1, c0:c1] = cv2.warpAffine(small, m, (c1 - c0, r1 - r0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                               borderMode=cv2.BORDER_CONSTANT)
    if progress is not None:
        progress(1.0)
    return out, (x0, y0, out_um, out_um)


## Time estimate ##

_TILE_MS: dict[tuple[str, str], float] = {}
END_TO_END = 2.25    # measured (36 mm² of tissue, 9 blocks): a run costs ~2.25× the isolated model time per tile
                     # (margins, edge tiles, data transfer, slower sustained GPU clocks)


def tile_ms(pkg: ModelPackage, device: str = "auto") -> float:
    """Model-only milliseconds per tile on this machine (measured once on realistic input)."""
    import time
    from modalytics.services import compute
    key = (pkg.path, compute.resolve(device).key)
    if key not in _TILE_MS:
        sess = session(pkg, device)
        tile, cin = int(pkg.manifest["tile"]), int(pkg.manifest["in_channels"])
        x = np.random.default_rng(0).normal(size=(1, cin, tile, tile)).astype(np.float32)
        sess.run(None, {"image": x})                              # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            sess.run(None, {"image": x})
        _TILE_MS[key] = (time.perf_counter() - t0) / 5 * 1e3
    return _TILE_MS[key]


def estimate(image, rect_local, pkg: ModelPackage, overlap: int | None = None, device: str = "auto",
             ignore_background: bool = True) -> dict:
    """How much work a segmentation will be, before running it.

    Returns:
        dict: area_mm2 (region), tissue_mm2 (what will be analysed), megapixels
        (at the model's resolution), tiles, seconds (on this machine).
    """
    um = pkg.um_per_px
    overlap = int(pkg.manifest["overlap"] if overlap is None else overlap)
    w_um, h_um = image.size_um
    x0, y0 = max(0.0, rect_local[0]), max(0.0, rect_local[1])
    x1, y1 = min(w_um, rect_local[2]), min(h_um, rect_local[3])
    area = max(0.0, (x1 - x0) * (y1 - y0))
    fraction = 1.0
    if ignore_background and area > 0:
        _stats, tissue, _f = _coarse(image, (x0, y0, x1, y1), pkg)
        if tissue is not None:
            fraction = float(tissue.mean())
    tissue_um2 = area * fraction
    stride = int(pkg.manifest["tile"]) - overlap
    tiles = tissue_um2 / (stride * um) ** 2
    return {"area_mm2": area / 1e6, "tissue_mm2": tissue_um2 / 1e6, "megapixels": tissue_um2 / um ** 2 / 1e6,
            "tiles": int(tiles), "seconds": tiles * tile_ms(pkg, device) * END_TO_END / 1e3}


def pick_model(models: list[ModelPackage], modality: str, profile=None, group: str = "") -> ModelPackage | None:
    """The default model for an image type.

    Args:
        models: Installed models.
        modality: Image type.
        profile: The project's profile: the model it names for this image type (for
            ``group``, or else for its first group that has one) comes first.
        group: The class being segmented ("" = any).

    Returns:
        ModelPackage | None: The profile's model if installed, else U-Net++ when several fit.
    """
    if profile is not None:
        groups = [group] if group else [g.name for g in profile.classes.groups]
        wanted = next((profile.model_for(g, modality) for g in groups if profile.model_for(g, modality)), None)
        chosen = next((m for m in models if m.id == wanted), None)
        if chosen is not None:
            return chosen
    fits = [m for m in models if m.modality == modality]
    fits.sort(key=lambda m: "unetpp" not in m.manifest.get("source", {}).get("architecture_id", ""))
    return fits[0] if fits else None
