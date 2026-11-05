"""
Helpers shared by processing step scripts (Qt-free).

Step scripts run tile by tile, so anything that depends on the *whole* image (a
background level, a fitted surface, stain colours, a contrast range) is
computed once in ``prepare`` from an **overview**: the finest level of at most
~4 Mpx. These helpers do the common parts:

* ``overview`` / ``overview_scale`` / ``sample`` — the image at a coarse level, or a full-resolution crop
  (for noise estimates, which need real pixels);
* ``robust_range``, ``background_level``, ``noise_sigma``, ``tissue_mask`` — statistics of an overview;
* ``fit_surface`` / ``eval_surface`` — a 2D polynomial in image fractions (0–1),
  so it is evaluated the same at every level (with ``process(..., where)``);
* ``luminance``, ``to_like`` — grey values from colour, results back to the input's type;
* ``aligned`` — the part of a block that starts and ends on a global grid (steps
  working in blocks, e.g. CLAHE, give the same result tile by tile);
* ``read_picture`` — a reference or blank image chosen in a step's options.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OVERVIEW_PIXELS = 4_000_000


# ---------------------------------------------------------------- reading
def overview(image, max_pixels: int = OVERVIEW_PIXELS) -> np.ndarray:
    """The image at its finest level of at most ``max_pixels`` (the coarsest if none is that small).

    Args:
        image: An ImageNode (or the stand-in a chain's earlier step makes).
    """
    from modalytics.processing.run import _read_level
    src = image.source
    lv = next((i for i, l in enumerate(src.levels) if l.shape[0] * l.shape[1] <= max_pixels), len(src.levels) - 1)
    h, w = src.levels[lv].shape
    return _read_level(src, lv, 0, h, 0, w)


def overview_scale(image, max_pixels: int = OVERVIEW_PIXELS) -> float:
    """µm per pixel of the level ``overview`` reads."""
    src = image.source
    lv = next((i for i, l in enumerate(src.levels) if l.shape[0] * l.shape[1] <= max_pixels), len(src.levels) - 1)
    return float(image.pixel_size_um[0] * src.levels[lv].downsample[1])


def sample(image, size: int = 1024) -> np.ndarray:
    """A full-resolution crop of the image's centre (at most ``size`` × ``size`` px)."""
    from modalytics.processing.run import _read_level
    h, w = image.source.levels[0].shape
    y0, x0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return _read_level(image.source, 0, y0, min(h, y0 + size), x0, min(w, x0 + size))


def read_picture(path: str, max_pixels: int = OVERVIEW_PIXELS) -> np.ndarray:
    """A picture file (PNG, JPEG, BMP, TIFF) as an array, shrunk to at most ``max_pixels``.

    Raises:
        ValueError: If no file is given, or it cannot be read.
    """
    if not path:
        raise ValueError("no image file is chosen in the step's options")
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"the image file is not found: {p}")
    try:
        if p.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            data = np.asarray(tifffile.imread(p))
            if data.ndim == 3 and data.shape[0] in (3, 4) and data.shape[2] not in (3, 4):
                data = np.moveaxis(data, 0, -1)                    # channels first → last
        else:
            from PIL import Image
            with Image.open(p) as im:
                data = np.asarray(im.convert("RGB") if im.mode not in ("L", "I;16", "I", "F") else im)
    except Exception as exc:
        raise ValueError(f"{p.name} could not be read ({exc})") from exc
    if data.ndim == 3:
        data = data[..., :3]
    step = int(np.ceil(np.sqrt(data.shape[0] * data.shape[1] / max_pixels)))
    return data[::step, ::step] if step > 1 else data


# ---------------------------------------------------------------- statistics
def robust_range(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """The ``low``–``high`` percentile range of the finite values (never empty)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [low, high])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def background_level(values: np.ndarray, bins: int = 512) -> float:
    """The most common value (the histogram's peak), e.g. the empty medium of a phase map."""
    lo, hi = robust_range(values, 0.5, 99.5)
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= lo) & (v <= hi)]
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    from scipy import ndimage
    counts = ndimage.gaussian_filter1d(counts.astype(float), 2)        # a smooth peak, not one noisy bin
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2)


def noise_sigma(values: np.ndarray) -> float:
    """The standard deviation of the pixel noise (Immerkær 1996: a Laplacian-like difference cancels the
    image's structure; colour channels averaged)."""
    from scipy import ndimage
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    v = np.asarray(values, np.float64)
    planes = [v[..., c] for c in range(v.shape[2])] if v.ndim == 3 else [v]
    sigmas = []
    for plane in planes:
        if min(plane.shape) < 3:
            continue
        response = np.abs(ndimage.convolve(plane, kernel))[1:-1, 1:-1]
        sigmas.append(np.sqrt(np.pi / 2) * response.mean() / 6)
    return float(np.mean(sigmas)) if sigmas else 0.0


def tissue_mask(rgb: np.ndarray, white: np.ndarray, min_od: float = 0.15) -> np.ndarray:
    """Pixels of a colour image that hold sample (optical density above ``min_od``), not empty glass."""
    od = -np.log10(np.clip(rgb[..., :3].astype(np.float64), 1, None) / np.asarray(white, float)[:3])
    return od.mean(axis=-1) > min_od


# ---------------------------------------------------------------- surfaces
def _terms(u: np.ndarray, v: np.ndarray, order: int) -> list[np.ndarray]:
    return [u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)]


def fit_surface(values: np.ndarray, order: int, mask: np.ndarray | None = None,
                clip: float | None = 2.5, rounds: int = 5) -> np.ndarray:
    """Fit a 2D polynomial to an overview, in image fractions (so it applies at every level).

    Args:
        values: 2D overview.
        order: Polynomial order (1 = a tilted plane, 2 = a bowl, …).
        mask: Pixels allowed in the fit (default: all finite ones).
        clip: Leave out, round after round, pixels farther than ``clip`` × the robust spread from the
            surface (the sample), so the fit follows the background; None = fit everything.
        rounds: How many rounds of leaving out.

    Returns:
        np.ndarray: The coefficients (for ``eval_surface``).
    """
    h, w = values.shape
    ys = (np.arange(h) + 0.5) / h
    xs = (np.arange(w) + 0.5) / w
    # 1. At most ~250 k points (a regular subsample is enough for a smooth surface)
    step = max(1, int(np.sqrt(h * w / 250_000)))
    vv = np.asarray(values, np.float64)[::step, ::step]
    yy, xx = np.meshgrid(ys[::step], xs[::step], indexing="ij")
    keep = np.isfinite(vv) if mask is None else (np.asarray(mask)[::step, ::step] & np.isfinite(vv))
    u, v, z = 2 * xx[keep] - 1, 2 * yy[keep] - 1, vv[keep]
    A = np.stack(_terms(u, v, order), axis=1)
    use = np.ones(z.size, bool)
    coef = np.zeros(A.shape[1])
    # 2. Least squares, then leave out what is far from the surface, and again
    for _ in range(rounds if clip else 1):
        if use.sum() < A.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(A[use], z[use], rcond=None)
        resid = z - A @ coef
        spread = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use]))) or np.std(resid[use]) or 1.0
        use = np.abs(resid - np.median(resid[use])) <= clip * spread
    return coef


def eval_surface(coef: np.ndarray, order: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The fitted surface on a grid of image fractions (``Where.fraction`` gives them for a tile)."""
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    terms = _terms(2 * xx - 1, 2 * yy - 1, order)
    return sum(c * t for c, t in zip(coef, terms)).astype(np.float32)


# ---------------------------------------------------------------- values
def luminance(pixels: np.ndarray) -> np.ndarray:
    """Grey values (float): colour → luminance (Rec. 709 weights), grey → as it is."""
    p = np.asarray(pixels)
    if p.ndim == 3 and p.shape[2] >= 3:
        return (p[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    return (p[..., 0] if p.ndim == 3 else p).astype(np.float32)


def dtype_range(dtype) -> tuple[float, float] | None:
    """(0, the largest value) for integer pixel types (8 / 16-bit), None for floats."""
    dt = np.dtype(dtype)
    return (0.0, float(np.iinfo(dt).max)) if np.issubdtype(dt, np.integer) else None


def to_like(result: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """``result`` in the input's pixel type: integer images are rounded and kept within their range."""
    rng = dtype_range(pixels.dtype)
    if rng is None:
        return result.astype(np.float32)
    return np.clip(np.rint(result), *rng).astype(pixels.dtype)


def per_channel(fn, pixels: np.ndarray) -> np.ndarray:
    """Apply ``fn`` (2D → 2D) to each colour channel (or to a grey image)."""
    if pixels.ndim == 3:
        return np.stack([fn(pixels[..., c]) for c in range(pixels.shape[2])], axis=-1)
    return fn(pixels)


# ---------------------------------------------------------------- blocks on a global grid
def aligned(where, block_shape: tuple[int, int], cell: int) -> tuple[slice, slice]:
    """Rows and columns of a block that start and end on multiples of ``cell`` in the whole image.

    Steps that work in cells (CLAHE's regions, a shrunk background) then see the same
    cells in every tile, so tiles join without seams. The block's margin must be
    at least 2 × ``cell`` for the tile itself to stay inside.
    """
    def one(start, n):
        a = (-start) % cell
        b = a + ((n - a) // cell) * cell
        return slice(a, b)
    return one(where.row, block_shape[0]), one(where.col, block_shape[1])
