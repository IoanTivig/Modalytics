"""
Optical density (slide scans) and brightness (brightfield photos): the pixel values the general
slide and brightfield parameters measure.

A phase map's pixels are a physical quantity (phase, rad); a slide's or a
brightfield camera's are grey levels, which depend on the scanner, the
exposure and the white balance.

**Slide scans: optical density.** The physical quantity behind a scan's grey
levels is how much light the sample absorbs (Beer–Lambert):

    OD = −log₁₀(I / I_white)          per colour channel, then averaged

where ``I_white`` is the intensity of the empty slide (glass, mounting medium,
coverslip; no sample). OD is 0 there and grows with the amount of absorbing
material (stain, pigment, dense structures), whatever the illumination. A scan
always shows empty slide around the tissue, so ``I_white`` is estimated per
image from its brightest background (the median colour of its brightest 1 % of
pixels, on a level of at most ~4 Mpx), or fixed by a parameter constant
("White level", 0 = estimated).

**Brightfield photos: brightness.** A camera field is often covered by the
sample with no empty background to take OD against, so their parameters use
brightness instead (``brightness``: 0 = black, 1 = the brightest value the file
can hold). It needs no reference, and compares images taken with the same
microscope, lamp and exposure.

Stain-specific parameters (haematoxylin, eosin, DAB…) will be added with the
profiles that need them.
"""

from __future__ import annotations

import numpy as np

_WHITE: dict = {}                      # source uid → estimated white level (per channel)
MIN_INTENSITY = 1.0                    # darker pixels are clipped here (OD stays finite)


def estimate_white(image) -> np.ndarray:
    """The intensity of clear glass per channel, estimated from an image's brightest pixels (cached).

    Args:
        image: The ImageNode (any size: a coarse level of at most ~4 Mpx is used).

    Returns:
        np.ndarray: One value per channel (shape (C,), C = 1 for grey images).
    """
    src = image.source
    if src.uid in _WHITE:
        return _WHITE[src.uid]
    # 1. A coarse level: the finest one of at most ~4 Mpx (the coarsest if none is that small)
    lv = next((i for i, l in enumerate(src.levels) if l.shape[0] * l.shape[1] <= 4_000_000), len(src.levels) - 1)
    level = src.levels[lv]
    gy, gx = level.grid
    data = np.concatenate([np.concatenate([src.read_tile(lv, ty, tx) for tx in range(gx)], axis=1)
                           for ty in range(gy)], axis=0)
    px = data.reshape(-1, data.shape[2] if data.ndim == 3 else 1)[:, :3].astype(np.float64)
    # 2. The brightest 1 % (by mean over channels): their median colour
    grey = px.mean(axis=1)
    bright = px[grey >= np.percentile(grey, 99)]
    white = np.maximum(np.median(bright, axis=0), MIN_INTENSITY)
    _WHITE[src.uid] = white
    return white


def white_level(image, fixed: float = 0.0) -> np.ndarray:
    """``fixed`` for every channel if it is > 0, else the estimate (``estimate_white``)."""
    if fixed and fixed > 0:
        return np.full(image.source.channels if image.source.channels <= 3 else 3, float(fixed))
    return estimate_white(image)


def optical_density(values: np.ndarray, white: np.ndarray, channels: int) -> np.ndarray:
    """Optical density of pixel values, averaged over the colour channels.

    Args:
        values: Grey ``(…)`` or colour ``(…, C)`` intensities (any leading shape: a crop or a list).
        white: Clear-glass intensity per channel (``white_level``).
        channels: Number of channels of the image (1 = grey; colour values have them on the last axis).

    Returns:
        np.ndarray: OD ≥ 0, the leading shape of ``values`` (the channel axis averaged away).
    """
    v = np.asarray(values, dtype=np.float64)
    if channels > 1:
        v = v[..., :3]
    w = np.asarray(white, dtype=np.float64)
    od = -np.log10(np.clip(v, MIN_INTENSITY, None) / (w if channels > 1 else w[0]))
    od = np.clip(od, 0.0, None)                          # brighter than glass: no absorption
    return od.mean(axis=-1) if channels > 1 else od


# ---------------------------------------------------------------- brightness (brightfield photos)
def brightness(values: np.ndarray, dtype: np.dtype, channels: int) -> np.ndarray:
    """Pixel brightness from 0 (black) to 1 (the brightest value the file can hold), colour channels averaged.

    The base signal of brightfield photos, which often show no empty background
    to take optical density against: it needs no reference, and is comparable
    between images taken with the same microscope, lamp and exposure.

    Args:
        values: Grey ``(…)`` or colour ``(…, C)`` pixel values.
        dtype: The image's data type (8-bit ÷ 255, 16-bit ÷ 65535).
        channels: Number of channels (1 = grey; colour values have them on the last axis).
    """
    v = np.asarray(values, dtype=np.float64)
    if channels > 1:
        v = v[..., :3].mean(axis=-1)
    top = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    return v / top
