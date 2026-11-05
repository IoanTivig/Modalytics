"""Contrast enhancement (CLAHE): local contrast, for looking (any image).

Contrast-limited adaptive histogram equalisation (Zuiderveld 1994): the contrast
is stretched region by region, so faint structures in dark and bright parts both
show; the limit keeps noise from being stretched too. Colour images are enhanced
on their lightness only (colours kept). Works on 256 grey levels (the standard
form; the contrast limit means the same as in other software).

It changes values non-linearly: use it to look and to segment, not to measure
intensities (a measured chain with it gives warped values).
"""

import cv2
import numpy as np
from skimage import color

from modalytics.processing import helpers as hp

KEY = 'clahe'
NAME = 'Contrast enhancement (CLAHE)'
SHORT = 'CLAHE'
ICON = 'mdi6.contrast-box'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Enhance'
DESCRIPTION = 'Contrast-limited adaptive histogram equalisation (Zuiderveld 1994): stretches the contrast region by region. For looking and segmenting, not for measuring intensities.'
OPTIONS = [
    {'key': 'region_um',
     'label': 'Region size',
     'kind': 'float',
     'default': 50.0,
     'minimum': 1.0,
     'maximum': 5000.0,
     'step': 5.0,
     'suffix': ' µm',
     'tooltip': 'Side of the regions whose contrast is stretched separately'},
    {'key': 'clip_limit',
     'label': 'Contrast limit',
     'kind': 'float',
     'default': 2.0,
     'minimum': 1.0,
     'maximum': 40.0,
     'step': 0.5,
     'tooltip': 'How much the contrast may be stretched (1 = not at all; higher = stronger, noisier)'},
]


def region_px(options, um_per_px):
    return max(8, round(options["region_um"] / um_per_px))


def margin(options, um_per_px):
    return 2 * region_px(options, um_per_px)


def prepare(image, options):
    """The value range mapped to the 256 levels CLAHE works on (the same for every tile)."""
    view = hp.overview(image)
    if view.ndim == 3 and view.shape[2] >= 3:
        return {"range": (0.0, 100.0)}                               # Lab lightness
    rng = hp.dtype_range(image.source.dtype)
    return {"range": rng if rng else hp.robust_range(hp.luminance(view), 0.5, 99.5)}


def process(pixels, options, state, um_per_px, where):
    # 1. The values on 256 levels (colour: the lightness)
    colour = pixels.ndim == 3 and pixels.shape[2] >= 3
    if colour:
        rng01 = hp.dtype_range(pixels.dtype)[1] if hp.dtype_range(pixels.dtype) else 1.0
        lab = color.rgb2lab(pixels[..., :3].astype(np.float64) / rng01)
        values = lab[..., 0]
    else:
        values = hp.luminance(pixels)
    lo, hi = state["range"]
    scaled = np.clip((values - lo) / (hi - lo), 0, 1) * 255
    # 2. CLAHE on whole regions of the image's grid (every tile sees the same regions)
    k = region_px(options, um_per_px)
    rows, cols = hp.aligned(where, scaled.shape, k)
    part = np.rint(scaled[rows, cols]).astype(np.uint8)
    out = scaled.copy()
    if part.size:
        clahe = cv2.createCLAHE(clipLimit=float(options["clip_limit"]),
                                tileGridSize=(part.shape[1] // k, part.shape[0] // k))
        out[rows, cols] = clahe.apply(np.ascontiguousarray(part))
    out = out / 255 * (hi - lo) + lo
    # 3. Back to the input's form
    if colour:
        lab[..., 0] = out
        rgb = color.lab2rgb(lab) * rng01
        if pixels.shape[2] > 3:
            rgb = np.concatenate([rgb, pixels[..., 3:].astype(float)], axis=-1)
        return hp.to_like(rgb, pixels)
    if pixels.ndim == 3:
        out = np.repeat(out[..., None], pixels.shape[2], axis=2)
    return hp.to_like(out, pixels)
