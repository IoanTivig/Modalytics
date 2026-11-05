"""Total-variation denoising: makes areas flat and edges sharp (any image).

Finds the image closest to the input with the least total variation (Rudin,
Osher & Fatemi 1992; Chambolle's algorithm): noise is removed and regions become
piecewise smooth while edges stay sharp. Used in quantitative phase imaging to
clean phase maps before segmentation.
"""

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from modalytics.processing import helpers as hp

KEY = 'tv_denoise'
NAME = 'Total-variation denoising'
SHORT = 'TV denoise'
ICON = 'mdi6.water-outline'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Denoise'
DESCRIPTION = 'Removes noise making regions piecewise smooth with sharp edges (Rudin, Osher & Fatemi 1992; Chambolle 2004).'
OPTIONS = [
    {'key': 'weight',
     'label': 'Strength',
     'kind': 'float',
     'default': 0.05,
     'minimum': 0.001,
     'maximum': 1.0,
     'step': 0.005,
     'tooltip': "Higher = flatter (as a fraction of the image's value range)"},
]

MARGIN = 24                                       # px of context: the result settles within this distance


def margin(options, um_per_px):
    return MARGIN


def prepare(image, options):
    rng = hp.dtype_range(image.source.dtype)
    return {"range": rng if rng else hp.robust_range(hp.overview(image), 0.5, 99.5)}


def process(pixels, options, state, um_per_px):
    lo, hi = state["range"]
    x = (pixels.astype(np.float32) - lo) / (hi - lo)              # the same scale for every tile
    out = denoise_tv_chambolle(x, weight=float(options["weight"]),
                               channel_axis=-1 if pixels.ndim == 3 else None)
    return hp.to_like(out * (hi - lo) + lo, pixels)
