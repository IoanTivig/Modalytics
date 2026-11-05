"""Local entropy: how disordered the values are around each pixel (any image).

The entropy (in bits) of the histogram of grey values in a disc around each
pixel: uniform areas are low, complex textures high. Often used to tell tissue
from empty background or to separate textures. Values are first put on a common
256-level scale for the whole image; colour images use their luminance.
"""

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

from modalytics.processing import helpers as hp

KEY = 'local_entropy'
NAME = 'Local entropy (texture)'
SHORT = 'Entropy'
ICON = 'mdi6.texture'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = 'bits'
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Entropy of the grey values in a disc around each pixel (bits): low in uniform areas, high in complex textures.'
OPTIONS = [
    {'key': 'radius_um',
     'label': 'Disc radius',
     'kind': 'float',
     'default': 3.0,
     'minimum': 0.1,
     'maximum': 100.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Size of the neighbourhood'},
]


def radius_px(options, um_per_px):
    return max(1, min(64, round(options["radius_um"] / um_per_px)))


def margin(options, um_per_px):
    return radius_px(options, um_per_px) + 1


def prepare(image, options):
    """The grey range mapped to 256 levels (the same for every tile)."""
    rng = hp.dtype_range(image.source.dtype)
    return {"range": rng if rng else hp.robust_range(hp.luminance(hp.overview(image)), 0.5, 99.5)}


def process(pixels, options, state, um_per_px):
    lo, hi = state["range"]
    levels = np.clip((hp.luminance(pixels) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return entropy(levels, disk(radius_px(options, um_per_px))).astype(np.float32)
