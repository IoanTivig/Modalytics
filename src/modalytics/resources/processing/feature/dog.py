"""Band-pass (difference of Gaussians): keeps structures between two sizes (any image).

Subtracts a strongly blurred copy from a lightly blurred one: details smaller
than the first width (noise) and larger than the second (uneven background)
are removed, structures in between remain. A fast approximation of the
Laplacian of Gaussian, common before spot or nucleus detection. Colour images
use their luminance.
"""

import math

import numpy as np
from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'dog'
NAME = 'Band-pass (difference of Gaussians)'
SHORT = 'Band-pass'
ICON = 'mdi6.chart-bell-curve'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Lightly blurred minus strongly blurred image: keeps structures between the two sizes, removes noise and background.'
OPTIONS = [
    {'key': 'small_um',
     'label': 'Smallest detail (σ)',
     'kind': 'float',
     'default': 0.5,
     'minimum': 0.05,
     'maximum': 100.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Details below this width are smoothed away (noise)'},
    {'key': 'large_um',
     'label': 'Largest structure (σ)',
     'kind': 'float',
     'default': 5.0,
     'minimum': 0.1,
     'maximum': 1000.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Structures above this width are removed (background)'},
]


def margin(options, um_per_px):
    return math.ceil(4 * max(options["small_um"], options["large_um"]) / um_per_px) + 1


def process(pixels, options, state, um_per_px):
    values = hp.luminance(pixels)
    small = max(0.3, options["small_um"] / um_per_px)
    large = max(small * 1.1, options["large_um"] / um_per_px)
    return ndimage.gaussian_filter(values, small) - ndimage.gaussian_filter(values, large).astype(np.float32)
