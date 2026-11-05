"""DIC-like relief: the slope along one direction, like differential interference contrast (any image).

Differential interference contrast shows the derivative of the optical path
along its shear direction: objects look lit from one side, in relief. From a
phase map this gives a DIC image numerically; on other images it is an
embossed relief. Colour images use their luminance.
"""

import math

import numpy as np
from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'dic'
NAME = 'DIC-like relief (directional slope)'
SHORT = 'DIC relief'
ICON = 'mdi6.image-filter-hdr'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = 'per µm'
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Slope along a chosen direction (Gaussian derivative): a DIC-like relief, numerically from phase maps.'
OPTIONS = [
    {'key': 'angle',
     'label': 'Direction',
     'kind': 'float',
     'default': 45.0,
     'minimum': 0.0,
     'maximum': 360.0,
     'step': 15.0,
     'suffix': '°',
     'tooltip': 'Shear direction (0° = to the right, 90° = up)'},
    {'key': 'sigma_um',
     'label': 'Smoothing (σ)',
     'kind': 'float',
     'default': 0.3,
     'minimum': 0.05,
     'maximum': 20.0,
     'step': 0.05,
     'suffix': ' µm',
     'tooltip': 'Scale of the derivative (the shear distance of a DIC prism)'},
]


def margin(options, um_per_px):
    return math.ceil(4 * options["sigma_um"] / um_per_px) + 1


def process(pixels, options, state, um_per_px):
    sigma = max(0.5, options["sigma_um"] / um_per_px)
    values = hp.luminance(pixels)
    dy = ndimage.gaussian_filter(values, sigma, order=(1, 0))
    dx = ndimage.gaussian_filter(values, sigma, order=(0, 1))
    a = math.radians(options["angle"])
    return (math.cos(a) * dx - math.sin(a) * dy) / np.float32(um_per_px)      # rows grow downwards
