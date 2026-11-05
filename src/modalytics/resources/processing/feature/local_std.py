"""Local variation (standard deviation): how much the values vary around each pixel (any image).

A simple texture map: smooth areas are dark, textured or granular ones bright.
Computed in a disc-like Gaussian window (σ in µm); colour images use their
luminance. In the image's unit.
"""

import math

import numpy as np
from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'local_std'
NAME = 'Local variation (standard deviation)'
SHORT = 'Variation'
ICON = 'mdi6.chart-scatter-plot'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Standard deviation of the values in a Gaussian window around each pixel: a texture map.'
OPTIONS = [
    {'key': 'sigma_um',
     'label': 'Window (σ)',
     'kind': 'float',
     'default': 2.0,
     'minimum': 0.1,
     'maximum': 200.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Size of the neighbourhood the variation is measured in'},
]


def margin(options, um_per_px):
    return math.ceil(4 * options["sigma_um"] / um_per_px) + 1


def process(pixels, options, state, um_per_px):
    s = max(0.5, options["sigma_um"] / um_per_px)
    v = hp.luminance(pixels).astype(np.float64)
    mean = ndimage.gaussian_filter(v, s)
    var = ndimage.gaussian_filter(v * v, s) - mean * mean
    return np.sqrt(np.clip(var, 0, None)).astype(np.float32)
