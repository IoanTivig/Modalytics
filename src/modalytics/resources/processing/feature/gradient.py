"""Edge map (gradient magnitude): how steeply the values change at each pixel (any image).

Boundaries of cells, nuclei and glands show as bright lines. The derivatives
are taken on a Gaussian-smoothed image (σ in µm), so the edge scale is chosen
and noise is not amplified; colour images use their luminance. Values are in
the image's unit per µm.
"""

import math

import numpy as np
from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'gradient'
NAME = 'Edge map (gradient magnitude)'
SHORT = 'Edges'
ICON = 'mdi6.vector-difference'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = 'per µm'
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Gradient magnitude of the smoothed image (Gaussian derivatives): bright where values change steeply, e.g. boundaries.'
OPTIONS = [
    {'key': 'sigma_um',
     'label': 'Smoothing (σ)',
     'kind': 'float',
     'default': 0.5,
     'minimum': 0.05,
     'maximum': 50.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Scale of the edges found (larger = coarser edges, less noise)'},
]


def margin(options, um_per_px):
    return math.ceil(4 * options["sigma_um"] / um_per_px) + 1


def process(pixels, options, state, um_per_px):
    sigma = max(0.5, options["sigma_um"] / um_per_px)
    return ndimage.gaussian_gradient_magnitude(hp.luminance(pixels), sigma) / np.float32(um_per_px)
