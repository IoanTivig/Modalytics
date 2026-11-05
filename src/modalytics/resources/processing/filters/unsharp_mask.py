"""Unsharp mask (sharpen): crisper edges, for looking (any image).

Adds back the difference between the image and a blurred copy of it, which
strengthens edges and fine detail of about the blur's width.
"""

import math

import numpy as np
from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'unsharp_mask'
NAME = 'Unsharp mask (sharpen)'
SHORT = 'Sharpen'
ICON = 'mdi6.image-filter-center-focus-strong'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Enhance'
DESCRIPTION = 'Sharpens: adds back the difference between the image and a blurred copy (detail of about the given width).'
OPTIONS = [
    {'key': 'sigma_um',
     'label': 'Detail width',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.05,
     'maximum': 50.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Width (σ) of the blur: the size of the detail strengthened'},
    {'key': 'amount',
     'label': 'Amount',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.1,
     'maximum': 10.0,
     'step': 0.1,
     'tooltip': 'How much of the detail is added back (1 = doubled)'},
]


def margin(options, um_per_px):
    return math.ceil(4 * options["sigma_um"] / um_per_px)


def process(pixels, options, state, um_per_px):
    sigma = options["sigma_um"] / um_per_px
    x = pixels.astype(np.float32)
    blurred = ndimage.gaussian_filter(x, (sigma, sigma) + (0,) * (pixels.ndim - 2))
    return hp.to_like(x + float(options["amount"]) * (x - blurred), pixels)
