"""Bilateral filter: smooths while keeping edges (any image).

Each pixel becomes a weighted mean of its neighbours, weighted both by distance
and by how similar their values are (Tomasi & Manduchi 1998): flat areas are
smoothed, edges between different values are not blurred across.
"""

import math

import cv2
import numpy as np

from modalytics.processing import helpers as hp

KEY = 'bilateral'
NAME = 'Bilateral filter (edge-preserving smoothing)'
SHORT = 'Bilateral'
ICON = 'mdi6.blur-radial'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Denoise'
DESCRIPTION = 'Smooths flat areas while keeping edges: neighbours are weighted by distance and by how similar their values are (Tomasi & Manduchi 1998).'
OPTIONS = [
    {'key': 'sigma_um',
     'label': 'Width',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.05,
     'maximum': 50.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'How far the smoothing reaches (σ of the distance weight)'},
    {'key': 'edge',
     'label': 'Edge threshold',
     'kind': 'float',
     'default': 0.1,
     'minimum': 0.005,
     'maximum': 1.0,
     'step': 0.01,
     'tooltip': "Differences larger than this fraction of the image's value range are kept as edges"},
]


def sigma_px(options, um_per_px):
    return max(0.5, options["sigma_um"] / um_per_px)


def margin(options, um_per_px):
    return math.ceil(3 * sigma_px(options, um_per_px)) + 1


def prepare(image, options):
    rng = hp.dtype_range(image.source.dtype)
    lo, hi = rng if rng else hp.robust_range(hp.overview(image))
    return {"value_sigma": float(options["edge"]) * (hi - lo)}


def process(pixels, options, state, um_per_px):
    s = sigma_px(options, um_per_px)
    d = 2 * math.ceil(2 * s) + 1

    def one(values):
        return cv2.bilateralFilter(np.ascontiguousarray(values, np.float32), d, state["value_sigma"], s)
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        out = one(pixels)                                    # colour: neighbours compared in all three channels
    else:
        out = hp.per_channel(one, pixels)
    return hp.to_like(out, pixels)
