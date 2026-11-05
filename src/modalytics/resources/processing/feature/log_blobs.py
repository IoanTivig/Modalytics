"""Blob map (Laplacian of Gaussian): highlights round spots of a chosen size (any image).

The scale-normalised Laplacian of a Gaussian (Marr & Hildreth 1980; Lindeberg
1998) responds most to blobs whose radius is about σ·√2: nuclei, cells, granules.
Also an edge detector (its zero crossings). The sign is chosen so the blobs you
look for (bright or dark) come out positive; colour images use their luminance.
"""

import math

from scipy import ndimage

from modalytics.processing import helpers as hp

KEY = 'log_blobs'
NAME = 'Blob map (Laplacian of Gaussian)'
SHORT = 'Blobs'
ICON = 'mdi6.dots-hexagon'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Scale-normalised Laplacian of Gaussian (Marr & Hildreth 1980): strongest at blobs of radius ≈ 1.4 σ, e.g. nuclei.'
OPTIONS = [
    {'key': 'radius_um',
     'label': 'Blob radius',
     'kind': 'float',
     'default': 3.0,
     'minimum': 0.1,
     'maximum': 200.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Radius of the blobs to highlight'},
    {'key': 'blobs',
     'label': 'Blobs are',
     'kind': 'choice',
     'default': 'Dark',
     'choices': ['Dark', 'Bright'],
     'tooltip': 'Dark: e.g. nuclei in brightfield. Bright: e.g. cells in a phase map'},
]


def sigma_px(options, um_per_px):
    return max(0.5, options["radius_um"] / math.sqrt(2) / um_per_px)


def margin(options, um_per_px):
    return math.ceil(4 * sigma_px(options, um_per_px)) + 1


def process(pixels, options, state, um_per_px):
    s = sigma_px(options, um_per_px)
    response = s * s * ndimage.gaussian_laplace(hp.luminance(pixels), s)          # > 0 at dark blobs
    return response if options["blobs"] == "Dark" else -response
