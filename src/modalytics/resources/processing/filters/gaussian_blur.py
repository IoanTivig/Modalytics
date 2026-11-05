"""Gaussian blur: smooths an image, keeping its type (any image).

The width is in µm, so the same blur applies at every zoom level and to images
of any pixel size. Colour images are blurred per channel.
"""

import math

from scipy import ndimage

KEY = 'gaussian_blur'
NAME = 'Gaussian blur'
SHORT = 'Gaussian'
ICON = 'mdi6.blur'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Denoise'
DESCRIPTION = 'Smooths the image with a Gaussian of the given width (µm), keeping its image type.'
OPTIONS = [
    {'key': 'sigma_um',
     'label': 'Width (σ)',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.05,
     'maximum': 100.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Standard deviation of the Gaussian, in µm'},
]


def margin(options, um_per_px):
    """Context needed around a tile: 4 standard deviations."""
    return math.ceil(4 * options["sigma_um"] / um_per_px)


def process(pixels, options, state, um_per_px):
    sigma = options["sigma_um"] / um_per_px
    sigmas = (sigma, sigma) + (0,) * (pixels.ndim - 2)
    return ndimage.gaussian_filter(pixels.astype("float32"), sigmas)
