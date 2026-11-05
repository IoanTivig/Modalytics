"""Median filter: removes noise and small specks while keeping edges, keeping the image type (any image).

Each pixel becomes the median of a square neighbourhood of the given size (µm).
Colour images are filtered per channel.
"""

from scipy import ndimage

KEY = 'median'
NAME = 'Median filter (denoise)'
SHORT = 'Median'
ICON = 'mdi6.blur-linear'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Denoise'
DESCRIPTION = 'Removes noise and small specks while keeping edges: each pixel becomes the median of a square neighbourhood of the given size (µm).'
OPTIONS = [
    {'key': 'size_um',
     'label': 'Neighbourhood',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.05,
     'maximum': 50.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Side of the square neighbourhood, in µm'},
]


def size_px(options, um_per_px):
    """The neighbourhood's side in pixels at this level (odd, at least 1)."""
    n = max(1, round(options["size_um"] / um_per_px))
    return n if n % 2 else n + 1


def margin(options, um_per_px):
    return size_px(options, um_per_px) // 2 + 1


def process(pixels, options, state, um_per_px):
    n = size_px(options, um_per_px)
    return ndimage.median_filter(pixels, size=(n, n) + (1,) * (pixels.ndim - 2), mode="reflect")
