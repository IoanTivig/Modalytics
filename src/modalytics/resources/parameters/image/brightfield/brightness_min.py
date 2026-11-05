"""Brightness Min of the image on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

import numpy as np

KEY = 'brightness_min'
LABEL = 'Brightness Min'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'image'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
ORDER = 110
DESCRIPTION = 'Minimum: the lowest pixel value of the whole image. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(np.min(ctx.brightness()))
