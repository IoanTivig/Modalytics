"""Brightness Variance of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

import numpy as np

KEY = 'variance'
LABEL = 'Brightness Variance'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 70
DESCRIPTION = 'Variance σ² = (1/N)·Σ (xᵢ − x̄)² over the N pixels inside the object. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(np.var(ctx.brightness()))
