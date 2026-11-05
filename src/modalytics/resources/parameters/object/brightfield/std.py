"""Brightness Std Dev of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

import numpy as np

KEY = 'std'
LABEL = 'Brightness Std Dev'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 95
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(np.std(ctx.brightness()))
