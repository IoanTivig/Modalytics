"""Uniformity Coef. of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

import numpy as np

KEY = 'uniformity'
LABEL = 'Uniformity Coef.'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 170
DESCRIPTION = 'Uniformity = x̄² / σ² of the pixels inside the object (a squared signal-to-noise ratio). Higher = flat, uniform values; lower = variable. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    b = ctx.brightness()
    m, v = np.mean(b), np.var(b)
    if v == 0:
        return 0.0
    return float(m ** 2 / v)
