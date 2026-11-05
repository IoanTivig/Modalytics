"""Brightness Skewness of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

from scipy.stats import skew

KEY = 'skewness'
LABEL = 'Brightness Skewness'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 80
DESCRIPTION = 'Skewness g₁ = [(1/N)·Σ (xᵢ − x̄)³] / σ³ over the pixels inside the object: 0 = symmetric distribution; positive = a tail towards high values; negative = a tail towards low values. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(skew(ctx.brightness()))
