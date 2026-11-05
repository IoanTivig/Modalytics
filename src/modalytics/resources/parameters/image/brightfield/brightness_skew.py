"""Brightness Skewness of the image on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

from scipy.stats import skew

KEY = 'brightness_skew'
LABEL = 'Brightness Skewness'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'image'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
ORDER = 90
DESCRIPTION = 'Skewness g₁ = [(1/N)·Σ (xᵢ − x̄)³] / σ³ over the pixels of the whole image: 0 = symmetric distribution; positive = a tail towards high values; negative = a tail towards low values. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(skew(ctx.brightness()))
