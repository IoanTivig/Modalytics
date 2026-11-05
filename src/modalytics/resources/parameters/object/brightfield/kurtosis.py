"""Brightness Kurtosis of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

from scipy.stats import kurtosis

KEY = 'kurtosis'
LABEL = 'Brightness Kurtosis'
UNIT = ''
CATEGORY = 'brightness'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 90
DESCRIPTION = 'Excess kurtosis g₂ = [(1/N)·Σ (xᵢ − x̄)⁴] / σ⁴ − 3 over the pixels inside the object: 0 = as peaked as a normal distribution; positive = a sharp peak with heavy tails (a few extreme pixels); negative = flat-topped. On the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = []


def compute(ctx):
    return float(kurtosis(ctx.brightness()))
