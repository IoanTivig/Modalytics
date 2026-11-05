"""Skewness (asymmetry) of the phase value distribution.

Ported from HoloPath resources/parameters/builtin/object/skewness.py.
"""

from scipy.stats import skew

KEY = 'skewness'
LABEL = 'Phase Skewness'
UNIT = ''
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 80
DESCRIPTION = 'Skewness g₁ = [(1/N)·Σ (xᵢ − x̄)³] / σ³ over the pixels inside the object: 0 = symmetric distribution; positive = a tail towards high values; negative = a tail towards low values.'


def compute(ctx):
    return skew(ctx.pixels)
