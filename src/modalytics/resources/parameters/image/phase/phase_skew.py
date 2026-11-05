"""Skewness of the whole-image phase distribution.

Ported from HoloPath resources/parameters/builtin/image/phase_skew.py.
"""

from scipy.stats import skew

KEY = 'phase_skew'
LABEL = 'Phase Skewness'
UNIT = ''
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 90
DESCRIPTION = 'Skewness g₁ = [(1/N)·Σ (xᵢ − x̄)³] / σ³ over the pixels of the whole phase map, background included: 0 = symmetric distribution; positive = a tail towards high values; negative = a tail towards low values.'


def compute(ctx):
    return float(skew(ctx.flat))
