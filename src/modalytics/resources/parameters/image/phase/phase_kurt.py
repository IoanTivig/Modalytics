"""Kurtosis of the whole-image phase distribution.

Ported from HoloPath resources/parameters/builtin/image/phase_kurt.py.
"""

from scipy.stats import kurtosis

KEY = 'phase_kurt'
LABEL = 'Phase Kurtosis'
UNIT = ''
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 100
DESCRIPTION = 'Excess kurtosis g₂ = [(1/N)·Σ (xᵢ − x̄)⁴] / σ⁴ − 3 over the pixels of the whole phase map, background included: 0 = as peaked as a normal distribution; positive = a sharp peak with heavy tails (a few extreme pixels); negative = flat-topped.'


def compute(ctx):
    return float(kurtosis(ctx.flat))
