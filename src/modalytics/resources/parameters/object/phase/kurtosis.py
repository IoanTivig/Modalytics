"""Kurtosis (tailedness) of the phase value distribution.

Ported from HoloPath resources/parameters/builtin/object/kurtosis.py.
"""

from scipy.stats import kurtosis

KEY = 'kurtosis'
LABEL = 'Phase Kurtosis'
UNIT = ''
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 90
DESCRIPTION = 'Excess kurtosis g₂ = [(1/N)·Σ (xᵢ − x̄)⁴] / σ⁴ − 3 over the pixels inside the object: 0 = as peaked as a normal distribution; positive = a sharp peak with heavy tails (a few extreme pixels); negative = flat-topped.'


def compute(ctx):
    return kurtosis(ctx.pixels)
