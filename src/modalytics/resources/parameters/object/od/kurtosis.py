"""OD Kurtosis of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

from scipy.stats import kurtosis

KEY = 'kurtosis'
LABEL = 'OD Kurtosis'
UNIT = ''
CATEGORY = 'optical density'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 90
DESCRIPTION = "Excess kurtosis g₂ = [(1/N)·Σ (xᵢ − x̄)⁴] / σ⁴ − 3 over the pixels inside the object: 0 = as peaked as a normal distribution; positive = a sharp peak with heavy tails (a few extreme pixels); negative = flat-topped. On the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
CONSTANTS = []


def compute(ctx):
    return float(kurtosis(ctx.pixels))
