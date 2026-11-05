"""Bimodality of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

import numpy as np
from scipy.stats import skew, kurtosis

KEY = 'bimodality'
LABEL = 'Bimodality'
UNIT = ''
CATEGORY = 'optical density'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 96
DESCRIPTION = "Sarle's bimodality coefficient BC = (g₁² + 1) / [g₂ + 3(N − 1)² / ((N − 2)(N − 3))] of the pixels inside the object (g₁ skewness, g₂ excess kurtosis, N pixels), 0–1. Above 5/9 ≈ 0.555 (the value of a uniform distribution) suggests two populations of values, e.g. dark nuclei in pale cytoplasm; lower = one population. Sensitive to skewed single populations as well. On the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
REFERENCES = [
    ('Pfister R, Schwarz KA, Janczyk M, Dale R, Freeman JB (2013). Good things peak in pairs: a note '
     'on the bimodality coefficient. Front Psychol 4:700. doi:10.3389/fpsyg.2013.00700'),
]
CONSTANTS = []


def compute(ctx):
    x = np.asarray(ctx.pixels, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float('nan')
    g = skew(x)
    k = kurtosis(x)            # excess kurtosis (Fisher)
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float('nan')
    return float((g ** 2 + 1.0) / denom)
