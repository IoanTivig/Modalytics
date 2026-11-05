"""Sarle's bimodality coefficient of the whole-image phase distribution (skewness + kurtosis).

Ported from HoloPath resources/parameters/builtin/image/bimodality.py.
"""

from scipy.stats import skew, kurtosis
import numpy as np

KEY = 'bimodality'
LABEL = 'Bimodality'
UNIT = ''
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 136
DESCRIPTION = "Sarle's bimodality coefficient BC = (g₁² + 1) / [g₂ + 3(N − 1)² / ((N − 2)(N − 3))] of the pixels of the whole phase map (g₁ skewness, g₂ excess kurtosis, N pixels), 0–1. Above 5/9 ≈ 0.555 (the value of a uniform distribution) suggests two populations of values, e.g. dark nuclei in pale cytoplasm; lower = one population. Sensitive to skewed single populations as well."
REFERENCES = [
    ('Pfister R, Schwarz KA, Janczyk M, Dale R, Freeman JB (2013). Good things peak in pairs: a note '
     'on the bimodality coefficient. Front Psychol 4:700. doi:10.3389/fpsyg.2013.00700'),
]


def compute(ctx):
    x = np.asarray(ctx.flat, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float('nan')
    g = skew(x)
    k = kurtosis(x)            # excess kurtosis (Fisher)
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float('nan')
    return float((g ** 2 + 1.0) / denom)
