"""Uniformity Coef. of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

import numpy as np

KEY = 'uniformity'
LABEL = 'Uniformity Coef.'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 170
DESCRIPTION = "Uniformity = x̄² / σ² of the pixels inside the object (a squared signal-to-noise ratio). Higher = flat, uniform values; lower = variable. On the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
CONSTANTS = []


def compute(ctx):
    m, v = np.mean(ctx.pixels), np.var(ctx.pixels)
    return float(m ** 2 / v) if v else 0.0
