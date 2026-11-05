"""OD Variance of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

import numpy as np

KEY = 'variance'
LABEL = 'OD Variance'
UNIT = 'OD²'
CATEGORY = 'optical density'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 70
DESCRIPTION = "Variance σ² = (1/N)·Σ (xᵢ − x̄)² over the N pixels inside the object. On the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
CONSTANTS = []


def compute(ctx):
    return float(np.var(ctx.pixels))
