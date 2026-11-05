"""Integrated OD of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

import numpy as np

KEY = 'integrated_od'
LABEL = 'Integrated OD'
UNIT = 'OD·µm²'
CATEGORY = 'optical density'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 75
DESCRIPTION = "Integrated optical density IOD = Σ ODᵢ·A_px over the object (OD·µm²; A_px = pixel area), the densitometric measure of the total amount of absorbing material (e.g. DNA content in a Feulgen-stained nucleus). On the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
CONSTANTS = []


def compute(ctx):
    px, py = ctx.image.pixel_size_um
    return float(np.sum(ctx.pixels) * px * py)
