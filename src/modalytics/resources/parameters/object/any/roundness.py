"""Roundness (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import math

KEY = 'roundness'
LABEL = 'Roundness'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 104
DESCRIPTION = 'Roundness = 4A / (π·D_max²): the area relative to the circle whose diameter is the maximum Feret diameter, 0–1. 1 = a circle; lower = elongated. Unlike circularity (4πA/P²), it does not fall with boundary roughness, only with elongation.'
CONSTANTS = []


def compute(ctx):
    fmax = ctx.feret[0]
    return 4 * ctx.area_um2 / (math.pi * fmax ** 2) if fmax > 0 else 0.0
