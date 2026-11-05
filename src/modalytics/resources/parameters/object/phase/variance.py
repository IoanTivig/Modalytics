"""Variance of the phase values inside the object.

Ported from HoloPath resources/parameters/builtin/object/variance.py.
"""

import numpy as np

KEY = 'variance'
LABEL = 'Phase Variance'
UNIT = 'rad²'
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 70
DESCRIPTION = 'Variance σ² = (1/N)·Σ (xᵢ − x̄)² over the N pixels inside the object. In rad².'


def compute(ctx):
    return np.var(ctx.pixels)
