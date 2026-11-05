"""Mean²/Variance of the phase values (squared signal-to-noise ratio).

Ported from HoloPath resources/parameters/builtin/object/uniformity.py.
"""

import numpy as np

KEY = 'uniformity'
LABEL = 'Uniformity Coef.'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 170
DESCRIPTION = 'Uniformity = x̄² / σ² of the pixels inside the object (a squared signal-to-noise ratio). Higher = flat, uniform values; lower = variable.'


def compute(ctx):
    m = np.mean(ctx.pixels)
    v = np.var(ctx.pixels)
    if v == 0:
        return 0.0
    return (m ** 2) / v
