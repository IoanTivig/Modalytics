"""Mean²/Variance over the whole-image phase distribution.

Ported from HoloPath resources/parameters/builtin/image/uniformity.py.
"""

import numpy as np

KEY = 'uniformity'
LABEL = 'Uniformity'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 200
DESCRIPTION = 'Uniformity = x̄² / σ² of the pixels of the whole phase map, background included (a squared signal-to-noise ratio). Higher = flat, uniform values; lower = variable.'


def compute(ctx):
    m = np.mean(ctx.flat)
    v = np.var(ctx.flat)
    if v == 0:
        return 0.0
    return float((m ** 2) / v)
