"""Mean phase over the whole image.

Ported from HoloPath resources/parameters/builtin/image/phase_mean.py.
"""

import numpy as np

KEY = 'phase_mean'
LABEL = 'Phase Mean'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 60
DESCRIPTION = 'Mean x̄ = (1/N)·Σ xᵢ over the N pixels of the whole phase map, background included. In rad.'


def compute(ctx):
    return float(np.mean(ctx.flat))
