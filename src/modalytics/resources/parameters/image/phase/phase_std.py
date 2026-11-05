"""Standard deviation of phase over the whole image.

Ported from HoloPath resources/parameters/builtin/image/phase_std.py.
"""

import numpy as np

KEY = 'phase_std'
LABEL = 'Phase Std'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 70
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels of the whole phase map, background included (population form). In rad.'


def compute(ctx):
    return float(np.std(ctx.flat))
