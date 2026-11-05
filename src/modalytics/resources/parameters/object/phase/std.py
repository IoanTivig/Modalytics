"""Standard deviation of the phase values inside the object.

Ported from HoloPath resources/parameters/builtin/object/std.py.
"""

import numpy as np

KEY = 'std'
LABEL = 'Phase Std Dev'
UNIT = 'rad'
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 95
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). In rad. Higher = more variation in thickness or density inside the object.'


def compute(ctx):
    return float(np.std(ctx.pixels))
