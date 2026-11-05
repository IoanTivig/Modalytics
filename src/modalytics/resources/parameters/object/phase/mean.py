"""Mean of the phase values inside the object.

Ported from HoloPath resources/parameters/builtin/object/mean.py.
"""

import numpy as np

KEY = 'mean'
LABEL = 'Phase Mean'
UNIT = 'rad'
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 60
DESCRIPTION = 'Mean x̄ = (1/N)·Σ xᵢ over the N pixels inside the object. Phase delay φ in rad: φ = (2π/λ)·∫Δn dz, so it grows with thickness and with refractive-index difference (density) of the material.'


def compute(ctx):
    return np.mean(ctx.pixels)
