"""Variance of phase over the whole image.

Ported from HoloPath resources/parameters/builtin/image/phase_var.py.
"""

import numpy as np

KEY = 'phase_var'
LABEL = 'Phase Variance'
UNIT = 'rad²'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 80
DESCRIPTION = 'Variance σ² = (1/N)·Σ (xᵢ − x̄)² over the N pixels of the whole phase map, background included. In rad².'


def compute(ctx):
    return float(np.var(ctx.flat))
