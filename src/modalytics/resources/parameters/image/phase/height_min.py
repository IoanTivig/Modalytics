"""Minimum height; 0 if no factor is set.

Ported from HoloPath resources/parameters/builtin/image/height_min.py.
"""

import numpy as np

KEY = 'height_min'
LABEL = 'Height Min'
UNIT = 'nm'
CATEGORY = 'height'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 160
DESCRIPTION = 'Minimum: the lowest pixel value of the whole map. On the height h = φ × the phase-to-height factor the file provides (nm; Lyncée files give it), 0 if the file has none. For a height from a known refractive-index difference, use the Height map step.'


def compute(ctx):
    if ctx.phase_to_height_factor_nm is None:
        return 0.0
    return float(np.min(ctx.height_flat))
