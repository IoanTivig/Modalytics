"""Minimum phase value in the image.

Ported from HoloPath resources/parameters/builtin/image/phase_min.py.
"""

import numpy as np

KEY = 'phase_min'
LABEL = 'Phase Min'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 110
DESCRIPTION = 'Minimum: the lowest pixel value of the whole phase map, background included. In rad.'


def compute(ctx):
    return float(np.min(ctx.flat))
