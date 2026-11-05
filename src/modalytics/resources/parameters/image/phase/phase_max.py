"""Maximum phase value in the image.

Ported from HoloPath resources/parameters/builtin/image/phase_max.py.
"""

import numpy as np

KEY = 'phase_max'
LABEL = 'Phase Max'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 120
DESCRIPTION = 'Maximum: the highest pixel value of the whole phase map, background included. In rad.'


def compute(ctx):
    return float(np.max(ctx.flat))
