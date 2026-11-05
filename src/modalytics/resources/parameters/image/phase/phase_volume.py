"""Sum of all phase values in the image (integrated phase).

Ported from HoloPath resources/parameters/builtin/image/phase_volume.py.
"""

import numpy as np

KEY = 'phase_volume'
LABEL = 'Phase Volume'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 135
DESCRIPTION = 'Integrated phase Σ φᵢ over all pixels of the phase map (rad, summed per pixel; depends on the pixel size and on the background level).'


def compute(ctx):
    return float(np.sum(ctx.flat))
