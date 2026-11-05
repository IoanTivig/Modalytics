"""Median phase value in the image.

Ported from HoloPath resources/parameters/builtin/image/phase_median.py.
"""

import numpy as np

KEY = 'phase_median'
LABEL = 'Phase Median'
UNIT = 'rad'
CATEGORY = 'phase'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 130
DESCRIPTION = 'Median: half the pixels of the whole phase map, background included are below it, half above (robust to outliers). In rad.'


def compute(ctx):
    return float(np.median(ctx.flat))
