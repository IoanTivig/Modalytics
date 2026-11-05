"""OPD Max of the object on OPD maps.

An OPD map's pixels are optical path difference, OPD = φ · λ / 2π (nm), made
from a phase map by the OPD map processing step
(modalytics/resources/processing/opd/opd.py).
"""

import numpy as np

KEY = 'max'
LABEL = 'OPD Max'
UNIT = 'nm'
CATEGORY = 'optical path'
SCOPE = 'object'
IMAGE_TYPE = 'opd'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 65
DESCRIPTION = 'Maximum optical path difference inside the object (nm): its thickest or densest point.'
CONSTANTS = []


def compute(ctx):
    return float(np.max(ctx.pixels))
