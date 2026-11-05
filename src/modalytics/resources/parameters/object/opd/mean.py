"""OPD Mean of the object on OPD maps.

An OPD map's pixels are optical path difference, OPD = φ · λ / 2π (nm), made
from a phase map by the OPD map processing step
(modalytics/resources/processing/opd/opd.py).
"""

import numpy as np

KEY = 'mean'
LABEL = 'OPD Mean'
UNIT = 'nm'
CATEGORY = 'optical path'
SCOPE = 'object'
IMAGE_TYPE = 'opd'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 60
DESCRIPTION = 'Mean optical path difference OPD = φ·λ/(2π) inside the object (nm): the extra optical path through the object relative to the medium, = Δn·h (refractive-index difference × thickness).'
CONSTANTS = []


def compute(ctx):
    return float(np.mean(ctx.pixels))
