"""Optical Volume of the object on OPD maps.

An OPD map's pixels are optical path difference, OPD = φ · λ / 2π (nm), made
from a phase map by the OPD map processing step
(modalytics/resources/processing/opd/opd.py).
"""

import numpy as np

KEY = 'optical_volume'
LABEL = 'Optical Volume'
UNIT = 'µm³'
CATEGORY = 'optical path'
SCOPE = 'object'
IMAGE_TYPE = 'opd'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 75
DESCRIPTION = 'Optical volume V_opt = Σ OPDᵢ·A_px over the object (µm³): the integrated optical path, equal to (refractive-index difference) × (volume). Proportional to dry mass.'
CONSTANTS = []


def compute(ctx):
    px, py = ctx.image.pixel_size_um
    return float(np.sum(ctx.pixels) * 1e-3 * px * py)
