"""Sum of all phase values inside the object (integrated phase, proportional to dry mass).

Ported from HoloPath resources/parameters/builtin/object/phase_volume.py.
"""

import numpy as np

KEY = 'phase_volume'
LABEL = 'Phase Volume'
UNIT = 'rad'
CATEGORY = 'intensity'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 75
DESCRIPTION = "Integrated phase Σ φᵢ over the pixels inside the object (rad, summed per pixel, as HoloPath; it depends on the pixel size). Proportional to the object's dry mass: for a physical value use the dry-mass map's Dry mass (pg)."


def compute(ctx):
    return float(np.sum(ctx.pixels))
