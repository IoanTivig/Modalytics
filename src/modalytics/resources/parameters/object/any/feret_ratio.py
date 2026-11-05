"""Feret aspect ratio (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'feret_ratio'
LABEL = 'Feret aspect ratio'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 103
DESCRIPTION = 'Feret aspect ratio = D_max / D_min (maximum over minimum Feret diameter), ≥ 1: 1 for a circle; larger = more elongated.'
CONSTANTS = []


def compute(ctx):
    fmax, fmin, _a = ctx.feret
    return fmax / fmin if fmin > 0 else float('nan')
