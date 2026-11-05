"""Hole fraction (lumen) (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'hole_fraction'
LABEL = 'Hole fraction (lumen)'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 151
DESCRIPTION = "Hole fraction = ΣA_holes / (A + ΣA_holes): the share of the object's filled outline taken by its holes, 0–1. For a gland outlined with its lumen as a hole, the lumen-to-gland area ratio."
CONSTANTS = []


def compute(ctx):
    _n, holes = ctx.holes
    total = ctx.area_um2 + holes
    return holes / total if total > 0 else 0.0
