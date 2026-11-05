"""Rectangularity (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'rectangularity'
LABEL = 'Rectangularity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 105
DESCRIPTION = 'Rectangularity = A / A_MBR: the area divided by the area of the smallest (rotated) rectangle enclosing the object, 0–1. 1 = a rectangle; π/4 ≈ 0.785 for an ellipse; lower = irregular or branched.'
CONSTANTS = []


def compute(ctx):
    box = ctx.polygon.minimum_rotated_rectangle
    return ctx.area_um2 / box.area if box.area > 0 else 0.0
