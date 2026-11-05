"""Feret diameter (max) (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'feret_max'
LABEL = 'Feret diameter (max)'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 41
DESCRIPTION = 'Maximum Feret diameter D_max (maximum caliper, µm) = max |pᵢ − pⱼ| over all pairs of outline points: the largest distance across the object, whatever its orientation (unlike the major axis, which belongs to a fitted ellipse).'
CONSTANTS = []


def compute(ctx):
    return ctx.feret[0]
