"""Feret diameter (min) (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'feret_min'
LABEL = 'Feret diameter (min)'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 42
DESCRIPTION = "Minimum Feret diameter D_min (minimum caliper, µm): the narrowest distance between two parallel lines that enclose the object, over all directions (rotating calipers on the convex hull). The object's true width."
CONSTANTS = []


def compute(ctx):
    return ctx.feret[1]
