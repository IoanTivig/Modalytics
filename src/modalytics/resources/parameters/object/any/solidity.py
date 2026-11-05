"""Solidity (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'solidity'
LABEL = 'Solidity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 101
DESCRIPTION = "Solidity = A / A_hull: the object's area divided by the area of its convex hull (the smallest convex outline around it, like a rubber band stretched around the object), 0–1. 1 = convex, no indentation; lower = concave: indented, lobed or budding outlines."
CONSTANTS = []


def compute(ctx):
    return ctx.solidity
