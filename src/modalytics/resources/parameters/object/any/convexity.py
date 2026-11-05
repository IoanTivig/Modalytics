"""Convexity (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'convexity'
LABEL = 'Convexity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 102
DESCRIPTION = "Convexity = P_hull / P: the perimeter of the convex hull divided by the object's perimeter, 0–1. 1 = convex with a smooth boundary; lower = a ragged, folded or crenellated boundary (it falls with small-scale roughness, unlike solidity, which falls with large indentations)."
CONSTANTS = []


def compute(ctx):
    return ctx.convexity
