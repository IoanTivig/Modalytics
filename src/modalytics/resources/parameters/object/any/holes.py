"""Holes (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'holes'
LABEL = 'Holes'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 150
DESCRIPTION = 'Number of holes in the outline: for a gland outlined with its lumen left out, the number of lumens.'
CONSTANTS = []


def compute(ctx):
    return ctx.holes[0]
