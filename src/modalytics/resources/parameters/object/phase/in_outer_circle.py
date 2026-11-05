"""1 if the object lies fully inside the centered outer circle, else 0.

Boolean (0/1) flag for a centered circular data window whose diameter is the
LONGER image side. Use whichever circle (inner/outer) matches your acquisition's
valid-data window.

Ported from HoloPath resources/parameters/builtin/object/in_outer_circle.py.
"""

KEY = 'in_outer_circle'
LABEL = 'Inside Outer Circle'
UNIT = ''
CATEGORY = 'flags'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = False
WHOLE_OBJECT = False
ORDER = 320
DESCRIPTION = "Position flag: 1 if the object's farthest pixel from the image centre lies within the circle of radius max(H, W)/2 − margin (constant, px); else 0."
CONSTANTS = [
    {'key': 'margin',
     'label': 'Margin',
     'kind': 'float',
     'default': 2.0,
     'minimum': 0.0,
     'maximum': 50.0,
     'step': 0.5,
     'suffix': ' px',
     'tooltip': 'Shrink the circle by this much so objects touching it count as outside (HoloPath: 2)'},
]


def compute(ctx):
    radius = max(ctx.full_shape) / 2.0 - ctx.constants["margin"]
    return 1.0 if ctx.max_radius_from_center <= radius else 0.0
