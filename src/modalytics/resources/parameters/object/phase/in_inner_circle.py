"""1 if the object lies fully inside the centered inner circle, else 0.

Boolean (0/1) flag for a centered circular data window whose diameter is the
SHORTER image side. Useful for Phasics phase maps where the valid data lies
inside a circle and objects touching it are geometrically truncated.

Ported from HoloPath resources/parameters/builtin/object/in_inner_circle.py.
"""

KEY = 'in_inner_circle'
LABEL = 'Inside Inner Circle'
UNIT = ''
CATEGORY = 'flags'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = False
WHOLE_OBJECT = False
ORDER = 310
DESCRIPTION = "Position flag: 1 if the object's farthest pixel from the image centre lies within the circle of radius min(H, W)/2 − margin (constant, px), i.e. inside the round field of view of many holographic microscopes; else 0. Use it to keep only objects in the valid area."
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
    radius = min(ctx.full_shape) / 2.0 - ctx.constants["margin"]
    return 1.0 if ctx.max_radius_from_center <= radius else 0.0
