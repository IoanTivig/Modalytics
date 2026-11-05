"""Diameter of a circle with the same area as the object (sqrt(4·Area/pi)).

Ported from HoloPath resources/parameters/builtin/object/length_avg.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'length_avg'
LABEL = 'Equivalent Diameter'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 20
DESCRIPTION = 'Equivalent diameter d_eq = √(4A/π) (µm): the diameter of the circle with the same area.'


def compute(ctx):
    return ctx.equivalent_diameter_um
