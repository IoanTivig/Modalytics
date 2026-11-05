"""Length of the object's outer outline.

Ported from HoloPath resources/parameters/builtin/object/perimeter.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image. The outer boundary only, as HoloPath.
"""

KEY = 'perimeter'
LABEL = 'Perimeter'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 50
DESCRIPTION = 'Perimeter P (µm): the length of the outer outline (holes not counted). Grows with size and with how convoluted the boundary is.'


def compute(ctx):
    return ctx.perimeter_um
