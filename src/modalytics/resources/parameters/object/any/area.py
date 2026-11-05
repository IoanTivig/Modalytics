"""Area of the object's outline, holes subtracted.

Ported from HoloPath resources/parameters/builtin/object/area.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'area'
LABEL = 'Area'
UNIT = 'µm²'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 10
DESCRIPTION = 'Area A of the outline (µm²), holes (e.g. a lumen) subtracted, computed exactly from the vector outline (shoelace formula), so it is the same on any image.'


def compute(ctx):
    return ctx.area_um2
