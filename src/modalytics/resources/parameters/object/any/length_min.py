"""Minor axis length of the best-fit ellipse (object width).

Ported from HoloPath resources/parameters/builtin/object/length_min.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'length_min'
LABEL = 'Minor Axis'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 30
DESCRIPTION = 'Minor axis (µm) of the ellipse with the same second moments of area as the outline: 4√λ₂, λ₂ the smaller eigenvalue (for an ellipse exactly its short axis). Its overall width; see also the minimum Feret diameter.'


def compute(ctx):
    return ctx.minor_axis_um
