"""Major axis length of the best-fit ellipse (object length).

Ported from HoloPath resources/parameters/builtin/object/length_max.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'length_max'
LABEL = 'Major Axis'
UNIT = 'µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 40
DESCRIPTION = 'Major axis (µm) of the ellipse with the same second moments of area as the outline: 4√λ₁, λ₁ the larger eigenvalue of the normalised second-moment matrix (for an ellipse it is exactly its long axis). Its overall length; see also the maximum Feret diameter.'


def compute(ctx):
    return ctx.major_axis_um
