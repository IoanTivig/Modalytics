"""1 - minor/major axis of the best-fit ellipse (0 = circular, approaching 1 = elongated).

Ported from HoloPath resources/parameters/builtin/object/ellipticity.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'ellipticity'
LABEL = 'Ellipticity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 115
DESCRIPTION = 'Ellipticity = 1 − minor axis / major axis of the ellipse with the same second moments, 0–1: 0 = round; towards 1 = elongated.'


def compute(ctx):
    major, minor = ctx.axes_um
    return 1.0 - minor / major if major > 0 else 0.0
