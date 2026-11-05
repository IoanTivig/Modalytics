"""(4·pi·Area)/Perimeter² — 1.0 for a perfect circle, lower for irregular shapes.

Ported from HoloPath resources/parameters/builtin/object/circularity.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

import math

KEY = 'circularity'
LABEL = 'Circularity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 100
DESCRIPTION = 'Circularity C = 4πA / P² (A area, P perimeter), 0–1: 1 for a circle; lower for elongated or ragged outlines (it falls with both elongation and boundary roughness).'


def compute(ctx):
    p = ctx.perimeter_um
    return 4.0 * math.pi * ctx.area_um2 / p ** 2 if p > 0 else 0.0
