"""Minimum curvature along the smoothed object contour.

Ported from HoloPath resources/parameters/builtin/object/curvature_min.py. Computed along the vector outline, in 1/µm (HoloPath: along the pixel contour, in 1/px).
"""

KEY = 'curvature_min'
LABEL = 'Curvature (min)'
UNIT = '1/µm'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 140
DESCRIPTION = "The curvature κ = |x′y″ − x″y′| / (x′² + y′²)^(3/2) (1/µm; 1/κ = the radius of the circle that fits the boundary there) is computed along the outer outline, sampled every 'spacing' µm (constant; 0 = automatic) and smoothed with a Savitzky–Golay filter. This is its minimum: near 0 = the boundary has a straight section."
CONSTANTS = [
    {'key': 'spacing',
     'label': 'Sample spacing',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 10.0,
     'step': 0.05,
     'suffix': ' µm',
     'tooltip': "Distance between outline points before smoothing; 0 = the measured image's pixel size "
                "(HoloPath's contour spacing)"},
]


def compute(ctx):
    return ctx.curvature_um(ctx.constants["spacing"])[2]
