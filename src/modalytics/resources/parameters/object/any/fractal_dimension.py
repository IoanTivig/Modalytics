"""Fractal dimension (outline) (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'fractal_dimension'
LABEL = 'Fractal dimension (outline)'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 145
DESCRIPTION = "Box-counting (Minkowski–Bouligand) fractal dimension D of the outline: the boundary is covered with square boxes of side ε, from the smallest size (constant; 0 = 2 pixels) to a quarter of the object's size (8 geometric steps); N(ε) = number of boxes it passes through, D = slope of log N(ε) against log(1/ε). 1 = a smooth curve; higher (up to 2) = a more complex, space-filling boundary (e.g. 1.26 for a Koch snowflake)."
CONSTANTS = [
    {'key': 'smallest_um',
     'label': 'Smallest box',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 100.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Smallest box side; 0 = 2 pixels of the measured image'},
]


def compute(ctx):
    return ctx.fractal_dimension(ctx.constants['smallest_um'])
