"""Correlation Length of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

KEY = 'correlation_length'
LABEL = 'Correlation Length'
UNIT = 'µm'
CATEGORY = 'fluctuation'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 220
DESCRIPTION = 'Correlation length l_c (µm): the typical size of the structures inside the object (nuclei, granules, fibres). A 2D polynomial of degree d (constant, default 2) is removed first (slow shading); the autocorrelation C(r) of what remains is averaged over directions and fitted with C(r) = exp(−r / l_c) (up to where C drops below 0.05; the 1/e crossing if the fit fails). Higher = larger, coarser structures. Computed on the brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged).'
CONSTANTS = [
    {'key': 'degree',
     'label': 'Detrend degree',
     'kind': 'int',
     'default': 2,
     'minimum': 0,
     'maximum': 4,
     'tooltip': 'Degree of the 2D polynomial removed from the brightness inside the object first '
                '(uneven illumination), so only the structures remain'},
]


def compute(ctx):
    return ctx.brightness_correlation_length(ctx.constants["degree"])
