"""Brightness Fluctuation Mean of the object on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

KEY = 'fluctuation_mean'
LABEL = 'Brightness Fluctuation Mean'
UNIT = ''
CATEGORY = 'fluctuation'
SCOPE = 'object'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 230
DESCRIPTION = "Mean local fluctuation of the brightness inside the object: the variance of b in a w × w window (constant, default 3 px) divided by the object's mean brightness squared, averaged over the object: ⟨var_w(b)⟩ / b̄². Dividing by the object's mean (not the local one) keeps it finite and independent of overall darkness. Higher = a more textured interior."
CONSTANTS = [
    {'key': 'window',
     'label': 'Fluctuation window',
     'kind': 'int',
     'default': 3,
     'minimum': 3,
     'maximum': 15,
     'step': 2,
     'suffix': ' px',
     'tooltip': 'Side of the square neighbourhood of the local variance'},
]


def compute(ctx):
    f = ctx.brightness_fluctuation_image(ctx.constants["window"])
    return float(f[ctx.mask].mean()) if ctx.mask.any() else 0.0
