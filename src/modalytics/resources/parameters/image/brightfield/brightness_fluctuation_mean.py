"""Brightness Fluctuation Mean of the image on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

import numpy as np

KEY = 'brightness_fluctuation_mean'
LABEL = 'Brightness Fluctuation Mean'
UNIT = ''
CATEGORY = 'fluctuation'
SCOPE = 'image'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
ORDER = 230
DESCRIPTION = "Mean local fluctuation of the brightness over the whole image: ⟨var_w(b)⟩ / b̄², the variance in a w × w window (default 3 px) divided by the image's mean brightness squared."
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
    return float(np.mean(ctx.brightness_fluctuation_image(ctx.constants["window"])))
