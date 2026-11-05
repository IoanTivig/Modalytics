"""Positive fraction (object, stain).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'positive_fraction'
LABEL = 'Positive fraction'
UNIT = '%'
CATEGORY = 'stain'
SCOPE = 'object'
IMAGE_TYPE = 'stain'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 63
DESCRIPTION = "Positive fraction = 100 · (pixels with OD ≥ t) / (all pixels inside the object) (%), t the positivity threshold (constant, default 0.2 OD). On the stain's optical density (colour deconvolution step: the amount of this one stain per pixel, Ruifrok & Johnston 2001). Set t from negative and positive controls stained and scanned the same way; the default is only a starting point."
REFERENCES = [
    ('Ruifrok AC, Johnston DA (2001). Quantification of histochemical staining by color deconvolution. '
     'Anal Quant Cytol Histol 23(4):291–299.'),
]
CONSTANTS = [
    {'key': 'threshold',
     'label': 'Positive from',
     'kind': 'float',
     'default': 0.2,
     'minimum': 0.0,
     'maximum': 5.0,
     'step': 0.05,
     'suffix': ' OD',
     'tooltip': 'Stain optical density from which a pixel counts as positive'},
]


def compute(ctx):
    v = np.asarray(ctx.pixels, float)
    return float(100.0 * np.mean(v >= ctx.constants['threshold'])) if v.size else float('nan')
