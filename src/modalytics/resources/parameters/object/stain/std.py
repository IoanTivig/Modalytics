"""Stain standard deviation (object, stain).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'std'
LABEL = 'Stain standard deviation'
UNIT = 'OD'
CATEGORY = 'stain'
SCOPE = 'object'
IMAGE_TYPE = 'stain'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 61
DESCRIPTION = "Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). On the stain's optical density (colour deconvolution step: the amount of this one stain per pixel, Ruifrok & Johnston 2001). Higher = unevenly stained."
REFERENCES = [
    ('Ruifrok AC, Johnston DA (2001). Quantification of histochemical staining by color deconvolution. '
     'Anal Quant Cytol Histol 23(4):291–299.'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.std(ctx.pixels))
