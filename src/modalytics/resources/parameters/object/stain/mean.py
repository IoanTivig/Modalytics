"""Stain mean (object, stain).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'mean'
LABEL = 'Stain mean'
UNIT = 'OD'
CATEGORY = 'stain'
SCOPE = 'object'
IMAGE_TYPE = 'stain'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 60
DESCRIPTION = "Mean x̄ = (1/N)·Σ xᵢ over the N pixels inside the object. On the stain's optical density (colour deconvolution step: the amount of this one stain per pixel, Ruifrok & Johnston 2001). Higher = more of the stain."
REFERENCES = [
    ('Ruifrok AC, Johnston DA (2001). Quantification of histochemical staining by color deconvolution. '
     'Anal Quant Cytol Histol 23(4):291–299.'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.mean(ctx.pixels))
