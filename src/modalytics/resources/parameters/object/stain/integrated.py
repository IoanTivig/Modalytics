"""Integrated stain (object, stain).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'integrated'
LABEL = 'Integrated stain'
UNIT = 'OD·µm²'
CATEGORY = 'stain'
SCOPE = 'object'
IMAGE_TYPE = 'stain'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 62
DESCRIPTION = "Integrated stain = Σ ODᵢ·A_px over the object (OD·µm², A_px = pixel area): the total amount of the stain in the object (bigger and/or darker = more). On the stain's optical density (colour deconvolution step: the amount of this one stain per pixel, Ruifrok & Johnston 2001)."
REFERENCES = [
    ('Ruifrok AC, Johnston DA (2001). Quantification of histochemical staining by color deconvolution. '
     'Anal Quant Cytol Histol 23(4):291–299.'),
]
CONSTANTS = []


def compute(ctx):
    px, py = ctx.image.pixel_size_um
    return float(np.sum(ctx.pixels) * px * py)
