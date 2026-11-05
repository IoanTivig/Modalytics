"""Mean object area (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'mean_area'
LABEL = 'Mean object area'
UNIT = 'µm²'
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 40
DESCRIPTION = "Mean object area = (1/N) · Σ Aᵢ over the class's objects on the image (µm²). Measured for each class of the profile on each image (one row per class and image)."
CONSTANTS = []


def compute(ctx):
    v = ctx.object_values('object:any:area')
    return float(np.mean(v)) if v.size else float('nan')
