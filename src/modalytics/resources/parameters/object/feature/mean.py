"""Feature mean (object, feature).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'mean'
LABEL = 'Feature mean'
UNIT = ''
CATEGORY = 'feature'
SCOPE = 'object'
IMAGE_TYPE = 'feature'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 60
DESCRIPTION = 'Mean x̄ = (1/N)·Σ xᵢ over the N pixels inside the object. On the feature map (edges, blobs, ridges or texture, depending on the step that made it; dimensionless, relative): e.g. how much edge, ridge or blob response the object holds.'
CONSTANTS = []


def compute(ctx):
    return float(np.mean(ctx.pixels))
