"""Feature standard deviation (object, feature).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'std'
LABEL = 'Feature standard deviation'
UNIT = ''
CATEGORY = 'feature'
SCOPE = 'object'
IMAGE_TYPE = 'feature'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 61
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). On the feature map (edges, blobs, ridges or texture, depending on the step that made it; dimensionless, relative).'
CONSTANTS = []


def compute(ctx):
    return float(np.std(ctx.pixels))
