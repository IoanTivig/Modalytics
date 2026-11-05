"""Refractive-index standard deviation (object, ri).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'std'
LABEL = 'Refractive-index standard deviation'
UNIT = ''
CATEGORY = 'refractive index'
SCOPE = 'object'
IMAGE_TYPE = 'ri'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 61
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). On the refractive-index map n = n_medium + φλ / (2π·L) (Refractive-index map step: the mean index along each line of sight through a sample of known thickness L).'
REFERENCES = [
    ('Wang Z, Tangella K, Balla A, Popescu G (2011). Tissue refractive index as marker of disease. J '
     'Biomed Opt 16(11):116017. doi:10.1117/1.3656732'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.std(ctx.pixels))
