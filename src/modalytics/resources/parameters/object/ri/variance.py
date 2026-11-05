"""Refractive-index variance (object, ri).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'variance'
LABEL = 'Refractive-index variance'
UNIT = ''
CATEGORY = 'refractive index'
SCOPE = 'object'
IMAGE_TYPE = 'ri'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 62
DESCRIPTION = 'Variance σ² = (1/N)·Σ (xᵢ − x̄)² over the N pixels inside the object. On the refractive-index map n = n_medium + φλ / (2π·L) (Refractive-index map step: the mean index along each line of sight through a sample of known thickness L). How heterogeneous the material is; the refractive-index variance of unstained biopsies was consistently higher in prostate tumours than in normal regions (Wang et al. 2011).'
REFERENCES = [
    ('Wang Z, Tangella K, Balla A, Popescu G (2011). Tissue refractive index as marker of disease. J '
     'Biomed Opt 16(11):116017. doi:10.1117/1.3656732'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.var(ctx.pixels))
