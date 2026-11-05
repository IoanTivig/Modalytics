"""Mean refractive index (object, ri).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'mean'
LABEL = 'Mean refractive index'
UNIT = ''
CATEGORY = 'refractive index'
SCOPE = 'object'
IMAGE_TYPE = 'ri'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 60
DESCRIPTION = 'Mean x̄ = (1/N)·Σ xᵢ over the N pixels inside the object. On the refractive-index map n = n_medium + φλ / (2π·L) (Refractive-index map step: the mean index along each line of sight through a sample of known thickness L). Higher = denser material (more protein, nucleic acid).'
REFERENCES = [
    ('Rappaz B, Marquet P, Cuche E, Emery Y, Depeursinge C, Magistretti P (2005). Measurement of the '
     'integral refractive index and dynamic cell morphometry of living cells with digital holographic '
     'microscopy. Opt Express 13(23):9361–9373. doi:10.1364/OPEX.13.009361'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.mean(ctx.pixels))
