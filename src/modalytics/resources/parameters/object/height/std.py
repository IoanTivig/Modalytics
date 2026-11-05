"""Height standard deviation (object, height).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'std'
LABEL = 'Height standard deviation'
UNIT = 'µm'
CATEGORY = 'height'
SCOPE = 'object'
IMAGE_TYPE = 'height'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 62
DESCRIPTION = 'Standard deviation σ = √[(1/N)·Σ (xᵢ − x̄)²] over the N pixels inside the object (population form). On the height map h = φλ / (2π·Δn) (µm; Height map step, Δn the sample–medium refractive-index difference: Rappaz et al. 2005). Higher = a rougher, more uneven object.'
REFERENCES = [
    ('Rappaz B, Marquet P, Cuche E, Emery Y, Depeursinge C, Magistretti P (2005). Measurement of the '
     'integral refractive index and dynamic cell morphometry of living cells with digital holographic '
     'microscopy. Opt Express 13(23):9361–9373. doi:10.1364/OPEX.13.009361'),
]
CONSTANTS = []


def compute(ctx):
    return float(np.std(ctx.pixels))
