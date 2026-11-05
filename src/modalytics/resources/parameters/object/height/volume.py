"""Volume (object, height).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'volume'
LABEL = 'Volume'
UNIT = 'µm³'
CATEGORY = 'height'
SCOPE = 'object'
IMAGE_TYPE = 'height'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 63
DESCRIPTION = "Volume V = Σ hᵢ·A_px over the object (µm³, A_px = pixel area): the object's volume from its height. On the height map h = φλ / (2π·Δn) (µm; Height map step, Δn the sample–medium refractive-index difference: Rappaz et al. 2005); as accurate as the Δn given to the step."
REFERENCES = [
    ('Rappaz B, Marquet P, Cuche E, Emery Y, Depeursinge C, Magistretti P (2005). Measurement of the '
     'integral refractive index and dynamic cell morphometry of living cells with digital holographic '
     'microscopy. Opt Express 13(23):9361–9373. doi:10.1364/OPEX.13.009361'),
]
CONSTANTS = []


def compute(ctx):
    px, py = ctx.image.pixel_size_um
    return float(np.sum(ctx.pixels) * px * py)
