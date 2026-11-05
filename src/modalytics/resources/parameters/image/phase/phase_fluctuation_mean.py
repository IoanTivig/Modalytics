"""Mean of local phase fluctuations (dimensionless) over the entire image.

Calculated as local variance divided by squared local mean in a 3x3 pixel window (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/image/phase_fluctuation_mean.py.
"""

import numpy as np

KEY = 'phase_fluctuation_mean'
LABEL = 'Phase Fluctuation Mean'
UNIT = ''
CATEGORY = 'disorder'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 220
DESCRIPTION = 'Mean phase fluctuation ⟨PF⟩ over the whole map, where the phase fluctuation image PF = local variance / local mean² of the whole phase map in a w × w window (constant, default 3 px), without detrending (as HoloPath).'
REFERENCES = [
    ('Eldridge WJ, Steelman ZA, Loomis B, Wax A (2017). Optical phase measurements of disorder '
     'strength link microstructure to cell stiffness. Biophys J 112(4):692–702. '
     'doi:10.1016/j.bpj.2016.12.016'),
]
CONSTANTS = [
    {'key': 'window',
     'label': 'Fluctuation window',
     'kind': 'int',
     'default': 3,
     'minimum': 3,
     'maximum': 15,
     'step': 2,
     'suffix': ' px',
     'tooltip': 'Side of the square neighbourhood of the local variance / mean² (HoloPath: 3)'},
]


def compute(ctx):
    return float(np.mean(ctx.fluctuation_image(ctx.constants["window"])))
