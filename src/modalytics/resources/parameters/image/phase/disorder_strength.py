"""Disorder strength Ld (um) computed across the full image.

Relates local phase fluctuations to spatial correlation length lc.
Eq. 5 (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/image/disorder_strength.py.
"""

from modalytics.parameters.measures import disorder_strength

KEY = 'disorder_strength'
LABEL = 'Disorder Strength'
UNIT = 'µm'
CATEGORY = 'disorder'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 210
DESCRIPTION = 'Disorder strength L_d = ⟨PF⟩·n̄²·l_c (µm) of the whole map (Eldridge et al. 2017, Eq. 5): the phase fluctuation image PF = local variance / local mean² of the whole phase map in a w × w window (constant, default 3 px), without detrending (as HoloPath), l_c its correlation length, n̄ the mean refractive index (constant, default 1.38).'
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
    {'key': 'refractive_index',
     'label': 'Refractive index',
     'kind': 'float',
     'default': 1.38,
     'minimum': 1.0,
     'maximum': 2.0,
     'step': 0.01,
     'tooltip': 'Mean refractive index n of the sample (Eldridge et al. 2017: 1.38)'},
]


def compute(ctx):
    w = ctx.constants["window"]
    return disorder_strength(ctx.fluctuation_image(w), ctx.correlation_length_for(w), None,
                             ctx.constants["refractive_index"])
