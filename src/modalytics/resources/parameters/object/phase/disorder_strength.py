"""Disorder strength Ld (um) computed across the object's mask.

Relates local phase fluctuations to spatial correlation length lc.
Eq. 5 (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/object/disorder_strength.py.
"""

from modalytics.parameters.measures import disorder_strength

KEY = 'disorder_strength'
LABEL = 'Disorder Strength'
UNIT = 'µm'
CATEGORY = 'disorder'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 210
DESCRIPTION = 'Disorder strength L_d = ⟨PF⟩·n̄²·l_c (µm), Eq. 5 of Eldridge et al. 2017: the phase fluctuation image PF = local variance / local mean² of the phase in a w × w window (constant, default 3 px), computed after removing a 2D polynomial of degree d (default 2) fitted to the phase inside the object (Eldridge et al. 2017), l_c its correlation length and n̄ the mean refractive index (constant, default 1.38 as in the paper). L_d = ⟨Δn²⟩·l_c combines how strongly and over what distance the refractive index varies; in the paper it was inversely related to cell stiffness. Higher = a more disordered interior.'
REFERENCES = [
    ('Eldridge WJ, Steelman ZA, Loomis B, Wax A (2017). Optical phase measurements of disorder '
     'strength link microstructure to cell stiffness. Biophys J 112(4):692–702. '
     'doi:10.1016/j.bpj.2016.12.016'),
]
CONSTANTS = [
    {'key': 'degree',
     'label': 'Detrend degree',
     'kind': 'int',
     'default': 2,
     'minimum': 0,
     'maximum': 4,
     'tooltip': 'Degree of the 2D polynomial removed from the phase inside the object first (HoloPath: '
                '2)'},
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
    s = ctx.constants
    pfi = ctx.fluctuation_image(s["degree"], s["window"])
    lc = ctx.correlation_length_for(s["degree"], s["window"])
    return disorder_strength(pfi, lc, ctx.mask, s["refractive_index"])
