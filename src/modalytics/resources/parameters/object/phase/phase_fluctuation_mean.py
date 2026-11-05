"""Mean of local phase fluctuations (dimensionless) over the object's mask.

Calculated as local variance divided by squared local mean in a 3x3 pixel window (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/object/phase_fluctuation_mean.py.
"""

KEY = 'phase_fluctuation_mean'
LABEL = 'Phase Fluctuation Mean'
UNIT = ''
CATEGORY = 'disorder'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 220
DESCRIPTION = 'Mean phase fluctuation ⟨PF⟩ inside the object, where the phase fluctuation image PF = local variance / local mean² of the phase in a w × w window (constant, default 3 px), computed after removing a 2D polynomial of degree d (default 2) fitted to the phase inside the object (Eldridge et al. 2017). Higher = a more heterogeneous, textured interior.'
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
]


def compute(ctx):
    pfi = ctx.fluctuation_image(ctx.constants["degree"], ctx.constants["window"])
    return float(pfi[ctx.mask].mean()) if ctx.mask.any() else 0.0
