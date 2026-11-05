"""Spatial correlation length lc (um) describing the characteristic size of cell structures.

Obtained by autocorrelation of the Phase Fluctuation Image (PFI) and exponential
fitting of the curve (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/object/correlation_length.py.
"""

KEY = 'correlation_length'
LABEL = 'Correlation Length'
UNIT = 'µm'
CATEGORY = 'disorder'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 200
DESCRIPTION = 'Correlation length l_c (µm): the typical size of the structures inside the object. The phase fluctuation image PF = local variance / local mean² of the phase in a w × w window (constant, default 3 px), computed after removing a 2D polynomial of degree d (default 2) fitted to the phase inside the object (Eldridge et al. 2017); its autocorrelation C(r) is averaged over directions and fitted with C(r) = exp(−r / l_c) (up to where C drops below 0.05; the 1/e crossing if the fit fails). Higher = larger, coarser structures.'
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
    return ctx.correlation_length_for(ctx.constants["degree"], ctx.constants["window"])
