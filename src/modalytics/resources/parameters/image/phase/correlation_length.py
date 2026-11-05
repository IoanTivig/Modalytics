"""Spatial correlation length lc (um) computed from the full image PFI.

Obtained by autocorrelation of the full-image Phase Fluctuation Image (PFI)
and exponential fitting of the curve (Eldridge et al. 2017).

Ported from HoloPath resources/parameters/builtin/image/correlation_length.py.
"""

KEY = 'correlation_length'
LABEL = 'Correlation Length'
UNIT = 'µm'
CATEGORY = 'disorder'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 200
DESCRIPTION = 'Correlation length l_c (µm) of the whole map: the phase fluctuation image PF = local variance / local mean² of the whole phase map in a w × w window (constant, default 3 px), without detrending (as HoloPath); its autocorrelation C(r), averaged over directions, is fitted with C(r) = exp(−r / l_c). Higher = larger structures.'
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
    return ctx.correlation_length_for(ctx.constants["window"])
