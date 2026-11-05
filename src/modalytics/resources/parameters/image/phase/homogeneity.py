"""Whole-image GLCM homogeneity.

Ported from HoloPath resources/parameters/builtin/image/homogeneity.py.
"""

KEY = 'homogeneity'
LABEL = 'Homogeneity'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 206
DESCRIPTION = 'Homogeneity (inverse difference moment) = Σᵢⱼ P(i, j) / (1 + (i − j)²), 0–1. Higher = neighbours have similar values (smooth); lower = abrupt local changes. Grey-level co-occurrence matrix (Haralick et al. 1973): the values of the whole image are rescaled to 254 grey levels (0 = its lowest, 253 = its highest value, so it is independent of the absolute level); P(i, j) is the fraction of neighbouring pixel pairs, at distance d (constant, default 1 px) in the 4 directions 0°, 45°, 90°, 135°, with grey levels i and j (symmetric, normalised); the feature is averaged over the 4 directions. Computed on the whole phase map (rad).'
REFERENCES = [
    ('Haralick RM, Shanmugam K, Dinstein I (1973). Textural features for image classification. IEEE '
     'Trans Syst Man Cybern SMC-3(6):610–621. doi:10.1109/TSMC.1973.4309314'),
]
CONSTANTS = [
    {'key': 'distance',
     'label': 'Neighbour distance',
     'kind': 'int',
     'default': 1,
     'minimum': 1,
     'maximum': 16,
     'suffix': ' px',
     'tooltip': 'Pixel distance of the pairs compared by the co-occurrence matrix (HoloPath: 1)'},
]


def compute(ctx):
    return float(ctx.glcm_features(ctx.constants["distance"])["homogeneity"])
