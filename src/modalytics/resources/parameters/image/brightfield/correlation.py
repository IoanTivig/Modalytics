"""Correlation of the image on brightfield images.

Brightness runs from 0 (black) to 1 (the brightest value the file can hold),
colour channels averaged; it needs no background reference, and compares images
taken with the same microscope, lamp and exposure
(modalytics/parameters/optical.py).
"""

KEY = 'correlation'
LABEL = 'Correlation'
UNIT = ''
CATEGORY = 'texture'
SCOPE = 'image'
IMAGE_TYPE = 'brightfield'
DEFAULT_ON = True
ORDER = 208
DESCRIPTION = "Correlation = Σᵢⱼ (i − μᵢ)(j − μⱼ)·P(i, j) / (σᵢσⱼ), −1 to 1. How well a pixel's value predicts its neighbour's: near 1 = smooth gradients and structures larger than the distance; near 0 = noise-like texture. Grey-level co-occurrence matrix (Haralick et al. 1973): the values of the whole image are rescaled to 254 grey levels (0 = its lowest, 253 = its highest value, so it is independent of the absolute level); P(i, j) is the fraction of neighbouring pixel pairs, at distance d (constant, default 1 px) in the 4 directions 0°, 45°, 90°, 135°, with grey levels i and j (symmetric, normalised); the feature is averaged over the 4 directions. Computed on the whole image's brightness b = pixel value / the largest value the file can hold (0 = black, 1 = white; colour channels averaged)."
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
     'tooltip': 'Pixel distance of the pairs compared by the co-occurrence matrix'},
]


def compute(ctx):
    return float(ctx.glcm_features(ctx.constants["distance"])["correlation"])
