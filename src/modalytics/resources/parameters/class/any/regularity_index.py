"""Regularity index (BAM) (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'regularity_index'
LABEL = 'Regularity index (BAM)'
UNIT = ''
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 102
DESCRIPTION = "Regularity index = (objects in the first two BAM histogram bins) / (all objects), 0–1: the share of regular objects, BAM < 2 × bin width (default 0.03). Awan et al. 2017 used it with the mean BAM and BAM entropy to grade colorectal adenocarcinoma (their bins start at the dataset's smallest BAM; here at 0). BAM = the best alignment metric of each object (the 'Irregularity (BAM)' object parameter); objects touching the image border and those smaller than a minimum area can be left out (constants), as Awan et al. 2017 did for border glands and tangentially cut crypts. Measured for each class of the profile on each image (one row per class and image)."
REFERENCES = [
    ('Awan R, Sirinukunwattana K, Epstein D, Jefferyes S, Qidwai U, Aftab Z, Mujeeb I, Snead D, '
     'Rajpoot N (2017). Glandular morphometrics for objective grading of colorectal adenocarcinoma '
     'histology images. Sci Rep 7:16852. doi:10.1038/s41598-017-16516-w'),
]
CONSTANTS = [
    {'key': 'exclude_edge',
     'label': 'Leave out objects at the border',
     'kind': 'bool',
     'default': True,
     'tooltip': 'Objects cut by the image border have incomplete outlines (and artificially high '
                'irregularity)'},
    {'key': 'min_area_um2',
     'label': 'Smallest object',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 10000000.0,
     'step': 10.0,
     'suffix': ' µm²',
     'tooltip': 'Leave out smaller objects (e.g. tangentially cut crypts)'},
    {'key': 'bin_width',
     'label': 'Histogram bin width',
     'kind': 'float',
     'default': 0.015,
     'minimum': 0.001,
     'maximum': 1.0,
     'step': 0.001,
     'tooltip': 'Width of the BAM histogram bins (Awan et al. 2017: 0.015)'},
]


def compute(ctx):
    c = ctx.constants
    v = ctx.object_values('object:any:best_alignment_metric', exclude_edge=c['exclude_edge'],
                          min_area_um2=c['min_area_um2'])
    v = v[np.isfinite(v)]
    return float(np.mean(v < 2 * c['bin_width'])) if v.size else float('nan')
