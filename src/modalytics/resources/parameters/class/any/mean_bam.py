"""Mean irregularity (BAM) (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'mean_bam'
LABEL = 'Mean irregularity (BAM)'
UNIT = ''
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 100
DESCRIPTION = "Mean BAM = (1/N) · Σ BAMᵢ over the class's objects on the image. BAM = the best alignment metric of each object (the 'Irregularity (BAM)' object parameter); objects touching the image border and those smaller than a minimum area can be left out (constants), as Awan et al. 2017 did for border glands and tangentially cut crypts. With the BAM entropy, one of the two image features Awan et al. 2017 used to grade colorectal adenocarcinoma from gland shape. Measured for each class of the profile on each image (one row per class and image)."
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
]


def compute(ctx):
    c = ctx.constants
    v = ctx.object_values('object:any:best_alignment_metric', exclude_edge=c['exclude_edge'],
                          min_area_um2=c['min_area_um2'])
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float('nan')
