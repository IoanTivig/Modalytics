"""Irregularity entropy (BAM) (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'bam_entropy'
LABEL = 'Irregularity entropy (BAM)'
UNIT = 'bits'
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 101
DESCRIPTION = "BAM entropy H = −Σₖ pₖ · log₂ pₖ, pₖ the share of the class's objects whose BAM falls in histogram bin k (bins of the given width, default 0.015 as in Awan et al. 2017; empty bins add nothing). How varied the objects' irregularity is. Awan et al. set the bins between the smallest and largest BAM of their whole dataset; here they start at 0, so values compare between images measured the same way. BAM = the best alignment metric of each object (the 'Irregularity (BAM)' object parameter); objects touching the image border and those smaller than a minimum area can be left out (constants), as Awan et al. 2017 did for border glands and tangentially cut crypts. Measured for each class of the profile on each image (one row per class and image)."
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
    if not v.size:
        return float('nan')
    counts = np.bincount(np.floor(v / c['bin_width']).astype(int))
    p = counts[counts > 0] / v.size
    return float(-np.sum(p * np.log2(p))) + 0.0          # never "-0.0"
