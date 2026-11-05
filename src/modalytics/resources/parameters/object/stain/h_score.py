"""H-score (pixels) (object, stain).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

import numpy as np

KEY = 'h_score'
LABEL = 'H-score (pixels)'
UNIT = ''
CATEGORY = 'stain'
SCOPE = 'object'
IMAGE_TYPE = 'stain'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 64
DESCRIPTION = "H-score = 1·%weak + 2·%moderate + 3·%strong, 0–300: the histochemical score of McCarty et al. 1985, computed here on the pixels inside the object (% of pixels with stain OD in [t₁, t₂), [t₂, t₃) and ≥ t₃; thresholds are constants, defaults 0.2 / 0.4 / 0.6 OD). The classical H-score counts cells: measured on cell or nucleus objects, each object's mean intensity class gives that. On the stain's optical density (colour deconvolution step: the amount of this one stain per pixel, Ruifrok & Johnston 2001). Calibrate the thresholds with controls; compare H-scores only between slides stained and scanned alike."
REFERENCES = [
    ('McCarty KS Jr, Miller LS, Cox EB, Konrath J, McCarty KS (1985). Estrogen receptor analyses. '
     'Correlation of biochemical and immunohistochemical methods using monoclonal antireceptor '
     'antibodies. Arch Pathol Lab Med 109(8):716–721.'),
    ('Ruifrok AC, Johnston DA (2001). Quantification of histochemical staining by color deconvolution. '
     'Anal Quant Cytol Histol 23(4):291–299.'),
]
CONSTANTS = [
    {'key': 'weak',
     'label': 'Weak from',
     'kind': 'float',
     'default': 0.2,
     'minimum': 0.0,
     'maximum': 5.0,
     'step': 0.05,
     'suffix': ' OD',
     'tooltip': 'Lowest OD counted as weak (1+)'},
    {'key': 'moderate',
     'label': 'Moderate from',
     'kind': 'float',
     'default': 0.4,
     'minimum': 0.0,
     'maximum': 5.0,
     'step': 0.05,
     'suffix': ' OD',
     'tooltip': 'Lowest OD counted as moderate (2+)'},
    {'key': 'strong',
     'label': 'Strong from',
     'kind': 'float',
     'default': 0.6,
     'minimum': 0.0,
     'maximum': 5.0,
     'step': 0.05,
     'suffix': ' OD',
     'tooltip': 'Lowest OD counted as strong (3+)'},
]


def compute(ctx):
    v = np.asarray(ctx.pixels, float)
    if not v.size:
        return float('nan')
    c = ctx.constants
    weak = np.mean((v >= c['weak']) & (v < c['moderate']))
    moderate = np.mean((v >= c['moderate']) & (v < c['strong']))
    strong = np.mean(v >= c['strong'])
    return float(100.0 * (weak + 2 * moderate + 3 * strong))
