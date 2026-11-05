"""White balance: makes the empty background neutral (slides and brightfield colour images).

Scanners and cameras tint the empty glass differently (bluish, yellowish). Each
colour channel is scaled so the background (the brightest 1 % of pixels) becomes
neutral grey — of the same brightness, or pure white.
"""

import numpy as np

from modalytics.parameters import optical as op
from modalytics.processing import helpers as hp

KEY = 'white_balance'
NAME = 'White balance'
SHORT = 'White balance'
ICON = 'mdi6.white-balance-auto'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Correction'
DESCRIPTION = 'Scales each colour channel so the empty background becomes neutral (same brightness) or pure white.'
OPTIONS = [
    {'key': 'target',
     'label': 'Background becomes',
     'kind': 'choice',
     'default': 'Neutral (same brightness)',
     'choices': ['Neutral (same brightness)', 'White'],
     'tooltip': "Neutral: grey of the background's own brightness. White: the brightest value the file "
                'holds'},
]


def prepare(image, options):
    """The gain of each channel."""
    if image.source.channels < 3:
        raise ValueError("white balance needs a colour image")
    white = np.asarray(op.estimate_white(image), float)[:3]
    rng = hp.dtype_range(image.source.dtype)
    target = rng[1] if (options["target"] == "White" and rng) else white.mean()
    return {"gain": (target / white).astype(np.float32)}


def process(pixels, options, state, um_per_px):
    out = pixels.astype(np.float32)
    out[..., :3] *= state["gain"]
    return hp.to_like(out, pixels)
