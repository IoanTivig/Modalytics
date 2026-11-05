"""Brightness, contrast and gamma: a fixed tone curve, for looking (slides and brightfield images).

Values between the chosen black and white points are stretched to the full
range, then a gamma curve brightens (< 1) or darkens (> 1) the mid-tones. The
points are taken from the image (1 % darkest and brightest, per channel or all
channels together) or given.
"""

import numpy as np

from modalytics.processing import helpers as hp

KEY = 'intensity'
NAME = 'Brightness, contrast and gamma'
SHORT = 'Brightness'
ICON = 'mdi6.tune-vertical'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Enhance'
DESCRIPTION = 'Stretches the values between a black and a white point to the full range, then applies a gamma curve. For looking.'
OPTIONS = [
    {'key': 'points',
     'label': 'Black and white points',
     'kind': 'choice',
     'default': 'From the image (all channels)',
     'choices': ['From the image (all channels)', 'From the image (each channel)', 'Fixed values'],
     'tooltip': 'From the image: the 1 % darkest and brightest values. Each channel: also balances '
                'colours'},
    {'key': 'black',
     'label': 'Black point',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 65535.0,
     'step': 1.0,
     'tooltip': 'Values at or below become black (used with "Fixed values")'},
    {'key': 'white',
     'label': 'White point',
     'kind': 'float',
     'default': 255.0,
     'minimum': 1.0,
     'maximum': 65535.0,
     'step': 1.0,
     'tooltip': 'Values at or above become white (used with "Fixed values")'},
    {'key': 'gamma',
     'label': 'Gamma',
     'kind': 'float',
     'default': 1.0,
     'minimum': 0.1,
     'maximum': 5.0,
     'step': 0.05,
     'tooltip': '1 = straight; below 1 brightens the mid-tones, above 1 darkens them'},
]


def prepare(image, options):
    top = (hp.dtype_range(image.source.dtype) or (0.0, 1.0))[1]
    if options["points"] == "Fixed values":
        lo, hi = float(options["black"]), float(options["white"])
        if hi <= lo:
            raise ValueError("the white point must be above the black point")
        return {"lo": np.float32(lo), "hi": np.float32(hi), "top": top}
    view = hp.overview(image).astype(np.float32)
    if options["points"].endswith("(each channel)") and view.ndim == 3:
        pts = [hp.robust_range(view[..., c]) for c in range(view.shape[2])]
        lo, hi = np.array([p[0] for p in pts], np.float32), np.array([p[1] for p in pts], np.float32)
    else:
        lo, hi = (np.float32(v) for v in hp.robust_range(view))
    return {"lo": lo, "hi": hi, "top": top}


def process(pixels, options, state, um_per_px):
    x = np.clip((pixels.astype(np.float32) - state["lo"]) / (state["hi"] - state["lo"]), 0, 1)
    x = x ** np.float32(options["gamma"])
    return hp.to_like(x * state["top"], pixels)
