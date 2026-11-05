"""Colour → grey: one grey value per pixel (slides and brightfield colour images).

Luminance (how bright it looks), the mean of the channels, or one channel: for
H&E, the green channel often shows nuclei with the most contrast.
"""

import numpy as np

from modalytics.processing import helpers as hp

KEY = 'to_grey'
NAME = 'Colour → grey'
SHORT = 'Grey'
ICON = 'mdi6.invert-colors-off'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Enhance'
DESCRIPTION = 'Makes a grey image: luminance, the mean of the channels, or the red, green or blue channel alone.'
OPTIONS = [
    {'key': 'mode',
     'label': 'Grey from',
     'kind': 'choice',
     'default': 'Luminance',
     'choices': ['Luminance', 'Mean of the channels', 'Red', 'Green', 'Blue'],
     'tooltip': 'Luminance: brightness as the eye sees it (Rec. 709). Green: often the best nuclei '
                'contrast in H&E'},
]


def process(pixels, options, state, um_per_px):
    if pixels.ndim == 2:
        return pixels
    mode = options["mode"]
    if mode == "Luminance":
        grey = hp.luminance(pixels)
    elif mode == "Mean of the channels":
        grey = pixels[..., :3].astype(np.float32).mean(axis=-1)
    else:
        grey = pixels[..., ["Red", "Green", "Blue"].index(mode)]
    return hp.to_like(grey, pixels)
