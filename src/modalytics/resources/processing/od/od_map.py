"""OD map: the optical density of a slide or a brightfield photo, OD = −log₁₀(I / I_background).

How much light the sample absorbs, per colour channel and averaged: 0 where
there is no sample, growing with the amount of stain or absorbing material,
independent of the scanner's or camera's brightness and white balance
(Beer–Lambert). The background (the brightness of the empty slide) is
estimated from the image's brightest pixels, which suits scans (they always
show empty slide); for photos covered by the sample, give a fixed value
measured on an empty field with the same settings.
"""

from modalytics.parameters import optical as op

KEY = 'od_map'
NAME = 'OD map'
SHORT = 'OD'
ICON = 'mdi6.invert-colors'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'od'
UNIT = 'OD'
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = "Optical density, OD = −log10(I / I_background): how much light the sample absorbs (0 = none), independent of brightness and white balance. The background is estimated from the image's brightest pixels (scans) or given as a fixed value (photos without empty background)."
OPTIONS = [
    {'key': 'background',
     'label': 'Background',
     'kind': 'choice',
     'default': 'Estimated from the image',
     'choices': ['Estimated from the image', 'Fixed value'],
     'tooltip': "Brightness of the empty slide (no sample): estimated from the image's brightest 1 % "
                'of pixels, or the fixed value below'},
    {'key': 'background_value',
     'label': 'Background value',
     'kind': 'float',
     'default': 240.0,
     'minimum': 1.0,
     'maximum': 65535.0,
     'step': 1.0,
     'tooltip': 'Brightness of an empty field taken with the same microscope settings (used with '
                '"Fixed value")'},
]


def prepare(image, options):
    """The background brightness per channel (once per image)."""
    if options["background"] == "Fixed value":
        return {"white": op.white_level(image, options["background_value"])}
    return {"white": op.estimate_white(image)}


def process(pixels, options, state, um_per_px):
    channels = pixels.shape[2] if pixels.ndim == 3 else 1
    return op.optical_density(pixels, state["white"], channels)
