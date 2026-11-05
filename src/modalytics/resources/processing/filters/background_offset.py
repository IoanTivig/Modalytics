"""Background offset: shifts the values so the empty medium is 0 (phase and derived maps).

Phase is only known up to a constant: dry mass, thickness and OPD are measured
relative to the medium around the sample, so the medium should read 0. The
level subtracted is the most common value (the histogram's peak: the medium,
which covers most of a typical field), the median, or a fixed value.
"""

import numpy as np

from modalytics.processing import helpers as hp

KEY = 'background_offset'
NAME = 'Background offset (medium = 0)'
SHORT = 'Offset'
ICON = 'mdi6.arrow-collapse-down'
INPUT_TYPES = ['phase', 'opd', 'dry_mass', 'height']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Correction'
DESCRIPTION = 'Subtracts the background level so the empty medium reads 0: the most common value (histogram peak), the median, or a fixed value.'
OPTIONS = [
    {'key': 'method',
     'label': 'Background level',
     'kind': 'choice',
     'default': 'Most common value',
     'choices': ['Most common value', 'Median', 'Fixed value'],
     'tooltip': 'Most common value: the histogram peak (the medium, when it covers most of the field); '
                'Median: half the pixels above, half below; Fixed value: the value below'},
    {'key': 'value',
     'label': 'Fixed value',
     'kind': 'float',
     'default': 0.0,
     'minimum': -1000000.0,
     'maximum': 1000000.0,
     'step': 0.01,
     'tooltip': 'Subtracted from every pixel (used with "Fixed value"), in the image\'s unit'},
]


def prepare(image, options):
    """The level subtracted (from an overview)."""
    if options["method"] == "Fixed value":
        return {"level": float(options["value"])}
    values = hp.luminance(hp.overview(image))
    if options["method"] == "Median":
        return {"level": float(np.nanmedian(values))}
    return {"level": hp.background_level(values)}


def process(pixels, options, state, um_per_px):
    return pixels.astype(np.float32) - np.float32(state["level"])
