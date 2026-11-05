"""OPD map: the optical path difference of a phase map, OPD = φ · λ / 2π, in nm.

The extra optical path light took through the sample (refractive index × thickness,
relative to the medium). It needs the wavelength of the light: from the file when
the importer recorded it (``wavelength_nm`` in the image's metadata), else the
value given here.
"""

import math

KEY = 'opd'
NAME = 'OPD map'
SHORT = 'OPD'
ICON = 'mdi6.wave'
INPUT_TYPES = ['phase']
OUTPUT_TYPE = 'opd'
UNIT = 'nm'
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = 'Optical path difference, OPD = phase × wavelength / 2π (nm): refractive index × thickness of the sample relative to the medium.'
OPTIONS = [
    {'key': 'wavelength_nm',
     'label': 'Wavelength',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 2000.0,
     'step': 1.0,
     'suffix': ' nm',
     'tooltip': 'Wavelength of the light; 0 = the one recorded in the file (if the importer read it)'},
]


def prepare(image, options):
    """The wavelength (nm): the option, else the file's."""
    wavelength = options["wavelength_nm"] or image.metadata.get("wavelength_nm")
    if not wavelength:
        raise ValueError("the wavelength is not known: give it in the step's options")
    return {"nm_per_rad": float(wavelength) / (2 * math.pi)}


def process(pixels, options, state, um_per_px):
    return pixels * state["nm_per_rad"]
