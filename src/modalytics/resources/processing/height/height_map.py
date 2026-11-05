"""Height map: the physical thickness of the sample from its phase, h = φ · λ / (2π · Δn), in µm.

Phase is the product of thickness and refractive-index difference: with the
difference known (cells in medium: about 0.03–0.06; Rappaz et al. 2005), the
thickness follows (Popescu 2011). The medium should read 0 first (Background
offset). The wavelength comes from the file when the importer recorded it,
else from the option.
"""

import math

KEY = 'height_map'
NAME = 'Height map (thickness)'
SHORT = 'Height'
ICON = 'mdi6.terrain'
INPUT_TYPES = ['phase']
OUTPUT_TYPE = 'height'
UNIT = 'µm'
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = 'Thickness h = phase × wavelength / (2π × Δn), in µm, for a known refractive-index difference Δn between sample and medium.'
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
    {'key': 'delta_n',
     'label': 'Refractive-index difference (Δn)',
     'kind': 'float',
     'default': 0.04,
     'minimum': 0.001,
     'maximum': 1.0,
     'step': 0.005,
     'tooltip': 'Sample minus medium (cells in culture medium: about 0.03–0.06)'},
]


def prepare(image, options):
    """µm of thickness per radian."""
    wavelength = options["wavelength_nm"] or image.metadata.get("wavelength_nm")
    if not wavelength:
        raise ValueError("the wavelength is not known: give it in the step's options")
    return {"um_per_rad": float(wavelength) / 1000.0 / (2 * math.pi * float(options["delta_n"]))}


def process(pixels, options, state, um_per_px):
    return pixels * state["um_per_rad"]
