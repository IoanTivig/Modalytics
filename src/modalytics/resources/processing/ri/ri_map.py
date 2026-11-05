"""Refractive-index map: the mean refractive index through the sample, n = n_medium + φ · λ / (2π · h).

When the thickness is known instead of the index — a tissue section of known
thickness, cells confined in a chamber of known height — the phase gives the
mean refractive index along each line of sight (Rappaz et al. 2005), a measure
of how dense the material is. The medium should read 0 first (Background
offset). Where there is no sample the result is the medium's index.
"""

import math

KEY = 'ri_map'
NAME = 'Refractive-index map'
SHORT = 'RI'
ICON = 'mdi6.circle-half-full'
INPUT_TYPES = ['phase']
OUTPUT_TYPE = 'ri'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = 'Mean refractive index n = n_medium + phase × wavelength / (2π × thickness), for a sample of known thickness (a section, a chamber).'
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
    {'key': 'thickness_um',
     'label': 'Sample thickness',
     'kind': 'float',
     'default': 4.0,
     'minimum': 0.1,
     'maximum': 1000.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Thickness of the section or chamber'},
    {'key': 'n_medium',
     'label': 'Medium refractive index',
     'kind': 'float',
     'default': 1.335,
     'minimum': 1.0,
     'maximum': 2.0,
     'step': 0.001,
     'tooltip': 'Index of what surrounds the sample (water / culture medium ≈ 1.333–1.337, mounting '
                'media ≈ 1.4–1.52)'},
]


def prepare(image, options):
    wavelength = options["wavelength_nm"] or image.metadata.get("wavelength_nm")
    if not wavelength:
        raise ValueError("the wavelength is not known: give it in the step's options")
    return {"per_rad": float(wavelength) / 1000.0 / (2 * math.pi * float(options["thickness_um"])),
            "n0": float(options["n_medium"])}


def process(pixels, options, state, um_per_px):
    return state["n0"] + pixels * state["per_rad"]
