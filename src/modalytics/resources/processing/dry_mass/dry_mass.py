"""Dry-mass density map: σ = OPD / α, in pg/µm² (quantitative phase imaging).

The non-aqueous mass of cells per area, from the optical path difference and the
specific refraction increment α (how much the refractive index rises per
concentration of dry matter: 0.18–0.21 µm³/pg for most proteins and nucleic acids,
0.19 a common average). Summed over a cell, it gives its dry mass in pg. From a
phase map, OPD = φ · λ / 2π needs the wavelength (the option, else the file's).

The phase of the empty medium should be 0 (background-corrected maps); otherwise
the background adds to the mass.
"""

import math

KEY = 'dry_mass'
NAME = 'Dry-mass map'
SHORT = 'Dry mass'
ICON = 'mdi6.weight'
INPUT_TYPES = ['phase', 'opd']
OUTPUT_TYPE = 'dry_mass'
UNIT = 'pg/µm²'
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = 'Dry-mass surface density σ = OPD / α (pg/µm²), with α the specific refraction increment (0.18–0.21 µm³/pg). Summed over a cell: its dry mass. Needs background-corrected phase.'
OPTIONS = [
    {'key': 'alpha',
     'label': 'Specific refraction increment α',
     'kind': 'float',
     'default': 0.19,
     'minimum': 0.1,
     'maximum': 0.3,
     'step': 0.005,
     'suffix': ' µm³/pg',
     'tooltip': 'Refractive-index increment per dry-mass concentration (0.18–0.21 µm³/pg = mL/g)'},
    {'key': 'wavelength_nm',
     'label': 'Wavelength',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 2000.0,
     'step': 1.0,
     'suffix': ' nm',
     'tooltip': "For phase maps: the light's wavelength; 0 = the one recorded in the file"},
]


def prepare(image, options):
    """µm of OPD per unit of the input (rad of phase, or nm of an OPD map)."""
    if image.modality == "opd":
        return {"um_per_unit": 1e-3}
    wavelength = options["wavelength_nm"] or image.metadata.get("wavelength_nm")
    if not wavelength:
        raise ValueError("the wavelength is not known: give it in the step's options")
    return {"um_per_unit": float(wavelength) / (2 * math.pi) * 1e-3}


def process(pixels, options, state, um_per_px):
    return pixels * (state["um_per_unit"] / options["alpha"])
