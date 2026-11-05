"""Dry Mass of the object on dry-mass maps.

A dry-mass map's pixels are dry-mass surface density, σ = OPD / α (pg/µm²), made
from a phase map by the Dry-mass map processing step
(modalytics/resources/processing/dry_mass/dry_mass.py).
"""

import numpy as np

KEY = 'dry_mass'
LABEL = 'Dry Mass'
UNIT = 'pg'
CATEGORY = 'dry mass'
SCOPE = 'object'
IMAGE_TYPE = 'dry_mass'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 60
DESCRIPTION = "Dry mass m = Σ σᵢ·A_px over the object (pg), σ = λφ / (2πα) the dry-mass surface density of the dry-mass map (α = specific refraction increment, about 0.18–0.21 µm³/pg for proteins; the step's default 0.19). The mass of everything but water: proteins, nucleic acids, lipids. Needs a background-corrected phase (medium = 0)."
REFERENCES = [
    ('Barer R (1952). Interference microscopy and mass determination. Nature 169(4296):366–367. '
     'doi:10.1038/169366b0'),
    ('Popescu G, Park Y, Lue N, Best-Popescu C, Deflores L, Dasari RR, Feld MS, Badizadegan K (2008). '
     'Optical imaging of cell mass and growth dynamics. Am J Physiol Cell Physiol 295(2):C538–C544. '
     'doi:10.1152/ajpcell.00121.2008'),
    ('Zangle TA, Teitell MA (2014). Live-cell mass profiling: an emerging approach in quantitative '
     'biophysics. Nat Methods 11(12):1221–1228. doi:10.1038/nmeth.3175'),
]
CONSTANTS = []


def compute(ctx):
    px, py = ctx.image.pixel_size_um
    return float(np.sum(ctx.pixels) * px * py)
