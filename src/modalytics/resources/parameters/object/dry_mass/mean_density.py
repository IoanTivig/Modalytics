"""Dry-Mass Density Mean of the object on dry-mass maps.

A dry-mass map's pixels are dry-mass surface density, σ = OPD / α (pg/µm²), made
from a phase map by the Dry-mass map processing step
(modalytics/resources/processing/dry_mass/dry_mass.py).
"""

import numpy as np

KEY = 'mean_density'
LABEL = 'Dry-Mass Density Mean'
UNIT = 'pg/µm²'
CATEGORY = 'dry mass'
SCOPE = 'object'
IMAGE_TYPE = 'dry_mass'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 70
DESCRIPTION = 'Mean dry-mass surface density σ̄ inside the object (pg/µm²), σ = λφ / (2πα). Higher = more non-aqueous material per area.'
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
    return float(np.mean(ctx.pixels))
