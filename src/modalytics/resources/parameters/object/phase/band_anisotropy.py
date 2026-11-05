"""Anisotropy g of the surrounding stroma (per object, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'band_anisotropy'
LABEL = 'Anisotropy g of the surrounding stroma'
UNIT = ''
CATEGORY = 'scattering'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 240
DESCRIPTION = "Scattering anisotropy g = 1 − ⟨|∇φ|²⟩ / (4k²⟨1 − cos Δφ⟩), k = 2π·n₀/λ (Xu 2011, Eq. 9), computed in the band around the object. The band runs from 'gap' to 'gap + width' µm outside the object's outer outline, kept only where class C (default: stroma) lies; e.g. the stroma immediately around each gland. Lower g in the stroma adjoining cancerous prostate glands identified patients at higher risk of recurrence (Sridharan et al. 2015; the band here is a constant, not the paper's exact region). Use a background-corrected phase map. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
REFERENCES = [
    ('Sridharan S, Macias V, Tangella K, Kajdacsy-Balla A, Popescu G (2015). Prediction of prostate '
     'cancer recurrence using quantitative phase imaging. Sci Rep 5:9976. doi:10.1038/srep09976'),
    ('Xu M (2011). Scattering-phase theorem: anomalous diffraction by forward-peaked scattering media. '
     'Opt Express 19(22):21643–21651. doi:10.1364/OE.19.021643'),
    ('Wang Z, Ding H, Popescu G (2011). Scattering-phase theorem. Opt Lett 36(7):1215–1217. '
     'doi:10.1364/OL.36.001215'),
]
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Lamina propria', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
    {'key': 'width_um',
     'label': 'Band width',
     'kind': 'float',
     'default': 20.0,
     'minimum': 0.5,
     'maximum': 1000.0,
     'step': 1.0,
     'suffix': ' µm',
     'tooltip': 'Width of the band around the object'},
    {'key': 'gap_um',
     'label': 'Gap',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 1000.0,
     'step': 1.0,
     'suffix': ' µm',
     'tooltip': 'Distance between the object and the band'},
    {'key': 'thickness_um',
     'label': 'Section thickness',
     'kind': 'float',
     'default': 4.0,
     'minimum': 0.1,
     'maximum': 1000.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Thickness L of the tissue section the light went through'},
    {'key': 'wavelength_nm',
     'label': 'Wavelength',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 2000.0,
     'step': 1.0,
     'suffix': ' nm',
     'tooltip': 'Wavelength λ of the light; 0 = the one the importer read from the file'},
    {'key': 'medium_index',
     'label': 'Medium refractive index',
     'kind': 'float',
     'default': 1.0,
     'minimum': 1.0,
     'maximum': 2.0,
     'step': 0.001,
     'tooltip': 'Refractive index n₀ of what surrounds the section (1.0 air, ≈1.33 water, ≈1.5 '
                'mounting media)'},
]


def compute(ctx):
    c = ctx.constants
    band = ctx.surroundings(c['width_um'], c['gap_um'], only=c['other'])
    if band.area_um2 <= 0:
        return float('nan')
    return band.scattering(c['thickness_um'], c['wavelength_nm'], c['medium_index'])['g']
