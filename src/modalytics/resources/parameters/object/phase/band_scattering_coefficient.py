"""Scattering coefficient μs of the surrounding stroma (per object, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'band_scattering_coefficient'
LABEL = 'Scattering coefficient μs of the surrounding stroma'
UNIT = '1/mm'
CATEGORY = 'scattering'
SCOPE = 'object'
IMAGE_TYPE = 'phase'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 241
DESCRIPTION = "Scattering coefficient μs = 2⟨1 − cos Δφ⟩ / L (1/mm; Xu 2011, Eq. 7; ≈ ⟨Δφ²⟩ / L, Wang, Ding & Popescu 2011), Δφ = φ − ⟨φ⟩, computed in the band around the object. The band runs from 'gap' to 'gap + width' µm outside the object's outer outline, kept only where class C (default: stroma) lies; L the section thickness. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
REFERENCES = [
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
]


def compute(ctx):
    c = ctx.constants
    band = ctx.surroundings(c['width_um'], c['gap_um'], only=c['other'])
    if band.area_um2 <= 0:
        return float('nan')
    return band.scattering(c['thickness_um'])['mu_s'] * 1000.0
