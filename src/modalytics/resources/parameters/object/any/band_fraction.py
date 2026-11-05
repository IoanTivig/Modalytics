"""Surroundings covered by a class (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'band_fraction'
LABEL = 'Surroundings covered by a class'
UNIT = '%'
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 405
DESCRIPTION = "Band coverage = 100 · A(band ∩ C) / A(band) (%): the band around the object, from 'gap' to 'gap + width' µm outside its outer outline (cut to the image), and the share of it covered by class C. E.g. how much of the tissue around each gland is stroma. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
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
]


def compute(ctx):
    c = ctx.constants
    band = ctx.surroundings(c['width_um'], c['gap_um'])
    if band.area_um2 <= 0:
        return float('nan')
    return min(100.0, 100.0 * band.area_of(c['other']) / band.area_um2)
