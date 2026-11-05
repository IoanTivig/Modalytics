"""Density of a class inside (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'density_inside'
LABEL = 'Density of a class inside'
UNIT = '1/mm²'
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 403
DESCRIPTION = "Density = N_C / A (objects per mm²): N_C objects of class C inside the object (counted as in 'Objects of a class inside'), A the object's area. E.g. glands per mm² of a region (crowding). The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    a = ctx.area_um2
    return len(ctx.inside(ctx.constants['other'])) / (a * 1e-6) if a > 0 else float('nan')
