"""Area fraction of a class (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'class_area_fraction'
LABEL = 'Area fraction of a class'
UNIT = '%'
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 401
DESCRIPTION = "Area fraction = 100 · A_C / A of a class C inside the object (%): A_C = the area of class C within the object's outline, A = the object's area. E.g. how much of a region is glands. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    a = ctx.area_um2
    return min(100.0, 100.0 * ctx.area_of(ctx.constants['other']) / a) if a > 0 else float('nan')
