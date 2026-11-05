"""Nucleus-to-cytoplasm ratio (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'inside_area_ratio'
LABEL = 'Nucleus-to-cytoplasm ratio'
UNIT = ''
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 406
DESCRIPTION = "Inside-to-rest area ratio = A_C / (A − A_C): A_C the area of class C inside the object, A the object's area. With cells as objects and nuclei outlined as class C, this is the nucleus-to-cytoplasm (N/C) area ratio; with glands and their nuclei, the share of nuclei in the epithelium. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Inner class',
     'kind': 'label',
     'default': ['Nucleus', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    inner = ctx.area_of(ctx.constants['other'])
    rest = ctx.area_um2 - inner
    return inner / rest if rest > 0 else float('nan')
