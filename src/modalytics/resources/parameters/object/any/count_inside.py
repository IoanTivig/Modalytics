"""Objects of a class inside (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'count_inside'
LABEL = 'Objects of a class inside'
UNIT = ''
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 402
DESCRIPTION = 'Number of objects of a class C lying inside the object (an object counts when its representative point, a point always inside it, is within the outline). E.g. glands in a region, nuclei in a gland. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration.'
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    return len(ctx.inside(ctx.constants['other']))
