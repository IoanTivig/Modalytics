"""Distance to the nearest (class) (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'nearest_distance'
LABEL = 'Distance to the nearest (class)'
UNIT = 'µm'
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 404
DESCRIPTION = "Distance d = min over the objects j of class C of the shortest distance between this object's outline and object j's (µm; 0 if they touch or overlap; this object itself is left out). E.g. gland-to-gland spacing (class Gland), or how far a gland is from a class outlining the tumour border. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    return ctx.nearest(ctx.constants['other'])
