"""Area fraction of a class (whole image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'class_area_fraction'
LABEL = 'Area fraction of a class'
UNIT = '%'
CATEGORY = 'relations'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 401
DESCRIPTION = "Area fraction = 100 · A_C / A_image (%): the area class C covers on the image over the image's area. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    return min(100.0, 100.0 * ctx.area_of(ctx.constants['other']) / ctx.image_area_um2)
