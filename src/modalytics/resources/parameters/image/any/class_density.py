"""Density of a class (whole image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'class_density'
LABEL = 'Density of a class'
UNIT = '1/mm²'
CATEGORY = 'relations'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 402
DESCRIPTION = "Density = N_C / A_image (objects per mm²): the number of objects of class C on the image over the image's area. The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
CONSTANTS = [
    {'key': 'other',
     'label': 'Class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'An annotation class of the profile (a group alone = all its subgroups)'},
]


def compute(ctx):
    return len(ctx.others(ctx.constants['other'])) / (ctx.image_area_um2 * 1e-6)
