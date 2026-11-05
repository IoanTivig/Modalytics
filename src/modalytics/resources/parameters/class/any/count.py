"""Number of objects (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'count'
LABEL = 'Number of objects'
UNIT = ''
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 10
DESCRIPTION = 'Number of objects N of the class on the image (each separate outline is one object). Measured for each class of the profile on each image (one row per class and image).'
CONSTANTS = []


def compute(ctx):
    return ctx.count
