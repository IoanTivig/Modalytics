"""Objects per mm² (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'density'
LABEL = 'Objects per mm²'
UNIT = '1/mm²'
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 20
DESCRIPTION = 'Density = N / A_image (objects per mm² of image). Measured for each class of the profile on each image (one row per class and image).'
CONSTANTS = []


def compute(ctx):
    return ctx.count / (ctx.image_area_um2 * 1e-6)
