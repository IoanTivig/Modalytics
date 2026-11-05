"""Area fraction (per class in an image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'area_fraction'
LABEL = 'Area fraction'
UNIT = '%'
CATEGORY = 'summary'
SCOPE = 'class'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 30
DESCRIPTION = 'Area fraction = 100 · A_class / A_image (%): the share of the image the class covers. Measured for each class of the profile on each image (one row per class and image).'
CONSTANTS = []


def compute(ctx):
    return min(100.0, 100.0 * ctx.area_um2 / ctx.image_area_um2)
