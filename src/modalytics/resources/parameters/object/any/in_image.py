"""1 if the object lies fully inside the image (does not touch the border), else 0.

Boolean (0/1) flag. An object touching the rectangular image border is likely
truncated by the field of view, so its geometry is unreliable.

Ported from HoloPath resources/parameters/builtin/object/in_image.py.
"""

KEY = 'in_image'
LABEL = 'Inside Image'
UNIT = ''
CATEGORY = 'flags'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 300
DESCRIPTION = 'Edge flag: 1 if the object lies fully inside the image; 0 if it touches the border (its measurements are then incomplete).'


def compute(ctx):
    return 0.0 if ctx.touches_image_edge else 1.0
