"""Image height in pixels.

Ported from HoloPath resources/parameters/builtin/image/height_px.py.
"""

KEY = 'height_px'
LABEL = 'Height'
UNIT = 'px'
CATEGORY = 'dimensions'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 20
DESCRIPTION = 'Image height H in pixels.'


def compute(ctx):
    return ctx.height_px
