"""Image width in pixels.

Ported from HoloPath resources/parameters/builtin/image/width_px.py.
"""

KEY = 'width_px'
LABEL = 'Width'
UNIT = 'px'
CATEGORY = 'dimensions'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 10
DESCRIPTION = 'Image width W in pixels.'


def compute(ctx):
    return ctx.width_px
