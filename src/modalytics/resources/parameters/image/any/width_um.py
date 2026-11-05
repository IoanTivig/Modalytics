"""Image width in micrometres.

Ported from HoloPath resources/parameters/builtin/image/width_um.py.
"""

KEY = 'width_um'
LABEL = 'Width'
UNIT = 'µm'
CATEGORY = 'dimensions'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 30
DESCRIPTION = 'Image width W·s in µm (s = pixel size).'


def compute(ctx):
    return ctx.width_px * ctx.pixel_size_um
