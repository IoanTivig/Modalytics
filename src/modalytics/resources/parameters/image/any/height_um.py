"""Image height in micrometres.

Ported from HoloPath resources/parameters/builtin/image/height_um.py.
"""

KEY = 'height_um'
LABEL = 'Height'
UNIT = 'µm'
CATEGORY = 'dimensions'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 40
DESCRIPTION = 'Image height H·s in µm (s = pixel size).'


def compute(ctx):
    return ctx.height_px * ctx.pixel_size_um
