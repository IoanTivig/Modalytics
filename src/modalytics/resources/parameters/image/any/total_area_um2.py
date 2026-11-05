"""Total image area in square micrometres.

Ported from HoloPath resources/parameters/builtin/image/total_area_um2.py.
"""

KEY = 'total_area_um2'
LABEL = 'Total Area'
UNIT = 'µm²'
CATEGORY = 'dimensions'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = True
ORDER = 50
DESCRIPTION = 'Imaged area W·H·s² in µm² (s = pixel size).'


def compute(ctx):
    return ctx.width_px * ctx.pixel_size_um * ctx.height_px * ctx.pixel_size_um
