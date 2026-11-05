"""Orientation (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'orientation'
LABEL = 'Orientation'
UNIT = '°'
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 106
DESCRIPTION = "Orientation (°, 0–180) of the maximum Feret diameter, measured from the image's horizontal axis towards its vertical axis (rows grow downwards, so on screen it turns clockwise). Meaningful relative to other objects in the same image (e.g. how aligned glands or fibres are), not as an absolute value."
CONSTANTS = []


def compute(ctx):
    return ctx.feret[2]
