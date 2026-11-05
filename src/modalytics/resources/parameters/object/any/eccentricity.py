"""Eccentricity of the best-fit ellipse (0 = circle, approaching 1 = line).

Ported from HoloPath resources/parameters/builtin/object/eccentricity.py. Computed from the vector outline (HoloPath counted mask pixels): exact, and the same on any image.
"""

KEY = 'eccentricity'
LABEL = 'Eccentricity'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 110
DESCRIPTION = 'Eccentricity e = √(1 − λ₂/λ₁) of the ellipse with the same second moments (λ₁ ≥ λ₂ its eigenvalues), 0–1: 0 = a circle; towards 1 = a line.'


def compute(ctx):
    return ctx.eccentricity_outline
