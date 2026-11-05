"""OD Fluctuation Mean of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

KEY = 'fluctuation_mean'
LABEL = 'OD Fluctuation Mean'
UNIT = ''
CATEGORY = 'fluctuation'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 230
DESCRIPTION = "Mean local fluctuation of the optical density inside the object: ⟨var_w(OD)⟩ / OD̄², the variance in a w × w window (constant, default 3 px) divided by the object's mean OD squared. Independent of how dark the object is overall; higher = a more textured interior."
CONSTANTS = [
    {'key': 'window',
     'label': 'Fluctuation window',
     'kind': 'int',
     'default': 3,
     'minimum': 3,
     'maximum': 15,
     'step': 2,
     'suffix': ' px',
     'tooltip': 'Side of the square neighbourhood of the local variance'},
]


def compute(ctx):
    f = ctx.relative_fluctuation_image(ctx.constants["window"])
    return float(f[ctx.mask].mean()) if ctx.mask.any() else 0.0
