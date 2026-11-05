"""Correlation Length of the object on OD maps.

An OD map's pixels are optical density, OD = −log₁₀(I / I_background): how much
light the sample absorbs (0 = none), independent of brightness and white
balance. It is made from a slide or a brightfield photo by the OD map processing
step (modalytics/resources/processing/od/od_map.py).
"""

KEY = 'correlation_length'
LABEL = 'Correlation Length'
UNIT = 'µm'
CATEGORY = 'fluctuation'
SCOPE = 'object'
IMAGE_TYPE = 'od'
DEFAULT_ON = True
WHOLE_OBJECT = False
ORDER = 220
DESCRIPTION = "Correlation length l_c (µm): the typical size of the structures inside the object (absorbing structures: nuclei, granules). A 2D polynomial of degree d (constant, default 2) is removed first (slow shading); the autocorrelation C(r) of what remains is averaged over directions and fitted with C(r) = exp(−r / l_c) (up to where C drops below 0.05; the 1/e crossing if the fit fails). Higher = larger, coarser structures. Computed on the optical density OD = −log₁₀(I / I₀), I the pixel's intensity and I₀ that of the empty glass (OD map step; Beer–Lambert: proportional to the amount of absorbing stain)."
CONSTANTS = [
    {'key': 'degree',
     'label': 'Detrend degree',
     'kind': 'int',
     'default': 2,
     'minimum': 0,
     'maximum': 4,
     'tooltip': 'Degree of the 2D polynomial removed from the values inside the object first (slow '
                'shading), so only the structures remain'},
]


def compute(ctx):
    return ctx.structure_length(ctx.constants["degree"])
