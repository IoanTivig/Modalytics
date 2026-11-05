"""Background flattening: removes a tilt or curvature of the background (phase and derived maps).

What is left of the reference wave in a phase map — a tilt, a bowl, a saddle —
adds to every value measured (dry mass, thickness). A 2D polynomial is fitted to
the background and subtracted (numerical aberration compensation, Colomb et al.
2006). With "Leave out the sample", the fit is repeated leaving out the pixels far
from the surface (the cells or tissue), so it follows the empty medium only.
The surface is fitted once per image on an overview and is the same at every zoom level.
"""

from modalytics.processing import helpers as hp

KEY = 'background_flatten'
NAME = 'Background flattening (polynomial)'
SHORT = 'Background'
ICON = 'mdi6.gradient-horizontal'
INPUT_TYPES = ['phase', 'opd', 'dry_mass', 'height']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Correction'
DESCRIPTION = 'Fits a 2D polynomial to the background (leaving out the sample) and subtracts it: removes the tilt or curvature left by the reference wave (Colomb et al. 2006).'
OPTIONS = [
    {'key': 'order',
     'label': 'Polynomial order',
     'kind': 'int',
     'default': 2,
     'minimum': 1,
     'maximum': 5,
     'step': 1,
     'tooltip': '1 = a tilted plane, 2 = also a bowl or saddle, higher = more complex shapes'},
    {'key': 'leave_out_sample',
     'label': 'Leave out the sample',
     'kind': 'bool',
     'default': True,
     'tooltip': 'Fit again and again leaving out pixels far from the surface (cells, tissue), so only '
                'the empty background is fitted'},
    {'key': 'keep_level',
     'label': 'Keep the background level',
     'kind': 'bool',
     'default': False,
     'tooltip': 'Off: the background becomes 0. On: only its shape is removed (its mean stays)'},
]


def prepare(image, options):
    """The background surface, fitted on an overview."""
    values = hp.luminance(hp.overview(image))
    order = int(options["order"])
    coef = hp.fit_surface(values, order, clip=2.5 if options["leave_out_sample"] else None)
    if options["keep_level"]:
        coef = coef.copy()
        coef[0] = 0.0                                         # the constant term stays in the image
    return {"coef": coef, "order": order}


def process(pixels, options, state, um_per_px, where):
    ys, xs = where.fraction(pixels.shape[:2])
    return pixels - hp.eval_surface(state["coef"], state["order"], ys, xs)
