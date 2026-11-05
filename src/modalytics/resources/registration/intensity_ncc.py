"""Intensity cross-correlation: places an image by matching its grey values with the reference's.

The classic template matching: both images in one grey channel, smoothed a little and
compared by masked normalised cross-correlation over position, rotation, scale and
mirroring (the shared search), then refined. Grey values must look alike, so it suits
two images of the same kind; for a phase map on a stained slide the contrast is
different — "Contrast: inverted / both" tries the reference inverted as well.
"""

KEY = 'intensity_ncc'
NAME = 'Intensity cross-correlation'
DESCRIPTION = ('Template matching on the grey values (normalised cross-correlation over position, rotation, scale and '
               'mirroring, then refined). Suits two images of the same kind; across kinds, try Contrast: both.')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = [
    {'text': 'Brunelli R (2009). Template Matching Techniques in Computer Vision: Theory and Practice. Wiley.',
     'doi': '10.1002/9780470744055'},
    {'text': 'Zitová B, Flusser J (2003). Image registration methods: a survey. Image and Vision Computing 21(11): '
             '977–1000.', 'doi': '10.1016/S0262-8856(03)00137-9'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 2.0, 'minimum': 0.5,
     'maximum': 10.0, 'step': 0.5, 'suffix': ' µm/px', 'tooltip': 'The images are compared at this resolution'},
    {'key': 'smoothing_um', 'label': 'Smoothing', 'kind': 'float', 'default': 2.0, 'minimum': 0.0,
     'maximum': 20.0, 'step': 0.5, 'suffix': ' µm', 'tooltip': 'Gaussian smoothing before comparing (σ)'},
    {'key': 'contrast', 'label': 'Contrast', 'kind': 'choice', 'default': 'same', 'choices': ['same', 'inverted', 'both'],
     'tooltip': 'Whether bright in one image is bright (same) or dark (inverted) in the other; both tries the two'},
]


def _prepare(pixels, sigma_px):
    import numpy as np
    from scipy import ndimage
    g = ndimage.gaussian_filter(pixels, sigma_px) if sigma_px > 0 else pixels
    return ((g - g.mean()) / (g.std() + 1e-6)).astype(np.float32)


def register(ctx, options):
    px = options['resolution_um']
    sigma = options['smoothing_um'] / px
    fixed, frame = ctx.read(ctx.reference, px)
    moving, _ = ctx.read(ctx.moving, px)
    fixed, moving = _prepare(ctx.grey(fixed), sigma), _prepare(ctx.grey(moving), sigma)
    signs = {'same': [1], 'inverted': [-1], 'both': [1, -1]}[options['contrast']]
    found = []
    for sign in signs:
        ctx.check()
        found += ctx.match(sign * fixed, frame, moving, coarse_factor=2.0)
    return sorted(found, key=lambda c: -c.score)
