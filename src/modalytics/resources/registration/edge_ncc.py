"""Edge correlation: places an image by matching the edges (outlines of structures) of both images.

Canny edges of both images, blurred a little so nearby edges still overlap, compared by
the shared search (masked normalised cross-correlation over position, rotation, scale
and mirroring, then refined). Edges ignore how bright things are, so it can work across
image kinds when both show the same outlines. (HoloPath's "template matching".)
"""

KEY = 'edge_ncc'
NAME = 'Edge correlation'
DESCRIPTION = ('Matches the Canny edges of both images (blurred, then normalised cross-correlation over position, '
               'rotation, scale and mirroring, refined). Independent of brightness; needs the same outlines in both.')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = [
    {'text': 'Canny J (1986). A Computational Approach to Edge Detection. IEEE TPAMI PAMI-8(6): 679–698.',
     'doi': '10.1109/TPAMI.1986.4767851'},
    {'text': 'Brunelli R (2009). Template Matching Techniques in Computer Vision: Theory and Practice. Wiley.',
     'doi': '10.1002/9780470744055'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 2.0, 'minimum': 0.5,
     'maximum': 10.0, 'step': 0.5, 'suffix': ' µm/px', 'tooltip': 'The images are compared at this resolution'},
    {'key': 'edge_sigma_um', 'label': 'Edge scale', 'kind': 'float', 'default': 3.0, 'minimum': 0.5,
     'maximum': 20.0, 'step': 0.5, 'suffix': ' µm', 'tooltip': 'Smoothing before edge detection: larger finds only '
                                                                  'coarse outlines'},
    {'key': 'tolerance_um', 'label': 'Tolerance', 'kind': 'float', 'default': 4.0, 'minimum': 0.0,
     'maximum': 20.0, 'step': 0.5, 'suffix': ' µm', 'tooltip': 'Edges this far apart still count as overlapping'},
]


def _edges(grey, sigma_px, tol_px):
    import numpy as np
    from scipy import ndimage
    from skimage import feature
    e = feature.canny(grey, sigma=max(sigma_px, 0.5)).astype(np.float32)
    if tol_px > 0:
        e = ndimage.gaussian_filter(e, tol_px)
    return ((e - e.mean()) / (e.std() + 1e-6)).astype(np.float32)


def register(ctx, options):
    px = options['resolution_um']
    fixed, frame = ctx.read(ctx.reference, px)
    moving, _ = ctx.read(ctx.moving, px)
    s, t = options['edge_sigma_um'] / px, options['tolerance_um'] / px
    return ctx.match(_edges(ctx.grey(fixed), s, t), frame, _edges(ctx.grey(moving), s, t), coarse_factor=2.0)
