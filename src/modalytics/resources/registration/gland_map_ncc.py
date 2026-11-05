"""Gland-map correlation: places an image by matching the glands both images show.

Each image is turned into a gland-probability map by the segmentation model for its own
type, so the two maps look alike even when the images do not (a phase map and a stained
slide). The moving map is searched in the reference's around the rough position over the
rotations, scale corrections and mirroring the tool asks for (masked normalised
cross-correlation, coarse), the best candidates are refined (fine) and polished (ECC).
"""

KEY = 'gland_map_ncc'
NAME = 'Gland-map correlation'
DESCRIPTION = ('Matches the gland maps the segmentation models make of both images (normalised cross-correlation '
               'over position, rotation, scale and mirroring, then refined). Works across image types (phase map ↔ '
               'slide) wherever both types have a gland model; needs glands in the field of view.')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = False
REFERENCES = [
    {'text': 'Ronneberger O, Fischer P, Brox T (2015). U-Net: Convolutional Networks for Biomedical Image '
             'Segmentation. MICCAI 2015, LNCS 9351: 234–241.', 'doi': '10.1007/978-3-319-24574-4_28'},
    {'text': 'Brunelli R (2009). Template Matching Techniques in Computer Vision: Theory and Practice. Wiley.',
     'doi': '10.1002/9780470744055'},
    {'text': 'Evangelidis GD, Psarakis EZ (2008). Parametric Image Alignment Using Enhanced Correlation Coefficient '
             'Maximization. IEEE TPAMI 30(10): 1858–1865.', 'doi': '10.1109/TPAMI.2008.113'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 2.0, 'minimum': 0.5,
     'maximum': 10.0, 'step': 0.5, 'suffix': ' µm/px',
     'tooltip': 'The maps are compared at this resolution (the first search at twice it)'},
    {'key': 'group', 'label': 'Structures', 'kind': 'text', 'default': 'Gland',
     'tooltip': 'The class group whose segmentation model makes the maps'},
]


def register(ctx, options):
    px = options['resolution_um']
    fixed, frame = ctx.feature_map(ctx.reference, px, group=options['group'])
    ctx.progress(0.5)
    moving, _ = ctx.feature_map(ctx.moving, px, group=options['group'])
    ctx.check()
    return ctx.match(fixed, frame, moving, coarse_factor=2.0)
