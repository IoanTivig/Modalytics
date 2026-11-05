"""ECC refinement: polishes the current placement by maximising the correlation coefficient.

No search: starting where the image is now, the enhanced correlation coefficient (ECC) is
maximised over a small shift and rotation, first coarse then fine. Use it after placing
the image by hand to within a few cells, or after another method. On the grey values (same
kind of images) or on the gland maps (any kinds with a gland model).
"""

KEY = 'ecc_refine'
NAME = 'ECC refinement (from the current position)'
DESCRIPTION = ('Refines where the image is now by maximising the enhanced correlation coefficient (shift and '
               'rotation, coarse then fine). No search: place the image within a few cells first.')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation']
EXPERIMENTAL = True
REFINES_ONLY = True
REFERENCES = [
    {'text': 'Evangelidis GD, Psarakis EZ (2008). Parametric Image Alignment Using Enhanced Correlation Coefficient '
             'Maximization. IEEE Transactions on Pattern Analysis and Machine Intelligence 30(10): 1858–1865.',
     'doi': '10.1109/TPAMI.2008.113'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'on', 'label': 'Compare', 'kind': 'choice', 'default': 'grey values', 'choices': ['grey values', 'gland maps'],
     'tooltip': 'Grey values: images of the same kind; gland maps: any kinds with a gland model'},
    {'key': 'resolution_um', 'label': 'Fine resolution', 'kind': 'float', 'default': 1.0, 'minimum': 0.25,
     'maximum': 8.0, 'step': 0.25, 'suffix': ' µm/px', 'tooltip': 'The last pass runs at this resolution, the first '
                                                                  'at four times it'},
]


def _maps(ctx, px, on):
    import numpy as np
    from scipy import ndimage
    if on == 'gland maps':
        fixed, frame = ctx.feature_map(ctx.reference, px)
        moving, _ = ctx.feature_map(ctx.moving, px)
        return fixed, frame, moving
    fixed, frame = ctx.read(ctx.reference, px)
    moving, _ = ctx.read(ctx.moving, px)
    z = lambda g: ((g - g.mean()) / (g.std() + 1e-6)).astype(np.float32)
    return (z(ndimage.gaussian_filter(ctx.grey(fixed), 1.0)), frame, z(ndimage.gaussian_filter(ctx.grey(moving), 1.0)))


def register(ctx, options):
    import numpy as np
    fine = options['resolution_um']
    placement = ctx.moving.placement
    cand = None
    for k, px in enumerate((4 * fine, 2 * fine, fine)):
        ctx.check()
        fixed, frame, moving = _maps(ctx, px, options['on'])
        cand = ctx.refine_ecc(placement, fixed, frame, moving)
        placement = cand.placement
        ctx.progress((k + 1) / 3)
    # Score: the correlation coefficient at the final placement, on the finest maps
    import cv2
    a = (np.linalg.inv(ctx.to_world(frame)) @ placement.local_to_world() @ ctx.moving_grid(moving.shape))[:2]
    h, w = moving.shape
    warped = cv2.warpAffine(fixed, a.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderValue=float('nan'))
    ok = np.isfinite(warped)
    x, y = moving[ok] - moving[ok].mean(), warped[ok] - warped[ok].mean()
    score = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9)) if ok.any() else 0.0
    return [ctx.candidate(placement=placement, score=score)]
