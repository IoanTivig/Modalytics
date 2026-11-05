"""Feature points: places an image by matching distinctive points (SIFT or ORB) found in both.

Keypoints are detected in both images and described by their surroundings; descriptors
that match (Lowe's ratio test) give point pairs, and RANSAC keeps the pairs that agree on
one similarity transform (shift, rotation, scale; mirroring by trying the mirrored image),
discarding the rest as outliers. Works at any rotation without searching it, when both
images show the same distinctive details.
"""

import math

KEY = 'feature_points'
NAME = 'Feature points (SIFT / ORB + RANSAC)'
DESCRIPTION = ('Detects keypoints (SIFT or ORB) in both images, matches their descriptors and keeps the pairs that '
               'agree on one transform (RANSAC). Any rotation, no search needed; for images showing the same '
               'distinctive details (usually the same kind of image).')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = [
    {'text': 'Lowe DG (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of '
             'Computer Vision 60(2): 91–110.', 'doi': '10.1023/B:VISI.0000029664.99615.94'},
    {'text': 'Rublee E, Rabaud V, Konolige K, Bradski G (2011). ORB: An efficient alternative to SIFT or SURF. '
             '2011 International Conference on Computer Vision: 2564–2571.', 'doi': '10.1109/ICCV.2011.6126544'},
    {'text': 'Fischler MA, Bolles RC (1981). Random sample consensus: a paradigm for model fitting with applications '
             'to image analysis and automated cartography. Communications of the ACM 24(6): 381–395.',
     'doi': '10.1145/358669.358692'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 1.0, 'minimum': 0.25,
     'maximum': 8.0, 'step': 0.25, 'suffix': ' µm/px', 'tooltip': 'Keypoints are found at this resolution'},
    {'key': 'detector', 'label': 'Detector', 'kind': 'choice', 'default': 'SIFT', 'choices': ['SIFT', 'ORB'],
     'tooltip': 'SIFT: more robust; ORB: faster'},
    {'key': 'ratio', 'label': 'Ratio test', 'kind': 'float', 'default': 0.75, 'minimum': 0.5, 'maximum': 0.95,
     'step': 0.05, 'tooltip': "Lowe's ratio: a match counts when clearly better than the second best"},
    {'key': 'inlier_um', 'label': 'RANSAC tolerance', 'kind': 'float', 'default': 4.0, 'minimum': 0.5,
     'maximum': 20.0, 'step': 0.5, 'suffix': ' µm', 'tooltip': 'How far a point may be from the transform and still agree'},
]


def _eight_bit(grey):
    import numpy as np
    lo, hi = np.percentile(grey, [1, 99])
    return np.clip((grey - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def register(ctx, options):
    import cv2
    import numpy as np
    px = options['resolution_um']
    fixed, frame = ctx.read(ctx.reference, px)
    moving, _ = ctx.read(ctx.moving, px)
    f8, m8 = _eight_bit(ctx.grey(fixed)), _eight_bit(ctx.grey(moving))
    if options['detector'] == 'SIFT':
        det, norm = cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    else:
        det, norm = cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING
    kf, df = det.detectAndCompute(f8, None)
    if df is None or len(kf) < 3:
        return []
    matcher = cv2.BFMatcher(norm)
    to_world = ctx.to_world(frame)                           # reference map px → world µm
    grid = ctx.moving_grid(moving.shape)                     # moving map px → moving local µm
    h, w = m8.shape
    s = ctx.search
    out = []
    _a, _s, mirrors = ctx.pose_grid()
    for k, mirror in enumerate(mirrors):
        ctx.check()
        img = m8[:, ::-1].copy() if mirror else m8
        km, dm = det.detectAndCompute(img, None)
        if dm is None or len(km) < 3:
            continue
        pairs = [m for m, n in (p for p in matcher.knnMatch(dm, df, k=2) if len(p) == 2)
                 if m.distance < options['ratio'] * n.distance]
        if len(pairs) < 3:
            continue
        src = np.float32([km[m.queryIdx].pt for m in pairs])
        dst = np.float32([kf[m.trainIdx].pt for m in pairs])
        a, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=options['inlier_um'] / px)
        if a is None:
            continue
        a3 = np.vstack([a, [0, 0, 1]])
        flip = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float) if mirror else np.eye(3)
        # moving local µm → moving px → (mirrored px) → reference map px → world µm (a mirror gives det < 0)
        m = to_world @ a3 @ flip @ np.linalg.inv(grid)
        n_in = int(inliers.sum()) if inliers is not None else 0
        cand = ctx.candidate(matrix=m, score=n_in / max(len(pairs), 1))
        centre = np.asarray(cand.placement.local_to_world() @ [ctx.moving.size_um[0] / 2, ctx.moving.size_um[1] / 2, 1])
        if n_in >= 4 and math.hypot(centre[0] - s.centre_world[0], centre[1] - s.centre_world[1]) <= s.radius_um:
            out.append(cand)
        ctx.progress((k + 1) / len(mirrors))
    return sorted(out, key=lambda c: -c.score)
