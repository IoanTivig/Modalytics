"""Mutual information: places an image where its grey values best predict the reference's.

Mutual information measures how much one image's intensities tell about the other's,
whatever the relation between them (bright may be dark, several shades may map to one),
which is why it is the standard measure for registering different modalities. Here the
normalised form (Studholme et al. 1999) is maximised: a coarse search over positions in the
circle, rotations and mirroring, then each of the best candidates refined by local
optimisation (Powell). (HoloPath's "mutual information".)
"""

import math

KEY = 'mutual_information'
NAME = 'Mutual information'
DESCRIPTION = ('Maximises the normalised mutual information of the grey values (any relation between the two '
               'images\' intensities): coarse search over position, rotation and mirroring, then local optimisation. '
               'The standard measure across modalities; slower than correlation.')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = [
    {'text': 'Viola P, Wells WM III (1997). Alignment by Maximization of Mutual Information. International Journal '
             'of Computer Vision 24(2): 137–154.', 'doi': '10.1023/A:1007958904918'},
    {'text': 'Maes F, Collignon A, Vandermeulen D, Marchal G, Suetens P (1997). Multimodality image registration by '
             'maximization of mutual information. IEEE Transactions on Medical Imaging 16(2): 187–198.',
     'doi': '10.1109/42.563664'},
    {'text': 'Studholme C, Hill DLG, Hawkes DJ (1999). An overlap invariant entropy measure of 3D medical image '
             'alignment. Pattern Recognition 32(1): 71–86.', 'doi': '10.1016/S0031-3203(98)00091-0'},
    {'text': 'Pluim JPW, Maintz JBA, Viergever MA (2003). Mutual-information-based registration of medical images: '
             'a survey. IEEE Transactions on Medical Imaging 22(8): 986–1004.', 'doi': '10.1109/TMI.2003.815867'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 4.0, 'minimum': 1.0,
     'maximum': 16.0, 'step': 0.5, 'suffix': ' µm/px', 'tooltip': 'The coarse search runs at this resolution, the '
                                                                  'optimisation at half of it'},
    {'key': 'step_um', 'label': 'Search step', 'kind': 'float', 'default': 15.0, 'minimum': 2.0,
     'maximum': 100.0, 'step': 1.0, 'suffix': ' µm', 'tooltip': 'Spacing of the positions tried in the circle'},
    {'key': 'angle_step', 'label': 'Rotation step', 'kind': 'float', 'default': 10.0, 'minimum': 1.0,
     'maximum': 45.0, 'step': 1.0, 'suffix': ' °', 'tooltip': 'Spacing of the rotations tried'},
    {'key': 'bins', 'label': 'Histogram bins', 'kind': 'int', 'default': 32, 'minimum': 8, 'maximum': 128,
     'step': 8, 'tooltip': 'Grey levels of the joint histogram'},
]


def _nmi(a, b, bins):
    """Normalised mutual information (H(A) + H(B)) / H(A, B) of two value arrays scaled to 0…bins."""
    import numpy as np
    joint = np.bincount(a * bins + b, minlength=bins * bins).astype(np.float64).reshape(bins, bins)
    joint /= joint.sum()
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    return (ent(pa) + ent(pb)) / max(ent(joint.ravel()), 1e-12)


def _levels(img, bins):
    import numpy as np
    lo, hi = np.percentile(img, [1, 99])
    return np.clip(((img - lo) / max(hi - lo, 1e-6) * bins).astype(np.int64), 0, bins - 1)


class _Scorer:
    """NMI of the moving image against the reference at a pose (centre world µm, angle, scale, mirror)."""

    def __init__(self, ctx, px, bins):
        import numpy as np
        self.ctx, self.bins = ctx, bins
        fixed, self.frame = ctx.read(ctx.reference, px)
        moving, _ = ctx.read(ctx.moving, px)
        self.fixed = ctx.grey(fixed).astype(np.float32)
        self.moving_levels = _levels(ctx.grey(moving), bins)
        self.fixed_lo, self.fixed_hi = np.percentile(self.fixed, [1, 99])
        self.to_fixed = np.linalg.inv(ctx.to_world(self.frame))
        self.grid = ctx.moving_grid(moving.shape)
        self.size = ctx.moving.size_um

    def placement(self, cx, cy, angle, scale, mirror):
        from modalytics.core import geometry as geo
        from modalytics.core.geometry import Placement
        import numpy as np
        lin = geo.rotation(angle) @ geo.scaling(scale) @ (geo.scaling(-1.0, 1.0) if mirror else np.eye(3))
        off = geo.apply(lin, (self.size[0] / 2, self.size[1] / 2))
        return Placement(float(cx - off[0]), float(cy - off[1]), float(angle), float(scale), bool(mirror))

    def score(self, cx, cy, angle, scale, mirror):
        import cv2
        import numpy as np
        p = self.placement(cx, cy, angle, scale, mirror)
        a = (self.to_fixed @ p.local_to_world() @ self.grid)[:2].astype(np.float32)   # moving px → fixed px
        h, w = self.moving_levels.shape
        warped = cv2.warpAffine(self.fixed, a, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        ok = np.isfinite(warped)
        if ok.mean() < 0.8:                                   # mostly outside the reference: no score
            return 0.0
        b = np.clip(((warped[ok] - self.fixed_lo) / max(self.fixed_hi - self.fixed_lo, 1e-6) * self.bins)
                    .astype(np.int64), 0, self.bins - 1)
        return _nmi(self.moving_levels[ok], b, self.bins)


def register(ctx, options):
    import numpy as np
    from scipy.optimize import minimize
    px, bins = options['resolution_um'], int(options['bins'])
    coarse = _Scorer(ctx, px, bins)
    s = ctx.search
    angles, scales, mirrors = ctx.pose_grid(options['angle_step'])
    step = options['step_um']
    cx0, cy0 = s.centre_world
    n = int(s.radius_um // step)
    offsets = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
               if math.hypot(i * step, j * step) <= s.radius_um]
    poses = [(m, sc, an) for m in mirrors for sc in scales for an in angles]
    # 1. Coarse: every position in the circle at every pose
    found = []
    for k, (mirror, scale, angle) in enumerate(poses):
        ctx.check()
        for dx, dy in offsets:
            found.append((coarse.score(cx0 + dx, cy0 + dy, angle, scale, mirror), cx0 + dx, cy0 + dy, angle, scale,
                          mirror))
        ctx.progress(0.7 * (k + 1) / len(poses))
    found.sort(key=lambda f: -f[0])
    starts, centres = [], []
    for f in found:                                          # the best distinct positions
        if all(math.hypot(f[1] - c[0], f[2] - c[1]) > 2 * step for c in centres):
            starts.append(f)
            centres.append((f[1], f[2]))
        if len(starts) >= s.top_k:
            break
    # 2. Each refined by local optimisation at twice the resolution
    fine = _Scorer(ctx, px / 2, bins)
    out = []
    for k, (_score, cx, cy, angle, scale, mirror) in enumerate(starts):
        ctx.check()
        free_scale = s.scale_range > 0
        x0 = np.array([0.0, 0.0, 0.0] + ([0.0] if free_scale else []))

        def cost(v):
            sc = scale * (1 + v[3] / 100) if free_scale else scale
            return -fine.score(cx + v[0], cy + v[1], angle + v[2], sc, mirror)
        res = minimize(cost, x0, method='Powell', options={'xtol': 0.05, 'ftol': 1e-5, 'maxiter': 2000})
        v = res.x
        sc = scale * (1 + v[3] / 100) if free_scale else scale
        out.append(ctx.candidate(placement=fine.placement(cx + v[0], cy + v[1], angle + v[2], sc, mirror),
                                 score=-float(res.fun)))
        ctx.progress(0.7 + 0.3 * (k + 1) / len(starts))
    return sorted(out, key=lambda c: -c.score)
