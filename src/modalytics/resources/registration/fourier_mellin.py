"""Phase correlation (Fourier–Mellin): finds shift, rotation and scale from the images' Fourier transforms.

The magnitude of a Fourier transform does not change when an image moves, so rotation
and scale are found first: in log-polar coordinates of the magnitudes they become
shifts, measured by phase correlation. The moving image is then turned and scaled
accordingly and phase correlation gives the shift. Each reading (the magnitude cannot
tell θ from θ + 180°) is scored by normalised cross-correlation and the best kept.
Fast and exact for two images of the same kind that overlap well; a small image in a
large, repetitive field confuses it.
"""

import math

KEY = 'fourier_mellin'
NAME = 'Phase correlation (Fourier–Mellin)'
DESCRIPTION = ('Finds rotation and scale from the log-polar Fourier magnitudes, then the shift by phase '
               'correlation; each reading scored by cross-correlation. Fast; for images of the same kind that overlap '
               'well (a small search radius helps).')
MOVING_TYPES = ['phase', 'brightfield']
REFERENCE_TYPES = ['wsi', 'brightfield', 'phase']
FINDS = ['translation', 'rotation', 'scale', 'mirror']
EXPERIMENTAL = True
REFERENCES = [
    {'text': 'Reddy BS, Chatterji BN (1996). An FFT-based technique for translation, rotation, and scale-invariant '
             'image registration. IEEE Transactions on Image Processing 5(8): 1266–1271.', 'doi': '10.1109/83.506761'},
    {'text': 'Zitová B, Flusser J (2003). Image registration methods: a survey. Image and Vision Computing 21(11): '
             '977–1000.', 'doi': '10.1016/S0262-8856(03)00137-9'},
]
REQUIRES = []
OPTIONS = [
    {'key': 'resolution_um', 'label': 'Resolution', 'kind': 'float', 'default': 2.0, 'minimum': 0.5,
     'maximum': 10.0, 'step': 0.5, 'suffix': ' µm/px', 'tooltip': 'The images are compared at this resolution'},
]


def _window(n):
    import numpy as np
    return np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)


def _log_polar_magnitude(img):
    """High-passed log-magnitude spectrum in log-polar coordinates (rows: angle 0…360°, columns: log radius)."""
    import cv2
    import numpy as np
    n = img.shape[0]
    f = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    y, x = np.mgrid[:n, :n]
    r = np.hypot(x - n / 2, y - n / 2) / (n / 2)
    f = np.log1p(f) * np.clip(r * 4, 0, 1)                    # damp the lowest frequencies (high-pass)
    return cv2.warpPolar(f.astype(np.float32), (n, n), (n / 2, n / 2), n / 2,
                         cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR)


def register(ctx, options):
    import cv2
    import numpy as np
    from modalytics.core import geometry as geo
    from modalytics.core.geometry import Placement
    px = options['resolution_um']
    s = ctx.search
    mw, mh = ctx.moving.size_um
    half = s.radius_um + 0.5 * math.hypot(mw, mh) * (1 + s.scale_range)
    n = int(2 ** math.ceil(math.log2(2 * half / px)))           # a square window, power of two
    cx, cy = s.centre_world
    loc = geo.apply(np.linalg.inv(ctx.reference.local_to_world()), (cx, cy))
    rect = (loc[0] - n * px / 2, loc[1] - n * px / 2, loc[0] + n * px / 2, loc[1] + n * px / 2)
    fixed, frame = ctx.read(ctx.reference, px, rect)
    fixed = ctx.grey(fixed)
    fixed = cv2.resize(fixed, (n, n)) if fixed.shape != (n, n) else fixed
    frame = (rect[0], rect[1], px, px)
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-6)
    moving, _ = ctx.read(ctx.moving, px)
    moving = ctx.grey(moving)
    moving = (moving - moving.mean()) / (moving.std() + 1e-6)
    win = _window(n)
    to_win = np.linalg.inv(ctx.to_world(frame))
    grid = ctx.moving_grid(moving.shape)

    def placed(angle, scale, mirror, ccx, ccy):
        lin = geo.rotation(angle) @ geo.scaling(scale) @ (geo.scaling(-1.0, 1.0) if mirror else np.eye(3))
        off = geo.apply(lin, (mw / 2, mh / 2))
        return Placement(float(ccx - off[0]), float(ccy - off[1]), float(angle), float(scale), bool(mirror))

    def warp(p):
        a = (to_win @ p.local_to_world() @ grid)[:2].astype(np.float32)
        img = cv2.warpAffine(moving, a, (n, n), flags=cv2.INTER_LINEAR, borderValue=0.0)
        mask = cv2.warpAffine(np.ones_like(moving), a, (n, n), flags=cv2.INTER_NEAREST, borderValue=0.0)
        return img, mask > 0

    def ncc(p):
        img, mask = warp(p)
        if mask.sum() < 0.5 * moving.size:
            return -1.0
        a, b = img[mask], fixed[mask]
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    found = []
    _angles, _scales, mirrors = ctx.pose_grid()
    base = ctx.moving.placement.rotation_deg
    for k, mirror in enumerate(mirrors):
        ctx.check()
        # 1. Rotation and scale: log-polar phase correlation of the magnitude spectra
        upright, _m = warp(placed(base, 1.0, mirror, cx, cy))
        (dlog, dang), _resp = cv2.phaseCorrelate(_log_polar_magnitude(fixed * win), _log_polar_magnitude(upright * win))
        d_angle = dang * 360.0 / n
        d_scale = math.exp(dlog * math.log(n / 2) / n)
        if s.angle_range <= 0:
            angle_options = [base]
        else:
            angle_options = [base + sign * d_angle + extra for sign in (1, -1) for extra in (0.0, 180.0)]
        scale_options = [1.0] if s.scale_range <= 0 else \
            sorted({float(np.clip(v, 1 - s.scale_range, 1 + s.scale_range)) for v in (d_scale, 1 / d_scale)})
        # 2. Shift for each reading, scored by cross-correlation
        for angle in angle_options:
            for scale in scale_options:
                img, _mask = warp(placed(angle, scale, mirror, cx, cy))
                (sx, sy), _r = cv2.phaseCorrelate(fixed * win, img * win)
                c_px = geo.apply(to_win, (cx, cy))                   # where the moving centre is now (window px)
                for sign in (1, -1):                                 # the shift's direction, whichever it is
                    c = geo.apply(ctx.to_world(frame), (c_px[0] + sign * sx, c_px[1] + sign * sy))
                    p = placed(angle, scale, mirror, c[0], c[1])
                    centre = geo.apply(p.local_to_world(), (mw / 2, mh / 2))
                    if math.hypot(centre[0] - cx, centre[1] - cy) <= s.radius_um:
                        found.append(ctx.candidate(placement=p, score=ncc(p)))
        ctx.progress((k + 1) / len(mirrors))
    return sorted(found, key=lambda c: -c.score)
