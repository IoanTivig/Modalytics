"""
The maths behind the parameters (Qt-free, no knowledge of projects or files).

Two families:

* **Outline measures** work on the vector outline itself (shapely polygon in
  µm): area, perimeter, ellipse axes, eccentricity, curvature. They are exact
  and do not depend on any image's resolution, so a gland gives the same shape
  values whether it was outlined on a slide (0.25 µm/px) or a phase map
  (0.31 µm/px). The built-in shape parameters use these.
* **Pixel measures** are ported unchanged from HoloPath
  (``src/analysis/statistics.py``) and work on a boolean mask plus the image
  pixels under it: intensity, texture (GLCM), phase disorder (Eldridge et al.
  2017), and HoloPath's pixel-mask shape measures (kept so that HoloPath's
  user parameter scripts keep working, and to prove the port in the tests).

Conventions: outlines are shapely polygons in µm with y pointing down (image
convention); masks are 2D boolean arrays, rows = y.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

## Outline measures (vector, µm) ##


def parts(geom) -> list[Polygon]:
    """The polygons of a Polygon or MultiPolygon (empty ones dropped)."""
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    return [geom] if isinstance(geom, Polygon) and not geom.is_empty else []


def polygon_moments(poly) -> tuple[float, float, float, float, float, float]:
    """Area, centroid and central second moments of a polygon with holes (Green's theorem).

    Args:
        poly: Shapely Polygon or MultiPolygon (holes are subtracted; pieces add up).

    Returns:
        tuple: (area, cx, cy, mu20, mu02, mu11) where mu20 = var(x), mu02 = var(y),
        mu11 = cov(x, y) of a uniform density over the polygon (µm²).
    """
    if poly.is_empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    x0, y0 = poly.bounds[0], poly.bounds[1]        # shift for numerical stability
    a = sx = sy = sxx = syy = sxy = 0.0
    rings = []
    for part in parts(poly):
        part = orient(part, 1.0)                   # exterior counter-clockwise, holes clockwise
        rings += [part.exterior, *part.interiors]
    for ring in rings:
        xy = np.asarray(ring.coords)[:, :2] - (x0, y0)
        x, y = xy[:-1, 0], xy[:-1, 1]
        xn, yn = xy[1:, 0], xy[1:, 1]
        c = x * yn - xn * y                        # signed cross product per edge
        a += c.sum() / 2
        sx += ((x + xn) * c).sum() / 6
        sy += ((y + yn) * c).sum() / 6
        sxx += ((x * x + x * xn + xn * xn) * c).sum() / 12
        syy += ((y * y + y * yn + yn * yn) * c).sum() / 12
        sxy += ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * c).sum() / 24
    if a <= 0:
        return 0.0, x0, y0, 0.0, 0.0, 0.0
    cx, cy = sx / a, sy / a
    return a, cx + x0, cy + y0, sxx / a - cx * cx, syy / a - cy * cy, sxy / a - cx * cy


def ellipse_axes(mu20: float, mu02: float, mu11: float) -> tuple[float, float]:
    """(major, minor) axis lengths of the ellipse with the same second moments.

    Same convention as HoloPath (4 · √eigenvalue): for an ellipse with semi-axes
    a ≥ b this gives exactly (2a, 2b).
    """
    mean = (mu20 + mu02) / 2.0
    half = 0.5 * math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    lam1, lam2 = mean + half, max(mean - half, 0.0)
    return 4 * math.sqrt(lam1), 4 * math.sqrt(lam2)


def eccentricity_from_moments(mu20: float, mu02: float, mu11: float) -> float:
    """0 for a circle, towards 1 for a line (√(1 − λ₂/λ₁), as HoloPath)."""
    mean = (mu20 + mu02) / 2.0
    half = 0.5 * math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    lam1, lam2 = mean + half, mean - half
    if lam1 <= 0 or lam2 < 0:
        return 0.0
    return math.sqrt(1 - lam2 / lam1)


def resample_ring(ring_xy: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` (µm) along a closed ring (first point not repeated).

    Args:
        ring_xy: (N, 2) ring coordinates (closed or not).
        step: Spacing between output points.

    Returns:
        np.ndarray: (M, 2), M ≥ 8.
    """
    xy = np.asarray(ring_xy, float)[:, :2]
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / step))) if step > 0 else 8
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def curvature_along(points: np.ndarray, window: int | None = None, polyorder: int = 3) -> np.ndarray:
    """Curvature at each point of a closed, evenly sampled curve (1 / coordinate unit).

    Savitzky–Golay smoothing and derivatives on the closed curve, exactly as
    HoloPath's ``curvature_on_mask`` does on a pixel contour (same adaptive window).

    Args:
        points: (N, 2) evenly spaced points, (x, y).
        window: Smoothing window (odd); None = HoloPath's adaptive choice.
        polyorder: Polynomial order.

    Returns:
        np.ndarray: (N,) curvature, ≥ 0.

    Raises:
        ValueError: If there are too few points.
    """
    from scipy.signal import savgol_filter
    x, y = points[:, 0], points[:, 1]
    window = _savgol_window(len(x), window, polyorder)
    xp = savgol_filter(x, window, polyorder, deriv=1, mode="wrap")
    yp = savgol_filter(y, window, polyorder, deriv=1, mode="wrap")
    xpp = savgol_filter(x, window, polyorder, deriv=2, mode="wrap")
    ypp = savgol_filter(y, window, polyorder, deriv=2, mode="wrap")
    num = np.abs(xp * ypp - xpp * yp)
    den = (xp ** 2 + yp ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def _savgol_window(n: int, window: int | None, polyorder: int) -> int:
    """HoloPath's window rule: adaptive, odd, larger than the order, at most n."""
    if n < polyorder + 2:
        raise ValueError(f"Contour too small ({n} points, need at least {polyorder + 2}).")
    if window is None:
        window = max(5, (n // 30) * 2 + 1)
    if window % 2 == 0:
        window += 1
    max_window = n if n % 2 == 1 else n - 1
    window = min(window, max_window)
    if window <= polyorder:
        window = min(polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2, max_window)
    if window <= polyorder:
        raise ValueError(f"Cannot compute curvature: window ({window}) must be > polyorder ({polyorder}).")
    return window


## Rasterising an outline onto an image grid ##

def rasterize(poly: Polygon, x0: float, y0: float, px: float, py: float, shape: tuple[int, int]) -> np.ndarray:
    """Pixels of a grid whose centres lie inside a polygon (exactly; holes excluded).

    Every pixel centre of the grid is tested against the outline (shapely,
    prepared geometry). Polygon-filling routines (e.g. OpenCV's) also paint the
    pixels the outline merely touches, which grows a mask by about half a pixel
    all round: a few % of a small object's area, and its pixel statistics would
    include a ring of pixels outside it.

    Args:
        poly: Outline in µm.
        x0, y0: µm position of the grid's top-left corner.
        px, py: Pixel size (µm).
        shape: (rows, cols) of the grid.

    Returns:
        np.ndarray: (rows, cols) bool mask.
    """
    import shapely
    rows, cols = shape
    if rows == 0 or cols == 0:
        return np.zeros(shape, bool)
    xs = x0 + (np.arange(cols) + 0.5) * px
    ys = y0 + (np.arange(rows) + 0.5) * py
    gx, gy = np.meshgrid(xs, ys)
    shapely.prepare(poly)
    return shapely.contains_xy(poly, gx, gy)


## Pixel measures: shape (ported from HoloPath, pixel units) ##

def _pixel_moments(object_mask: np.ndarray):
    ys, xs = np.where(object_mask)
    n = len(xs)
    if n < 4:
        return None
    xb, yb = xs.mean(), ys.mean()
    return n, np.sum((xs - xb) ** 2), np.sum((ys - yb) ** 2), np.sum((xs - xb) * (ys - yb))


def major_axis_px(object_mask: np.ndarray) -> float:
    """Major axis of the best-fit ellipse (px). HoloPath ``calculate_major_axis_length``."""
    m = _pixel_moments(object_mask)
    if m is None:
        return 0.0
    n, m20, m02, m11 = m
    a, b = (m20 / n + m02 / n) / 2.0, 0.5 * np.sqrt(4 * (m11 / n) ** 2 + (m20 / n - m02 / n) ** 2)
    if a + b == 0 or a - b < 0:
        return 0.0
    return float(4 * np.sqrt(a + b))


def minor_axis_px(object_mask: np.ndarray) -> float:
    """Minor axis of the best-fit ellipse (px). HoloPath ``calculate_minor_axis_length``."""
    m = _pixel_moments(object_mask)
    if m is None:
        return 0.0
    n, m20, m02, m11 = m
    a, b = (m20 / n + m02 / n) / 2.0, 0.5 * np.sqrt(4 * (m11 / n) ** 2 + (m20 / n - m02 / n) ** 2)
    if a + b == 0 or a - b < 0:
        return 0.0
    return float(4 * np.sqrt(a - b))


def equivalent_diameter_px(object_mask: np.ndarray) -> float:
    """Diameter of the circle with the same pixel area (px)."""
    area = np.sum(object_mask)
    return float(np.sqrt(4 * area / np.pi)) if area else 0.0


def perimeter_px(object_mask: np.ndarray, window: int | None = None, polyorder: int = 3) -> float:
    """Perimeter of the Savitzky–Golay-smoothed outer pixel contour (px). HoloPath ``calculate_perimeter``."""
    from scipy.signal import savgol_filter
    from skimage import measure
    padded = np.pad(object_mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded.astype(float), level=0.5)
    if not contours:
        return 0.0
    coords = max(contours, key=len)
    y, x = coords[:, 0], coords[:, 1]
    n = len(x)
    if window is None:
        window = max(5, (n // 30) * 2 + 1)
    if window % 2 == 0:
        window += 1
    window = min(window, n - (1 - n % 2))
    if window <= polyorder:
        window = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
        window = min(window, n - (1 - n % 2))
    if n <= window:                                 # too small to smooth: the jagged perimeter
        return float(np.sum(np.hypot(np.diff(y), np.diff(x))) + np.hypot(y[-1] - y[0], x[-1] - x[0]))
    xs = savgol_filter(x, window, polyorder, deriv=0, mode="wrap")
    ys = savgol_filter(y, window, polyorder, deriv=0, mode="wrap")
    return float(np.sum(np.hypot(np.diff(ys), np.diff(xs))) + np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))


def eccentricity_px(object_mask: np.ndarray) -> float:
    """Eccentricity of the pixel mask. HoloPath ``calculate_eccentricity``."""
    m = _pixel_moments(object_mask)
    if m is None:
        return 0.0
    _n, m20, m02, m11 = m
    a, b = (m20 + m02) / 2.0, 0.5 * np.sqrt(4 * m11 ** 2 + (m20 - m02) ** 2)
    if a + b == 0 or a - b < 0:
        return 0.0
    return float(np.sqrt(1 - (a - b) / (a + b)))


def curvature_px(object_mask: np.ndarray, window: int | None = None, polyorder: int = 3) -> tuple[float, float, float]:
    """(mean, max, min) curvature along the outer pixel contour (1/px), NaN if too small.

    HoloPath ``calculate_curvature`` / ``curvature_on_mask``.
    """
    from skimage import measure
    contours = measure.find_contours(object_mask.astype(float), level=0.5)
    if not contours:
        return math.nan, math.nan, math.nan
    coords = max(contours, key=len)
    try:
        kappa = curvature_along(np.column_stack([coords[:, 1], coords[:, 0]]), window, polyorder)
    except ValueError:
        return math.nan, math.nan, math.nan
    kappa = kappa[np.isfinite(kappa)]
    if kappa.size == 0:
        return math.nan, math.nan, math.nan
    return float(kappa.mean()), float(kappa.max()), float(kappa.min())


## Pixel measures: texture ##

def uniformity(pixels: np.ndarray) -> float:
    """Mean² / variance (0 if the variance is 0)."""
    v = np.var(pixels)
    return float(np.mean(pixels) ** 2 / v) if v else 0.0


def glcm_features(object_mask: np.ndarray, image_crop: np.ndarray, distance: int = 1) -> dict:
    """Haralick texture features of a masked region, averaged over 4 directions.

    HoloPath ``calculate_glcm_features``: pixels quantised to 1–255 (0 = outside),
    co-occurrences with the outside removed, then energy, entropy, contrast,
    homogeneity, dissimilarity and correlation.

    Args:
        object_mask: Boolean mask.
        image_crop: Pixel values under it (2D).
        distance: Neighbour distance in pixels (HoloPath: 1).

    Returns:
        dict: The six features.
    """
    from skimage.feature import graycomatrix, graycoprops
    pixels = image_crop[object_mask]
    if pixels.size < 2 or np.all(pixels == pixels[0]):
        return {"energy": 1.0, "entropy": 0.0, "contrast": 0.0, "homogeneity": 1.0,
                "dissimilarity": 0.0, "correlation": 0.0}
    lo, hi = np.min(pixels), np.max(pixels)
    patch = np.zeros(object_mask.shape, np.uint8)
    patch[object_mask] = ((pixels - lo) / (hi - lo) * 254.0 + 1.0).astype(np.uint8)
    glcm = graycomatrix(patch, distances=[int(distance)], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    g = glcm[1:, 1:, :, :]                         # drop pairs involving the outside
    total = np.sum(g, axis=(0, 1))
    total[total == 0] = 1.0
    g = g / total
    energy = float(np.mean(np.sum(g ** 2, axis=(0, 1))))
    entropy = float(np.mean(np.sum(-g * np.log2(g + 1e-9), axis=(0, 1))))

    def prop(name: str) -> float:
        try:
            value = float(np.nanmean(graycoprops(g, name)))
            return value if np.isfinite(value) else 0.0
        except Exception:
            return 0.0

    return {"energy": energy, "entropy": entropy, "contrast": prop("contrast"),
            "homogeneity": prop("homogeneity"), "dissimilarity": prop("dissimilarity"),
            "correlation": prop("correlation")}


## Pixel measures: phase disorder (Eldridge et al. 2017) ##

def detrend_phase(phase_crop: np.ndarray, mask: np.ndarray, degree: int = 2) -> np.ndarray:
    """Subtract a 2D polynomial fitted to the masked phase; zero outside the mask.

    HoloPath ``detrend_phase_2d`` (falls back to mean subtraction when there are
    fewer pixels than polynomial terms).
    """
    ys, xs = np.where(mask)
    n = len(ys)
    if n < (degree + 1) * (degree + 2) // 2:
        mean = np.mean(phase_crop[mask]) if n > 0 else 0.0
        return np.where(mask, phase_crop - mean, 0.0)
    ym, ysd = ys.mean(), (ys.std() or 1.0)
    xm, xsd = xs.mean(), (xs.std() or 1.0)
    yn, xn = (ys - ym) / ysd, (xs - xm) / xsd
    terms = [(j, i) for j in range(degree + 1) for i in range(degree + 1 - j)]
    a = np.column_stack([(yn ** j) * (xn ** i) for j, i in terms])
    coeffs = np.linalg.lstsq(a, phase_crop[ys, xs], rcond=None)[0]
    yg, xg = np.indices(phase_crop.shape)
    ygn, xgn = (yg - ym) / ysd, (xg - xm) / xsd
    fit = np.zeros_like(phase_crop)
    for c, (j, i) in zip(coeffs, terms):
        fit += c * (ygn ** j) * (xgn ** i)
    return np.where(mask, phase_crop - fit, 0.0)


def relative_local_variance(values: np.ndarray, mask: np.ndarray, window: int = 3) -> np.ndarray:
    """Local variance over a ``window``×``window`` neighbourhood divided by the masked region's mean².

    The fluctuation measure of the optical-density parameters: like the phase
    fluctuation image, but normalised by the region's mean instead of each
    neighbourhood's, which stays finite for positive quantities with pale spots.
    Returns zeros if the region's mean is not positive.
    """
    from scipy import ndimage
    mean_all = float(np.mean(values[mask])) if mask.any() else 0.0
    if mean_all <= 0:
        return np.zeros(values.shape)
    local = ndimage.uniform_filter(values, size=window, mode="reflect")
    sq = ndimage.uniform_filter(values ** 2, size=window, mode="reflect")
    return np.maximum(sq - local ** 2, 0.0) / mean_all ** 2


def phase_fluctuation_image(phase: np.ndarray, window: int = 3, epsilon: float = 1e-5) -> np.ndarray:
    """Local variance / mean² over a ``window``×``window`` neighbourhood (HoloPath ``calculate_pfi``)."""
    from scipy import ndimage
    mean = ndimage.uniform_filter(phase, size=window, mode="reflect")
    sq_mean = ndimage.uniform_filter(phase ** 2, size=window, mode="reflect")
    var = np.maximum(sq_mean - mean ** 2, 0.0)
    denom = mean ** 2
    return np.divide(var, denom, out=np.zeros_like(var), where=denom > epsilon)


def correlation_length(pfi: np.ndarray, mask: np.ndarray | None = None, pixel_size_um: float = 1.0) -> float:
    """Spatial correlation length l_c (µm) of a fluctuation image (HoloPath ``calculate_correlation_length``).

    1. Autocorrelation of the (masked, mean-removed) image by FFT.
    2. Its radial average.
    3. Exponential fit up to where it drops below 0.05, with a 1/e
       interpolation as fallback when the fit is unusable.
    """
    if mask is not None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0.0
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask_crop = mask[y0:y1, x0:x1]
        roi = np.where(mask_crop, pfi[y0:y1, x0:x1], 0.0)
    else:
        roi, mask_crop = pfi, np.ones_like(pfi, dtype=bool)
    h, w = roi.shape
    if h < 3 or w < 3:
        return 0.0
    valid = roi[mask_crop]
    if valid.size == 0:
        return 0.0
    centred = np.where(mask_crop, roi - np.mean(valid), 0.0)
    auto = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(centred)) ** 2)))
    cy, cx = h // 2, w // 2
    if auto[cy, cx] < 1e-12:
        return 0.0
    auto = auto / auto[cy, cx]
    yi, xi = np.indices(auto.shape)
    r = np.round(np.sqrt((yi - cy) ** 2 + (xi - cx) ** 2)).astype(int).ravel()
    max_r = min(h, w) // 2
    if max_r < 2:
        return 0.0
    keep = r <= max_r
    sums = np.bincount(r[keep], weights=auto.ravel()[keep])
    radial = (sums / np.maximum(np.bincount(r[keep]), 1))[:max_r + 1]

    n_fit = 0
    for v in radial:
        if v <= 0.05:
            break
        n_fit += 1
    lc = 0.0
    if n_fit >= 2:
        rr = np.arange(n_fit)[1:]
        denom = np.sum(rr ** 2)
        if denom > 0:
            beta = -np.sum(rr * np.log(radial[1:n_fit])) / denom
            if beta > 1e-5:
                lc = 1.0 / beta
    if lc <= 0 or lc > max_r or np.isnan(lc):            # fallback: where it crosses 1/e
        target = 1.0 / np.e
        for i in range(len(radial) - 1):
            if radial[i] >= target >= radial[i + 1]:
                diff = radial[i] - radial[i + 1]
                lc = i + (radial[i] - target) / diff if diff > 1e-12 else float(i)
                break
        else:
            lc = next((float(i) for i, v in enumerate(radial) if v < target), 0.0)
    return float(lc * pixel_size_um)


def disorder_strength(pfi: np.ndarray, correlation_length_um: float, mask: np.ndarray | None = None,
                      refractive_index: float = 1.38) -> float:
    """L_d = ⟨PF⟩ · n² · l_c (µm), Eq. 5 of Eldridge et al. 2017 (HoloPath ``calculate_disorder_strength``)."""
    vals = pfi[mask] if mask is not None else pfi
    if vals.size == 0 or np.isnan(correlation_length_um) or correlation_length_um <= 0:
        return 0.0
    return float(np.mean(vals) * refractive_index ** 2 * correlation_length_um)


## Outline geometry: convex hull, Feret diameters, fractal dimension, holes, best alignment metric ##

def _hull_points(geom) -> np.ndarray:
    """(N, 2) vertices of the convex hull (counter-clockwise, not repeated)."""
    hull = geom.convex_hull
    if hull.geom_type != "Polygon":
        return np.asarray(hull.coords) if hasattr(hull, "coords") else np.zeros((0, 2))
    return np.asarray(hull.exterior.coords)[:-1]


def feret_diameters(geom) -> tuple[float, float, float]:
    """(maximum, minimum, angle of the maximum in degrees 0–180) caliper diameters of an outline.

    The maximum Feret diameter is the largest distance between two points of the
    outline (between two hull vertices); the minimum is the narrowest width
    between two parallel lines enclosing it (rotating calipers: the smallest,
    over the hull's edges, of the largest distance of a hull vertex from that edge).
    """
    pts = _hull_points(geom)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    fmax = float(dist[i, j])
    dx, dy = pts[j] - pts[i]
    angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    if len(pts) < 3:
        return fmax, 0.0, angle
    widths = []
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        e = b - a
        length = math.hypot(*e)
        if length == 0:
            continue
        normal = np.array([-e[1], e[0]]) / length
        widths.append(float(np.max(np.abs((pts - a) @ normal))))
    return fmax, (min(widths) if widths else 0.0), angle


def box_counting_dimension(geom, smallest: float, sizes: int = 8) -> float:
    """Box-counting (Minkowski–Bouligand) dimension of an outline's boundary.

    The boundary is sampled densely; for box sizes ε from ``smallest`` to a quarter
    of the outline's larger side (geometric steps), N(ε) = number of boxes the
    boundary passes through; the dimension is the slope of log N(ε) against
    log(1/ε). 1 for a smooth curve; up to 2 for a curve that fills the plane.
    """
    rings = [np.asarray(p.exterior.coords) for p in parts(geom)] + \
        [np.asarray(r.coords) for p in parts(geom) for r in p.interiors]
    if not rings:
        return math.nan
    minx, miny, maxx, maxy = geom.bounds
    largest = max(maxx - minx, maxy - miny)
    if largest <= 0 or smallest <= 0:
        return math.nan
    top = largest / 4.0
    if top <= smallest * 2:
        return math.nan
    eps = np.geomspace(smallest, top, sizes)
    pts = np.concatenate([resample_ring(r, smallest / 4.0) for r in rings if len(r) >= 3])
    counts = []
    for e in eps:
        # Average over a few grid offsets: the count depends a little on where the grid falls
        n = [len(np.unique(np.floor((pts - [minx - o * e, miny - o * e]) / e).astype(np.int64), axis=0))
             for o in (0.0, 0.25, 0.5, 0.75)]
        counts.append(np.mean(n))
    slope = np.polyfit(np.log(1.0 / eps), np.log(counts), 1)[0]
    return float(slope)


def minimum_enclosing_ellipse(points: np.ndarray, tolerance: float = 1e-5, rounds: int = 2000):
    """The minimum-area ellipse enclosing points (Khachiyan's algorithm).

    Returns:
        tuple: (centre c (2,), matrix A (2, 2)) with the ellipse {x : (x − c)ᵀ A (x − c) ≤ 1}.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    q = np.vstack([p.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    for _ in range(rounds):
        x = q @ (u[:, None] * q.T)
        m = np.einsum("ij,ji->i", q.T, np.linalg.solve(x, q))
        j = int(np.argmax(m))
        step = (m[j] - 3.0) / (3.0 * (m[j] - 1.0))
        new = (1.0 - step) * u
        new[j] += step
        done = np.linalg.norm(new - u) < tolerance
        u = new
        if done:
            break
    c = p.T @ u
    a = np.linalg.inv(p.T @ (u[:, None] * p) - np.outer(c, c)) / 2.0
    return c, a


def _equal_steps(curve: np.ndarray, n: int) -> np.ndarray:
    """n points equally spaced by arc length along a closed curve (complex numbers)."""
    closed = np.vstack([curve, curve[:1]])
    seg = np.sqrt((np.diff(closed, axis=0) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    x = np.interp(t, s, closed[:, 0])
    y = np.interp(t, s, closed[:, 1])
    return x + 1j * y


def best_alignment_metric(geom, n: int = 256) -> float:
    """Awan et al. (2017) best alignment metric of an outline against its normalised reference ellipse.

    1. The minimum-area ellipse enclosing the outline is found and the plane is
       stretched so that ellipse becomes a circle (the same stretch is applied to
       the outline: cutting angle removed).
    2. Outline and circle are scaled to unit length, centred, and sampled at n
       equally spaced points (complex numbers u_j, v_j), both counter-clockwise.
    3. d = (1/n) · min over shifts r and rotations θ of Σ_j |v_j − e^{iθ} u_{(j+r) mod n}|²
       (for each shift the best rotation is explicit, and all shifts at once come from an FFT).
    """
    largest = max(parts(geom), key=lambda p: p.area)
    if largest.length <= 0:
        return math.nan
    ring = resample_ring(np.asarray(largest.exterior.coords), largest.length / (4 * n))   # corners kept, sides filled
    if len(ring) < 5:
        return math.nan
    # 1. The enclosing ellipse turned into a circle
    c, a = minimum_enclosing_ellipse(_hull_points(largest))
    w, vecs = np.linalg.eigh(a)
    stretch = vecs @ np.diag(np.sqrt(np.maximum(w, 0))) @ vecs.T      # maps the ellipse onto the unit circle
    curve = (ring - c) @ stretch.T
    if _signed_area(curve) < 0:
        curve = curve[::-1]
    # 2. Unit length, centred, n equal steps; the reference circle the same way
    u = _equal_steps(curve, n)
    u = u - u.mean()
    u = u / np.sum(np.abs(np.diff(np.append(u, u[0]))))
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    v = np.exp(1j * angles) / (2 * np.pi)                                  # circumference 1
    # 3. Best shift and rotation: |Σ conj(v_j) u_(j+r)| is largest at the best rotation for each r
    corr = np.fft.ifft(np.conj(np.fft.fft(v)) * np.fft.fft(u))
    best = np.max(np.abs(corr))
    return float((np.sum(np.abs(v) ** 2) + np.sum(np.abs(u) ** 2) - 2 * best) / n)


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


## Scattering-phase theorem (Wang, Ding & Popescu 2011; relations as corrected by Xu 2011) ##

def scattering_parameters(phase: np.ndarray, mask: np.ndarray, pixel_size_um: float, wavelength_um: float | None,
                          thickness_um: float, medium_index: float = 1.0) -> dict:
    """Scattering coefficients of a thin slice from its phase map (Xu 2011, Eqs. 7–9).

    μs·L = 2⟨1 − cos Δφ⟩, Δφ = φ − ⟨φ⟩ (≈ ⟨Δφ²⟩ for small phases);
    μs′·L = ⟨|∇φ|²⟩ / (2k²), k = 2π·n₀/λ;  g = 1 − μs′/μs.
    Averages ⟨⟩ are over the masked pixels; the gradient only where a pixel's four
    neighbours are inside too (no step at the outline).

    μs and lₛ need only the thickness; μs′ and g also the wavelength (None: they are NaN).

    Returns:
        dict: mu_s (1/µm), l_s (µm), mu_s_reduced (1/µm), g, phase_variance (rad²).
    """
    from scipy import ndimage
    values = phase[mask].astype(np.float64)
    if values.size < 9 or thickness_um <= 0:
        return {k: math.nan for k in ("mu_s", "l_s", "mu_s_reduced", "g", "phase_variance")}
    delta = values - values.mean()
    mu_s = 2.0 * float(np.mean(1.0 - np.cos(delta))) / thickness_um
    mu_s_reduced = math.nan
    if wavelength_um:
        gy, gx = np.gradient(phase.astype(np.float64), pixel_size_um)
        inner = ndimage.binary_erosion(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        grad2 = float(np.mean((gx ** 2 + gy ** 2)[inner])) if inner.any() else math.nan
        k = 2.0 * math.pi * medium_index / wavelength_um
        mu_s_reduced = grad2 / (2.0 * k ** 2) / thickness_um
    return {"mu_s": mu_s, "l_s": 1.0 / mu_s if mu_s > 0 else math.inf, "mu_s_reduced": mu_s_reduced,
            "g": 1.0 - mu_s_reduced / mu_s if mu_s > 0 else math.nan,
            "phase_variance": float(np.mean(delta ** 2))}
