"""
What a parameter script's ``compute(ctx)`` receives.

``ObjectContext`` (object scope) describes one object — one polygon of a
mask — measured on one image. ``ImageContext`` (image scope) describes a
whole image. Everything is computed on first use and kept, so twenty
parameters that all need the same texture matrix compute it once.

Each context offers two vocabularies:

* **Modalytics (outline, µm)**: ``polygon``, ``area_um2``, ``perimeter_um``,
  ``major_axis_um``… exact, from the vector outline.
* **HoloPath (pixels)**: ``mask``, ``image_crop``, ``pixels``, ``area_px``,
  ``perimeter_px``, ``glcm``, ``pfi``… the same names and values as HoloPath's
  ``ObjectParamContext`` / ``ImageParamContext``, so HoloPath user parameter
  scripts run unchanged. The mask is the outline rasterised onto the measured
  image (pixels whose centre lies inside), trimmed to its bounding box like
  HoloPath's object slices.

``ctx.constants`` holds the constants of the parameter being computed (their
declared defaults, overridden by the profile); the engine sets them before each
``compute`` call. ``ctx.settings`` is the same dict under its older name.
Intermediates that depend on constants are methods with arguments
(``fluctuation_image(degree, window)``); the HoloPath-named properties are
those methods at HoloPath's fixed values.
"""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box

from modalytics.parameters import measures as ms
from modalytics.parameters import optical as op
from modalytics.parameters.classes import ClassIndex, as_label, same_object

MAX_IMAGE_PIXELS = 100_000_000      # whole-image parameters refuse larger images (a slide has billions)


def read_native(image, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Level-0 pixels rows r0:r1, columns c0:c1 of an image (no resampling).

    Raises:
        ValueError: For an image whose file is not found (its pixels are a placeholder).
    """
    src = image.source
    if getattr(src, "missing", False):
        raise ValueError(f"{image.name}: the image file is not found (relink it in Properties)")
    lv = src.levels[0]
    th, tw = lv.tile
    rows = []
    for ty in range(r0 // th, (r1 - 1) // th + 1):
        rows.append(np.concatenate([src.read_tile(0, ty, tx) for tx in range(c0 // tw, (c1 - 1) // tw + 1)], axis=1))
    block = np.concatenate(rows, axis=0)
    oy, ox = (r0 // th) * th, (c0 // tw) * tw
    return block[r0 - oy:r1 - oy, c0 - ox:c1 - ox]


def _wavelength(image, wavelength_nm: float | None) -> float | None:
    """The wavelength (nm) to use: None if not needed, the given one, else the importer's (0 = from the file)."""
    if wavelength_nm is None:
        return None
    wavelength = wavelength_nm or image.metadata.get("wavelength_nm")
    if not wavelength:
        raise ValueError("the wavelength is not known: set it in the parameter's constants")
    return float(wavelength)


class ObjectContext:
    """One object measured on one image.

    Args:
        polygon: The object's outline in the measured image's local µm (a
            MultiPolygon for regions in pieces, e.g. surroundings cut by the image edge).
        image: The ImageNode whose pixels are measured (it gives the pixel grid,
            the image size and the phase-to-height factor).
        constants: The current parameter's constants.
        images: The project's other images, for ``on`` (none: ``on`` finds nothing).
        classes: The annotation classes around it (``ClassIndex``; default: the image's own masks, and those
            of ``images`` registered onto it).
    """

    def __init__(self, polygon: Polygon, image, constants: dict | None = None, images=(), classes=None) -> None:
        self.polygon = polygon
        self.image = image
        self.constants = dict(constants or {})
        self.images = list(images)
        self.pixel_size_um = float(image.pixel_size_um[0])
        self._classes = classes
        self._cache: dict = {}

    @property
    def settings(self) -> dict:
        """The constants under their older name (scripts written before the rename)."""
        return self.constants

    # ------------------------------------------------------------------ outline (µm)
    @cached_property
    def moments(self) -> tuple[float, float, float, float, float, float]:
        """(area, cx, cy, mu20, mu02, mu11) of the outline (holes subtracted)."""
        return ms.polygon_moments(self.polygon)

    @property
    def area_um2(self) -> float:
        """Area of the outline (µm²), holes subtracted."""
        return self.moments[0]

    @cached_property
    def perimeter_um(self) -> float:
        """Length of the outer boundary (holes not counted, as HoloPath's outer contour)."""
        return float(sum(p.exterior.length for p in ms.parts(self.polygon)))

    @property
    def equivalent_diameter_um(self) -> float:
        """Diameter of the circle with the same area (µm)."""
        return math.sqrt(4 * self.area_um2 / math.pi) if self.area_um2 > 0 else 0.0

    @cached_property
    def axes_um(self) -> tuple[float, float]:
        """(major, minor) axes of the ellipse with the outline's second moments."""
        return ms.ellipse_axes(*self.moments[3:])

    @property
    def major_axis_um(self) -> float:
        """Major axis of the ellipse with the outline's second moments (µm)."""
        return self.axes_um[0]

    @property
    def minor_axis_um(self) -> float:
        """Minor axis of the ellipse with the outline's second moments (µm)."""
        return self.axes_um[1]

    @property
    def eccentricity_outline(self) -> float:
        """Eccentricity of the outline: 0 for a circle, towards 1 for a line."""
        return ms.eccentricity_from_moments(*self.moments[3:])

    @cached_property
    def hull(self):
        """Convex hull of the outline (shapely Polygon)."""
        return self.polygon.convex_hull

    @property
    def solidity(self) -> float:
        """Area / convex-hull area (1 = convex; lower = concave, indented)."""
        return min(1.0, self.area_um2 / self.hull.area) if self.hull.area > 0 else 0.0     # 1 + rounding → 1

    @property
    def convexity(self) -> float:
        """Convex-hull perimeter / perimeter (1 = convex; lower = a ragged, folded boundary)."""
        return min(1.0, self.hull.length / self.perimeter_um) if self.perimeter_um > 0 else 0.0

    @cached_property
    def feret(self) -> tuple[float, float, float]:
        """(maximum, minimum Feret diameter in µm, angle of the maximum in degrees)."""
        return ms.feret_diameters(self.polygon)

    @cached_property
    def holes(self) -> tuple[int, float]:
        """(number of holes, their total area in µm²) — e.g. a gland's lumen."""
        rings = [r for p in ms.parts(self.polygon) for r in p.interiors]
        return len(rings), float(sum(Polygon(r).area for r in rings))

    def fractal_dimension(self, smallest_um: float = 0.0) -> float:
        """Box-counting dimension of the outline (smallest box: ``smallest_um``, 0 = 2 pixels)."""
        key = ("fractal", float(smallest_um))
        if key not in self._cache:
            self._cache[key] = ms.box_counting_dimension(self.polygon, smallest_um or 2 * self.pixel_size_um)
        return self._cache[key]

    def best_alignment_metric(self, points: int = 256) -> float:
        """Awan et al. 2017 best alignment metric of the outline (0 = an ellipse; larger = more irregular)."""
        key = ("bam", int(points))
        if key not in self._cache:
            self._cache[key] = ms.best_alignment_metric(self.polygon, int(points))
        return self._cache[key]

    def curvature_um(self, spacing_um: float = 0.0) -> tuple[float, float, float]:
        """(mean, max, min) curvature of the outer boundary (1/µm), NaN if too small.

        Args:
            spacing_um: Distance between outline points before smoothing; 0 uses
                the measured image's pixel size (HoloPath's contour spacing).
        """
        step = spacing_um if spacing_um > 0 else self.pixel_size_um
        key = ("curvature_um", step)
        if key not in self._cache:
            try:
                largest = max(ms.parts(self.polygon), key=lambda p: p.area)
                k = ms.curvature_along(ms.resample_ring(np.asarray(largest.exterior.coords), step))
                k = k[np.isfinite(k)]
                self._cache[key] = (float(k.mean()), float(k.max()), float(k.min())) if k.size else (math.nan,) * 3
            except ValueError:
                self._cache[key] = (math.nan,) * 3
        return self._cache[key]

    # ------------------------------------------------------------------ other classes
    @property
    def classes(self) -> ClassIndex:
        """The annotation classes on this image (and registered onto it)."""
        if self._classes is None:
            self._classes = ClassIndex(self.image, self.images)
        return self._classes

    def other(self, label):
        """All of a class (e.g. ``ctx.other("Lamina propria")``) as one geometry in this image's µm."""
        return self.classes.geometry(label)

    def others(self, label) -> list[Polygon]:
        """The objects of a class (one polygon each), this object left out."""
        return [p for p in self.classes.polygons(label) if not same_object(p, self.polygon)]

    def inside(self, label) -> list[Polygon]:
        """The objects of a class lying in this object (their representative point inside), e.g. its nuclei."""
        return [p for p in self.classes.near(label, self.polygon)
                if not same_object(p, self.polygon) and self.polygon.contains(p.representative_point())]

    def area_of(self, label) -> float:
        """Area (µm²) of a class inside this object."""
        other = self.other(label)
        return float(other.intersection(self.polygon).area) if not other.is_empty else 0.0

    def nearest(self, label) -> float:
        """Distance (µm) from this object's outline to the nearest object of a class (0 if they touch; NaN if
        there is none)."""
        polys = self.others(label)
        if not polys:
            return math.nan
        return float(min(self.polygon.distance(p) for p in self.classes.near(label, self.polygon, 1e12)
                         if not same_object(p, self.polygon)))

    # ------------------------------------------------------------------ around it, and on other images
    def surroundings(self, width_um: float, gap_um: float = 0.0, only=None) -> "ObjectContext":
        """The band around the object, as an object of its own.

        The band runs from ``gap_um`` to ``gap_um + width_um`` outside the
        object's outer outline (its holes, e.g. a gland's lumen, are inside the
        object, not around it), cut to the image. Everything a context offers works
        on it: ``ring.pixels``, ``ring.mask``, ``ring.area_um2``, ``ring.glcm_features(1)``…

        Args:
            width_um: Width of the band (µm).
            gap_um: Distance kept between the object and the band (µm).
            only: Keep only the part of the band covered by this class (e.g. the stroma around a gland).

        Returns:
            ObjectContext: The band on the same image, with the same constants.
        """
        key = ("surroundings", float(width_um), float(gap_um), as_label(only) if only is not None else None)
        if key not in self._cache:
            filled = Polygon(max(ms.parts(self.polygon), key=lambda p: p.area).exterior) \
                if not isinstance(self.polygon, Polygon) else Polygon(self.polygon.exterior)
            inner = filled.buffer(gap_um, 64) if gap_um > 0 else filled
            band = filled.buffer(gap_um + width_um, 64).difference(inner)
            band = band.intersection(box(0, 0, *self.image.size_um))
            if only is not None:
                band = band.intersection(self.other(only))
            self._cache[key] = ObjectContext(band, self.image, self.constants, self.images, self._classes)
        return self._cache[key]

    def on(self, image) -> "ObjectContext | None":
        """The same object on another image of the project (e.g. the slide under a phase map).

        Args:
            image: An image type ("wsi", "phase", "brightfield"), an image name, or an ImageNode.

        Returns:
            ObjectContext | None: The object's outline moved onto that image (through
            both images' placements, so it lands on the same tissue), with the same
            constants; None if no such image covers the object. With several, the one
            covering most of the object; the measured image itself if it matches.
        """
        if image is self.image or image in (self.image.modality, self.image.name):
            return self
        candidates = [image] if not isinstance(image, str) else             [i for i in self.images if i is not self.image and image in (i.modality, i.name)]
        best, best_area = None, 0.0
        for other in candidates:
            moved = self._moved_to(other)
            overlap = moved.polygon.intersection(box(0, 0, *other.size_um)).area
            if overlap > best_area:
                best, best_area = moved, overlap
        return best

    def _moved_to(self, other) -> "ObjectContext":
        """This object on another image (one context per image, shared by every parameter)."""
        key = ("on", id(other))
        if key not in self._cache:
            m = np.linalg.inv(other.local_to_world()) @ self.image.local_to_world()   # this image's µm → other's
            moved = affinity.affine_transform(self.polygon, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]])
            self._cache[key] = ObjectContext(moved, other, self.constants, self.images)
        return self._cache[key]

    @cached_property
    def image_size_um(self) -> tuple[float, float]:
        """(width, height) of the measured image in µm."""
        return self.image.size_um

    # ------------------------------------------------------------------ pixels (HoloPath names)
    @cached_property
    def _raster(self):
        """(mask, crop, (y0, y1, x0, x1)) of the outline on the image's level-0 grid, trimmed to the mask."""
        px, py = self.image.pixel_size_um
        h, w = self.image.source.levels[0].shape[:2]
        minx, miny, maxx, maxy = self.polygon.bounds
        c0, r0 = max(0, int(math.floor(minx / px))), max(0, int(math.floor(miny / py)))
        c1, r1 = min(w, int(math.ceil(maxx / px)) + 1), min(h, int(math.ceil(maxy / py)) + 1)
        if c1 <= c0 or r1 <= r0:
            return np.zeros((0, 0), bool), np.zeros((0, 0)), (0, 0, 0, 0)
        mask = ms.rasterize(self.polygon, c0 * px, r0 * py, px, py, (r1 - r0, c1 - c0))
        if not mask.any():
            return np.zeros((0, 0), bool), np.zeros((0, 0)), (0, 0, 0, 0)
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        crop = read_native(self.image, r0 + y0, r0 + y1, c0 + x0, c0 + x1)
        return mask[y0:y1, x0:x1], crop, (r0 + y0, r0 + y1, c0 + x0, c0 + x1)

    @property
    def mask(self) -> np.ndarray:
        """Boolean object mask (its bounding box)."""
        return self._raster[0]

    @property
    def image_crop(self) -> np.ndarray:
        """Image pixels of the same bounding box (2D, or 3D for RGB)."""
        return self._raster[1]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of the mask in the full image (y1/x1 exclusive)."""
        return self._raster[2]

    @cached_property
    def pixels(self) -> np.ndarray:
        """Pixel values inside the object: 1D, or (N, C) for RGB."""
        return self.image_crop[self.mask]

    @property
    def area_px(self) -> int:
        """Number of pixels in the mask."""
        return int(self.mask.sum())

    @property
    def full_shape(self) -> tuple[int, int]:
        """(rows, cols) of the whole measured image at full resolution."""
        return tuple(self.image.source.levels[0].shape[:2])

    @cached_property
    def touches_image_edge(self) -> bool:
        """True if the object reaches the image border (it is clipped by the field of view)."""
        if self.area_px == 0:
            return True
        y0, y1, x0, x1 = self.bbox
        h, w = self.full_shape
        return y0 <= 0 or x0 <= 0 or y1 >= h or x1 >= w

    @cached_property
    def max_radius_from_center(self) -> float:
        """Distance (px) from the image centre to the object's farthest pixel."""
        if self.area_px == 0:
            return 0.0
        h, w = self.full_shape
        ys, xs = np.where(self.mask)
        d2 = (ys + self.bbox[0] - (h - 1) / 2.0) ** 2 + (xs + self.bbox[2] - (w - 1) / 2.0) ** 2
        return float(np.sqrt(d2.max()))

    @cached_property
    def equivalent_diameter_px(self) -> float:
        """Equivalent diameter of the pixel mask (px), as HoloPath."""
        return ms.equivalent_diameter_px(self.mask)

    @cached_property
    def major_axis_px(self) -> float:
        """Major axis of the pixel mask's ellipse (px), as HoloPath."""
        return ms.major_axis_px(self.mask)

    @cached_property
    def minor_axis_px(self) -> float:
        """Minor axis of the pixel mask's ellipse (px), as HoloPath."""
        return ms.minor_axis_px(self.mask)

    @cached_property
    def perimeter_px(self) -> float:
        """Perimeter of the smoothed pixel contour (px), as HoloPath."""
        return ms.perimeter_px(self.mask)

    @cached_property
    def eccentricity(self) -> float:
        """Eccentricity of the pixel mask (HoloPath's; the outline's is ``eccentricity_outline``)."""
        return ms.eccentricity_px(self.mask)

    @cached_property
    def curvature(self) -> tuple[float, float, float]:
        """(mean, max, min) curvature of the pixel contour (1/px), as HoloPath."""
        return ms.curvature_px(self.mask)

    def glcm_features(self, distance: int = 1) -> dict:
        """Haralick texture features at a neighbour distance (cached per distance)."""
        key = ("glcm", int(distance))
        if key not in self._cache:
            self._cache[key] = ms.glcm_features(self.mask, self._grey_crop, distance)
        return self._cache[key]

    @property
    def glcm(self) -> dict:
        """Texture features at neighbour distance 1 (HoloPath's): dict of energy, entropy, contrast…"""
        return self.glcm_features(1)

    @cached_property
    def _grey_crop(self) -> np.ndarray:
        crop = self.image_crop
        return crop if crop.ndim == 2 else crop[..., :3].astype(np.float64).mean(axis=2)

    @cached_property
    def regionprops(self):
        """skimage regionprops of the mask (for user scripts), or None."""
        from skimage.measure import label, regionprops
        props = regionprops(label(self.mask.astype(np.uint8)))
        return props[0] if props else None

    def scattering(self, thickness_um: float, wavelength_nm: float | None = None, medium_index: float = 1.0) -> dict:
        """Scattering parameters of the object's phase (``measures.scattering_parameters``; cached).

        Args:
            thickness_um: Thickness of the section.
            wavelength_nm: Light wavelength for μs′ and g; 0 = the one the importer recorded;
                None = not needed (μs and lₛ only).
            medium_index: Refractive index n₀ of the medium around the section.

        Raises:
            ValueError: If a wavelength is needed and not known.
        """
        wavelength = _wavelength(self.image, wavelength_nm)
        key = ("scattering", float(thickness_um), wavelength, float(medium_index))
        if key not in self._cache:
            self._cache[key] = ms.scattering_parameters(self._grey_crop, self.mask, self.pixel_size_um,
                                                        wavelength / 1000.0 if wavelength else None,
                                                        thickness_um, medium_index)
        return self._cache[key]

    def fluctuation_image(self, degree: int = 2, window: int = 3) -> np.ndarray:
        """Phase fluctuation image of the detrended phase (cached per pair of values)."""
        key = ("pfi", int(degree), int(window))
        if key not in self._cache:
            detrended = ms.detrend_phase(self._grey_crop, self.mask, int(degree))
            self._cache[key] = ms.phase_fluctuation_image(detrended, int(window))
        return self._cache[key]

    def correlation_length_for(self, degree: int = 2, window: int = 3) -> float:
        """Correlation length l_c (µm) of the fluctuation image for these values."""
        key = ("lc", int(degree), int(window))
        if key not in self._cache:
            self._cache[key] = ms.correlation_length(self.fluctuation_image(degree, window), self.mask,
                                                     self.pixel_size_um)
        return self._cache[key]

    @property
    def pfi(self) -> np.ndarray:
        """Fluctuation image at HoloPath's values (detrend degree 2, window 3)."""
        return self.fluctuation_image(2, 3)

    @property
    def correlation_length_um(self) -> float:
        """Correlation length l_c (µm) at HoloPath's values."""
        return self.correlation_length_for(2, 3)

    @property
    def disorder_strength_um(self) -> float:
        """Disorder strength L_d (µm) at HoloPath's values (n = 1.38)."""
        return ms.disorder_strength(self.pfi, self.correlation_length_um, self.mask)

    @property
    def phase_fluctuation_mean(self) -> float:
        """Mean of the fluctuation image inside the object (HoloPath's values)."""
        return float(np.mean(self.pfi[self.mask])) if self.mask.any() else 0.0

    # ------------------------------------------------------------------ any single-value image
    def relative_fluctuation_image(self, window: int = 3) -> np.ndarray:
        """Local variance of the pixel values over ``window``² px, divided by the object's mean value².

        For positive quantities (OD, brightness, dry mass): unlike the phase fluctuation image (local
        variance / *local* mean² of the detrended phase, whose values hover around zero) the denominator
        is the object's mean, which keeps it finite and independent of how dark or dense it is overall.
        """
        key = ("rel_pfi", int(window))
        if key not in self._cache:
            self._cache[key] = ms.relative_local_variance(self._grey_crop, self.mask, int(window))
        return self._cache[key]

    def structure_length(self, degree: int = 2) -> float:
        """Typical size (µm) of the structures inside the object: the autocorrelation length of its pixel
        values, with a fitted 2D polynomial (slow shading) removed first."""
        key = ("structure_lc", int(degree))
        if key not in self._cache:
            detrended = ms.detrend_phase(self._grey_crop.astype(np.float64), self.mask, int(degree))
            self._cache[key] = ms.correlation_length(detrended, self.mask, self.pixel_size_um)
        return self._cache[key]

    # ------------------------------------------------------------------ brightness (brightfield photos)
    def brightness_image(self) -> np.ndarray:
        """Brightness of the bounding box (2D), 0 = black … 1 = the brightest value the file can hold,
        colour channels averaged (``optical.brightness``)."""
        if "brightness" not in self._cache:
            src = self.image.source
            self._cache["brightness"] = op.brightness(self.image_crop, src.dtype, src.channels)
        return self._cache["brightness"]

    def brightness(self) -> np.ndarray:
        """Brightness of the pixels inside the object (1D, 0–1)."""
        return self.brightness_image()[self.mask]

    def brightness_fluctuation_image(self, window: int = 3) -> np.ndarray:
        """Local variance of the brightness over ``window``² px, divided by the object's mean brightness²."""
        key = ("bright_pfi", int(window))
        if key not in self._cache:
            self._cache[key] = ms.relative_local_variance(self.brightness_image(), self.mask, int(window))
        return self._cache[key]

    def brightness_correlation_length(self, degree: int = 2) -> float:
        """Typical size (µm) of the structures inside the object: the autocorrelation length of its
        brightness, with a fitted 2D polynomial (uneven illumination) removed first."""
        key = ("bright_lc", int(degree))
        if key not in self._cache:
            detrended = ms.detrend_phase(self.brightness_image(), self.mask, int(degree))
            self._cache[key] = ms.correlation_length(detrended, self.mask, self.pixel_size_um)
        return self._cache[key]



class ClassContext:
    """One class in one image (class scope): its objects together, e.g. the glands of an image.

    Args:
        image: The ImageNode.
        label: The class: (group, subgroup), subgroup "" = the whole group.
        constants: The current parameter's constants.
        images: The project's images (classes registered onto this one).
        classes: A shared ``ClassIndex`` (default: made here).
        library: The parameter library (``object_values`` measures its object parameters).
        object_constants: {parameter id: constants} the profile uses for object parameters.
        value_cache: {(parameter id, constants, outline WKB): value} measured already (the object rows of the
            same measurement), so summaries do not measure the same objects again.
    """

    def __init__(self, image, label, constants: dict | None = None, images=(), classes=None, library=None,
                 object_constants: dict | None = None, value_cache: dict | None = None) -> None:
        self.image = image
        self.label = as_label(label)
        self.constants = dict(constants or {})
        self.images = list(images)
        self.pixel_size_um = float(image.pixel_size_um[0])
        self._classes = classes
        self.library = library
        self.object_constants = dict(object_constants or {})
        self.value_cache = value_cache if value_cache is not None else {}
        self._cache: dict = {}

    @property
    def settings(self) -> dict:
        return self.constants

    @property
    def classes(self) -> ClassIndex:
        """The annotation classes on this image (and registered onto it)."""
        if self._classes is None:
            self._classes = ClassIndex(self.image, self.images)
        return self._classes

    @property
    def objects(self) -> list[Polygon]:
        """The class's objects on this image (one polygon each)."""
        return self.classes.polygons(self.label)

    @property
    def count(self) -> int:
        """Number of objects of the class on the image."""
        return len(self.objects)

    @property
    def area_um2(self) -> float:
        """Area (µm²) the class covers on the image."""
        return float(self.classes.geometry(self.label).intersection(box(0, 0, *self.image.size_um)).area)

    @property
    def image_area_um2(self) -> float:
        """Area of the whole image (µm²)."""
        w, h = self.image.size_um
        return float(w * h)

    def other(self, label):
        """All of another class as one geometry, cut to the image."""
        return self.classes.geometry(label).intersection(box(0, 0, *self.image.size_um))

    def area_of(self, label) -> float:
        """Area (µm²) of another class on the image."""
        return float(self.other(label).area)

    def object_values(self, uid: str, constants: dict | None = None, exclude_edge: bool = False,
                      min_area_um2: float = 0.0) -> np.ndarray:
        """One object parameter measured on each of the class's objects (cached).

        Args:
            uid: The object parameter's id (e.g. "object:any:best_alignment_metric").
            constants: Its constants (default: the profile's, else its defaults).
            exclude_edge: Leave out objects touching the image border (cut by the field of view).
            min_area_um2: Leave out objects smaller than this.

        Returns:
            np.ndarray: The values (NaN where the parameter could not be computed).

        Raises:
            ValueError: If the parameter is unknown or the image has no version of its type.
        """
        from modalytics.parameters.registry import ParameterLibrary
        library = self.library or ParameterLibrary()
        spec = library.get(uid)
        if spec is None or spec.scope != "object":
            raise ValueError(f"'{uid}' is not an object parameter of the library")
        resolved = spec.resolve(constants if constants is not None else self.object_constants.get(uid))
        key = ("values", uid, tuple(sorted(resolved.items())), bool(exclude_edge), float(min_area_um2))
        if key not in self._cache:
            target = self.image.as_type(spec.image_type)
            if target is None:
                raise ValueError(f"{self.image.name} has no {spec.image_type} version")
            eps = self.pixel_size_um
            frame = box(eps, eps, self.image.size_um[0] - eps, self.image.size_um[1] - eps)
            out = []
            frozen = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in resolved.items()))
            import shapely
            for poly in self.objects:
                if poly.area < min_area_um2 or (exclude_edge and not frame.contains(poly)):
                    continue
                known = (uid, frozen, shapely.to_wkb(poly))
                if known in self.value_cache:                       # measured already (an object row)
                    out.append(float(self.value_cache[known]))
                    continue
                ctx = ObjectContext(poly, target, resolved, self.images, self.classes)
                try:
                    out.append(float(spec.compute(ctx)))
                except Exception:
                    out.append(math.nan)
                self.value_cache[known] = out[-1]
            self._cache[key] = np.asarray(out, dtype=np.float64)
        return self._cache[key]


class ImageContext:
    """A whole image (HoloPath's ``ImageParamContext`` names).

    Args:
        image: The ImageNode.
        constants: The current parameter's constants.

    Raises:
        ValueError: (on first pixel access) if the image is too large to load whole.
    """

    def __init__(self, image, constants: dict | None = None, images=(), classes=None) -> None:
        self.image = image
        self.images = list(images)
        self._classes = classes
        self.constants = dict(constants or {})
        self.pixel_size_um = float(image.pixel_size_um[0])
        self.phase_to_height_factor_nm = image.metadata.get("phase_to_height_nm")
        h, w = image.source.levels[0].shape[:2]
        self.height_px, self.width_px = int(h), int(w)
        self._cache: dict = {}

    @property
    def settings(self) -> dict:
        """The constants under their older name (scripts written before the rename)."""
        return self.constants

    # ------------------------------------------------------------------ classes (no pixels needed)
    @property
    def classes(self) -> ClassIndex:
        """The annotation classes on this image (and registered onto it)."""
        if self._classes is None:
            self._classes = ClassIndex(self.image, self.images)
        return self._classes

    @property
    def image_area_um2(self) -> float:
        """Area of the whole image (µm²)."""
        return self.width_px * self.height_px * self.pixel_size_um ** 2

    def other(self, label):
        """All of a class as one geometry, cut to the image."""
        return self.classes.geometry(label).intersection(box(0, 0, *self.image.size_um))

    def others(self, label) -> list[Polygon]:
        """The objects of a class on the image."""
        return self.classes.polygons(label)

    def area_of(self, label) -> float:
        """Area (µm²) of a class on the image."""
        return float(self.other(label).area)

    @cached_property
    def pixels(self) -> np.ndarray:
        """All level-0 pixels as stored: 2D, or (H, W, C) for colour."""
        if self.height_px * self.width_px > MAX_IMAGE_PIXELS:
            raise ValueError(f"{self.image.name} is too large for whole-image parameters "
                             f"({self.width_px}×{self.height_px} px)")
        return read_native(self.image, 0, self.height_px, 0, self.width_px)

    @cached_property
    def phase(self) -> np.ndarray:
        """All level-0 pixels (2D; RGB is averaged to grey)."""
        data = self.pixels
        return data if data.ndim == 2 else data[..., :3].astype(np.float64).mean(axis=2)

    @cached_property
    def flat(self) -> np.ndarray:
        """All pixel values as a 1D float array."""
        return self.phase.ravel().astype(np.float64)

    @cached_property
    def height_flat(self):
        """Per-pixel optical height (nm), or None without a conversion factor."""
        if self.phase_to_height_factor_nm is None:
            return None
        return self.flat * float(self.phase_to_height_factor_nm)

    def glcm_features(self, distance: int = 1) -> dict:
        """Haralick texture features of the whole image at a neighbour distance (cached)."""
        key = ("glcm", int(distance))
        if key not in self._cache:
            self._cache[key] = ms.glcm_features(np.ones(self.phase.shape, bool), self.phase, distance)
        return self._cache[key]

    @property
    def glcm(self) -> dict:
        """Texture features at neighbour distance 1 (HoloPath's)."""
        return self.glcm_features(1)

    def scattering(self, thickness_um: float, wavelength_nm: float | None = None, medium_index: float = 1.0) -> dict:
        """Scattering parameters of the whole phase map (``measures.scattering_parameters``; cached).

        Raises:
            ValueError: If a wavelength is needed and not known.
        """
        wavelength = _wavelength(self.image, wavelength_nm)
        key = ("scattering", float(thickness_um), wavelength, float(medium_index))
        if key not in self._cache:
            self._cache[key] = ms.scattering_parameters(self.phase, np.ones(self.phase.shape, bool),
                                                        self.pixel_size_um, wavelength / 1000.0 if wavelength else None,
                                                        thickness_um, medium_index)
        return self._cache[key]

    def fluctuation_image(self, window: int = 3) -> np.ndarray:
        """Phase fluctuation image of the whole (not detrended) phase, as HoloPath."""
        key = ("pfi", int(window))
        if key not in self._cache:
            self._cache[key] = ms.phase_fluctuation_image(self.phase, int(window))
        return self._cache[key]

    def correlation_length_for(self, window: int = 3) -> float:
        """Correlation length l_c (µm) of the whole image for a fluctuation window."""
        key = ("lc", int(window))
        if key not in self._cache:
            self._cache[key] = ms.correlation_length(self.fluctuation_image(window), None, self.pixel_size_um)
        return self._cache[key]

    @property
    def pfi(self) -> np.ndarray:
        """Fluctuation image of the whole phase at HoloPath's window (3)."""
        return self.fluctuation_image(3)

    @property
    def correlation_length_um(self) -> float:
        """Correlation length l_c (µm) at HoloPath's window."""
        return self.correlation_length_for(3)

    @property
    def disorder_strength_um(self) -> float:
        """Disorder strength L_d (µm) at HoloPath's values (n = 1.38)."""
        return ms.disorder_strength(self.pfi, self.correlation_length_um, None)

    @property
    def phase_fluctuation_mean(self) -> float:
        """Mean of the whole fluctuation image."""
        return float(np.mean(self.pfi))


    # ------------------------------------------------------------------ brightness (brightfield photos)
    def brightness_image(self) -> np.ndarray:
        """Brightness of the whole image (2D), 0 = black … 1 = the brightest value the file can hold."""
        if "brightness" not in self._cache:
            src = self.image.source
            self._cache["brightness"] = op.brightness(self.pixels, src.dtype, src.channels)
        return self._cache["brightness"]

    def brightness(self) -> np.ndarray:
        """Brightness of every pixel (1D, 0–1)."""
        return self.brightness_image().ravel()

    def brightness_fluctuation_image(self, window: int = 3) -> np.ndarray:
        """Local variance of the brightness over ``window``² px, divided by the image's mean brightness²."""
        key = ("bright_pfi", int(window))
        if key not in self._cache:
            b = self.brightness_image()
            self._cache[key] = ms.relative_local_variance(b, np.ones(b.shape, bool), int(window))
        return self._cache[key]

    def brightness_correlation_length(self) -> float:
        """Typical size (µm) of the structures across the whole image (brightness autocorrelation length)."""
        if "bright_lc" not in self._cache:
            self._cache["bright_lc"] = ms.correlation_length(self.brightness_image(), None, self.pixel_size_um)
        return self._cache["bright_lc"]


## What the parameter editor's helper list offers (Settings › Parameters › Code) ##

# Instance attributes (introspection cannot see their documentation)
ATTRIBUTE_HELP = {
    "label": "The class measured: (group, subgroup); subgroup \"\" = the whole group.",
    "polygon": "The object's outline: a shapely Polygon in the measured image's local µm (y down).",
    "image": "The ImageNode measured on: .name, .modality, .pixel_size_um, .size_um, .metadata.",
    "constants": "This parameter's constants, e.g. ctx.constants[\"window\"] (declared on the Constants tab).",
    "images": "The project's other images (ctx.on picks one of them).",
    "pixel_size_um": "Size of one pixel of the measured image, in µm.",
    "phase_to_height_factor_nm": "Radians → optical height (nm) factor of the phase map, or None.",
    "height_px": "Image height in pixels.",
    "width_px": "Image width in pixels.",
}

OBJECT_HELP = {
    "Outline (µm, exact)": ["polygon", "area_um2", "perimeter_um", "equivalent_diameter_um", "axes_um",
                            "major_axis_um", "minor_axis_um", "eccentricity_outline", "curvature_um", "moments",
                            "image_size_um"],
    "Outline: hull, calipers, complexity": ["hull", "solidity", "convexity", "feret", "holes", "fractal_dimension",
                                            "best_alignment_metric"],
    "Pixels under the object": ["pixels", "mask", "image_crop", "bbox", "area_px", "pixel_size_um", "full_shape",
                                "touches_image_edge", "max_radius_from_center"],
    "Texture": ["glcm_features", "glcm"],
    "Phase disorder": ["fluctuation_image", "correlation_length_for", "pfi", "correlation_length_um",
                       "disorder_strength_um", "phase_fluctuation_mean"],
    "Light scattering (phase)": ["scattering"],
    "Any single-value image (OD, dry mass…)": ["relative_fluctuation_image", "structure_length"],
    "Brightness (brightfield photos)": ["brightness", "brightness_image", "brightness_fluctuation_image",
                                        "brightness_correlation_length"],
    "HoloPath pixel shape": ["equivalent_diameter_px", "major_axis_px", "minor_axis_px", "perimeter_px",
                             "eccentricity", "curvature", "regionprops"],
    "Around it, and on other images": ["surroundings", "on"],
    "Other classes": ["other", "others", "inside", "nearest", "area_of", "classes"],
    "This parameter": ["constants", "image"],
}

IMAGE_HELP = {
    "Pixels": ["phase", "flat", "pixels", "height_flat", "pixel_size_um", "height_px", "width_px",
               "phase_to_height_factor_nm"],
    "Texture": ["glcm_features", "glcm"],
    "Phase disorder": ["fluctuation_image", "correlation_length_for", "pfi", "correlation_length_um",
                       "disorder_strength_um", "phase_fluctuation_mean"],
    "Brightness (brightfield photos)": ["brightness", "brightness_image", "brightness_fluctuation_image",
                                        "brightness_correlation_length"],
    "Light scattering (phase)": ["scattering"],
    "Other classes": ["other", "others", "area_of", "image_area_um2", "classes"],
    "This parameter": ["constants", "image"],
}

CLASS_HELP = {
    "This class": ["label", "objects", "count", "area_um2", "image_area_um2", "object_values"],
    "Other classes": ["other", "area_of", "classes"],
    "This parameter": ["constants", "image"],
}

# Functions usable in compute() besides ctx: (name, text inserted, import it needs, help)
LIBRARY_HELP = [
    ("np.mean(x)", "np.mean()", "import numpy as np", "Mean of an array (np.median, np.std, np.var, np.sum…)."),
    ("np.percentile(x, q)", "np.percentile(, 90)", "import numpy as np", "q-th percentile of an array."),
    ("skew(x)", "skew()", "from scipy.stats import skew", "Skewness of the values (scipy)."),
    ("kurtosis(x)", "kurtosis()", "from scipy.stats import kurtosis", "Excess (Fisher) kurtosis (scipy)."),
    ("math.pi", "math.pi", "import math", "π and the rest of Python's math module."),
    ("ms.polygon_moments(polygon)", "ms.polygon_moments()", "from modalytics.parameters import measures as ms",
     "(area, cx, cy, mu20, mu02, mu11) of any polygon."),
    ("ms.glcm_features(mask, image, distance)", "ms.glcm_features()",
     "from modalytics.parameters import measures as ms", "Haralick texture features of a masked region."),
    ("ms.detrend_phase(phase, mask, degree)", "ms.detrend_phase()",
     "from modalytics.parameters import measures as ms", "Phase minus a fitted 2D polynomial."),
    ("ms.phase_fluctuation_image(phase, window)", "ms.phase_fluctuation_image()",
     "from modalytics.parameters import measures as ms", "Local variance / mean² image."),
]


def help_entries(scope: str) -> list[tuple[str, str, str, str]]:
    """The helper list for a scope: (group, label, text to insert, help), in display order.

    Args:
        scope: "object" or "image" (anything else gets the object list).

    Returns:
        list[tuple[str, str, str, str]]
    """
    import inspect
    cls, groups = {"image": (ImageContext, IMAGE_HELP), "class": (ClassContext, CLASS_HELP)}.get(
        scope, (ObjectContext, OBJECT_HELP))
    out = []
    for group, names in groups.items():
        for name in names:
            member = inspect.getattr_static(cls, name, None)
            if name in ATTRIBUTE_HELP:
                out.append((group, f"ctx.{name}", f"ctx.{name}", ATTRIBUTE_HELP[name]))
                continue
            if member is None:
                raise AttributeError(f"{cls.__name__} has no '{name}' (update the helper list)")
            func = member.func if isinstance(member, cached_property) else \
                member.fget if isinstance(member, property) else member
            doc = inspect.getdoc(func) or ""
            if callable(member) and not isinstance(member, (property, cached_property)):
                params = [p for p in inspect.signature(member).parameters.values() if p.name != "self"]
                sig = ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                                for p in params)
                out.append((group, f"ctx.{name}({sig})", f"ctx.{name}()", doc.split("\n\n")[0]))
            else:
                out.append((group, f"ctx.{name}", f"ctx.{name}", doc.split("\n\n")[0]))
    out += [("Libraries", label, insert, f"{text}\nNeeds: {imp}") for label, insert, imp, text in LIBRARY_HELP]
    return out
