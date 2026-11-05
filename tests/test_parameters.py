"""Parameter library: loading, the outline maths against exact geometry, and the port against HoloPath."""
import json
import math
import os

from pathlib import Path

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode
from modalytics.io.sources import ArraySource
from modalytics.parameters import engine
from modalytics.parameters import measures as ms
from modalytics.parameters.context import ObjectContext
from modalytics.parameters.registry import ParameterLibrary

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "holopath_parameters.json")
SHAPE = ("area", "perimeter", "length_avg", "length_max", "length_min", "circularity", "eccentricity",
         "ellipticity", "curvature_avg", "curvature_max", "curvature_min")


def _phase_image(h=200, w=240, px=0.5):
    data = np.random.default_rng(3).normal(1.0, 0.3, (h, w)).astype(np.float32)
    return ImageNode("phase", ArraySource(data), (px, px), "phase", Placement(0, 0),
                     metadata={"phase_to_height_nm": 200.0})


# 1. The library -------------------------------------------------------------------------------

def test_builtin_library_loads_completely():
    lib = ParameterLibrary()
    # HoloPath's 60 (31 per object, 29 per image) + brightfield (15 + 18), OD map (16), OPD map (3), dry mass (2)
    # + shape (12), scattering (4 + 4), stain (5), height (4), refractive index (3), feature maps (2)
    # + relations between classes (7 + 2 per object, 3 per image) and per-class summaries (7)
    assert lib.errors == {} and len(lib.of_scope("object")) == 31 + 15 + 16 + 3 + 2 + 12 + 4 + 5 + 4 + 3 + 2 + 9
    assert len(lib.of_scope("image")) == 29 + 18 + 4 + 3 and len(lib.of_scope("class")) == 7
    area, ds = lib.get("object:any:area"), lib.get("object:phase:disorder_strength")
    assert area.accepts("wsi") and area.accepts("phase") and not ds.accepts("wsi")
    assert lib.get("object:area") is area                       # an id from before image types still works
    assert len(lib.of_type("object", "*")) == 31 and len(lib.of_type("object", "phase")) == 25
    assert len(lib.of_type("image", "*")) == 8 and len(lib.of_type("image", "phase")) == 28
    assert len(lib.of_type("object", "od")) == 16 and len(lib.of_type("object", "brightfield")) == 15
    assert len(lib.of_type("object", "wsi")) == 0                  # slides are measured through their OD map
    assert len(lib.of_type("image", "brightfield")) == 18 and len(lib.of_type("image", "wsi")) == 0
    assert [s.key for s in ds.constants] == ["degree", "window", "refractive_index"]
    assert ds.resolve({"degree": "1", "window": 99}) == {"degree": 1, "window": 15, "refractive_index": 1.38}
    assert "object:phase:in_inner_circle" not in lib.defaults() and "object:any:area" in lib.defaults()


def test_user_scripts_add_replace_and_fail_politely(tmp_path):
    user = tmp_path / "user"
    user.mkdir()
    # A HoloPath-format script (no IMAGE_TYPES / CONSTANTS) still loads: phase maps by default
    (user / "aspect_ratio.py").write_text(
        "KEY = 'aspect_ratio'\nLABEL = 'Aspect ratio'\n\ndef compute(ctx):\n"
        "    return ctx.major_axis_px / ctx.minor_axis_px\n", encoding="utf-8")
    (user / "area.py").write_text("KEY = 'area'\nIMAGE_TYPES = '*'\ndef compute(ctx):\n    return 1.0\n",
                                  encoding="utf-8")
    (user / "broken.py").write_text("KEY = 'broken'\nraise RuntimeError('oops')\n", encoding="utf-8")
    (user / "bad_setting.py").write_text("KEY = 'bad'\nCONSTANTS = [{'key': 'x'}]\ndef compute(ctx):\n    return 0\n",
                                         encoding="utf-8")
    (user / "EXAMPLE_skipped.py").write_text("KEY = ", encoding="utf-8")
    lib = ParameterLibrary(user=user)
    assert lib.get("object:phase:aspect_ratio").image_type == "phase"
    assert lib.get("object:any:area").source == "user" and "object:any:area" in lib.overridden
    assert sorted(os.path.basename(p) for p in lib.errors) == ["bad_setting.py", "broken.py"]
    assert "oops" in next(e for p, e in lib.errors.items() if p.endswith("broken.py"))
    assert set(lib.user_files()) == {"object:phase:aspect_ratio", "object:any:area"}


def test_engine_skips_other_image_types_and_survives_failures(tmp_path):
    (tmp_path / "boom.py").write_text("KEY = 'boom'\nIMAGE_TYPES = ['*']\ndef compute(ctx):\n    return 1 / 0\n",
                                      encoding="utf-8")
    lib = ParameterLibrary(user=tmp_path)
    choices, missing = engine.select(lib, {"object:area": {}, "object:mean": {}, "object:boom": {},
                                           "object:gone": {}})
    assert missing == ["object:gone"]
    rgb = ImageNode("slide", ArraySource(np.zeros((50, 50, 3), np.uint8)), (1, 1), "wsi")
    values, problems = engine.measure_object(box(5, 5, 25, 15), rgb, choices)
    assert values["object:any:area"] == pytest.approx(200) and "object:phase:mean" not in values
    assert problems["object:phase:mean"] == "not for wsi images" and math.isnan(values["object:any:boom"])
    assert "ZeroDivisionError" in problems["object:any:boom"]


# 2. Outline maths against exact geometry -----------------------------------------------------

def test_moments_and_axes_are_exact():
    ring = box(0, 0, 40, 20).difference(box(10, 5, 20, 15))          # rectangle with a hole
    a, cx, cy, mu20, mu02, mu11 = ms.polygon_moments(ring)
    assert a == pytest.approx(700) and cy == pytest.approx(10)
    assert cx == pytest.approx((800 * 20 - 100 * 15) / 700)
    ell = affinity.rotate(affinity.scale(Point(0, 0).buffer(1.0, 1024), 30, 12), 40)
    major, minor = ms.ellipse_axes(*ms.polygon_moments(ell)[3:])
    assert (major, minor) == (pytest.approx(60, rel=1e-4), pytest.approx(24, rel=1e-4))
    assert ms.eccentricity_from_moments(*ms.polygon_moments(ell)[3:]) == pytest.approx(math.sqrt(1 - (12 / 30) ** 2),
                                                                                       rel=1e-4)


def test_outline_curvature_matches_the_exact_values():
    img = _phase_image()
    circle = ObjectContext(Point(60, 50).buffer(20, 512), img)
    assert circle.curvature_um(0.0) == pytest.approx((0.05, 0.05, 0.05), rel=0.01)       # 1 / r
    a, b = 30.0, 14.0
    ell = ObjectContext(affinity.scale(Point(60, 50).buffer(1.0, 1024), a, b), img)
    mean, kmax, kmin = ell.curvature_um(0.0)
    t = np.linspace(0, 2 * np.pi, 200001)[:-1]
    perim = np.mean(np.hypot(a * np.sin(t), b * np.cos(t))) * 2 * np.pi
    assert mean == pytest.approx(2 * np.pi / perim, rel=0.03)       # ∫κ ds = 2π for a convex curve
    assert kmin == pytest.approx(b / a ** 2, rel=0.02) and kmax == pytest.approx(a / b ** 2, rel=0.12)


def test_rasterize_takes_exactly_the_pixel_centres_inside():
    poly = Polygon([(3.2, 1.1), (17.9, 4.4), (12.3, 18.6), (2.1, 11.7)]).difference(Point(9, 9).buffer(3))
    mask = ms.rasterize(poly, 0.0, 0.0, 0.5, 0.5, (40, 40))
    expected = np.array([[poly.contains(Point((c + 0.5) * 0.5, (r + 0.5) * 0.5)) for c in range(40)] for r in range(40)])
    assert (mask == expected).all()
    disk = ms.rasterize(Point(50, 50).buffer(30, 512), 0, 0, 0.25, 0.25, (400, 400))
    assert disk.sum() * 0.25 ** 2 == pytest.approx(math.pi * 900, rel=0.002)            # no outward bias


# 3. The port against HoloPath's own output -------------------------------------------------------

@pytest.fixture(scope="module")
def reference():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from reference_objects import label_images, reference_image_and_objects
    found = reference_image_and_objects()
    if found is None or not os.path.exists(REFERENCE):
        pytest.skip("HoloPath test data or tests/data/holopath_parameters.json not available")
    image, objects = found
    with open(REFERENCE, encoding="utf-8") as f:
        ref = json.load(f)
    labels, _ = label_images(image, objects)
    import hashlib
    assert hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()[:16] == ref["labels_sha256_16"], \
        "the test masks changed: rerun scripts/make_parameter_reference.py"
    return image, objects, ref


def holopath_set(lib) -> dict:
    """HoloPath's parameters: the scripts ported from it (their docstring says so)."""
    return {u: {} for u, s in lib.specs.items() if "Ported from HoloPath" in Path(s.path).read_text(encoding="utf-8")}


def test_pixel_parameters_equal_holopath(reference):
    image, objects, ref = reference
    lib = ParameterLibrary()
    choices, _ = engine.select(lib, holopath_set(lib), "object")
    for name, poly in objects:
        values, problems = engine.measure_object(poly, image, choices)
        assert problems == {}
        for uid, v in values.items():
            key = uid.split(":")[-1]
            if key in SHAPE:
                continue
            assert v == pytest.approx(ref["objects"][name][key], rel=1e-9, abs=1e-12, nan_ok=True), (name, key)


def test_holopath_pixel_vocabulary_is_ported_exactly(reference):
    """HoloPath user scripts read ctx.area_px, ctx.perimeter_px… : same values as HoloPath's context."""
    image, objects, ref = reference
    px = image.pixel_size_um[0]
    for name, poly in objects:
        ctx, hp = ObjectContext(poly, image), ref["objects"][name]
        got = {"area": ctx.area_px * px * px, "perimeter": ctx.perimeter_px * px,
               "length_avg": ctx.equivalent_diameter_px * px, "length_max": ctx.major_axis_px * px,
               "length_min": ctx.minor_axis_px * px, "eccentricity": ctx.eccentricity,
               "curvature_avg": ctx.curvature[0], "curvature_max": ctx.curvature[1], "curvature_min": ctx.curvature[2]}
        for key, v in got.items():
            assert v == pytest.approx(hp[key], rel=1e-9, abs=1e-12, nan_ok=True), (name, key)


def test_outline_shape_agrees_with_holopath_within_pixel_error(reference):
    """Outline (exact) vs pixel-mask shape values, for objects inside the image.

    Measured on the HoloPath test project: area and axes within 0.5 %; HoloPath's
    smoothed pixel perimeter is 0.5–2.5 % shorter than the outline, so its
    circularity is up to 5 % higher. Curvature is not compared: HoloPath's follows
    the pixel staircase (on a circle of radius 25 µm it reports 0.10–0.20 µm⁻¹
    instead of 0.04; see test_outline_curvature_matches_the_exact_values).
    """
    image, objects, ref = reference
    lib = ParameterLibrary()
    choices, _ = engine.select(lib, {f"object:{k}": {} for k in SHAPE})
    tolerance = {"area": 0.005, "length_avg": 0.005, "length_max": 0.005, "length_min": 0.006,
                 "perimeter": 0.03, "circularity": 0.06}
    for name, poly in objects:
        if ref["objects"][name]["in_image"] != 1.0:
            continue                                    # cut by the border: the outline keeps the whole object
        values, _ = engine.measure_object(poly, image, choices)
        by_key = {uid.split(":")[-1]: v for uid, v in values.items()}
        for key, tol in tolerance.items():
            assert by_key[key] == pytest.approx(ref["objects"][name][key], rel=tol), (name, key)


def test_whole_image_parameters_equal_holopath(reference):
    image, _objects, ref = reference
    lib = ParameterLibrary()
    choices, _ = engine.select(lib, holopath_set(lib), "image")
    values, problems = engine.measure_image(image, choices)
    assert problems == {} and len(values) == 29
    for uid, v in values.items():
        assert v == pytest.approx(ref["image"][uid.split(":")[-1]], rel=1e-9, abs=1e-12), uid


def test_constants_change_the_result():
    img = _phase_image()
    lib = ParameterLibrary()
    poly = Point(60, 50).buffer(25, 256)
    base, _ = engine.measure_object(poly, img, engine.select(lib, {"object:disorder_strength": {}})[0])
    other, _ = engine.measure_object(poly, img, engine.select(lib, {"object:disorder_strength": {"window": 5}})[0])
    n15, _ = engine.measure_object(poly, img, engine.select(lib, {"object:disorder_strength":
                                                                   {"refractive_index": 1.5}})[0])
    ds = "object:phase:disorder_strength"
    assert other[ds] != base[ds]
    assert n15[ds] == pytest.approx(base[ds] * (1.5 / 1.38) ** 2)


def test_helper_list_matches_the_context_and_documents_everything():
    from modalytics.parameters.context import help_entries
    for scope in ("object", "image"):
        entries = help_entries(scope)                # raises if a listed member no longer exists
        assert all(doc.strip() for _g, _l, _i, doc in entries), [l for _g, l, _i, d in entries if not d.strip()]
        assert any(label.startswith("ctx.constants") for _g, label, _i, _d in entries)


# 4. Around the object, and on another image ----------------------------------------------

def test_surroundings_is_the_band_around_the_outer_outline():
    img = _phase_image(h=200, w=240, px=0.5)                   # 120 × 100 µm
    gland = Point(60, 50).buffer(10, 512).difference(Point(60, 50).buffer(4, 512))   # with a lumen
    ctx = ObjectContext(gland, img)
    ring = ctx.surroundings(5)
    assert ring.area_um2 == pytest.approx(math.pi * (15 ** 2 - 10 ** 2), rel=1e-3)   # lumen not included
    assert ring.area_px * 0.25 == pytest.approx(ring.area_um2, rel=0.02)
    assert ring.polygon.intersection(Point(60, 50).buffer(10, 512)).area < 1e-6     # nothing of the gland
    far = ctx.surroundings(5, gap_um=3)
    assert far.area_um2 == pytest.approx(math.pi * (18 ** 2 - 13 ** 2), rel=1e-3)
    # Cut by the image edge
    edge = ObjectContext(Point(5, 50).buffer(4, 256), img).surroundings(6)
    assert edge.area_um2 == pytest.approx(Point(5, 50).buffer(10, 256).difference(Point(5, 50).buffer(4, 256))
                                          .intersection(box(0, 0, 120, 100)).area, rel=1e-3)
    assert ring is ctx.surroundings(5)                          # computed once per object


def test_on_finds_the_same_tissue_on_another_image():
    # A: identity placement. B: shifted, rotated, other pixel size; its pixels hold their own world x
    a = _phase_image(h=200, w=240, px=0.5)
    pb = Placement(20, -10, 30.0, 1.0, False)
    hb, wb, sb = 300, 300, 0.4
    ys, xs = np.mgrid[0:hb, 0:wb]
    world = pb.local_to_world() @ np.stack([(xs.ravel() + 0.5) * sb, (ys.ravel() + 0.5) * sb, np.ones(xs.size)])
    b = ImageNode("b", ArraySource(world[0].reshape(hb, wb).astype(np.float32)), (sb, sb), "phase2", pb)
    cx, cy, _ = pb.local_to_world() @ [60, 60, 1]               # the middle of B, in world µm (inside A too)
    gland = Point(cx, cy).buffer(8, 256)
    ctx = ObjectContext(gland, a, images=[a, b])
    other = ctx.on("phase2")
    assert other is not None and other.image is b and other.area_um2 == pytest.approx(gland.area, rel=1e-9)
    assert other.polygon.centroid.distance(Point(60, 60)) < 1e-9
    assert float(other.pixels.mean()) == pytest.approx(cx, abs=0.05)   # B's pixels there hold world x = cx
    assert ctx.on("phase") is ctx and ctx.on("wsi") is None and ctx.on(b) is other


def test_a_parameter_can_use_another_image_through_the_engine(tmp_path):
    (tmp_path / "pair.py").write_text(
        "KEY = 'pair'\nIMAGE_TYPES = ['*']\n\ndef compute(ctx):\n"
        "    slide = ctx.on('wsi')\n    return float(slide.pixels.mean()) if slide else float('nan')\n",
        encoding="utf-8")
    lib = ParameterLibrary(user=tmp_path)
    phase = _phase_image()
    slide = ImageNode("slide", ArraySource(np.full((100, 120), 7.0, np.float32)), (1, 1), "wsi")
    choices, _ = engine.select(lib, {"object:pair": {}})
    v, problems = engine.measure_object(Point(30, 30).buffer(5), phase, choices, images=[phase, slide])
    assert problems == {} and v["object:any:pair"] == pytest.approx(7.0)


# 5. Shape completion, scattering, the new image types, references ------------------------------

def _measure(uid, polygon, image, constants=None):
    lib = ParameterLibrary()
    spec = lib.get(uid)
    ctx = ObjectContext(polygon, image)
    ctx.constants = spec.resolve(constants)
    return spec.compute(ctx)


def test_shape_completion_known_answers():
    img = _phase_image(400, 400, 0.5)
    rect = box(20, 20, 120, 40)                                       # 100 × 20 µm
    assert _measure("object:any:feret_max", rect, img) == pytest.approx(math.hypot(100, 20))
    assert _measure("object:any:feret_min", rect, img) == pytest.approx(20)
    assert _measure("object:any:feret_ratio", rect, img) == pytest.approx(math.hypot(100, 20) / 20)
    assert _measure("object:any:rectangularity", rect, img) == pytest.approx(1.0)
    assert _measure("object:any:solidity", rect, img) == pytest.approx(1.0)
    circle = Point(100, 100).buffer(40, 512)
    assert _measure("object:any:roundness", circle, img) == pytest.approx(1.0, abs=1e-3)
    assert _measure("object:any:fractal_dimension", circle, img) == pytest.approx(1.0, abs=0.03)
    assert _measure("object:any:best_alignment_metric", circle, img) == pytest.approx(0.0, abs=1e-6)
    ellipse = affinity.scale(Point(100, 100).buffer(20, 512), 3.0, 1.0)
    assert _measure("object:any:best_alignment_metric", ellipse, img) == pytest.approx(0.0, abs=1e-5)
    l_shape = Polygon([(20, 20), (120, 20), (120, 40), (40, 40), (40, 120), (20, 120)])
    assert _measure("object:any:solidity", l_shape, img) == pytest.approx(l_shape.area / l_shape.convex_hull.area)
    assert _measure("object:any:best_alignment_metric", l_shape, img) > 0.005
    gland = Point(100, 100).buffer(40, 256).difference(Point(100, 100).buffer(20, 256))
    assert _measure("object:any:holes", gland, img) == 1
    assert _measure("object:any:hole_fraction", gland, img) == pytest.approx(0.25, abs=0.01)


def test_scattering_phase_theorem():
    """μs = 2⟨1 − cos Δφ⟩/L (≈ ⟨Δφ²⟩/L), lₛ = 1/μs, μs′ = ⟨|∇φ|²⟩/(2k²L), g = 1 − μs′/μs (Xu 2011)."""
    from scipy import ndimage
    rng = np.random.default_rng(1)
    phi = ndimage.gaussian_filter(rng.standard_normal((300, 300)), 4)
    phi = (0.15 * phi / phi.std()).astype(np.float32)
    img = ImageNode("p", ArraySource(phi), (0.25, 0.25), "phase", metadata={"wavelength_nm": 633.0})
    region = box(0, 0, 75, 75)
    inside = phi.astype(float)
    mu_s = _measure("object:phase:scattering_coefficient", region, img, {"thickness_um": 5.0})
    assert mu_s == pytest.approx(2 * np.mean(1 - np.cos(inside - inside.mean())) / 5.0 * 1000, rel=0.02)
    assert mu_s == pytest.approx(inside.var() / 5.0 * 1000, rel=0.02)                  # the small-phase form
    ls = _measure("object:phase:scattering_mean_free_path", region, img, {"thickness_um": 5.0})
    assert ls == pytest.approx(1000 / mu_s, rel=1e-6)
    gy, gx = np.gradient(inside, 0.25)
    k = 2 * math.pi / 0.633
    expected = np.mean(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2) / (2 * k ** 2) / 5.0 * 1000
    mu_r = _measure("object:phase:reduced_scattering_coefficient", region, img, {"thickness_um": 5.0})
    assert mu_r == pytest.approx(expected, rel=0.05)
    g = _measure("object:phase:anisotropy", region, img, {"thickness_um": 5.0})
    assert g == pytest.approx(1 - mu_r / mu_s, rel=1e-6) and 0 < g < 1
    no_lambda = ImageNode("p", ArraySource(phi), (0.25, 0.25), "phase")
    assert _measure("object:phase:scattering_coefficient", region, no_lambda) > 0       # μs needs no wavelength
    with pytest.raises(ValueError, match="wavelength"):
        _measure("object:phase:anisotropy", region, no_lambda)


def test_stain_height_refractive_index_parameters():
    stain = np.zeros((100, 100), np.float32)
    stain[:, :25], stain[:, 25:50], stain[:, 50:75] = 0.1, 0.3, 0.5         # negative, weak, moderate
    stain[:, 75:] = 0.8                                                        # strong
    img = ImageNode("s", ArraySource(stain), (1.0, 1.0), "stain")
    whole = box(0, 0, 100, 100)
    assert _measure("object:stain:h_score", whole, img) == pytest.approx(100 * (0.25 + 2 * 0.25 + 3 * 0.25))
    assert _measure("object:stain:positive_fraction", whole, img) == pytest.approx(75.0)
    assert _measure("object:stain:integrated", whole, img) == pytest.approx(stain.sum(), rel=1e-6)
    height = ImageNode("h", ArraySource(np.full((50, 50), 2.0, np.float32)), (0.5, 0.5), "height")
    assert _measure("object:height:volume", box(0, 0, 10, 10), height) == pytest.approx(2.0 * 100)
    ri = ImageNode("r", ArraySource(np.tile(np.array([1.36, 1.40], np.float32), (40, 20))), (1, 1), "ri")
    assert _measure("object:ri:variance", box(0, 0, 40, 40), ri) == pytest.approx(0.02 ** 2, rel=1e-3)


def test_every_parameter_has_a_description_and_well_formed_references():
    import re
    lib = ParameterLibrary()
    for uid, spec in lib.specs.items():
        assert len(spec.description) >= 20, uid
        for ref in spec.references:
            assert re.search(r"\(\d{4}\)\.", ref) and len(ref) > 40, (uid, ref)
        if spec.references:
            assert "References:" in spec.help_text()
    assert lib.get("object:phase:anisotropy").references[1].startswith("Xu M (2011)")
