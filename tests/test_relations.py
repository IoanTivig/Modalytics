"""Parameters that relate annotation classes: the class index (own masks, registered images, views), per-object
relations, whole-image ratios and the per-class summaries, with known answers."""
import math
import os

import numpy as np
import pytest
from shapely.geometry import Point, box

from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode, MaskNode
from modalytics.core.settings import Setting
from modalytics.io.sources import ArraySource
from modalytics.parameters import engine
from modalytics.parameters.classes import ClassIndex, as_label
from modalytics.parameters.context import ClassContext, ObjectContext
from modalytics.parameters.registry import ParameterLibrary

GLANDS = [Point(30, 30).buffer(10, 256), Point(70, 30).buffer(10, 256), Point(50, 70).buffer(12, 256)]


def add_mask(image, group, subgroup, geometry):
    mask = MaskNode(geometry=geometry, group=group, subgroup=subgroup)
    mask.parent = image
    image.children.append(mask)
    return mask


def project(px=0.5, size=200):
    """A 100 × 100 µm phase map: three glands (two Normal, one Dysplasia), stroma around them, a Region."""
    rng = np.random.default_rng(0)
    image = ImageNode("p", ArraySource(rng.normal(0, 0.1, (size, size)).astype(np.float32)), (px, px), "phase",
                      Placement(0, 0), metadata={"wavelength_nm": 633.0})
    add_mask(image, "Gland", "Normal", GLANDS[:2])
    add_mask(image, "Gland", "Dysplasia high grade", [GLANDS[2]])
    region = box(10, 10, 90, 90)
    stroma = region
    for g in GLANDS:
        stroma = stroma.difference(g)
    add_mask(image, "Lamina propria", "Normal", stroma)
    add_mask(image, "Region", "", region)
    return image, region, stroma


@pytest.fixture(scope="module")
def lib():
    return ParameterLibrary()


def measure(lib, uid, polygon, image, constants=None, images=()):
    values, problems = engine.measure_object(polygon, image, [(lib.get(uid), lib.get(uid).resolve(constants))],
                                             images)
    assert not problems, problems
    return values[uid]


def test_labels_and_matching():
    assert as_label("Gland › Normal") == ("Gland", "Normal") and as_label(["Gland", ""]) == ("Gland", "")
    assert as_label("Lamina propria") == ("Lamina propria", "")
    s = Setting.from_dict({"key": "c", "label": "Class", "kind": "label", "default": ["Gland", ""]})
    assert s.clean(["Gland", "Normal"]) == ("Gland", "Normal") and s.clean("Gland") == ("Gland", "")
    image, _r, _s = project()
    index = ClassIndex(image)
    assert len(index.polygons(("Gland", ""))) == 3                           # the group: every subgroup
    assert len(index.polygons(("Gland", "Normal"))) == 2
    assert len(index.polygons("Gland › Dysplasia high grade")) == 1
    assert index.polygons(("Nucleus", "")) == []


def test_per_object_relations(lib):
    image, region, stroma = project()
    gland_area = sum(g.area for g in GLANDS)
    assert measure(lib, "object:any:stroma_ratio", region, image) == pytest.approx(
        100 * stroma.area / (stroma.area + gland_area))
    assert measure(lib, "object:any:count_inside", region, image) == 3
    assert measure(lib, "object:any:density_inside", region, image) == pytest.approx(3 / (region.area * 1e-6))
    assert measure(lib, "object:any:class_area_fraction", region, image) == pytest.approx(
        100 * gland_area / region.area)
    # Gland-to-gland: the two Normal ones are 40 µm apart centre to centre, radius 10 → 20 µm
    assert measure(lib, "object:any:nearest_distance", GLANDS[0], image, {"other": ["Gland", "Normal"]}) == \
        pytest.approx(20.0, abs=0.01)
    # A gland in stroma: its 5 µm band is all stroma (the neighbours are farther away)
    assert measure(lib, "object:any:band_fraction", GLANDS[0], image, {"width_um": 5.0}) == pytest.approx(100, abs=0.2)
    # N/C: a 20 µm cell with a 10 µm nucleus → 100 / (400 − 100)
    cell, nucleus = box(0, 0, 20, 20), box(5, 5, 15, 15)
    add_mask(image, "Nucleus", "", nucleus)
    assert measure(lib, "object:any:inside_area_ratio", cell, image) == pytest.approx(100 / 300)


def test_stroma_band_scattering_uses_only_the_stroma(lib):
    image, _region, stroma = project()
    ctx = ObjectContext(GLANDS[0], image)
    band = ctx.surroundings(5.0, only=("Lamina propria", ""))
    assert band.area_um2 == pytest.approx(GLANDS[0].buffer(5, 256).difference(GLANDS[0]).intersection(stroma).area,
                                          rel=1e-3)
    g = measure(lib, "object:phase:band_anisotropy", GLANDS[0], image)
    mu = measure(lib, "object:phase:band_scattering_coefficient", GLANDS[0], image)
    assert 0 < g < 1 and mu > 0


def test_whole_image_and_per_class(lib):
    image, region, stroma = project()
    choices, _ = engine.select(lib, {u: {} for u, s in lib.specs.items()
                                     if s.scope == "image" and s.category == "relations"}, "image")
    values, problems = engine.measure_image(image, choices)
    assert not problems
    gland_area = sum(g.area for g in GLANDS)
    assert values["image:any:stroma_ratio"] == pytest.approx(100 * stroma.area / (stroma.area + gland_area))
    assert values["image:any:class_density"] == pytest.approx(3 / (100 * 100 * 1e-6))
    choices, _ = engine.select(lib, {u: {} for u in lib.specs if u.startswith("class:")}, "class")
    values, problems = engine.measure_class(image, ("Gland", ""), choices, library=lib)
    assert not problems, problems
    assert values["class:any:count"] == 3
    assert values["class:any:area_fraction"] == pytest.approx(100 * gland_area / 10000)
    assert values["class:any:mean_area"] == pytest.approx(gland_area / 3, rel=1e-6)
    assert values["class:any:mean_bam"] == pytest.approx(0, abs=1e-5)          # circles: perfectly regular
    assert values["class:any:regularity_index"] == 1.0 and values["class:any:bam_entropy"] == 0.0


def test_border_objects_left_out_of_summaries(lib):
    image, _r, _s = project()
    add_mask(image, "Gland", "Normal", [box(0, 40, 6, 60)])                    # cut by the left border
    ctx = ClassContext(image, ("Gland", ""), library=lib)
    assert ctx.object_values("object:any:area").size == 4
    assert ctx.object_values("object:any:area", exclude_edge=True).size == 3
    assert ctx.object_values("object:any:area", min_area_um2=400).size == 1    # only the radius-12 gland (452 µm²)


def test_classes_found_through_registration_and_shared_with_views(lib):
    # The stroma is outlined on a second image placed 20 µm to the right; measured on the first
    image, region, _stroma = project()
    for m in list(image.masks):
        if m.group == "Lamina propria":
            image.children.remove(m)
    other = ImageNode("slide", ArraySource(np.zeros((400, 400), np.uint8)), (0.5, 0.5), "brightfield",
                      Placement(20, 0))
    add_mask(other, "Lamina propria", "", box(-10, -10, 70, 90))               # in the slide's own µm
    found = ClassIndex(image, [image, other]).geometry(("Lamina propria", ""))
    assert found.bounds == pytest.approx((10, -10, 90, 90))                     # moved by the placements
    assert measure(lib, "object:any:stroma_ratio", region, image, images=[image, other]) > 0
    # A measured view of the image has no masks of its own: the image's classes are used
    from modalytics.processing import run
    from modalytics.processing.registry import ProcessingLibrary
    image2, _r2, _s2 = project()
    view = run.make_view(image2, ProcessingLibrary().get("filters:gaussian_blur"), {"sigma_um": 1.0}, measure=True)
    image2.views.append(view)
    image2.metadata["measure_original"] = False                                 # only the view is included
    assert image2.as_type("phase") is not image2                                # measured through the view
    g = measure(lib, "object:phase:band_anisotropy", GLANDS[0], image2)
    assert 0 < g < 1


def test_every_included_version_measured_the_others_labelled(lib):
    """Original and a filtered view both included: plain results from the original, the view's labelled."""
    from modalytics.parameters import engine
    from modalytics.processing import run
    from modalytics.processing.registry import ProcessingLibrary
    image, _r, _s = project()
    view = run.make_view(image, ProcessingLibrary().get("filters:gaussian_blur"), {"sigma_um": 1.0}, measure=True)
    view.name = "Blurred"
    image.views.append(view)
    assert image.original_measured and image.measured_view("phase") is None   # the original gives the plain ones
    assert [label for label, _n in image.extra_versions("phase")] == ["Blurred"]
    choices, _m = engine.select(lib, {"object:phase:band_anisotropy": {}}, "object")
    values, _p = engine.measure_object(GLANDS[0], image, choices)
    plain, blurred = values["object:phase:band_anisotropy"], values["object:phase:band_anisotropy@Blurred"]
    assert 0 < plain < 1 and 0 < blurred < 1 and plain != blurred
    assert set(engine.measure_object(GLANDS[0], image, choices, extras=False)[0]) == {"object:phase:band_anisotropy"}
    # The original left out: the view gives the plain results, nothing labelled
    image.metadata["measure_original"] = False
    assert image.measured_view("phase") is view and image.extra_versions("phase") == []
    only, _p = engine.measure_object(GLANDS[0], image, choices)
    assert only == {"object:phase:band_anisotropy": pytest.approx(blurred)}
    # Nothing of the type included: skipped, said
    view.chain = dict(view.chain, measure=False)
    none, problems = engine.measure_object(GLANDS[0], image, choices)
    assert "left out of the parameters" in problems["object:phase:band_anisotropy"]


def test_profile_keeps_class_constants(tmp_path):
    from modalytics.profiles.model import Profile
    p = Profile("Lab", images=["phase"], parameters={"object:any:stroma_ratio": {"stroma": ("Lamina propria", ""),
                                                                                 "tumour": ("Gland", "Dysplasia")}})
    again = Profile.from_yaml(p.to_yaml())
    spec = ParameterLibrary().get("object:any:stroma_ratio")
    resolved = spec.resolve(again.parameters["object:any:stroma_ratio"])
    assert resolved == {"stroma": ("Lamina propria", ""), "tumour": ("Gland", "Dysplasia")}


def test_real_glands_and_stroma():
    """HoloPath's colon phase map: its gland and lamina propria outlines, related."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from reference_objects import reference_image_and_objects
    found = reference_image_and_objects()
    if found is None:
        pytest.skip("HoloPath test data not available")
    image, _objects = found
    lib = ParameterLibrary()
    choices, _ = engine.select(lib, {u: {} for u in ("image:any:stroma_ratio", "image:any:class_density")}, "image")
    values, problems = engine.measure_image(image, choices)
    assert not problems and 0 < values["image:any:stroma_ratio"] < 100
    gland = next(p for m in image.masks if m.group == "Gland" for p in m.geometry.geoms)
    v = measure(lib, "object:any:band_fraction", gland, image, {"width_um": 10.0})
    assert 0 <= v <= 100
    counts, _ = engine.measure_class(image, ("Gland", ""), engine.select(lib, {"class:any:count": {}}, "class")[0])
    assert counts["class:any:count"] == sum(len(m.geometry.geoms) for m in image.masks if m.group == "Gland")
    assert math.isfinite(values["image:any:class_density"])
