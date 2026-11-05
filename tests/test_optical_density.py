"""The general slide and brightfield parameters, through the real parameter scripts and processing step:
optical density (an OD-map view of a slide, made by the OD map step) from Beer–Lambert on synthetic images
whose OD is known, and brightness (brightfield photos)."""
import math

import numpy as np
import pytest
from shapely.geometry import Point, box

from modalytics.core.model import ImageNode
from modalytics.io.sources import ArraySource
from modalytics.parameters import engine, measures as ms
from modalytics.parameters import optical as op
from modalytics.parameters.registry import ParameterLibrary

# Uniform test regions make skewness / kurtosis meaningless (scipy warns, as it would for phase maps)
pytestmark = pytest.mark.filterwarnings("ignore:Precision loss occurred in moment calculation")

GLASS = np.array([238.0, 232.0, 244.0])          # a scanner's white balance: not 255, not neutral


def _image(od_map: np.ndarray, image_type: str = "wsi", px: float = 0.5, glass=GLASS, background=None) -> ImageNode:
    """RGB image whose optical density is ``od_map`` (Beer–Lambert: I = I_white · 10^-OD), as 8-bit, with
    the OD-map view the OD map processing step makes (background estimated, or the fixed ``background``)."""
    from modalytics.processing.registry import ProcessingLibrary
    from modalytics.processing.run import make_view
    rgb = np.clip(np.round(glass[None, None, :] * 10.0 ** (-od_map[..., None])), 0, 255).astype(np.uint8)
    node = ImageNode("synthetic", ArraySource(rgb), (px, px), image_type)
    if image_type == "wsi":
        options = {"background": "Fixed value", "background_value": background} if background else {}
        node.views.append(make_view(node, ProcessingLibrary().get("od:od_map"), options, measure=True))
    return node


def _measure(scope: str, image_type: str, node, polygon=None) -> dict:
    lib = ParameterLibrary()
    chosen = {uid: {} for uid, s in lib.specs.items() if s.scope == scope and s.image_type == image_type}
    choices, missing = engine.select(lib, chosen, scope)
    assert not missing and choices
    values, problems = (engine.measure_object(polygon, node, choices) if scope == "object"
                        else engine.measure_image(node, choices))
    assert not problems, problems
    return {uid.rsplit(":", 1)[1]: v for uid, v in values.items()}


# ---------------------------------------------------------------- the conversion itself
def test_optical_density_inverts_beer_lambert():
    od = np.linspace(0.0, 1.5, 200).reshape(10, 20)
    rgb = GLASS * 10.0 ** (-od[..., None])                    # exact (not rounded)
    back = op.optical_density(rgb, GLASS, channels=3)
    assert np.allclose(back, od, atol=1e-12)
    assert op.optical_density(np.array([300.0, 300, 300]), GLASS, 3) == 0.0       # brighter than glass: 0
    assert np.isfinite(op.optical_density(np.zeros(3), GLASS, 3))                 # black: finite
    assert np.allclose(op.optical_density(np.array([100.0]), np.array([200.0]), 1), math.log10(2))


def test_white_level_is_the_glass():
    od = np.zeros((400, 600))
    od[100:300, 150:450] = 0.6                                                    # tissue in the middle
    node = _image(od)
    assert np.allclose(op.estimate_white(node), GLASS, atol=0.5)
    assert np.allclose(op.white_level(node, 200.0), [200.0, 200.0, 200.0])         # a fixed value wins


# ---------------------------------------------------------------- the parameters, on a known object
def test_object_parameters_on_a_uniform_disc():
    image_type = "od"
    od = np.zeros((300, 300))
    yy, xx = np.mgrid[0:300, 0:300]
    od[(yy - 150) ** 2 + (xx - 150) ** 2 <= 80 ** 2] = 0.5                      # a disc of OD 0.5 on glass
    node = _image(od)                                                           # a slide, with its OD map
    disc = Point(75.25, 75.25).buffer(30, quad_segs=64)                        # inside the disc (µm; 0.5 µm/px)
    v = _measure("object", image_type, node, disc)
    assert v["mean"] == pytest.approx(0.5, abs=0.003)                          # 8-bit rounding only
    assert v["std"] < 0.003 and v["variance"] < 1e-5
    assert v["integrated_od"] == pytest.approx(0.5 * disc.area, rel=0.02)     # OD × µm²
    assert v["uniformity"] > 1e4                                               # flat
    assert set(v) >= {"skewness", "kurtosis", "bimodality", "contrast", "correlation", "dissimilarity",
                      "energy", "entropy", "homogeneity", "correlation_length", "fluctuation_mean"}


def test_texture_and_two_populations():
    rng = np.random.default_rng(4)
    od = np.zeros((300, 300))
    od[50:250, 50:250] = 0.2 + 0.05 * rng.standard_normal((200, 200))           # pale cytoplasm
    yy, xx = np.mgrid[0:300, 0:300]
    for cy, cx in ((100, 100), (100, 200), (200, 100), (200, 200), (150, 150)):  # dark nuclei
        od[(yy - cy) ** 2 + (xx - cx) ** 2 <= 18 ** 2] = 1.0
    node = _image(od)
    region = box(25.5, 25.5, 124.5, 124.5)                                      # inside the square (µm)
    v = _measure("object", "od", node, region)
    assert v["bimodality"] > 0.555                                              # nuclei vs cytoplasm
    assert v["skewness"] > 0
    # GLCM on OD = the HoloPath GLCM routine applied to the OD image
    from modalytics.parameters.context import ObjectContext
    ctx = ObjectContext(region, node)                                          # the slide's RGB pixels
    expect = ms.glcm_features(ctx.mask, op.optical_density(ctx.image_crop, op.estimate_white(node), 3), 1)
    assert v["contrast"] == pytest.approx(expect["contrast"]) and v["entropy"] == pytest.approx(expect["entropy"])
    assert 0 < v["correlation_length"] < 50


def test_grey_slides_too():
    od = np.full((100, 100), 0.3)
    od[:10] = 0.0                                                               # a strip of empty slide
    grey = np.clip(np.round(200.0 * 10.0 ** (-od)), 0, 255).astype(np.uint8)
    from modalytics.processing.registry import ProcessingLibrary
    from modalytics.processing.run import make_view
    node = ImageNode("grey", ArraySource(grey), (1.0, 1.0), "wsi")
    node.views.append(make_view(node, ProcessingLibrary().get("od:od_map"), measure=True))
    v = _measure("object", "od", node, box(20, 20, 80, 80))
    assert v["mean"] == pytest.approx(0.3, abs=0.003)


def test_fixed_background_option():
    od = np.full((100, 100), 0.4)                                               # no empty slide in view
    auto = _measure("object", "od", _image(od), box(10, 10, 40, 40))["mean"]
    fixed = _measure("object", "od", _image(od, background=float(GLASS.mean())), box(10, 10, 40, 40))["mean"]
    assert auto == pytest.approx(0.0, abs=0.003)                                # estimated white = the tissue
    assert fixed == pytest.approx(0.4, abs=0.01)                                # the constant fixes it


# ---------------------------------------------------------------- brightfield photos: brightness
def _photo(values: np.ndarray, px: float = 0.5) -> ImageNode:
    return ImageNode("photo", ArraySource(values), (px, px), "brightfield")


def test_brightness_conversion():
    assert op.brightness(np.array([0, 51, 255], np.uint8), np.dtype(np.uint8), 1).tolist() == [0.0, 0.2, 1.0]
    assert op.brightness(np.array([65535], np.uint16), np.dtype(np.uint16), 1)[0] == 1.0
    rgb = np.array([[[30, 60, 90]]], np.uint8)
    assert op.brightness(rgb, np.dtype(np.uint8), 3)[0, 0] == pytest.approx(60 / 255)    # channels averaged


@pytest.mark.parametrize("dtype, top", [(np.uint8, 255), (np.uint16, 65535)])
def test_brightness_parameters_on_a_photo_without_background(dtype, top):
    """A field covered by the sample: no empty background is needed."""
    img = np.full((300, 300, 3), int(0.40 * top), dtype)
    yy, xx = np.mgrid[0:300, 0:300]
    img[(yy - 150) ** 2 + (xx - 150) ** 2 <= 40 ** 2] = int(0.10 * top)            # a dark nucleus
    node = _photo(img)
    v = _measure("object", "brightfield", node, box(20, 20, 130, 130))             # the whole field (µm)
    nucleus = np.pi * 40 ** 2 / (220 * 220)
    assert v["mean"] == pytest.approx(0.40 - 0.30 * nucleus, abs=0.002)
    assert v["bimodality"] > 0.555 and v["skewness"] < 0                           # a tail of dark pixels
    assert set(v) == {"mean", "variance", "skewness", "kurtosis", "std", "bimodality", "uniformity", "energy",
                      "entropy", "contrast", "homogeneity", "dissimilarity", "correlation", "correlation_length",
                      "fluctuation_mean"}
    flat = _measure("object", "brightfield", node, box(5, 5, 30, 30))              # cytoplasm only
    assert flat["mean"] == pytest.approx(0.40, abs=0.002) and flat["std"] < 1e-9


def test_brightness_whole_image_and_grey_photos():
    grey = np.full((200, 300), 200, np.uint8)
    grey[50:150, 100:200] = 50
    v = _measure("image", "brightfield", _photo(grey, px=0.25))
    frac = 100 * 100 / (200 * 300)
    assert v["brightness_mean"] == pytest.approx((200 - 150 * frac) / 255)
    assert v["brightness_min"] == pytest.approx(50 / 255) and v["brightness_max"] == pytest.approx(200 / 255)
    assert v["brightness_median"] == pytest.approx(200 / 255)
    assert v["bimodality"] > 0.555 and len(v) == 18


def test_brightness_texture_and_size():
    from scipy import ndimage
    rng = np.random.default_rng(5)
    lengths = []
    for sigma in (2.0, 8.0):
        field = ndimage.gaussian_filter(rng.standard_normal((400, 400)), sigma)
        img = np.clip(0.5 + 0.12 * field / field.std(), 0, 1)
        node = _photo((img * 255).round().astype(np.uint8))
        v = _measure("object", "brightfield", node, box(10, 10, 190, 190))
        lengths.append(v["correlation_length"])
        assert v["fluctuation_mean"] > 0
    assert lengths[1] / lengths[0] == pytest.approx(4.0, rel=0.35)              # ∝ the structure size


def test_correlation_length_follows_the_size_of_the_structures():
    """Blobs of a known size (smoothed noise): the OD correlation length grows in proportion."""
    from scipy import ndimage
    rng = np.random.default_rng(11)
    lengths = []
    for sigma in (2.0, 4.0, 8.0):
        field = ndimage.gaussian_filter(rng.standard_normal((400, 400)), sigma)
        od = 0.4 + 0.15 * field / field.std()
        node = _image(np.clip(od, 0.02, None), "wsi", px=0.5)
        lengths.append(_measure("object", "od", node, box(10, 10, 190, 190))["correlation_length"])
    assert lengths[0] < lengths[1] < lengths[2]
    assert lengths[2] / lengths[0] == pytest.approx(4.0, rel=0.35)              # ∝ the structure size
