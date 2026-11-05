"""The built-in processing steps: each runs on every image type it accepts, tile by tile equals the whole
image, and each method does what it says on images with a known answer."""
import math

import numpy as np
import pytest
from PIL import Image

from modalytics.core.model import ACQUIRED_TYPES, ImageNode
from modalytics.io.sources import ArraySource, Level, PixelSource
from modalytics.processing import run
from modalytics.processing.registry import ProcessingLibrary, Where
from modalytics.processing.types import MADE

STANDARD = {"H": np.array([0.650, 0.704, 0.286]), "E": np.array([0.072, 0.990, 0.105]),
            "DAB": np.array([0.268, 0.570, 0.776])}
WHITE = np.array([238.0, 240.0, 236.0])


class Pyramid(PixelSource):
    """A two-level pyramid with 64 px tiles (a slide in miniature)."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = [data, data[::2, ::2].copy()]
        self.dtype, self.channels = data.dtype, (data.shape[2] if data.ndim == 3 else 1)
        h, w = data.shape[:2]
        self.levels = [Level((h, w), (64, 64), (1.0, 1.0)),
                       Level((h // 2, w // 2), (64, 64), (h / (h // 2), w / (w // 2)))]

    def read_tile(self, level, ty, tx):
        return self.data[level][ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64]


def _unit(v):
    return v / np.linalg.norm(v)


def he_slide(size=256, seed=0, tint=(1.0, 1.0, 1.0)):
    """A synthetic H&E slide: nuclei (haematoxylin) in eosin tissue on empty glass, plus the stain amounts."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    tissue = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.35) ** 2
    e = 0.45 * tissue + 0.03 * rng.standard_normal((size, size)) * tissue
    h = np.zeros((size, size))
    for _ in range(25):
        cy, cx = rng.uniform(0.25, 0.75, 2) * size
        h += 0.9 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    h *= tissue
    e = e * (1 - np.clip(h / 0.9, 0, 1))                       # nuclei push the cytoplasm aside
    od = h[..., None] * _unit(STANDARD["H"]) + np.clip(e, 0, None)[..., None] * _unit(STANDARD["E"])
    rgb = WHITE * np.array(tint) * 10 ** (-od)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8), h, np.clip(e, 0, None)


def phase_map(size=256, tilt=True, offset=0.4, seed=1):
    """Cells (2 rad bumps) on a medium at ``offset``, with a tilt and a bowl from the reference wave."""
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:size, 0:size] + 0.5) / size
    cells = np.zeros((size, size))
    for cy, cx in [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7), (0.25, 0.75)]:
        cells += 2.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 0.04 ** 2))
    background = offset + ((0.8 * (2 * xx - 1) + 0.5 * (2 * yy - 1) ** 2) if tilt else 0)
    noise = 0.01 * rng.standard_normal((size, size))
    return (cells + background + noise).astype(np.float32), cells


def image_of(kind):
    """A test image of each acquired type (with a wavelength for the phase steps)."""
    if kind == "phase":
        return ImageNode("p", Pyramid(phase_map()[0]), (0.25, 0.25), "phase", metadata={"wavelength_nm": 666.0})
    rgb = he_slide()[0]
    return ImageNode("s", Pyramid(rgb), (0.5, 0.5), kind)


def whole(source, level=0):
    gy, gx = source.levels[level].grid
    return np.concatenate([np.concatenate([source.read_tile(level, ty, tx) for tx in range(gx)], 1)
                           for ty in range(gy)], 0)


@pytest.fixture(scope="module")
def lib():
    return ProcessingLibrary(user="nonexistent-user-folder")


@pytest.fixture
def blank(tmp_path):
    """A blank (empty-field) picture with vignetting, and the vignetting itself."""
    yy, xx = (np.mgrid[0:256, 0:256] + 0.5) / 256
    vignette = 1.0 - 0.35 * ((2 * xx - 1) ** 2 + (2 * yy - 1) ** 2) / 2
    picture = np.clip(np.rint(WHITE * vignette[..., None]), 0, 255).astype(np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(picture).save(path)
    return str(path), vignette


def options_for(uid, tmp_path, blank_path):
    """Options a step needs to run on the test images (files, choices that need them)."""
    if uid == "filters:flat_field":
        return {"blank": blank_path}
    if uid == "filters:stain_normalisation":
        ref = tmp_path / "ref.png"
        Image.fromarray(he_slide(seed=5)[0]).save(ref)
        return {"reference": str(ref)}
    return {}


# ---------------------------------------------------------------- every step runs
def test_library(lib):
    assert not lib.errors
    assert len(lib.specs) == 29
    assert {"stain:colour_deconvolution", "height:height_map", "ri:ri_map", "filters:background_flatten",
            "filters:phase_unwrap", "feature:ridges", "filters:clahe", "filters:nl_means"} <= set(lib.specs)
    assert {s.output_type for s in lib.specs.values()} - {"same"} <= set(MADE)


@pytest.mark.parametrize("kind", ACQUIRED_TYPES)
def test_every_step_runs_on_every_type_it_accepts(lib, kind, tmp_path, blank):
    """Each step makes a view of the right type, finite, at both levels (tile by tile)."""
    image = image_of(kind)
    ran = []
    for spec in lib.for_type(kind):
        view = run.make_view(image, spec, options_for(spec.uid, tmp_path, blank[0]))
        for level in range(len(view.source.levels)):
            data = whole(view.source, level)
            assert data.shape[:2] == view.source.levels[level].shape, spec.uid
            assert np.isfinite(data).all(), spec.uid
        assert view.image_type == spec.output_for(kind)
        ran.append(spec.key)
    assert len(ran) >= 15


@pytest.mark.parametrize("uid", ["feature:gradient", "feature:dic", "feature:log_blobs", "feature:dog",
                                 "feature:local_std", "feature:local_entropy", "feature:ridges",
                                 "filters:bilateral", "filters:unsharp_mask", "filters:nl_means",
                                 "filters:rolling_ball", "filters:clahe", "filters:background_flatten",
                                 "filters:flat_field", "stain:colour_deconvolution",
                                 "filters:stain_normalisation"])
def test_tiles_equal_the_whole_image(lib, uid, tmp_path, blank):
    """Computed tile by tile (with margins, told where each tile is) = computed on the whole image at once."""
    spec = lib.get(uid)
    kind = "phase" if spec.accepts("phase") and uid != "filters:clahe" else "wsi"
    image = image_of(kind)
    options = spec.resolve(options_for(uid, tmp_path, blank[0]))
    if uid == "filters:rolling_ball":
        options["radius_um"] = 8.0                                 # 32 px: rolled on a copy shrunk 4 times
    if uid == "filters:clahe":
        options["region_um"] = 16.0                                # 32 px regions
    tiled = whole(run.make_view(image, spec, options).source)
    state = spec.prepare(image, options) if spec.prepare else {}
    data = image.source.data[0]
    extra = {"where": Where(0, 0, data.shape[:2], 0)} if spec.wants_where else {}
    at_once = spec.process(data, options, state, image.pixel_size_um[0], **extra)
    m = min(spec.margin_px(options, image.pixel_size_um[0]), 40)
    inner = (slice(m, -m or None), slice(m, -m or None))           # borders: the edges are mirrored differently
    a, b = tiled[inner].astype(float), np.asarray(at_once)[inner].astype(float)
    scale = float(np.ptp(b)) or 1.0
    assert np.abs(a - b).max() <= 1e-3 * scale + (1.0 if data.dtype == np.uint8 else 0), uid


# ---------------------------------------------------------------- phase maps
def test_background_flattening_and_offset(lib):
    data, cells = phase_map()
    image = ImageNode("p", Pyramid(data), (0.25, 0.25), "phase")
    flat = whole(run.make_view(image, lib.get("filters:background_flatten")).source)
    medium = cells < 0.01
    assert np.abs(np.median(data[medium])) > 0.3 and np.std(data[medium]) > 0.3       # before: offset + tilt
    assert np.abs(np.median(flat[medium])) < 0.01 and np.std(flat[medium]) < 0.02     # after: flat at 0
    assert np.abs((flat - cells)[~medium]).max() < 0.08                                # the cells are kept
    # Offset only (no tilt): the medium moves to 0, the cells keep their height
    data, cells = phase_map(tilt=False, offset=1.3)
    image = ImageNode("p", Pyramid(data), (0.25, 0.25), "phase")
    shifted = whole(run.make_view(image, lib.get("filters:background_offset")).source)
    assert abs(np.median(shifted[cells < 0.01])) < 0.01
    assert abs(shifted.max() - (data.max() - 1.3)) < 0.02


def test_phase_unwrapping(lib):
    yy, xx = np.mgrid[0:200, 0:200]
    true = (0.12 * xx + 0.05 * yy).astype(np.float32)                    # up to 34 rad: wraps many times
    wrapped = np.angle(np.exp(1j * true)).astype(np.float32)
    spec = lib.get("filters:phase_unwrap")
    out = whole(run.make_view(ImageNode("p", ArraySource(wrapped), (1, 1), "phase"), spec).source)
    diff = out - true
    assert np.ptp(diff) < 1e-3 and abs(diff.mean() / (2 * np.pi) - round(diff.mean() / (2 * np.pi))) < 1e-3
    again = whole(run.make_view(ImageNode("p", ArraySource(true), (1, 1), "phase"), spec).source)
    assert np.abs(again - true).max() < 1e-3                               # already unwrapped: unchanged


def test_height_and_refractive_index(lib):
    phase = np.full((64, 64), 2 * np.pi * 0.04 * 5.0 / 0.666, np.float32)   # 5 µm thick, Δn 0.04, λ 666 nm
    image = ImageNode("p", ArraySource(phase), (0.25, 0.25), "phase", metadata={"wavelength_nm": 666.0})
    height = run.make_view(image, lib.get("height:height_map"), {"delta_n": 0.04})
    assert height.image_type == "height" and height.unit == "µm"
    assert np.allclose(whole(height.source), 5.0, atol=1e-4)
    ri = run.make_view(image, lib.get("ri:ri_map"), {"thickness_um": 5.0, "n_medium": 1.335})
    assert ri.image_type == "ri" and np.allclose(whole(ri.source), 1.375, atol=1e-5)


# ---------------------------------------------------------------- colour images
@pytest.mark.parametrize("vectors", ["Standard (Ruifrok & Johnston)", "Estimated from the slide (Macenko)"])
def test_colour_deconvolution_recovers_the_stains(lib, vectors):
    rgb, h, e = he_slide()
    image = ImageNode("s", Pyramid(rgb), (0.5, 0.5), "wsi")
    spec = lib.get("stain:colour_deconvolution")
    got_h = whole(run.make_view(image, spec, {"stain": "Haematoxylin", "vectors": vectors}).source)
    got_e = whole(run.make_view(image, spec, {"stain": "Eosin", "vectors": vectors}).source)
    tissue = e > 0.2
    assert np.corrcoef(got_h[tissue], h[tissue])[0, 1] > 0.97
    assert np.corrcoef(got_e[tissue], e[tissue])[0, 1] > 0.9
    if vectors.startswith("Standard"):
        assert np.abs(got_h - h)[tissue].mean() < 0.03                      # the amounts themselves
    with pytest.raises(ValueError, match="not among the stains"):
        run.make_view(image, spec, {"stains": "H&E", "stain": "DAB"})


def test_stain_normalisation_brings_slides_together(lib, tmp_path):
    """Two scans of the same slide with different tints end up much closer after normalisation."""
    spec = lib.get("filters:stain_normalisation")
    a = he_slide(tint=(1.0, 1.0, 1.0))[0]
    b = he_slide(tint=(0.92, 1.0, 1.06))[0]
    for method, options in (("Macenko", {}), ("Reinhard", {"reference": str(tmp_path / "ref.png")})):
        Image.fromarray(he_slide(seed=3)[0]).save(tmp_path / "ref.png")
        na, nb = (whole(run.make_view(ImageNode("s", Pyramid(x), (0.5, 0.5), "wsi"), spec,
                                      {"method": method, **options}).source).astype(float) for x in (a, b))
        tissue = he_slide()[2] > 0.2
        before = np.abs(a.astype(float) - b)[tissue].mean()
        after = np.abs(na - nb)[tissue].mean()
        assert after < 0.5 * before, (method, before, after)


def test_flat_field_correction(lib, blank, tmp_path):
    path, vignette = blank
    rgb = he_slide()[0].astype(float) * vignette[..., None]
    image = ImageNode("s", Pyramid(np.clip(np.rint(rgb), 0, 255).astype(np.uint8)), (0.5, 0.5), "brightfield")
    glass = he_slide()[2] == 0
    corner = np.zeros_like(glass)
    corner[:30, :30] = corner[-30:, -30:] = True
    spec = lib.get("filters:flat_field")
    for options in ({"blank": path}, {"method": "Estimated from the image"}):
        out = whole(run.make_view(image, spec, options).source).astype(float)
        before = np.ptp([rgb[glass & corner].mean(), rgb[glass & ~corner].mean()])
        after = np.ptp([out[glass & corner].mean(), out[glass & ~corner].mean()])
        assert after < 0.15 * before, (options, before, after)


def test_white_balance_grey_and_tone(lib):
    rgb = he_slide(tint=(1.0, 0.95, 0.85))[0]
    image = ImageNode("s", Pyramid(rgb), (0.5, 0.5), "brightfield")
    glass = he_slide()[2] == 0
    out = whole(run.make_view(image, lib.get("filters:white_balance")).source).astype(float)
    assert np.ptp(out[glass].mean(axis=0)) < 2.0 < np.ptp(rgb[glass].mean(axis=0))   # neutral background
    green = whole(run.make_view(image, lib.get("filters:to_grey"), {"mode": "Green"}).source)
    assert green.ndim == 2 and np.array_equal(green, rgb[..., 1])
    tone = whole(run.make_view(image, lib.get("filters:intensity"),
                               {"points": "Fixed values", "black": 50, "white": 150}).source)
    assert tone[rgb == 50].max() == 0 and tone[rgb >= 150].min() == 255


# ---------------------------------------------------------------- filters and feature maps
def test_rolling_ball_removes_background_keeps_spots(lib):
    yy, xx = np.mgrid[0:256, 0:256]
    background = 40 + 60 * xx / 255
    spots = np.zeros((256, 256))
    for cy, cx in [(60, 60), (128, 200), (200, 90)]:
        spots += 80 * (((yy - cy) ** 2 + (xx - cx) ** 2) < 5 ** 2)
    image = ImageNode("b", Pyramid((background + spots).astype(np.float32)), (1, 1), "phase")
    out = whole(run.make_view(image, lib.get("filters:rolling_ball"), {"radius_um": 20.0}).source)
    assert np.abs(out[spots == 0]).max() < 3.0                            # the ramp is gone
    assert np.abs(out[spots > 0] - 80).mean() < 5.0                       # the spots stay


@pytest.mark.parametrize("uid", ["filters:bilateral", "filters:nl_means", "filters:tv_denoise"])
def test_denoising_keeps_edges(lib, uid):
    rng = np.random.default_rng(2)
    clean = np.where(np.arange(256)[None, :] < 128, 0.0, 1.0) * np.ones((256, 1))
    noisy = (clean + 0.1 * rng.standard_normal((256, 256))).astype(np.float32)
    image = ImageNode("p", Pyramid(noisy), (0.25, 0.25), "phase")
    out = whole(run.make_view(image, lib.get(uid)).source)
    left, right = out[20:-20, 20:100], out[20:-20, 156:-20]
    assert left.std() < 0.6 * noisy[20:-20, 20:100].std(), uid            # less noise
    assert right.mean() - left.mean() > 0.9, uid                         # the step is kept
    assert abs(out[:, 126].mean() - out[:, 129].mean()) > 0.5, uid       # and stays sharp


def test_feature_maps(lib):
    yy, xx = np.mgrid[0:256, 0:256]
    blob = np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 8.0 ** 2)).astype(np.float32)
    image = ImageNode("p", Pyramid(blob), (1, 1), "phase")

    def view(uid, **options):
        v = run.make_view(image, lib.get(uid), options)
        assert v.image_type == "feature"
        return whole(v.source)
    log = view("feature:log_blobs", radius_um=11.0, blobs="Bright")
    assert np.unravel_index(np.argmax(log), log.shape) == (128, 128)
    grad = view("feature:gradient", sigma_um=1.0)
    ring = grad[128, 128:].argmax()
    assert 6 <= ring <= 10 and grad[128, 128] < 1e-3                     # steepest at σ from the centre
    dic = view("feature:dic", angle=0.0, sigma_um=1.0)
    assert dic[128, 120] > 0 > dic[128, 136]                              # lit from one side
    dog = view("feature:dog", small_um=1.0, large_um=20.0)
    assert dog[128, 128] > 0 and dog[128, 128] == dog.max()
    std = view("feature:local_std", sigma_um=3.0)
    assert std[5, 5] < 1e-4 < std[128, 136]
    ent = view("feature:local_entropy", radius_um=3.0)
    assert ent[5, 5] == 0 and ent[128, 146] > 2.0
    line = np.zeros((256, 256), np.float32)
    line[:, 126:130] = 1.0
    ridge = whole(run.make_view(ImageNode("p", Pyramid(line), (1, 1), "phase"), lib.get("feature:ridges"),
                                {"min_um": 1.0, "max_um": 3.0, "ridges": "Bright"}).source)
    assert ridge[:, 127:129].mean() > 5 * ridge[:, 60:100].mean()


def test_unsharp_and_clahe(lib):
    edge = np.where(np.arange(256)[None, :] < 128, 0.3, 0.7) * np.ones((256, 1))
    from scipy import ndimage
    soft = ndimage.gaussian_filter(edge, 3).astype(np.float32)
    image = ImageNode("p", Pyramid(soft), (1, 1), "phase")
    sharp = whole(run.make_view(image, lib.get("filters:unsharp_mask"), {"sigma_um": 3.0}).source)
    assert sharp[:, 129].mean() - sharp[:, 126].mean() > soft[:, 129].mean() - soft[:, 126].mean()
    low_contrast = (100 + 20 * he_slide()[2] / 0.5).astype(np.uint8)
    image = ImageNode("b", Pyramid(np.repeat(low_contrast[..., None], 3, 2)), (0.5, 0.5), "brightfield")
    out = whole(run.make_view(image, lib.get("filters:clahe"), {"region_um": 32.0}).source)
    assert out.dtype == np.uint8 and np.ptp(out[..., 0]) > 1.5 * np.ptp(low_contrast)


def test_file_option():
    from modalytics.core.settings import Setting
    s = Setting.from_dict({"key": "blank", "label": "Blank image", "kind": "file", "default": ""})
    assert s.clean(None) == "" and s.clean("C:/x.png") == "C:/x.png"


def test_categories(lib):
    """Menus list steps by category, in a fixed order; every built-in has one."""
    assert all(s.category for s in lib.specs.values())
    groups = dict(lib.grouped("wsi"))
    assert list(groups) == ["Quantitative maps", "Feature maps", "Correction", "Denoise", "Enhance"]
    assert {s.key for s in groups["Correction"]} == {"flat_field", "white_balance", "stain_normalisation",
                                                       "rolling_ball"}
    assert "phase_unwrap" in {s.key for s in dict(lib.grouped("phase"))["Correction"]}


def test_rolling_ball_on_slides_keeps_the_glass_white(lib):
    """Automatic background: light for slides, so the empty glass becomes white, not coloured."""
    rgb, _h, e = he_slide()
    yy, xx = np.mgrid[0:256, 0:256]
    uneven = np.clip(rgb.astype(float) * (0.85 + 0.15 * xx / 255)[..., None], 0, 255).astype(np.uint8)
    out = whole(run.make_view(ImageNode("s", Pyramid(uneven), (0.5, 0.5), "wsi"), lib.get("filters:rolling_ball"),
                              {"radius_um": 20.0}).source).astype(float)
    glass = e == 0
    assert out[glass].mean() > 250 and np.ptp(out[glass].mean(axis=0)) < 3          # white and neutral
    assert out[~glass].mean() < out[glass].mean() - 30                              # tissue still darker


def test_macenko_to_itself_changes_little(lib, tmp_path):
    """A slide normalised to a picture of itself keeps its look (units as published: natural-log OD)."""
    rgb = he_slide()[0]
    Image.fromarray(rgb).save(tmp_path / "self.png")
    out = whole(run.make_view(ImageNode("s", Pyramid(rgb), (0.5, 0.5), "wsi"), lib.get("filters:stain_normalisation"),
                              {"reference": str(tmp_path / "self.png")}).source).astype(float)
    tissue = he_slide()[2] > 0.2
    assert np.abs(out - rgb)[tissue].mean() < 6


def test_display_range_from_full_resolution():
    """A feature map's first contrast comes from a full-resolution tile where the structure is, not from the
    mostly empty coarsest level."""
    lib = ProcessingLibrary(user="nonexistent-user-folder")
    rgb = np.full((512, 512, 3), 240, np.uint8)
    rgb[300:400, 300:400] = he_slide(size=100)[0]                                      # tissue in one corner
    view = run.make_view(ImageNode("s", Pyramid(rgb), (0.5, 0.5), "wsi"), lib.get("feature:gradient"))
    full = whole(view.source, 0)
    lo, hi = view.display.clim
    assert hi > 0.5 * np.percentile(full[300:400, 300:400], 99)
