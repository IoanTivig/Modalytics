"""Processing: the step library, views computed tile by tile (equal to the whole-image result), contracts,
views in projects and profiles, and measuring through views."""
import math

import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import box

from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io.sources import ArraySource, Level, PixelSource
from modalytics.parameters import engine
from modalytics.parameters.registry import ParameterLibrary
from modalytics.processing import run
from modalytics.processing.registry import ProcessingLibrary, load_script
from modalytics.processing.types import ContractError


class Pyramid(PixelSource):
    """A small two-level pyramid with 64 px tiles (like a slide, in miniature)."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = [data, data[::2, ::2].copy()]
        self.dtype, self.channels = data.dtype, (data.shape[2] if data.ndim == 3 else 1)
        h, w = data.shape[:2]
        self.levels = [Level((h, w), (64, 64), (1.0, 1.0)),
                       Level((h // 2, w // 2), (64, 64), (h / (h // 2), w / (w // 2)))]

    def read_tile(self, level, ty, tx):
        return self.data[level][ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64]


def _whole(source, level=0):
    gy, gx = source.levels[level].grid
    return np.concatenate([np.concatenate([source.read_tile(level, ty, tx) for tx in range(gx)], 1)
                           for ty in range(gy)], 0)


@pytest.fixture
def lib():
    return ProcessingLibrary()


def test_builtin_steps(lib):
    assert {"od:od_map", "opd:opd", "dry_mass:dry_mass", "filters:gaussian_blur", "filters:median"} <= set(lib.specs)
    assert not lib.errors
    wsi, phase = {s.uid for s in lib.for_type("wsi")}, {s.uid for s in lib.for_type("phase")}
    assert {"od:od_map", "stain:colour_deconvolution", "filters:median"} <= wsi and "opd:opd" not in wsi
    assert {"opd:opd", "dry_mass:dry_mass", "height:height_map", "filters:median"} <= phase and "od:od_map" not in phase
    from modalytics.parameters.script_file import ProcessingScript
    for spec in lib.specs.values():                     # in the editor's canonical form
        text = open(spec.path, encoding="utf-8").read()
        assert ProcessingScript.parse(text).render() == text


@pytest.mark.parametrize("uid, options", [("filters:gaussian_blur", {"sigma_um": 3.0}),
                                          ("filters:median", {"size_um": 5.0})])
def test_tile_by_tile_equals_the_whole_image(lib, uid, options):
    """Neighbourhood steps get the context they need around each tile: no seams, borders included."""
    rng = np.random.default_rng(2)
    data = rng.random((200, 260)).astype(np.float32)
    image = ImageNode("p", Pyramid(data), (1.0, 1.0), "phase")
    view = run.make_view(image, lib.get(uid), options)
    tiled = _whole(view.source)
    if uid == "filters:gaussian_blur":
        expect = ndimage.gaussian_filter(data, 3.0)
    else:
        expect = ndimage.median_filter(data, size=5, mode="reflect")
    assert np.allclose(tiled, expect, atol=1e-6)
    # On the next level, sizes stay in µm (σ = 3 µm = 1.5 px there)
    assert view.source.read_tile(1, 0, 0).shape == (64, 64)


def test_od_map_view_of_a_slide(lib):
    glass = np.array([240.0, 236.0, 246.0])
    od = np.zeros((300, 300))
    od[100:200, 100:200] = 0.7
    rgb = np.clip(np.round(glass * 10.0 ** (-od[..., None])), 0, 255).astype(np.uint8)
    slide = ImageNode("s", Pyramid(rgb), (0.5, 0.5), "wsi")
    view = run.make_view(slide, lib.get("od:od_map"))
    assert view.image_type == "od" and view.unit == "OD" and view.display.colormap == "magma"
    out = _whole(view.source)
    assert out.dtype == np.float32 and out.ndim == 2
    assert out[150, 150] == pytest.approx(0.7, abs=0.003) and out[10, 10] == pytest.approx(0.0, abs=0.003)


def test_opd_and_dry_mass(lib):
    phase = np.full((50, 60), math.pi, np.float32)                     # π rad
    image = ImageNode("p", ArraySource(phase), (0.3, 0.3), "phase", metadata={"wavelength_nm": 600.0})
    opd = run.make_view(image, lib.get("opd:opd"))
    assert _whole(opd.source)[0, 0] == pytest.approx(300.0)            # λ/2
    dm = run.make_view(image, lib.get("dry_mass:dry_mass"), {"alpha": 0.2})
    assert _whole(dm.source)[0, 0] == pytest.approx(0.3 / 0.2)         # 0.3 µm OPD / α → pg/µm²
    no_lambda = ImageNode("p", ArraySource(phase), (0.3, 0.3), "phase")
    with pytest.raises(ValueError, match="wavelength is not known"):
        run.make_view(no_lambda, lib.get("opd:opd"))
    with pytest.raises(ValueError, match="does not process"):
        run.make_view(no_lambda, lib.get("od:od_map"))


def test_contracts_and_whole_image_steps(tmp_path, monkeypatch):
    def step(name, body, output="od", kind="pixel"):
        f = tmp_path / f"{name}.py"
        f.write_text(f"KEY = '{name}'\nINPUT_TYPES = ['phase']\nOUTPUT_TYPE = '{output}'\nKIND = '{kind}'\n"
                     f"def process(pixels, options, state, um_per_px):\n    {body}\n", encoding="utf-8")
        return load_script(f)
    image = ImageNode("p", ArraySource(np.ones((40, 40), np.float32)), (1, 1), "phase")
    with pytest.raises(ContractError, match="keeps the size"):
        run.make_view(image, step("crop", "return pixels[:10]"))
    with pytest.raises(ContractError, match="≥ 0"):
        run.make_view(image, step("neg", "return -pixels"))
    whole = step("fft", "import numpy as np\n    return np.abs(np.fft.fft2(pixels)).astype('float32')", "opd", "whole")
    assert _whole(run.make_view(image, whole).source).shape == (40, 40)
    monkeypatch.setattr(run, "MAX_WHOLE_PIXELS", 100)
    with pytest.raises(ValueError, match="too large"):
        run.make_view(image, whole)


def test_views_in_projects(tmp_path, lib):
    from modalytics.io import project_file as pf
    image = ImageNode("p", ArraySource(np.full((60, 80), 2.0, np.float32)), (0.5, 0.5), "phase",
                      metadata={"wavelength_nm": 650.0})
    project = Project()
    project.add_image(image)
    project.add_view(image, run.make_view(image, lib.get("dry_mass:dry_mass")))
    project.add_view(image, run.make_view(image, lib.get("filters:gaussian_blur"), {"sigma_um": 2.0}), show=True)
    image.views.append(run.make_view(image, lib.get("opd:opd")))
    image.views[-1].chain["steps"][0]["step"] = "opd:not_here"        # a step this computer does not have
    pf.save_project(project, str(tmp_path / "case"))
    data = pf.load_project(str(tmp_path / "case.modalytics"))
    back = data.images[0]
    assert [v.steps[0]["step"] for v in back.views] == ["dry_mass:dry_mass", "filters:gaussian_blur"]
    assert back.shown is back.views[1] and back.views[1].steps[0]["options"] == {"sigma_um": 2.0}
    assert any("not available here" in w for w in data.warnings)
    # Saved again: the unavailable recipe is still there
    project2 = Project()
    project2.add_image(back)
    pf.save_project(project2, str(tmp_path / "case"))
    assert any("not available" in w for w in pf.load_project(str(tmp_path / "case.modalytics")).warnings)


def test_measuring_through_views_and_profiles(lib):
    from modalytics.profiles.library import default_profile
    profile, _pid = default_profile()                                   # Colonic tissue: slides → OD map
    rgb = np.full((200, 200, 3), 240, np.uint8)
    rgb[50:150, 50:150] = 120                                           # OD ≈ 0.30 (240 → 120)
    slide = ImageNode("s", ArraySource(rgb), (1.0, 1.0), "wsi")
    plib = ParameterLibrary()
    choices, _ = engine.select(plib, {"object:od:mean": {}, "object:any:area": {}}, "object")
    values, problems = engine.measure_object(box(60, 60, 140, 140), slide, choices)
    assert "no od view" in problems["object:od:mean"] and values["object:any:area"] == pytest.approx(6400)
    views, why = run.profile_views(profile, slide, lib)
    assert not why and [v.image_type for v in views] == ["od"] and views[0].standard and views[0].measure
    slide.views += views
    values, problems = engine.measure_object(box(60, 60, 140, 140), slide, choices)
    assert not problems and values["object:od:mean"] == pytest.approx(math.log10(2), abs=0.003)


def test_views_follow_the_image(lib, tmp_path):
    """A view reads the image's current pixels: a relinked file, and a missing one (nothing to measure)."""
    from modalytics.io.sources import MissingSource
    from modalytics.parameters.context import read_native
    image = ImageNode("s", ArraySource(np.full((64, 64, 3), 200, np.uint8)), (1, 1), "wsi")
    view = run.make_view(image, lib.get("od:od_map"), {"background": "Fixed value", "background_value": 200.0})
    assert view.source.read_tile(0, 0, 0)[0, 0] == pytest.approx(0.0)
    project = Project()
    project.add_image(image)
    project.set_source(image, ArraySource(np.full((64, 64, 3), 100, np.uint8)))
    assert view.source.read_tile(0, 0, 0)[0, 0] == pytest.approx(math.log10(2), abs=1e-6)
    project.set_source(image, MissingSource("tiff", "C:/gone.tif", None, (64, 64)))
    stand_in = ImageNode("x", view.source, (1, 1), "od")
    assert view.source.missing
    with pytest.raises(ValueError, match="not found"):
        read_native(stand_in, 0, 4, 0, 4)


def test_show_and_remove_views(lib):
    image = ImageNode("p", ArraySource(np.ones((32, 32), np.float32)), (1, 1), "phase")
    project = Project()
    project.add_image(image)
    project.add_mask(image, MaskNode(geometry=box(2, 2, 10, 10), color=(1, 0, 0, 1), group="Cell", subgroup=""))
    seen = []
    project.node_changed.connect(lambda n, f: seen.append(f))
    v = run.make_view(image, lib.get("filters:median"))
    project.add_view(image, v, show=True)
    assert image.view_source is v.source and image.view_type == "phase" and seen == ["views", "source"]
    project.set_display(image, colormap="viridis")
    assert v.display.colormap == "viridis" and image.display.colormap == "gray"     # the shown view's own
    project.remove_view(image, v)
    assert image.shown is None and image.view_source is image.source and image.masks   # masks untouched



# ---------------------------------------------------------------- chains
def test_a_chain_equals_its_steps_done_by_hand(lib):
    """Median → OD map, tile by tile, is the OD map of the median-filtered slide."""
    from modalytics.processing.chains import Chain
    from modalytics.parameters import optical as op
    rng = np.random.default_rng(3)
    rgb = rng.integers(120, 245, (200, 260, 3)).astype(np.uint8)
    slide = ImageNode("s", Pyramid(rgb), (1.0, 1.0), "wsi")
    chain = Chain("Denoised OD", [{"step": "filters:median", "options": {"size_um": 3.0}},
                                  {"step": "od:od_map", "options": {"background": "Fixed value",
                                                                    "background_value": 250.0}}])
    view = run.make_chain_view(slide, chain, lib)
    assert view.image_type == "od" and view.name == "Denoised OD" and view.measure
    by_hand = op.optical_density(ndimage.median_filter(rgb, size=(3, 3, 1), mode="reflect"),
                                 np.full(3, 250.0), 3)
    assert np.allclose(_whole(view.source), by_hand, atol=1e-6)
    with pytest.raises(ValueError, match="does not process od images"):
        run.make_chain_view(slide, Chain("wrong", [{"step": "od:od_map"}, {"step": "od:od_map"}]), lib)
    assert chain.summary(lib) == "Median filter (denoise) → OD map"
    assert chain.output_type("wsi", lib) == "od" and chain.problems("phase", lib)


def test_which_view_parameters_use(lib):
    """The standard (profile's) measured view first, else the first measured one; never a look-only one;
    for the image's own type, a measured filter chain takes the original's place."""
    from modalytics.processing.chains import Chain
    phase = ImageNode("p", ArraySource(np.ones((40, 40), np.float32)), (1, 1), "phase",
                      metadata={"wavelength_nm": 600.0})
    look = run.make_chain_view(phase, Chain("blur", [{"step": "filters:gaussian_blur"}], measure=False), lib)
    phase.views.append(look)
    assert phase.as_type("phase") is phase                              # look-only: the original is measured
    mine = run.make_chain_view(phase, Chain("mine", [{"step": "dry_mass:dry_mass"}]), lib)
    std = run.make_chain_view(phase, Chain("std", [{"step": "dry_mass:dry_mass"}]), lib, standard=True)
    phase.views += [mine, std]
    assert phase.measured_view("dry_mass") is std                       # the profile's wins
    assert phase.as_type("dry_mass").source is std.source
    assert phase.as_type("dry_mass", view=mine).source is mine.source  # its own, labelled results (stage 6)
    denoise = run.make_chain_view(phase, Chain("denoise", [{"step": "filters:median"}]), lib, standard=True)
    phase.views.append(denoise)
    assert phase.as_type("phase").source is denoise.source              # measured filter replaces the original
    assert phase.as_type("od") is None and phase.as_type("*") is phase


def test_my_chains_library_and_profile_checks(tmp_path, lib):
    from modalytics.processing.chains import Chain, ChainLibrary
    from modalytics.profiles.model import Profile
    chains = ChainLibrary(tmp_path)
    chains.save(Chain("Denoised OD", [{"step": "filters:median", "options": {"size_um": 1.0}},
                                      {"step": "od:od_map"}]))
    chains.save(Chain("Look", [{"step": "filters:gaussian_blur"}], measure=False))
    assert [c.name for c in chains.chains()] == ["Denoised OD", "Look"]
    assert chains.get("Look").measure is False
    chains.delete("Look")
    assert [c.name for c in chains.chains()] == ["Denoised OD"]
    # A profile: two measured chains making OD maps from slides is a problem (one standard per type)
    p = Profile("t", images=["wsi"], processing={"wsi": [Chain("A", [{"step": "od:od_map"}]).to_dict(),
                                                         chains.get("Denoised OD").to_dict()]})
    assert any("both measured" in x for x in p.chain_problems(lib))
    p.processing["wsi"][1]["measure"] = False
    assert p.chain_problems(lib) == [] and p.made_types(lib) == ["od"]
    # Older files: one step per entry
    old = Profile.from_dict({"name": "o", "images": ["wsi"], "processing": {"wsi": [{"step": "od:od_map",
                                                                                     "options": {}}]}})
    assert old.chains_for("wsi")[0].steps == [{"step": "od:od_map", "options": {}}]


def test_older_view_recipes_still_open(lib):
    image = ImageNode("p", ArraySource(np.ones((30, 30), np.float32)), (1, 1), "phase")
    problems = run.restore_views(image, [{"step": "filters:median", "options": {"size_um": 2.0}, "shown": True}], lib)
    assert not problems and image.views[0].steps[0]["step"] == "filters:median" and image.shown is image.views[0]


def test_chains_in_profiles(tmp_path, lib):
    """Profiles keep their own copy of a chain; updating replaces it on request, keeping their measured choice."""
    from modalytics.processing.chains import Chain, input_types, profiles_using, update_profiles
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.profiles.model import Profile
    profiles = ProfileLibrary(tmp_path / "profiles", builtin=tmp_path / "none")
    old = Chain("Denoised OD", [{"step": "filters:median", "options": {"size_um": 1.0}}, {"step": "od:od_map"}],
                measure=False)
    pid = profiles.save(Profile("Lab", images=["wsi"], processing={"wsi": [old.to_dict()]}))
    assert input_types(old, lib) == ["wsi", "brightfield"]
    [(found_pid, name, image_type, copy)] = profiles_using("Denoised OD", profiles)
    assert (found_pid, name, image_type) == (pid, "Lab", "wsi") and copy.same_steps(old, lib)
    assert old.same_steps(Chain("x", [{"step": "filters:median", "options": {"size_um": 1.0}},
                                      {"step": "od:od_map", "options": {"background": "Estimated from the image"}}]),
                          lib)                                          # defaults written out or not: the same
    mine = Chain("Denoised OD", [{"step": "filters:median", "options": {"size_um": 3.0}}, {"step": "od:od_map"}])
    assert profiles.get(pid).chains_for("wsi")[0].steps[0]["options"]["size_um"] == 1.0   # unchanged until asked
    assert update_profiles(mine, [(pid, "wsi")], profiles) == 1
    [updated] = profiles.get(pid).chains_for("wsi")
    assert updated.steps[0]["options"]["size_um"] == 3.0 and updated.measure is False      # its own choice kept
