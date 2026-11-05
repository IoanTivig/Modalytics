"""Profiles: file format, the user library, sharing as .mprofile, and the project's own copy."""
import zipfile

import numpy as np
import pytest

from modalytics.core.model import ImageNode, Project
from modalytics.io import project_file as pf
from modalytics.io.sources import ArraySource
from modalytics.profiles import Profile, ProfileError
from modalytics.profiles import bundle
from modalytics.profiles.library import ProfileLibrary, default_profile, slug
from modalytics.profiles.model import hex_to_rgba, rgba_to_hex

PARAM_SCRIPT = """KEY = 'aspect_ratio'
LABEL = 'Aspect ratio'
SCOPE = 'object'

def compute(ctx):
    return 1.0
"""


def _colonic():
    return default_profile(ProfileLibrary())[0]


# 1. File format ---------------------------------------------------------------------------

def test_shipped_profile_round_trips_and_is_consistent():
    p = _colonic()
    assert p.name == "Colonic tissue" and p.kind == "tissue" and p.images == ["wsi", "phase"]
    # One segmentation (Gland), two grades; older names (HoloPath's three grades) read as the current two
    assert [g.name for g in p.classes.groups] == ["Gland"]
    assert [s.name for s in p.classes.groups[0].subgroups] == ["Low grade dysplasia", "High grade dysplasia"]
    assert p.label_for("Gland", "Normal") == ("Gland", "Low grade dysplasia")
    assert p.label_for("Gland", "Dysplasia low grade") == ("Gland", "Low grade dysplasia")
    assert p.label_for("Gland", "Dysplasia high grade") == ("Gland", "High grade dysplasia")
    assert p.label_for("Gland", "High grade dysplasia") == ("Gland", "High grade dysplasia")
    assert p.model_for("Gland", "wsi") == "wsi_effnetb3_L00_gland_01" and p.model_for("Gland", "brightfield") is None
    assert p.problems() == []
    assert Profile.from_yaml(p.to_yaml()) == p
    # It measures HoloPath's default parameter set, every one of them in the library
    from modalytics.parameters.registry import ParameterLibrary
    lib = ParameterLibrary()
    # HoloPath's 58 + the 16 OD-map ones (slides get their OD map at import) + 10 shape completion ones
    # + 7 per-class summaries (counts, densities, Awan's gland grading features)
    assert p.problems(known_parameters=set(lib.specs)) == [] and len(p.parameters) == 58 + 16 + 10 + 7
    [chain] = p.chains_for("wsi")
    assert chain.name == "OD map" and chain.measure
    assert chain.steps == [{"step": "od:od_map", "options": {"background": "Estimated from the image"}}]
    assert p.made_types() == ["od"]
    assert p.parameters["object:phase:disorder_strength"] == {"degree": 2, "window": 3, "refractive_index": 1.38}


def test_colours_and_problems():
    assert hex_to_rgba("#33cc66") == (0.2, 0.8, 0.4, 1.0) and rgba_to_hex((0.2, 0.8, 0.4, 1.0)) == "#33cc66"
    assert rgba_to_hex((1, 0, 0, 0.5)) == "#ff000080"
    with pytest.raises(ProfileError):
        hex_to_rgba("green")
    p = _colonic()
    p.images = ["phase", "x-ray"]
    p.segmentation["Crypt"] = {"phase": "m1"}
    found = " | ".join(p.problems(known_models={"m1"}, known_parameters=set()))
    assert "x-ray" in found and "Crypt" in found and "wsi images, which the profile does not accept" in found
    assert "unet_efficientnet_b0__imagenet_micronet" in found                       # not in known_models


def test_bad_files_are_refused_with_a_reason():
    with pytest.raises(ProfileError, match="YAML"):
        Profile.from_yaml("name: [unclosed")
    with pytest.raises(ProfileError, match="Not a Modalytics profile"):
        Profile.from_yaml("format: something-else")
    with pytest.raises(ProfileError, match="newer"):
        Profile.from_yaml("format_version: 99\nname: x")


# 2. Library -------------------------------------------------------------------------------

def test_builtins_are_installed_once_and_stay_editable(private_user_library):
    lib = ProfileLibrary()
    assert [e.id for e in lib.entries()] == ["colonic_tissue"]
    p = lib.get("colonic_tissue")
    p.description = "edited here"
    lib.save(p, "colonic_tissue")
    assert lib.get("colonic_tissue").revision == 3                # changed → revision + 1 (shipped as 2)
    lib.save(lib.get("colonic_tissue"), "colonic_tissue")
    assert lib.get("colonic_tissue").revision == 3                # unchanged → same revision
    # A deleted shipped profile is not brought back by the next start…
    lib.delete("colonic_tissue")
    assert ProfileLibrary().entries() == []
    # …but "Reset to built-in" brings the shipped version back
    assert lib.reset_builtin("colonic_tissue").description != "edited here"


def test_new_duplicate_names_and_unreadable_files():
    lib = ProfileLibrary()
    pid = lib.duplicate("colonic_tissue")
    assert pid == "colonic_tissue_copy" and lib.get(pid).name == "Colonic tissue (copy)"
    assert lib.unique_name("Colonic tissue") == "Colonic tissue (2)"
    cells = Profile("Blood cells", kind="cells", images=["phase"])
    assert lib.save(cells) == "blood_cells" and lib.save(cells) == "blood_cells_2"
    assert [e.id for e in lib.entries(kind="cells")] == ["blood_cells", "blood_cells_2"]
    (lib.folder / "broken.yaml").write_text("name: [", encoding="utf-8")
    broken = next(e for e in lib.entries() if e.id == "broken")
    assert broken.profile is None and "YAML" in broken.error
    assert slug(" Colonic tissue (v2) ") == "colonic_tissue_v2"


# 3. Sharing ---------------------------------------------------------------------------------

def _fake_model(path, payload: bytes):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", "{}")
        z.writestr("model.onnx", payload)
    return path


def test_export_then_import_on_another_computer(tmp_path, monkeypatch):
    # This computer: a profile using one model and one user parameter
    model = _fake_model(tmp_path / "gland_net.mmodel", b"weights-A")
    script = tmp_path / "aspect_ratio.py"
    script.write_text(PARAM_SCRIPT, encoding="utf-8")
    p = Profile("Gastric tissue", images=["wsi"], segmentation={"Gland": {"wsi": "gland_net"}},
                parameters={"object:aspect_ratio": {}, "object:circularity": {}})
    p.classes.ensure("Gland")
    out = tmp_path / "gastric"
    warnings = bundle.export_profile(p, out, models={"gland_net": model},
                                     parameters={"object:aspect_ratio": script})
    assert warnings == [] and (tmp_path / "gastric.mprofile").exists()

    # Another computer: empty library
    monkeypatch.setenv("MODALYTICS_HOME", str(tmp_path / "other"))
    lib = ProfileLibrary()
    models_dir, params_dir = tmp_path / "other" / "models", tmp_path / "other" / "parameters"
    known_m, known_p = {}, {}
    r = bundle.import_profile(tmp_path / "gastric.mprofile", lib, models_dir, params_dir, known_m, known_p)
    assert r.name == "Gastric tissue" and sorted(r.added) == ["gland_net", "object:aspect_ratio"]
    assert (models_dir / "gland_net.mmodel").read_bytes() == model.read_bytes()
    assert lib.get(r.profile_id).parameters == p.parameters

    # Same bundle again: identical items reused, profile name made unique
    r2 = bundle.import_profile(tmp_path / "gastric.mprofile", lib, models_dir, params_dir, known_m, known_p)
    assert sorted(r2.reused) == ["gland_net", "object:aspect_ratio"] and r2.name == "Gastric tissue (2)"


def test_import_never_overwrites_a_different_item(tmp_path):
    lib = ProfileLibrary()
    models_dir, params_dir = tmp_path / "m", tmp_path / "p"
    # Here already: a *different* model and parameter with the same names
    models_dir.mkdir()
    params_dir.mkdir()
    mine = _fake_model(models_dir / "gland_net.mmodel", b"weights-MINE")
    my_param = params_dir / "aspect_ratio.py"                    # (where it lives does not matter here)
    my_param.write_text(PARAM_SCRIPT.replace("1.0", "2.0"), encoding="utf-8")
    known_m, known_p = {"gland_net": mine}, {"object:phase:aspect_ratio": my_param}
    # The bundle from elsewhere
    theirs = _fake_model(tmp_path / "gland_net.mmodel", b"weights-THEIRS")
    their_param = tmp_path / "aspect_ratio.py"
    their_param.write_text(PARAM_SCRIPT, encoding="utf-8")
    p = Profile("Gastric tissue", images=["wsi"], segmentation={"Gland": {"wsi": "gland_net"}},
                parameters={"object:phase:aspect_ratio": {"x": 1}})
    bundle.export_profile(p, tmp_path / "g.mprofile", {"gland_net": theirs}, {"object:phase:aspect_ratio": their_param})

    r = bundle.import_profile(tmp_path / "g.mprofile", lib, models_dir, params_dir, known_m, known_p)
    assert r.renamed == {"gland_net": "gland_net_2", "object:phase:aspect_ratio": "object:phase:aspect_ratio_2"}
    assert b"MINE" in zipfile.ZipFile(mine).read("model.onnx")                # untouched
    assert "2.0" in my_param.read_text(encoding="utf-8")                    # untouched
    assert "KEY = 'aspect_ratio_2'" in (params_dir / "object" / "phase" / "aspect_ratio_2.py").read_text(encoding="utf-8")
    got = lib.get(r.profile_id)
    assert got.model_for("Gland", "wsi") == "gland_net_2" and got.parameters == {"object:phase:aspect_ratio_2": {"x": 1}}


def test_a_file_that_is_not_a_bundle(tmp_path):
    bad = tmp_path / "x.mprofile"
    bad.write_bytes(b"hello")
    with pytest.raises(ProfileError, match="not a profile bundle"):
        bundle.import_profile(bad, ProfileLibrary(), tmp_path, tmp_path, {}, {})


# 4. A project keeps its own copy --------------------------------------------------------------

def test_project_reopens_with_its_own_profile_copy(tmp_path):
    lib = ProfileLibrary()
    profile = lib.get("colonic_tissue")
    project = Project()
    project.reset([], None, None, profile.copy(), "colonic_tissue")
    assert [g.name for g in project.labels.groups] == ["Gland"]
    project.add_image(ImageNode("p", ArraySource(np.zeros((4, 4), np.float32)), (1, 1), "phase"))
    pf.save_project(project, str(tmp_path / "case"))
    # The library profile changes afterwards…
    profile.description = "changed later"
    profile.images = ["phase"]
    lib.save(profile, "colonic_tissue")
    # …the project still has the analysis it was made with
    data = pf.load_project(str(tmp_path / "case.modalytics"))
    assert data.profile_id == "colonic_tissue" and data.profile.images == ["wsi", "phase"]
    assert data.profile.description != "changed later" and data.profile.revision == 2
    assert lib.get("colonic_tissue").revision == 3



def test_aliases_rename_older_annotations_on_import():
    """A HoloPath project's three grades come in as the profile's two (one mask per class, objects kept)."""
    from shapely.geometry import Point
    from modalytics.core.model import MaskNode
    from modalytics.services.classes import rename_to_profile
    profile = _colonic()
    img = ImageNode("p", ArraySource(np.zeros((40, 40), np.float32)), (1, 1), "phase")
    for sub, centre in (("Normal", (5, 5)), ("Dysplasia low grade", (20, 5)), ("Dysplasia high grade", (5, 20))):
        m = MaskNode(geometry=[Point(*centre).buffer(3, 32)], group="Gland", subgroup=sub)
        m.parent = img
        img.children.append(m)
    rename_to_profile(img, profile)
    classes = {(m.group, m.subgroup): len(m.geometry.geoms) for m in img.masks}
    assert classes == {("Gland", "Low grade dysplasia"): 2, ("Gland", "High grade dysplasia"): 1}
    problems = Profile("x", images=["phase"], aliases={"Gland": {"Old": "Missing"}}).problems()
    assert any("Older names for 'Gland'" in p for p in problems)
