"""Profile bundles: steps and library requirements travel with the profile; the plan before importing; replacing."""

from __future__ import annotations

import zipfile

from modalytics.profiles import Profile
from modalytics.profiles import bundle
from modalytics.profiles.library import ProfileLibrary

STEP = """KEY = 'smooth_more'
LABEL = 'Smooth more'
REQUIRES = ['FakeLib']

def process(pixels, options, state, um_per_px):
    return pixels
"""
PARAM = """KEY = 'wobble'
LABEL = 'Wobble'
SCOPE = 'object'
REQUIRES = ['FakeLib']

def compute(ctx):
    return 1.0
"""


def _model(path, with_weights=True):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("model.onnx", b"onnx-bytes")
        z.writestr("manifest.json", "{}")
        if with_weights:
            z.writestr(bundle.WEIGHTS, b"w" * 1000)
    return path


def _profile():
    return Profile("Gastric tissue", images=["wsi"], segmentation={"Gland": {"wsi": "gland_net"}},
                   parameters={"object:any:area": {}, "object:wsi:wobble": {}},
                   processing={"wsi": [{"name": "Smoothed", "steps": [{"step": "filters:smooth_more", "options": {}},
                                                                    {"step": "od:od_map", "options": {}}]}]},
                   classification={"Gland": {"kind": "ranges", "group": "Gland", "grades": ["Low", "High"],
                                             "rules": [{"grade": "High", "combine": "all", "conditions": [
                                                 {"parameter": "object:wsi:wobble", "op": ">=", "value": 2.0}]}]}})


def _export(tmp_path, weights=True):
    model = _model(tmp_path / "gland_net.mmodel")
    step = tmp_path / "smooth_more.py"
    step.write_text(STEP, encoding="utf-8")
    param = tmp_path / "wobble.py"
    param.write_text(PARAM, encoding="utf-8")
    out = tmp_path / "gastric.mprofile"
    bundle.export_profile(_profile(), out, {"gland_net": model}, {"object:wsi:wobble": param},
                          steps={"filters:smooth_more": step}, requires={"FakeLib": "1.2.3"}, weights=weights,
                          app_version="0.9")
    return out


def test_export_carries_steps_requirements_and_optionally_weights(tmp_path):
    out = _export(tmp_path)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"steps/filters/smooth_more.py", "parameters/object/wsi/wobble.py", "models/gland_net.mmodel"} <= names
        m = zipfile.ZipFile(z.open("models/gland_net.mmodel"))
        assert bundle.WEIGHTS in m.namelist()
    (tmp_path / "light").mkdir()
    light = _export(tmp_path / "light", weights=False)
    with zipfile.ZipFile(light) as z:
        m = zipfile.ZipFile(z.open("models/gland_net.mmodel"))
        assert bundle.WEIGHTS not in m.namelist() and m.read("model.onnx") == b"onnx-bytes"
    assert bundle.steps_used(_profile()) == {"filters:smooth_more", "od:od_map"}


def test_plan_then_import_with_renames_and_replace(tmp_path):
    out = _export(tmp_path)
    lib = ProfileLibrary()
    here = tmp_path / "here"
    # Here already: a different step of the same id, and a profile of the same name
    mine = here / "steps" / "filters" / "smooth_more.py"
    mine.parent.mkdir(parents=True)
    mine.write_text(STEP.replace("pixels\n", "pixels * 2\n"), encoding="utf-8")
    old_id = lib.save(Profile("Gastric tissue", images=["wsi"]))
    known = {"models": {}, "parameters": {}, "steps": {"filters:smooth_more": mine}, "importers": {}}
    # 1. The plan: what is new, what differs, what is missing, which libraries are needed
    plan = bundle.inspect(out, lib, known, builtin={"parameters": {"object:any:area"}, "steps": set()},
                          available_libraries=set())
    status = {(k, u): s for k, u, s in plan.items}
    assert status == {("models", "gland_net"): "new", ("parameters", "object:wsi:wobble"): "new",
                      ("steps", "filters:smooth_more"): "different"}
    assert plan.missing == ["step 'od:od_map'"] and plan.requires == {"FakeLib": "1.2.3"}
    assert plan.existing == old_id and plan.weights and plan.app_version == "0.9" and plan.count("new") == 2
    # 2. Import, replacing the profile of the same name: the different step is added as _2 and the chain uses it
    r = bundle.import_profile(out, lib, here / "models", here / "parameters", {}, {}, steps_dir=here / "steps",
                              known_steps=dict(known["steps"]), replace=old_id, available_libraries=set())
    assert r.replaced and r.profile_id == old_id and r.renamed == {"filters:smooth_more": "filters:smooth_more_2"}
    assert r.requires == ["FakeLib"]
    got = lib.get(old_id)
    assert got.processing["wsi"][0]["steps"][0]["step"] == "filters:smooth_more_2"
    assert "KEY = 'smooth_more_2'" in (here / "steps" / "filters" / "smooth_more_2.py").read_text(encoding="utf-8")
    assert "pixels * 2" in mine.read_text(encoding="utf-8")                         # yours untouched
    assert got.classification["Gland"]["rules"][0]["conditions"][0]["parameter"] == "object:wsi:wobble"


def test_a_renamed_parameter_is_renamed_in_the_classifier_too(tmp_path):
    out = _export(tmp_path)
    lib = ProfileLibrary()
    mine = tmp_path / "mine.py"
    mine.write_text(PARAM.replace("1.0", "3.0"), encoding="utf-8")
    r = bundle.import_profile(out, lib, tmp_path / "m", tmp_path / "p", {}, {"object:wsi:wobble": mine},
                              available_libraries={"fakelib"})
    got = lib.get(r.profile_id)
    assert r.renamed["object:wsi:wobble"] == "object:wsi:wobble_2" and r.requires == []
    assert "object:wsi:wobble_2" in got.parameters
    assert got.classification["Gland"]["rules"][0]["conditions"][0]["parameter"] == "object:wsi:wobble_2"
    assert got.name == "Gastric tissue"                                      # no profile of that name here before
