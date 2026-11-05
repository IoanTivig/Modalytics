"""
Registration methods as scripts: the library, the contract, the synthetic test, the profile's default.
"""

import pytest

from modalytics.profiles.model import Profile
from modalytics.registration.registry import DEFAULT_METHOD, RegistrationLibrary, load_script
from modalytics.registration.synthetic import evaluate_method, synthetic_tissue

SHIPPED = {"gland_map_ncc", "intensity_ncc", "edge_ncc", "mutual_information", "fourier_mellin", "feature_points",
           "ecc_refine"}


def test_the_shipped_methods_load_with_their_references():
    lib = RegistrationLibrary()
    assert set(lib.specs) == SHIPPED and not lib.errors
    assert all(s.references and all(r.get("doi") for r in s.references) for s in lib.specs.values())
    assert lib.get(DEFAULT_METHOD) is not None and not lib.get(DEFAULT_METHOD).experimental
    pair = [s.key for s in lib.for_pair("phase", "wsi")]
    assert pair[0] == DEFAULT_METHOD and set(pair) == SHIPPED                  # validated first
    assert not any(s.can_move("wsi") for s in lib.specs.values())              # a slide never moves
    assert lib.get("ecc_refine").refines_only and "rotation" in lib.get("ecc_refine").finds


def test_a_method_script_must_keep_to_the_contract(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("KEY = 'm'\nNAME = 'M'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="register"):
        load_script(f)
    f.write_text("KEY = 'm'\nFINDS = ['rotation']\ndef register(ctx, options):\n    return []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="translation"):
        load_script(f)
    f.write_text("KEY = 'm'\nMOVING_TYPES = ['nonsense']\ndef register(ctx, options):\n    return []\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="MOVING_TYPES"):
        load_script(f)


def test_a_users_method_replaces_the_built_in_of_the_same_key(tmp_path):
    (tmp_path / "edge_ncc.py").write_text(
        "KEY = 'edge_ncc'\nNAME = 'My edges'\ndef register(ctx, options):\n    return []\n", encoding="utf-8")
    lib = RegistrationLibrary(user=tmp_path)
    assert lib.get("edge_ncc").name == "My edges" and lib.overridden == ["edge_ncc"]


def test_synthetic_patches_are_found_again():
    lib = RegistrationLibrary()
    tissue = synthetic_tissue(size_um=900.0, seed=3)
    for key in ("intensity_ncc", "ecc_refine"):
        report = evaluate_method(lib.get(key), tissue, trials=2, radius_um=100.0, rotation=True, seed=2)
        assert all(t.success() for t in report.trials), report.summary()
    report = evaluate_method(lib.get("gland_map_ncc"), tissue, trials=1)          # no brightfield gland model
    assert report.trials[0].found is None and "segmentation model" in report.trials[0].error


def test_the_profile_chooses_its_methods_experimental_ones_off_until_ticked():
    p = Profile("Test", images=["phase", "wsi"])
    assert p.registration_default() == DEFAULT_METHOD and "registration" not in p.to_dict()
    assert p.registration_offered("gland_map_ncc", False) and not p.registration_offered("edge_ncc", True)
    p.set_registration_offered("edge_ncc", True, experimental=True)          # ticked on
    p.set_registration_offered("gland_map_ncc", False, experimental=False)   # the default stays offered
    assert p.registration_offered("edge_ncc", True) and p.registration_offered("gland_map_ncc", False)
    p.set_registration_default("mutual_information", experimental=True)      # a default is offered too
    assert p.registration_default() == "mutual_information" and p.registration_offered("mutual_information", True)
    p.set_registration_offered("edge_ncc", False, experimental=True)         # back to the rule: nothing written
    back = Profile.from_dict(p.to_dict())
    assert back.registration == {"method": "mutual_information", "on": ["mutual_information"],
                                 "off": ["gland_map_ncc"]}
    assert not back.registration_offered("edge_ncc", True) and not back.registration_offered("gland_map_ncc", False)


def test_the_register_tool_offers_what_the_profile_offers():
    from modalytics.tools.arrange import offered_methods
    p = Profile("Test", images=["phase", "wsi"])
    assert [s.key for s in offered_methods(p)] == [DEFAULT_METHOD]          # experimental ones off
    p.set_registration_offered("edge_ncc", True, experimental=True)
    p.set_registration_default("edge_ncc", experimental=True)
    assert [s.key for s in offered_methods(p)][0] == "edge_ncc" and len(offered_methods(p)) == 2
