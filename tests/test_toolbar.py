"""
The Process tab's buttons: a step's / chain's label and icon, the profile's choice (toolbar / available / hidden),
and the icons offered (Material Design, no brands).
"""

import pytest

from modalytics import icons
from modalytics.parameters.script_file import ProcessingScript
from modalytics.processing.chains import Chain
from modalytics.processing.registry import CATEGORY_ICONS, CHAIN_ICON, DEFAULT_TOOLBAR, ProcessingLibrary, load_script
from modalytics.profiles.model import Profile

STEP = '''
KEY = "my_step"
NAME = "My smoothing step"
{extra}
INPUT_TYPES = ["phase"]
CATEGORY = "Denoise"


def process(pixels, options, state, um_per_px):
    return pixels
'''


def test_shipped_steps_carry_their_label_and_icon():
    library = ProcessingLibrary()
    spec = library.get("filters:gaussian_blur")
    assert (spec.button_label, spec.button_icon) == ("Gaussian", "mdi6.blur")
    assert all(s.icon and icons.allowed(s.icon) for s in library.specs.values())     # every icon in the set
    assert all(library.get(uid) is not None for uid in DEFAULT_TOOLBAR)


def test_a_step_without_icon_takes_its_category_icon_and_a_bad_icon_is_refused(tmp_path):
    f = tmp_path / "my_step.py"
    f.write_text(STEP.format(extra=""), encoding="utf-8")
    spec = load_script(f)
    assert (spec.button_label, spec.button_icon) == ("My smoothing step", CATEGORY_ICONS["Denoise"])
    f.write_text(STEP.format(extra='SHORT = "Smooth"\nICON = "mdi6.water-outline"'), encoding="utf-8")
    spec = load_script(f)
    assert (spec.button_label, spec.button_icon) == ("Smooth", "mdi6.water-outline")
    f.write_text(STEP.format(extra='ICON = "fa5.star"'), encoding="utf-8")
    with pytest.raises(ValueError, match="ICON"):
        load_script(f)


def test_the_step_editor_writes_label_and_icon_after_the_name():
    script = ProcessingScript.parse(STEP.format(extra=""))
    script.properties.update({"SHORT": "Smooth", "ICON": "mdi6.blur"})
    text = script.render()
    lines = [l.split(" =")[0] for l in text.splitlines() if " = " in l]
    assert lines[:4] == ["KEY", "NAME", "SHORT", "ICON"]
    assert ProcessingScript.parse(text).properties["ICON"] == "mdi6.blur"


def test_a_chain_keeps_its_button():
    chain = Chain("Phase denoising", [{"step": "filters:gaussian_blur"}], short="Denoise", icon="mdi6.auto-fix")
    back = Chain.from_dict(chain.to_dict())
    assert (back.button_label, back.button_icon) == ("Denoise", "mdi6.auto-fix")
    plain = Chain.from_dict({"name": "Plain", "steps": []})
    assert (plain.button_label, plain.button_icon) == ("Plain", CHAIN_ICON) and "icon" not in plain.to_dict()


def test_the_profile_chooses_toolbar_available_or_hidden():
    p = Profile("Test", images=["phase"])
    assert p.toolbar() == list(DEFAULT_TOOLBAR) and p.step_place("filters:median") == "toolbar"
    assert p.step_place("filters:nl_means") == "available"                    # not a default button
    p.set_step_place("filters:median", "available")
    p.set_step_place("filters:nl_means", "toolbar")
    p.set_step_place("filters:clahe", "hidden")
    p.set_step_place("chain:Phase denoising", "toolbar")
    assert [p.step_place(x) for x in ("filters:median", "filters:nl_means", "filters:clahe", "chain:Phase denoising")] \
        == ["available", "toolbar", "hidden", "toolbar"]
    back = Profile.from_dict(p.to_dict())                                     # kept in the profile file
    assert back.toolbar() == p.toolbar() and back.step_place("filters:clahe") == "hidden"
    p.set_step_place("filters:clahe", "available")
    assert "hidden" not in p.steps and p.step_place("filters:clahe") == "available"
    assert "steps" not in Profile("Plain").to_dict()                          # untouched: the defaults


def test_icons_offered_are_searchable_and_never_brands():
    assert icons.search("blur")[:2] == ["mdi6.blur", "mdi6.blur-off"]
    assert "mdi6.microscope" in icons.search("microscope")
    assert icons.search("github") == [] and not icons.allowed("mdi6.github") and not icons.allowed("mdi6.android")
    assert icons.allowed("mdi6.blur") and not icons.allowed("blur") and len(icons.names()) > 6000
