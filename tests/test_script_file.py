"""Parameter scripts split into properties and code, and joined back without touching the code."""
import glob
import os

import pytest

from modalytics.parameters.registry import load_script
from modalytics.parameters.script_file import MARKER, ParameterScript

BUILTIN = os.path.join(os.path.dirname(__file__), "..", "src", "modalytics", "resources", "parameters")

SCRIPT = '''"""Ratio of the axes."""

import math
# a comment that must survive

KEY = 'aspect'   # a property: replaced on save
LABEL = 'Aspect'
UNIT = ''
SCOPE = 'object'
LEVEL = 'optimal'          # HoloPath's tier: not a property here, stays in the code
ORDER = 10 * 5             # computed: stays in the code


def helper(a, b):
    return a / b if b else math.nan


def compute(ctx):
    return helper(ctx.major_axis_um, ctx.minor_axis_um)
'''


def test_every_builtin_round_trips_byte_for_byte():
    files = glob.glob(os.path.join(BUILTIN, "*", "*", "*.py"))
    # HoloPath's + brightfield, OD map, OPD map, dry mass + shape, scattering, stain, height, refractive index,
    # feature maps, relations between classes, per-class summaries
    assert len(files) == 60 + 33 + 16 + 3 + 2 + 12 + 8 + 5 + 4 + 3 + 2 + 12 + 7
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        assert ParameterScript.parse(text).render() == text, f


def test_properties_change_and_the_code_stays_as_written(tmp_path):
    s = ParameterScript.parse(SCRIPT)
    assert s.properties == {"KEY": "aspect", "LABEL": "Aspect", "UNIT": "", "SCOPE": "object"}
    assert "ORDER = 10 * 5" in s.code and "LEVEL = 'optimal'" in s.code and MARKER in s.code
    assert "# a comment that must survive" in s.code
    s.properties.update(LABEL="Aspect ratio", UNIT="", IMAGE_TYPE="*", DESCRIPTION="Major / minor.",
                        CONSTANTS=[{"key": "eps", "label": "Epsilon", "kind": "float", "default": 0.001}])
    text = s.render()
    assert "LABEL = 'Aspect ratio'" in text and "def helper(a, b):" in text and MARKER not in text
    f = tmp_path / "aspect.py"
    f.write_text(text, encoding="utf-8")
    spec = load_script(f)
    assert spec.label == "Aspect ratio" and spec.image_type == "*" and spec.order == 50
    assert [x.key for x in spec.constants] == ["eps"]
    assert ParameterScript.parse(text).render() == text                    # stable


def test_without_marker_the_properties_go_after_the_imports():
    s = ParameterScript.parse(SCRIPT)
    s.code = s.code.replace(MARKER + "\n", "")
    text = s.render()
    assert text.index("KEY = 'aspect'") > text.index("import math") and text.index("KEY") < text.index("def helper")
    bare = ParameterScript({"KEY": "x"}, "def compute(ctx):\n    return 1\n").render()
    assert bare.startswith("KEY = 'x'\n")


def test_invalid_python_is_refused():
    with pytest.raises(SyntaxError):
        ParameterScript.parse("def compute(ctx):\n    return (\n")


def test_older_name_settings_still_works_and_is_saved_as_constants(tmp_path):
    from shapely.geometry import box

    import numpy as np
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.parameters.context import ObjectContext
    old = ("KEY = 'scaled'\nIMAGE_TYPES = ['*']\n"
           "SETTINGS = [{'key': 'k', 'label': 'Factor', 'kind': 'float', 'default': 2.0}]\n\n"
           "def compute(ctx):\n    return ctx.settings['k'] * ctx.area_um2\n")
    f = tmp_path / "scaled.py"
    f.write_text(old, encoding="utf-8")
    spec = load_script(f)                                    # the older names load…
    img = ImageNode("x", ArraySource(np.zeros((20, 20), np.float32)), (1, 1), "phase")
    assert spec.compute(ObjectContext(box(0, 0, 5, 2), img, spec.resolve())) == 20.0   # …and measure
    s = ParameterScript.parse(old)
    assert s.properties["CONSTANTS"][0]["key"] == "k"
    assert "CONSTANTS = [" in s.render() and "SETTINGS" not in s.render()           # saved under the new name



def test_older_image_types_lists_become_one_image_type(tmp_path):
    from modalytics.parameters.registry import image_type_from_list
    assert image_type_from_list(["*"]) == ("*", "") and image_type_from_list("wsi") == ("wsi", "")
    t, note = image_type_from_list(["phase", "brightfield"])
    assert t == "phase" and "several" in note
    s = ParameterScript.parse("KEY = 'x'\nIMAGE_TYPES = ['brightfield']\n\ndef compute(ctx):\n    return 0\n")
    assert s.properties["IMAGE_TYPE"] == "brightfield" and "IMAGE_TYPE = 'brightfield'" in s.render()
    f = tmp_path / "two.py"
    f.write_text("KEY = 'two'\nIMAGE_TYPES = ['phase', 'wsi']\n\ndef compute(ctx):\n    return 0\n", encoding="utf-8")
    spec = load_script(f)
    assert spec.uid == "object:phase:two" and "several" in spec.note


def test_the_same_key_can_exist_once_per_image_type(tmp_path):
    from modalytics.parameters.registry import ParameterLibrary
    for t in ("phase", "brightfield"):
        (tmp_path / f"{t}.py").write_text(f"KEY = 'mean'\nIMAGE_TYPE = '{t}'\nUNIT = 'u'\n\ndef compute(ctx):\n"
                                          f"    return 0\n", encoding="utf-8")
    lib = ParameterLibrary(user=tmp_path)
    assert {"object:phase:mean", "object:brightfield:mean"} <= set(lib.specs) and lib.errors == {}
    assert lib.find("object:mean") == "object:phase:mean"       # an old id prefers the phase map
    assert lib.migrate({"object:area": {}, "object:nothing": {}}) == {"object:any:area": {}, "object:nothing": {}}
