"""Libraries for parameter scripts: the rules, planning an install, removing, REQUIRES (no network)."""
import sys

import pytest

from modalytics.parameters import libraries as libs
from modalytics.parameters.registry import ParameterLibrary


def test_rules_for_what_may_be_installed():
    assert "Training" in libs.check_name("torch") and "Training" in libs.check_name("TensorFlow")
    assert "included" in libs.check_name("numpy") and "included" in libs.check_name("scikit_image")
    assert libs.check_name("PyWavelets") == "" and libs.check_name("statsmodels>=0.14") == ""
    assert "not a package name" in libs.check_name("; rm -rf /")
    assert {lib.name.lower() for lib in libs.included()} >= {"numpy", "scipy", "scikit-image", "shapely"}


def test_plan_installs_only_what_is_missing_and_refuses_changing_an_included_library():
    numpy = next(lib for lib in libs.included() if lib.name.lower() == "numpy")
    report = {"install": [{"metadata": {"name": "PyWavelets", "version": "1.10.0"}},
                          {"metadata": {"name": "numpy", "version": numpy.version}}]}
    assert libs.plan_from_report(report) == (["PyWavelets==1.10.0"], [])
    clash = {"install": [{"metadata": {"name": "numpy", "version": "1.0.0"}}]}
    todo, problems = libs.plan_from_report(clash)
    assert todo == [] and "numpy 1.0.0" in problems[0] and numpy.version in problems[0]
    stage = libs.staging()
    cmd = libs.install_command(["PyWavelets==1.10.0"], stage)
    assert "--only-binary=:all:" in cmd and "--no-deps" in cmd and str(stage) in cmd


def _fake_library(name="tinylib", version="0.1"):
    root = libs.folder()
    (root / name).mkdir()
    (root / name / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: test\n",
                                   encoding="utf-8")
    (info / "RECORD").write_text(f"{name}/__init__.py,,\n{info.name}/METADATA,,\n{info.name}/RECORD,,\n",
                                 encoding="utf-8")


def test_your_libraries_are_listed_importable_and_removable():
    _fake_library()
    assert [lib.name for lib in libs.yours()] == ["tinylib"] and libs.missing(["tinylib", "nothere"]) == ["nothere"]
    libs.activate()
    libs.activate()
    assert sys.path.count(str(libs.folder())) == 1
    import importlib
    assert importlib.import_module("tinylib").VALUE == 42
    sys.modules.pop("tinylib", None)
    assert sorted(libs.remove("tinylib")) == ["tinylib", "tinylib-0.1.dist-info"]
    assert libs.yours() == [] and not (libs.folder() / "tinylib").exists()
    with pytest.raises(KeyError):
        libs.remove("tinylib")


def test_a_parameter_whose_library_is_missing_says_what_to_install(tmp_path):
    (tmp_path / "wavelet.py").write_text(
        "import pywt_not_installed_here\nKEY = 'wavelet'\nREQUIRES = ['PyWavelets-Not-Here']\n"
        "def compute(ctx):\n    return 0.0\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("import numpy\nKEY = 'fine'\nREQUIRES = ['numpy']\n"
                                      "def compute(ctx):\n    return 1.0\n", encoding="utf-8")
    lib = ParameterLibrary(user=tmp_path)
    err = next(e for p, e in lib.errors.items() if p.endswith("wavelet.py"))
    assert "needs PyWavelets-Not-Here" in err and "Libraries" in err
    assert lib.get("object:fine").requires == ("numpy",)
    missing = lib.missing_libraries()
    assert list(missing.values()) == [["PyWavelets-Not-Here"]] and next(iter(missing)).endswith("wavelet.py")


def test_a_staged_install_replaces_the_older_version_completely():
    _fake_library("tinylib", "0.1")
    (libs.folder() / "tinylib" / "old_only.py").write_text("", encoding="utf-8")   # a file 0.2 no longer has
    (libs.folder() / "tinylib-0.1.dist-info" / "RECORD").write_text(
        "tinylib/__init__.py,,\ntinylib/old_only.py,,\ntinylib-0.1.dist-info/METADATA,,\n", encoding="utf-8")
    stage = libs.staging()
    (stage / "tinylib").mkdir()
    (stage / "tinylib" / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    info = stage / "tinylib-0.2.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: tinylib\nVersion: 0.2\n", encoding="utf-8")
    assert libs.finish_install(stage) == [("tinylib", "0.1", "0.2")]
    assert [(lib.name, lib.version) for lib in libs.yours()] == [("tinylib", "0.2")]
    assert not (libs.folder() / "tinylib" / "old_only.py").exists() and not stage.exists()
