"""What the installed program relies on, checked from the source (packaging/build.py smoke-tests the build)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import modalytics
from modalytics import paths
from modalytics.parameters import libraries

ROOT = Path(__file__).resolve().parents[1]


def _build():
    spec = importlib.util.spec_from_file_location("packaging_build", ROOT / "packaging" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_version():
    """The program says the version the installer is named after."""
    assert _build().version() == modalytics.__version__


def test_windows_version_numbers():
    numeric = _build().numeric
    assert numeric("0.1.0.dev0") == "0.1.0.0"
    assert numeric("1.2") == "1.2.0.0"
    assert numeric("2.0.3rc1") == "2.0.3.0"


def test_from_the_source():
    assert not paths.installed()
    assert paths.app_root() == ROOT and (paths.app_root() / "models").is_dir()
    assert paths.helper_command() == [sys.executable, "-m", "modalytics"]
    assert libraries.pip() == [sys.executable, "-m", "pip"]


def test_installed(monkeypatch, tmp_path):
    """Installed: the program's files are in the bundle; child processes run the console twin."""
    exe = tmp_path / "Modalytics.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert paths.app_root() == tmp_path / "_internal"
    assert paths.helper_command() == [str(tmp_path / paths.CLI_EXE)]
    assert libraries.pip() == [str(tmp_path / paths.CLI_EXE), "--pip"]


def test_pip_through_the_program(capsys, monkeypatch):
    """``modalytics --pip …`` is ``python -m pip …`` (Settings › Libraries in the installed program)."""
    from modalytics.__main__ import main
    monkeypatch.setattr(sys, "argv", list(sys.argv))          # pip's run replaces it (its own process, normally)
    assert main(["--pip", "--version"]) == 0
    assert capsys.readouterr().out.startswith("pip ")
