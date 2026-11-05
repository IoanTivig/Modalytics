"""Shared fixtures: paths to the real test files (``testdata``; tests needing them are skipped if absent),
and a private user library so tests never touch the real one (%APPDATA%/Modalytics on Windows)."""

import pytest

import testdata


@pytest.fixture(autouse=True)
def private_user_library(tmp_path, monkeypatch):
    """Every test gets an empty user library of its own."""
    home = tmp_path / "modalytics_home"
    monkeypatch.setenv("MODALYTICS_HOME", str(home))
    return home


@pytest.fixture
def wsi_zarr():
    return testdata.need(testdata.SLIDE_ZARR)


@pytest.fixture
def phase_txt():
    return testdata.need(testdata.PHASE_TXT)


@pytest.fixture
def hpro():
    return testdata.need(testdata.HPRO)
