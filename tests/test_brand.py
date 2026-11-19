"""
The brand: the palette in the themes, the logo files, the start-up loader and the loading window.
"""

import os

import pytest

from modalytics import brand, startup


def test_the_palette_and_the_logo_files():
    assert brand.PALETTE == ("#B964A6", "#864CA0", "#653988", "#3A2665", "#B72E3B", "#A2142F")
    assert brand.GRADIENT[0] == brand.GRADIENT[-1]                 # the turning border has no seam
    for negative in (False, True):
        assert brand.source_path(negative).exists()
        for mark_only in (False, True):
            assert brand.logo_path(negative, mark_only).exists(), "run python packaging/make_icon.py"
    assert brand.logo_data_uri().startswith("data:image/png;base64,")
    assert "<img" in brand.logo_img() and "alt='Modalytics'" in brand.logo_img()


def test_the_themes_take_their_accent_from_the_palette():
    from modalytics.ui.theme import TOKENS
    assert TOKENS["dark"]["accent"] == brand.VIOLET and TOKENS["dark"]["selection"] == brand.INDIGO
    assert TOKENS["light"]["accent"] == brand.PURPLE and TOKENS["light"]["danger"] == brand.CRIMSON


def test_the_loader_runs_the_steps_in_order_and_reports_results():
    ran = []
    steps = [startup.Step("One", "first", lambda: ran.append(1)),
             startup.Step("Two", "second", lambda: ran.append(2) or "found it")]
    state = startup.Loader(steps, extra=2).start().wait(10)
    assert ran == [1, 2] and state.done and state.error is None
    assert [label for label, _s in state.log] == ["One", "Two"]
    assert state.detail == "found it"                             # a step's result replaces its detail
    assert (state.index, state.total, state.fraction) == (2, 4, 0.5)   # the caller's 2 steps are still to come


def test_the_loader_stops_at_a_failing_step():
    def broken():
        raise ImportError("no module named nowhere")
    steps = [startup.Step("Fails", "", broken), startup.Step("Never", "", lambda: pytest.fail("ran after a failure"))]
    state = startup.Loader(steps).start().wait(10)
    assert state.done and isinstance(state.error, ImportError) and state.log == []


def test_the_loading_window_paints_and_follows_a_loader(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    from modalytics.ui import splash as sp
    monkeypatch.setattr(sp, "MIN_SECONDS", 0.0)
    window = sp.SplashWindow("9.9.9")
    window.show()
    loader = startup.Loader([startup.Step("Loading", "things", lambda: "done")], extra=1).start()
    state = window.wait_for(loader)
    assert state.done and window.detail == "done" and (window.index, window.total) == (0, 2)
    window.step("Building the workspace")
    assert (window.label, window.index) == ("Building the workspace", 1)
    image = window.grab().toImage()                               # paints without error
    assert image.width() == sp.SIZE.width()
    window.finish()
    window.close()
