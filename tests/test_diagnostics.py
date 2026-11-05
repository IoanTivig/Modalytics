"""The diagnostic log: off by default; when on, messages, events and errors (also in threads) are written."""

from __future__ import annotations

import sys
import threading

from modalytics import diagnostics


def _text() -> str:
    for h in diagnostics.LOGGER.handlers:
        h.flush()
    return diagnostics.log_file().read_text(encoding="utf-8")


def test_off_by_default_then_on_and_off_again():
    assert not diagnostics.wanted() and not diagnostics.is_on()
    diagnostics.event("nothing is written while off")
    out_before = sys.stdout
    diagnostics.set_wanted(True)
    try:
        assert diagnostics.is_on() and diagnostics.wanted()
        print("✅ a printed message")
        diagnostics.event("task finished", task="Measuring", seconds=1.5)
        try:
            raise ValueError("caught and shown")
        except ValueError as exc:
            diagnostics.error("could not open x", exc)

        def boom():
            raise RuntimeError("in a thread")
        t = threading.Thread(target=boom, name="worker")
        t.start()
        t.join()
        text = _text()
        assert "session" in text and '"modalytics"' in text                  # versions of the session
        assert "✅ a printed message" in text
        assert 'task finished {"task": "Measuring", "seconds": 1.5}' in text
        assert "could not open x" in text and "ValueError: caught and shown" in text and "Traceback" in text
        assert "Uncaught error in thread worker" in text and "RuntimeError: in a thread" in text
        assert "nothing is written while off" not in text
    finally:
        diagnostics.set_wanted(False)
    assert not diagnostics.is_on() and sys.stdout is out_before
    size = len(_text())
    print("not logged any more")
    assert len(_text()) - size < 200 and "not logged any more" not in _text()
