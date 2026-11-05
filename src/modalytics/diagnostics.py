"""
The diagnostic log (Settings › General › Diagnostics): what the program did and what went wrong, for testing.

Off by default. When on, ``logs/modalytics.log`` in the user library (a new file
after 5 MB, the five previous kept as ``modalytics.log.1`` … ``.5``) records:

* the session: Modalytics, Python, Qt and library versions, the system;
* every message the program prints (✅ ⚠️ ❌ …), including Python's own error output;
* Python errors with their traceback: uncaught ones, those in background threads,
  and those caught and shown in a dialog (``error``);
* Qt's own warnings (``install_qt``);
* each background task: what, how long, how it ended (``ui.tasks``);
* the actions used (ribbon, menus, tools), projects opened and saved, and screen
  updates that took longer than half a second.

Nothing leaves the computer: it is a text file to look at, or to send along with a
bug report. Qt-free, except ``install_qt``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import platform
import sys
import threading
from pathlib import Path

LOGGER = logging.getLogger("modalytics")
MAX_BYTES = 5 * 1024 * 1024
KEEP = 5
_state: dict = {"handler": None, "stdout": None, "stderr": None, "excepthook": None, "threadhook": None,
                "qt": False}


def log_dir() -> Path:
    from modalytics import paths
    return paths.user_dir("logs")


def log_file() -> Path:
    return log_dir() / "modalytics.log"


def is_on() -> bool:
    """Whether the log is being written now."""
    return _state["handler"] is not None


def wanted() -> bool:
    """Whether Settings › General asks for the log."""
    from modalytics import app_settings
    try:
        return bool(app_settings.get("diagnostics"))
    except KeyError:
        return False


def set_wanted(on: bool) -> None:
    """Switch the log on or off (now and at the next starts)."""
    from modalytics import app_settings
    app_settings.put("diagnostics", bool(on))
    start() if on else stop()


class _Tee:
    """A stream that also sends each complete line it is given to the log."""

    def __init__(self, stream, level: int) -> None:
        self.stream, self.level, self._buf = stream, level, ""

    def write(self, text: str) -> int:
        if self.stream is not None:
            try:
                self.stream.write(text)
            except (OSError, ValueError, UnicodeError):
                pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                LOGGER.log(self.level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.stream is not None:
            try:
                self.stream.flush()
            except (OSError, ValueError):
                pass

    def __getattr__(self, name):                          # isatty, encoding, fileno… of the real stream
        return getattr(self.stream, name)


def start() -> None:
    """Start writing the log (no-op if it already is)."""
    if is_on():
        return
    # 1. The file
    handler = logging.handlers.RotatingFileHandler(log_file(), maxBytes=MAX_BYTES, backupCount=KEEP,
                                                   encoding="utf-8", delay=False)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(handler)
    _state["handler"] = handler
    # 2. What is printed, and errors nobody caught (the previous hooks still run)
    _state["stdout"], _state["stderr"] = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, logging.INFO)
    sys.stderr = _Tee(sys.stderr, logging.WARNING)
    _state["excepthook"], _state["threadhook"] = sys.excepthook, threading.excepthook

    def excepthook(kind, value, tb):
        LOGGER.error("Uncaught error", exc_info=(kind, value, tb))
        _state["excepthook"](kind, value, tb)

    def threadhook(args):
        LOGGER.error(f"Uncaught error in thread {args.thread.name if args.thread else '?'}",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        _state["threadhook"](args)
    sys.excepthook, threading.excepthook = excepthook, threadhook
    # 3. The session
    from modalytics import __version__
    versions = {}
    for mod in ("numpy", "scipy", "shapely", "PySide6", "pygfx", "wgpu", "onnxruntime", "cv2", "sklearn", "torch"):
        m = sys.modules.get(mod)
        if m is None:
            try:
                m = __import__(mod)
            except Exception:                              # noqa: BLE001 — not installed: not listed
                continue
        versions[mod] = getattr(m, "__version__", "?")
    LOGGER.info("=" * 20 + " session " + "=" * 20)
    event("session", modalytics=__version__, python=platform.python_version(),
          system=f"{platform.system()} {platform.release()} ({platform.version()})", machine=platform.machine(),
          **versions)


def stop() -> None:
    """Stop writing the log and put everything back."""
    handler = _state["handler"]
    if handler is None:
        return
    event("log switched off")
    if isinstance(sys.stdout, _Tee):
        sys.stdout = _state["stdout"]
    if isinstance(sys.stderr, _Tee):
        sys.stderr = _state["stderr"]
    sys.excepthook, threading.excepthook = _state["excepthook"], _state["threadhook"]
    LOGGER.removeHandler(handler)
    logging.getLogger("py.warnings").removeHandler(handler)
    handler.close()
    _state["handler"] = None


def start_if_wanted() -> None:
    """At start-up: begin the log if Settings › General asks for it."""
    if wanted():
        start()


def event(what: str, **fields) -> None:
    """Record something that happened (no-op while the log is off)."""
    if is_on():
        LOGGER.info(what + (" " + json.dumps(fields, ensure_ascii=False, default=str) if fields else ""))


def error(what: str, exc: BaseException | None = None) -> None:
    """Record an error that was caught (e.g. shown to the user in a dialog), with its traceback."""
    if is_on():
        LOGGER.error(what, exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None)


def install_qt() -> None:
    """Send Qt's own messages (warnings, critical errors) to the log too."""
    if _state["qt"]:
        return
    from PySide6 import QtCore
    levels = {QtCore.QtMsgType.QtDebugMsg: logging.DEBUG, QtCore.QtMsgType.QtInfoMsg: logging.INFO,
              QtCore.QtMsgType.QtWarningMsg: logging.WARNING, QtCore.QtMsgType.QtCriticalMsg: logging.ERROR,
              QtCore.QtMsgType.QtFatalMsg: logging.CRITICAL}
    previous = None

    def handler(kind, context, message):
        if is_on():
            LOGGER.log(levels.get(kind, logging.INFO), f"Qt: {message}")
        if previous is not None:
            previous(kind, context, message)
        elif kind != QtCore.QtMsgType.QtDebugMsg and _state["stderr"] is None:
            print(message, file=sys.__stderr__)
    previous = QtCore.qInstallMessageHandler(handler)
    # The warnings and errors shown to the user in a dialog: their text in the log too (the dialog is unchanged)
    from PySide6 import QtWidgets
    for name, level in (("warning", logging.WARNING), ("critical", logging.ERROR)):
        original = getattr(QtWidgets.QMessageBox, name)

        def shown(parent, title, text, *args, _original=original, _level=level, **kwargs):
            if is_on():
                LOGGER.log(_level, f"Dialog «{title}»: {' '.join(str(text).split())}")
            return _original(parent, title, text, *args, **kwargs)
        setattr(QtWidgets.QMessageBox, name, staticmethod(shown))
    _state["qt"] = True
