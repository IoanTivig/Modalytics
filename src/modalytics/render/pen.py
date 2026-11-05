"""
Pen input (graphics tablets: XP-Pen, Wacom Bamboo / Intuos…) and the pointer points the canvas would lose.

The canvas gets its pointer events from rendercanvas, which (1) knows nothing of
tablets — Qt turns the pen into a mouse, so the tip draws but its pressure and its
eraser end are lost — and (2) keeps only the last pointer move between two screen
updates, so a quick stroke (a pen reports 100–200+ points a second) loses most of
its points.

``PenInput`` watches the render widget's events before rendercanvas sees them:

* tablet events (``QTabletEvent``): the pressure (0–1) and which end of the pen
  touches (``"pen"`` or ``"eraser"``); they are not consumed, so Qt still makes the
  mouse events the canvas works with;
* every mouse move: its position (and the pen's pressure then) in a *trail*, which
  the canvas replays to the active tool, so no point of a stroke is lost.

On Windows Qt reads the pen through Windows Ink; the tablet driver's "Windows Ink"
option must be on (it is by default for XP-Pen and Wacom). Without a tablet nothing
changes: the mouse is ``"mouse"`` with no pressure.
"""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui

TABLET_FRESH_S = 0.25        # a mouse event this soon after a tablet event comes from the pen


class PenInput(QtCore.QObject):
    """Watches a widget's tablet and mouse events (see the module docstring).

    Args:
        widget: The render widget.
    """

    def __init__(self, widget) -> None:
        super().__init__(widget)
        self.pointer = "mouse"
        self.pressure: float | None = None
        self._tablet_t = -1.0
        self.trail: list[tuple[float, float, float | None]] = []
        self.press: tuple[str, float | None] = ("mouse", None)   # the pointer and pressure when a button went down
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TabletTracking, True)     # the pen hovering moves too
        widget.installEventFilter(self)

    def current(self) -> tuple[str, float | None]:
        """("pen" | "eraser" | "mouse", pressure or None) of the pointer now."""
        if time.perf_counter() - self._tablet_t < TABLET_FRESH_S:
            return self.pointer, self.pressure
        return "mouse", None

    def take_trail(self) -> list[tuple[float, float, float | None]]:
        """The pointer positions (widget px) and pressures since the last call."""
        out, self.trail = self.trail, []
        return out

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        t = event.type()
        E = QtCore.QEvent.Type
        if t in (E.TabletPress, E.TabletMove, E.TabletRelease):
            # 1. The pen: its pressure and which end touches (the event goes on: Qt makes the mouse event from it)
            self._tablet_t = time.perf_counter()
            self.pressure = float(event.pressure()) if t != E.TabletRelease else self.pressure
            device = event.pointingDevice()
            eraser = device is not None and device.pointerType() == QtGui.QPointingDevice.PointerType.Eraser
            self.pointer = "eraser" if eraser else "pen"
            return False
        if t == E.MouseButtonPress:
            # 2. A new gesture: its trail starts here (events are recorded as Qt delivers them, before the canvas
            #    handles them: the press's pointer and pressure are kept for it)
            pos = event.position()
            self.press = self.current()
            self.trail = [(pos.x(), pos.y(), self.press[1])]
        elif t == E.MouseMove:
            pos = event.position()
            self.trail.append((pos.x(), pos.y(), self.current()[1]))
        return False
