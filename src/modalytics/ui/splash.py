"""
The loading window shown while Modalytics starts: the logo, the version, what is being loaded, and a border whose
brand colours slowly turn around the window.

    ╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮   the border: a conical gradient of the brand colours
    ┃                                              ┃   (``brand.GRADIENT``), turning once every few seconds
    ┃              ◺  Modalytics                   ┃
    ┃     Multimodal imaging analysis · 0.1.1      ┃
    ┃                                              ┃
    ┃  Loading the rendering engine          2 / 6 ┃   the step running, and how many there are
    ┃  wgpu · pygfx                                ┃   what it loads
    ┃  ▬▬▬▬▬▬▬▬▬▬▬▬▬────────────────────────────── ┃   progress, in the brand colours
    ╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯

Everything is painted (no child widgets), always in the brand's dark colours whatever the theme. The window
follows a ``startup.Loader`` (``wait_for``), which loads in a background thread so the animation stays smooth;
steps the caller runs itself on the main thread are announced with ``step`` (the animation pauses during them,
as Qt cannot paint while the main thread is busy). ``finish`` fades the window out once the main window shows.
"""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import brand
from modalytics.startup import Loader, LoadState
from modalytics.ui.logo import logo_pixmap

# Colours (always dark: the white logo on the brand's darkest tones)
TOP = brand.DARK_TOP            # background, top (a dark indigo)
BOTTOM = brand.DARK_BOTTOM      # background, bottom
TEXT = brand.DARK_TEXT
MUTED = brand.DARK_MUTED
TRACK = "#2c2239"               # the progress bar's empty part

SIZE = QtCore.QSize(560, 330)
RADIUS = 18.0                   # corner radius (px)
BORDER = 3.0                    # border width (px)
TURN_SECONDS = 4.0              # one full turn of the border's colours
MIN_SECONDS = 0.9               # shown at least this long (a shorter flash looks like a glitch)


class SplashWindow(QtWidgets.QWidget):
    """The loading window.

    Args:
        version: The version shown under the logo.
        tagline: The words before the version.
    """

    def __init__(self, version: str, tagline: str = brand.TAGLINE) -> None:
        super().__init__(None, QtCore.Qt.WindowType.SplashScreen | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)    # the corners stay see-through
        self.setWindowTitle("Modalytics")
        self.setFixedSize(SIZE)
        self.version = version
        self.tagline = tagline
        self.label = "Starting"
        self.detail = ""
        self.index, self.total = 0, 0
        self._progress = 0.0            # the bar's width as drawn (eases towards the target)
        self._target = 0.0              # the share of the steps done
        self._shown_at = time.perf_counter()
        self._clock = QtCore.QElapsedTimer()
        self._clock.start()
        self._logo: QtGui.QPixmap | None = None
        self._fade: QtCore.QPropertyAnimation | None = None
        # About 60 frames a second: the border turns, the bar eases forward
        self._timer = QtCore.QTimer(self, interval=16, timeout=self._tick)
        self._timer.start()

    # ------------------------------------------------------------ showing progress
    def show(self) -> None:
        """Show the window in the middle of the screen under the mouse (where the user is looking)."""
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos()) or QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            self.move(screen.availableGeometry().center() - self.rect().center())
        super().show()
        self._shown_at = time.perf_counter()
        self.raise_()

    def set_state(self, state: LoadState) -> None:
        """Show a loader's state: the step running, what it loads, the progress."""
        self.label, self.detail = state.label, state.detail
        # Once done, the last step stays on show (with its result, e.g. the GPU found) until ``step`` moves on
        self.index, self.total = (state.index - 1 if state.done else state.index), state.total
        self._target = state.fraction

    def step(self, label: str, detail: str = "") -> None:
        """Announce a step the caller runs itself (on the main thread), and paint it before the work starts.

        Args:
            label: What it is: "Building the workspace".
            detail: What it involves.
        """
        self.label, self.detail = label, detail
        self.index = min(self.index + 1, max(self.total - 1, 0))       # the step after the one shown
        self._target = self.index / self.total if self.total else 0.0   # the steps before it are done
        self._progress = self._target                                   # no easing: frames pause during the step
        self.repaint()                                     # now: the main thread is about to be busy
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def wait_for(self, loader: Loader) -> LoadState:
        """Animate until the loader has finished (and the window has been up for ``MIN_SECONDS``).

        Args:
            loader: A started ``startup.Loader``.

        Returns:
            LoadState: Its final state (``error`` set if a step failed).
        """
        loop = QtCore.QEventLoop()
        result: list[LoadState] = []

        def poll() -> None:
            state = loader.state()
            self.set_state(state)
            if state.done and time.perf_counter() - self._shown_at >= MIN_SECONDS:
                result.append(state)
                loop.quit()

        timer = QtCore.QTimer(interval=30, timeout=poll)
        timer.start()
        poll()
        if not result:
            loop.exec()
        timer.stop()
        return result[0]

    def finish(self, window: QtWidgets.QWidget | None = None) -> None:
        """Fade out and close (after the main window has been shown).

        Args:
            window: The main window, raised above where this one was.
        """
        self.label, self.detail = "Ready", ""
        self._target = self._progress = 1.0
        if window is not None:
            window.raise_()
            window.activateWindow()
        self._fade = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(260)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.close)
        self._fade.finished.connect(self.deleteLater)
        self._fade.start()

    # ------------------------------------------------------------ animation
    def _tick(self) -> None:
        # The bar moves a fraction of the way to its target each frame: smooth, never jumping
        self._progress += (self._target - self._progress) * 0.12
        self.update()

    def _angle(self) -> float:
        """The border's turn now (degrees): one full turn every ``TURN_SECONDS``."""
        return (self._clock.elapsed() / 1000.0 / TURN_SECONDS * 360.0) % 360.0

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        outer = QtCore.QRectF(self.rect()).adjusted(BORDER / 2, BORDER / 2, -BORDER / 2, -BORDER / 2)

        # 1. Background: a dark indigo gradient, and a soft violet glow behind the logo
        background = QtGui.QLinearGradient(outer.topLeft(), outer.bottomLeft())
        background.setColorAt(0.0, QtGui.QColor(TOP))
        background.setColorAt(1.0, QtGui.QColor(BOTTOM))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(background)
        p.drawRoundedRect(outer, RADIUS, RADIUS)
        glow = QtGui.QRadialGradient(QtCore.QPointF(outer.center().x(), 112), outer.width() * 0.45)
        glow_colour = QtGui.QColor(brand.VIOLET)
        glow_colour.setAlpha(55)
        glow.setColorAt(0.0, glow_colour)
        glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        p.setBrush(glow)
        p.drawRoundedRect(outer, RADIUS, RADIUS)

        # 2. The border: the brand colours around a cone whose angle turns with time
        cone = QtGui.QConicalGradient(outer.center(), self._angle())
        stops = brand.GRADIENT
        for i, colour in enumerate(stops):
            cone.setColorAt(i / (len(stops) - 1), QtGui.QColor(colour))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QBrush(cone), BORDER))
        p.drawRoundedRect(outer, RADIUS, RADIUS)

        # 3. The white logo, centred, then the tagline and version under it
        if self._logo is None or self._logo.devicePixelRatio() != self.devicePixelRatioF():
            self._logo = logo_pixmap(54, negative=True, ratio=self.devicePixelRatioF())
        logo_w = self._logo.width() / self._logo.devicePixelRatio()
        logo_h = self._logo.height() / self._logo.devicePixelRatio()
        p.drawPixmap(QtCore.QPointF((self.width() - logo_w) / 2, 78), self._logo)
        font = QtGui.QFont(self.font())
        font.setPointSizeF(10)
        p.setFont(font)
        p.setPen(QtGui.QColor(MUTED))
        p.drawText(QtCore.QRectF(0, 78 + logo_h + 16, self.width(), 22), QtCore.Qt.AlignmentFlag.AlignCenter,
                   f"{self.tagline}  ·  version {self.version}")

        # 4. The step running (and how many), what it loads, then the progress bar
        margin = 34.0
        bar_y = self.height() - 40.0
        font.setPointSizeF(10.5)
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QtGui.QColor(TEXT))
        p.drawText(QtCore.QRectF(margin, bar_y - 50, self.width() - 2 * margin - 60, 22),
                   QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{self.label}…"
                   if self.label and self.label != "Ready" else self.label)
        font.setWeight(QtGui.QFont.Weight.Normal)
        font.setPointSizeF(9)
        p.setFont(font)
        p.setPen(QtGui.QColor(MUTED))
        if self.total:
            p.drawText(QtCore.QRectF(self.width() - margin - 60, bar_y - 50, 60, 22),
                       QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                       f"{min(self.index + 1, self.total)} / {self.total}")
        p.drawText(QtCore.QRectF(margin, bar_y - 28, self.width() - 2 * margin, 18),
                   QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, self.detail)
        track = QtCore.QRectF(margin, bar_y, self.width() - 2 * margin, 4)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(TRACK))
        p.drawRoundedRect(track, 2, 2)
        if self._progress > 0.002:
            fill = QtCore.QRectF(track.left(), track.top(), track.width() * min(self._progress, 1.0), track.height())
            colours = QtGui.QLinearGradient(track.topLeft(), track.topRight())
            for i, colour in enumerate((brand.ORCHID, brand.VIOLET, brand.PURPLE, brand.RED)):
                colours.setColorAt(i / 3, QtGui.QColor(colour))
            p.setBrush(colours)
            p.drawRoundedRect(fill, 2, 2)
        p.end()
