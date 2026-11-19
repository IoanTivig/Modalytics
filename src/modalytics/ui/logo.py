"""
The Modalytics logo as a widget: the white logo on a dark theme, the purple one on a light theme.

    LogoLabel(height=20)                 the ribbon's corner
    LogoLabel(height=46)                 the start screen's title
    LogoLabel(height=22, mark_only=True) the mark alone

It follows the theme by itself: when the application's palette changes (``ui.theme.apply_theme``), it picks
the other logo. The pixmap is made at the screen's pixel ratio, so it stays sharp on high-DPI screens.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import brand


def logo_pixmap(height: int, negative: bool, mark_only: bool = False, ratio: float = 1.0) -> QtGui.QPixmap:
    """The logo scaled to a height (in logical pixels).

    Args:
        height: Height on screen (logical px); the width follows the logo's proportions.
        negative: True for the white logo (dark backgrounds), False for the purple one.
        mark_only: True for the mark alone.
        ratio: The screen's device pixel ratio (the pixmap is made that much larger, then marked so).

    Returns:
        QtGui.QPixmap: The logo (a null pixmap if the file is missing).
    """
    pix = QtGui.QPixmap(str(brand.logo_path(negative, mark_only)))      # cropped to the drawing (brand)
    if pix.isNull():
        return pix
    scaled = pix.scaledToHeight(max(1, round(height * ratio)), QtCore.Qt.TransformationMode.SmoothTransformation)
    scaled.setDevicePixelRatio(ratio)
    return scaled


def is_dark(widget: QtWidgets.QWidget | None = None) -> bool:
    """True when the application's (or the widget's) window colour is dark: then the white logo is the one to show."""
    palette = widget.palette() if widget is not None else QtWidgets.QApplication.instance().palette()
    return palette.color(QtGui.QPalette.ColorRole.Window).lightness() < 128


class LogoLabel(QtWidgets.QLabel):
    """The logo, white on a dark theme and purple on a light one.

    Args:
        height: Height on screen (logical px).
        mark_only: True for the mark alone, False for the mark and the name.
        negative: Always the white (True) or the purple (False) logo; None follows the theme.
    """

    def __init__(self, height: int = 22, mark_only: bool = False, negative: bool | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AppLogo")
        self.setToolTip("Modalytics")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._height = height
        self._mark_only = mark_only
        self._fixed = negative
        self._shown: tuple | None = None          # (negative, ratio) of the pixmap shown, to redo it only on change
        self._refresh()

    def _refresh(self) -> None:
        """Pick the logo for the current theme and screen (only when one of them changed)."""
        negative = is_dark() if self._fixed is None else self._fixed
        ratio = self.devicePixelRatioF()
        if self._shown == (negative, ratio):
            return
        self._shown = (negative, ratio)
        self.setPixmap(logo_pixmap(self._height, negative, self._mark_only, ratio))

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 (Qt)
        # The theme changed (new palette or style sheet): maybe the other logo
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.StyleChange,
                            QtCore.QEvent.Type.ApplicationPaletteChange):
            self._refresh()
        super().changeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 (Qt)
        self._refresh()                                     # the screen (pixel ratio) is known once shown
        super().showEvent(event)
