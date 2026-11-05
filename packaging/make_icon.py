"""
The application icon: the app's own icon style (Material Design Icons, white) on its accent colour.

Writes ``src/modalytics/resources/icons/modalytics.png`` (256 px: the window icon) and ``packaging/modalytics.ico``
(16–256 px: the program, its shortcuts and the installer). Run again after changing it::

    python packaging/make_icon.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLYPH = "mdi6.microscope"
ACCENT = ("#24a07f", "#17654f")          # the theme's accent, top to bottom
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> int:
    from PySide6 import QtCore, QtGui, QtWidgets
    import qtawesome
    from PIL import Image

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])   # noqa: F841 (qtawesome)
    # 1. 256 px: a rounded square in the accent colour, the glyph in white
    size = 256
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    grad = QtGui.QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0, QtGui.QColor(ACCENT[0]))
    grad.setColorAt(1, QtGui.QColor(ACCENT[1]))
    p.setBrush(grad)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.drawRoundedRect(QtCore.QRectF(8, 8, size - 16, size - 16), 48, 48)
    glyph = qtawesome.icon(GLYPH, color="white").pixmap(176, 176)
    p.drawPixmap((size - 176) // 2, (size - 176) // 2, glyph)
    p.end()
    png = os.path.join(ROOT, "src", "modalytics", "resources", "icons", "modalytics.png")
    img.save(png)
    # 2. The .ico with every size Windows asks for
    ico = os.path.join(ROOT, "packaging", "modalytics.ico")
    Image.open(png).save(ico, sizes=[(s, s) for s in SIZES])
    print(f"✅ {png}\n✅ {ico}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
