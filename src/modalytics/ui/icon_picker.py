"""
Choosing an icon for a processing step's or chain's toolbar button.

    ┌ Choose an icon ─────────────────────────────────────────┐
    │ [ blur                                  ]  4 icons       │
    │  ▦ blur   ▦ blur-linear   ▦ blur-off   ▦ blur-radial     │
    │                                                          │
    │ mdi6.blur-linear                          [OK] [Cancel]  │
    └──────────────────────────────────────────────────────────┘

The icons are Material Design Icons (``modalytics.icons``: brands and logos left
out), drawn in the theme's text colour like every other button. At most
``SHOWN`` are drawn at a time; typing narrows them.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import icons
from modalytics.ui.ribbon import icon

SHOWN = 300
# What the picker searches first for a step of a category that has no icon yet
CATEGORY_HINTS = {"Denoise": "blur", "Correction": "tune", "Enhance": "contrast", "Quantitative maps": "chart",
                  "Feature maps": "texture", "Chains": "link"}


class IconPicker(QtWidgets.QDialog):
    """A searchable grid of icons; ``chosen`` holds the picked name ("mdi6.…") once accepted."""

    def __init__(self, current: str = "", hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose an icon")
        self.resize(640, 520)
        self.chosen = current
        col = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.query = QtWidgets.QLineEdit()
        self.query.setPlaceholderText("Search: blur, wave, drop, microscope, cell…")
        self.query.setClearButtonEnabled(True)
        row.addWidget(self.query, 1)
        self.count = QtWidgets.QLabel("")
        self.count.setObjectName("Muted")
        row.addWidget(self.count)
        col.addLayout(row)
        self.grid = QtWidgets.QListWidget()
        self.grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.grid.setIconSize(QtCore.QSize(28, 28))
        self.grid.setGridSize(QtCore.QSize(104, 72))
        self.grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.grid.setMovement(QtWidgets.QListView.Movement.Static)
        self.grid.setWordWrap(True)
        self.grid.setUniformItemSizes(True)
        col.addWidget(self.grid, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLabel(current or "No icon chosen")
        bottom.addWidget(self.name, 1)
        note = QtWidgets.QLabel("Material Design Icons (Apache License 2.0)")
        note.setObjectName("Muted")
        bottom.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        col.addLayout(bottom)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self._fill)
        self.query.textChanged.connect(lambda _t: self._timer.start())
        self.grid.currentItemChanged.connect(self._picked)
        self.grid.itemDoubleClicked.connect(lambda _i: self.accept())
        self.query.setText(current[len(icons.PREFIX):].replace("-", " ") if current else hint)
        self._fill()

    def _fill(self) -> None:
        """The icons matching the search (at most ``SHOWN``), the current one selected."""
        found = icons.search(self.query.text())
        color = self.palette().color(self.foregroundRole()).name()
        self.grid.clear()
        for name in found[:SHOWN]:
            item = QtWidgets.QListWidgetItem(icon(name, color), name[len(icons.PREFIX):])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setToolTip(name)
            self.grid.addItem(item)
            if name == self.chosen:
                self.grid.setCurrentItem(item)
        more = f" (the first {SHOWN}: type to narrow)" if len(found) > SHOWN else ""
        self.count.setText(f"{len(found)} icon{'s' if len(found) != 1 else ''}{more}")

    def _picked(self, item, _prev=None) -> None:
        if item is not None:
            self.chosen = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.name.setText(self.chosen)


def pick_icon(parent, current: str = "", hint: str = "") -> str | None:
    """Ask for an icon; returns its name ("mdi6.…"), or None when cancelled."""
    dlg = IconPicker(current, hint, parent)
    return dlg.chosen if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.chosen else None


class IconField(QtWidgets.QWidget):
    """An icon setting: a preview, the name (editable), Choose… and Clear (empty: the default icon).

    Signals:
        changed(str): the icon name changed ("" = none).
    """

    changed = QtCore.Signal(str)

    def __init__(self, hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self.hint = hint                            # what the picker searches first (e.g. the step's category)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(26, 26)
        row.addWidget(self.preview)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Default: its category's icon")
        row.addWidget(self.edit, 1)
        choose = QtWidgets.QPushButton("Choose…")
        choose.clicked.connect(self._choose)
        row.addWidget(choose)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda: self.set_icon("", emit=True))
        row.addWidget(clear)
        self.edit.editingFinished.connect(lambda: self.set_icon(self.edit.text().strip(), emit=True))

    def icon_name(self) -> str:
        return self.edit.text().strip()

    def set_icon(self, name: str, emit: bool = False) -> None:
        self.edit.setText(name)
        color = self.palette().color(self.foregroundRole()).name()
        ok = not name or icons.allowed(name)
        self.preview.setPixmap(icon(name, color).pixmap(22, 22) if name and ok else _blank())
        self.edit.setToolTip("" if ok else "Not an icon of the set: Choose… picks one")
        if emit:
            self.changed.emit(name)

    def _choose(self) -> None:
        name = pick_icon(self, self.icon_name(), self.hint)
        if name:
            self.set_icon(name, emit=True)


def _blank() -> QtGui.QPixmap:
    pix = QtGui.QPixmap(22, 22)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    return pix
