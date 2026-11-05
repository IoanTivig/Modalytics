"""
Small widgets shared by the Settings pages.

``setting_widget`` builds the input for one declared ``Setting`` (the same
kinds as the Tool Settings panel: float, int, choice, bool), so a parameter's
settings in a profile look and behave like a tool's.
"""

from __future__ import annotations

import math
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.core.settings import Setting


def setting_widget(s: Setting, value, on_change: Callable[[object], None], classes=None) -> QtWidgets.QWidget:
    """An input for one setting.

    Args:
        s: The declared setting.
        value: Its current value.
        on_change: Called with the new value whenever the user changes it.
        classes: The annotation classes a "label" setting chooses from (a ``LabelSchema``; default: the colonic
            default schema).

    Returns:
        QWidget: Spin box, combo box or check box.
    """
    if s.kind in ("float", "int"):
        w = QtWidgets.QDoubleSpinBox() if s.kind == "float" else QtWidgets.QSpinBox()
        if s.kind == "float":
            w.setDecimals(min(6, max(2, 1 - math.floor(math.log10(s.step or 1)))))
        w.setRange(s.minimum if s.minimum is not None else -1e9, s.maximum if s.maximum is not None else 1e9)
        w.setSingleStep(s.step or 1)
        w.setSuffix(s.suffix)
        w.setValue(value)
        w.valueChanged.connect(on_change)
    elif s.kind == "choice":
        w = QtWidgets.QComboBox()
        w.addItems(s.choices() if callable(s.choices) else s.choices)
        w.setCurrentText(str(value))
        w.currentTextChanged.connect(on_change)
    elif s.kind == "bool":
        w = QtWidgets.QCheckBox()
        w.setChecked(bool(value))
        w.toggled.connect(on_change)
    elif s.kind == "label":
        # An annotation class: every group (= all its subgroups) and each subgroup
        from modalytics.core.annotations import default_schema, label_text
        w = QtWidgets.QComboBox()
        current = s.clean(value)
        choices = (classes or default_schema()).choices()
        if current not in choices:
            choices = [current] + choices
        for group, sub in choices:
            w.addItem(label_text(group, sub) + ("  (all)" if not sub else ""), [group, sub])
        w.setCurrentIndex(choices.index(current))
        w.currentIndexChanged.connect(lambda _i: on_change(tuple(w.currentData())))
    elif s.kind == "text":
        w = QtWidgets.QLineEdit(str(value or ""))
        w.textChanged.connect(lambda t: on_change(t))
    elif s.kind in ("file", "folder"):
        # A path, typed or chosen with "…"
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(str(value or ""))
        edit.setPlaceholderText("Choose a file…" if s.kind == "file" else "Choose a folder…")
        edit.editingFinished.connect(lambda: on_change(edit.text().strip()))
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(32)

        def choose():
            if s.kind == "folder":
                path = QtWidgets.QFileDialog.getExistingDirectory(w, s.label, edit.text())
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(w, s.label, edit.text())
            if path:
                edit.setText(path)
                on_change(path)
        browse.clicked.connect(choose)
        row.addWidget(edit, 1)
        row.addWidget(browse)
    else:
        w = QtWidgets.QLabel(str(value))
    if s.tooltip:
        w.setToolTip(s.tooltip)
    return w


def header_tips(tree: QtWidgets.QTreeWidget, tips: list[str]) -> QtWidgets.QTreeWidget:
    """One tooltip per column title (short lines: "\\n" breaks them), in the columns' order; "" for none."""
    head = tree.headerItem()
    for c, tip in enumerate(tips):
        if tip:
            head.setToolTip(c, tip)
    return tree


def list_tree(headers: list[str] | None = None) -> QtWidgets.QTreeWidget:
    """A tree used as a list: whole rows are selected; group rows open and close on a click."""
    tree = QtWidgets.QTreeWidget()
    tree.setObjectName("ListTree")
    tree.setAllColumnsShowFocus(True)
    tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    if headers:
        tree.setHeaderLabels(headers)
    else:
        tree.setHeaderHidden(True)

    def toggle(item, _col):
        if not item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable and item.childCount():
            item.setExpanded(not item.isExpanded())
    tree.itemClicked.connect(toggle)
    return tree


def clear_tree(tree: QtWidgets.QTreeWidget) -> None:
    """Empty a tree so that Python references still held to its old rows stay safe.

    ``QTreeWidget.clear()`` deletes the rows in C++ while Python may still hold
    them (a slot's ``cur`` / ``prev``, a list kept by a script); touching or
    collecting those later crashes now and then (seen on Windows, with a real
    screen). Taking the rows out instead hands them to Python, which deletes
    them when the last reference goes.
    """
    while tree.topLevelItemCount():
        tree.takeTopLevelItem(tree.topLevelItemCount() - 1)


def is_group(item) -> bool:
    """True for a group row (not selectable): selection handlers ignore it."""
    return item is not None and not item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable


def keep_on(tree: QtWidgets.QTreeWidget, item) -> None:
    """Make ``item`` current again without notifying anyone (after a click on a group row)."""
    tree.blockSignals(True)
    tree.setCurrentItem(item)
    tree.blockSignals(False)


def group_item(parent, text: str) -> QtWidgets.QTreeWidgetItem:
    """A group row (scope, category…): shown in bold, not selectable, opens and closes on a click."""
    item = QtWidgets.QTreeWidgetItem(parent, [text]) if not isinstance(parent, QtWidgets.QTreeWidget) \
        else QtWidgets.QTreeWidgetItem([text])
    if isinstance(parent, QtWidgets.QTreeWidget):
        parent.addTopLevelItem(item)
    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
    font = item.font(0)
    font.setBold(True)
    item.setFont(0, font)
    return item


def swatch(rgba, size: int = 14) -> QtGui.QIcon:
    """A small square of colour (RGBA floats 0–1)."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor.fromRgbF(*[float(c) for c in rgba[:3]]))
    return QtGui.QIcon(pm)


def muted(text: str, wrap: bool = True) -> QtWidgets.QLabel:
    """A secondary-text label."""
    label = QtWidgets.QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def section(text: str) -> QtWidgets.QLabel:
    """A small upper-case section title."""
    label = QtWidgets.QLabel(text.upper())
    label.setObjectName("SectionTitle")
    return label


def open_in_system(path: str) -> None:
    """Open a file or folder with the system's default application."""
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
