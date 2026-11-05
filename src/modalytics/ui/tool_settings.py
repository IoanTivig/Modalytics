"""
Tool Settings panel: the settings of the active tool, built from its
declared ``Setting`` list (tools never build their own widgets).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.core.annotations import label_text
from modalytics.tools.base import Setting
from modalytics.tools.manager import ToolManager


def _swatch(rgba) -> QtGui.QIcon:
    pm = QtGui.QPixmap(12, 12)
    pm.fill(QtGui.QColor.fromRgbF(*rgba[:3]))
    return QtGui.QIcon(pm)


def _fit_combo(combo: QtWidgets.QComboBox) -> None:
    """Let a drop-down shrink with the dock (long entries such as GPU names are elided)
    while its open list stays wide enough to show every entry in full."""
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 24)


class ToolSettingsPanel(QtWidgets.QWidget):
    """Shows the active tool's name, usage hint and settings form.

    Args:
        tools: The tool manager.
        project: The project (for the annotation classes).
    """

    def __init__(self, tools: ToolManager, project, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.tools = tools
        self.project = project
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._updating = False

        self.title = QtWidgets.QLabel("No tool")
        self.title.setStyleSheet("font-weight: 600; font-size: 11pt;")
        self.hint = QtWidgets.QLabel("")
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        self.info = QtWidgets.QLabel("")              # live status line set by the tool
        self.info.setWordWrap(True)
        self.info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.info.hide()
        self.form = QtWidgets.QFormLayout()
        self.form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addWidget(self.info)
        layout.addLayout(self.form)
        layout.addStretch(1)

        tools.active_changed.connect(lambda _t: self.rebuild())
        tools.value_changed.connect(self._value_changed)
        tools.info_changed.connect(lambda t, text: t is self.tools.active and self._show_info(text))
        project.structure_changed.connect(self._refresh_labels)     # new classes from imports
        self.rebuild()

    # 1. Build the form for the active tool
    def rebuild(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets.clear()
        tool = self.tools.active
        if tool is None:
            self.title.setText("No tool")
            self.hint.setText("")
            return
        self.title.setText(tool.label + (f"  ({tool.shortcut})" if tool.shortcut else ""))
        self.hint.setText(tool.hint)
        self._show_info(self.tools.info(tool))
        for s in tool.settings:
            w = self._make_widget(tool, s)
            if s.tooltip:
                w.setToolTip(s.tooltip)
            self._widgets[s.key] = w
            self.form.addRow(s.label, w)

    def _show_info(self, text: str) -> None:
        self.info.setText(text)
        self.info.setVisible(bool(text))

    def _make_widget(self, tool, s: Setting) -> QtWidgets.QWidget:
        value = self.tools.value(tool, s.key)
        if s.kind in ("float", "int"):
            w = QtWidgets.QDoubleSpinBox() if s.kind == "float" else QtWidgets.QSpinBox()
            if s.kind == "float":
                w.setDecimals(2)
            w.setRange(s.minimum, s.maximum)
            w.setSingleStep(s.step or 1)
            w.setSuffix(s.suffix)
            w.setValue(value)
            w.valueChanged.connect(lambda v, k=s.key: self._set(k, v))
        elif s.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems(s.choices() if callable(s.choices) else s.choices)
            w.setCurrentText(value)
            w.currentTextChanged.connect(lambda v, k=s.key: self._set(k, v))
            _fit_combo(w)
        elif s.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, k=s.key: self._set(k, v))
        elif s.kind == "label":
            w = QtWidgets.QComboBox()
            w.setProperty("allow_empty", tuple(s.default or ()) == ("", ""))
            self._fill_labels(w, value)
            self._adopt_shown(tool, s.key, w)
            w.currentIndexChanged.connect(lambda _i, k=s.key, c=w: self._set(k, c.currentData()))
            _fit_combo(w)
        else:
            w = QtWidgets.QLabel(str(value))
        return w

    def _fill_labels(self, combo: QtWidgets.QComboBox, current) -> None:
        combo.blockSignals(True)
        combo.clear()
        if combo.property("allow_empty"):
            combo.addItem("Model's own class", ("", ""))
        for group, sub in self.project.labels.choices():
            combo.addItem(_swatch(self.project.labels.color(group, sub)), label_text(group, sub), (group, sub))
        # Compared in Python: Qt's findData does not match tuples, which showed the first class whatever was set
        want = tuple(current) if current else None
        idx = next((i for i in range(combo.count())
                    if combo.itemData(i) is not None and tuple(combo.itemData(i)) == want), -1)
        combo.setCurrentIndex(max(idx, 0))
        combo.blockSignals(False)
        _fit_combo(combo)

    def _refresh_labels(self) -> None:
        tool = self.tools.active
        for s in (tool.settings if tool else []):
            if s.kind == "label" and s.key in self._widgets:
                combo = self._widgets[s.key]
                self._fill_labels(combo, self.tools.value(tool, s.key))
                self._adopt_shown(tool, s.key, combo)

    def _adopt_shown(self, tool, key: str, combo) -> None:
        """If the remembered class is not one of the project's, the tool takes the one the box shows (else it
        would keep drawing a class the profile no longer has)."""
        shown = combo.currentData()
        if shown is not None and tuple(self.tools.value(tool, key) or ()) != tuple(shown):
            self.tools.set_value(tool, key, shown)

    # 2. Widget ↔ manager
    def _set(self, key: str, value) -> None:
        if not self._updating and self.tools.active is not None:
            self.tools.set_value(self.tools.active, key, value)

    def _value_changed(self, tool, key: str, value) -> None:
        """Reflect changes made elsewhere (e.g. [ and ] change the brush size)."""
        w = self._widgets.get(key)
        if tool is not self.tools.active or w is None:
            return
        self._updating = True
        if isinstance(w, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
            w.setValue(value)
        elif isinstance(w, QtWidgets.QComboBox) and isinstance(value, str):
            w.setCurrentText(value)
        self._updating = False
