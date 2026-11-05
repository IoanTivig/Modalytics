"""
Ribbon toolbar (Autodesk-style): tabs, each holding titled groups of buttons.

The ribbon only *shows* actions; what an action does is defined by the main
window (commands) or by the tool manager (tools). Tool settings never live
here: they appear in the Tool Settings panel.

    ┌ Modalytics │ Arrange  Process  Annotate  Measure  Classify  Report ─── ▦ ☀ │ 🐞 ? ┐
    │ [▣ Select] [✥ Move]  │ [⟲ Reset] [⇡ Front] [⇣ Back] │ [⧉ Register…]  │
    │        Tools         │          Placement           │  Registration  │
    └──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def icon(name: str, color: str) -> QtGui.QIcon:
    """A Material Design icon (qtawesome) in the given colour; empty icon if unavailable."""
    try:
        import qtawesome as qta
        return qta.icon(name, color=color, color_active=color, color_disabled="#6b7685")
    except Exception:
        return QtGui.QIcon()


class RibbonButton(QtWidgets.QToolButton):
    """A large ribbon button: never narrower than its label (nor than ``MIN_WIDTH``), so labels are never cut; a
    tab that does not fit scrolls sideways instead (``RibbonScroll``)."""

    MIN_WIDTH = 68

    def sizeHint(self) -> QtCore.QSize:
        s = super().sizeHint()
        return QtCore.QSize(max(s.width(), self.MIN_WIDTH), s.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()


class RibbonGroup(QtWidgets.QFrame):
    """A titled group of large buttons inside a ribbon tab."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonGroup")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._row = QtWidgets.QHBoxLayout()
        self._row.setContentsMargins(4, 2, 4, 0)
        self._row.setSpacing(2)
        label = QtWidgets.QLabel(title)
        label.setObjectName("RibbonGroupTitle")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(4, 0, 4, 2)
        col.setSpacing(0)
        col.addLayout(self._row)
        col.addWidget(label)

    def add(self, action: QtGui.QAction) -> QtWidgets.QToolButton:
        """Add a large button for an action (text under icon)."""
        btn = RibbonButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIconSize(QtCore.QSize(24, 24))
        btn.setObjectName("RibbonButton")
        btn.setAutoRaise(True)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        self._row.addWidget(btn)
        return btn


class RibbonScroll(QtWidgets.QScrollArea):
    """A ribbon tab's buttons, scrolling sideways when the window is too narrow for them (a thin scrollbar
    appears only then; the mouse wheel scrolls too)."""

    def __init__(self, page: QtWidgets.QWidget) -> None:
        super().__init__()
        self.setObjectName("RibbonScroll")
        self.setWidget(page)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.horizontalScrollBar().setFixedHeight(8)

    def _fit_height(self) -> None:
        page = self.widget()
        overflow = page.minimumSizeHint().width() > self.viewport().width() + 1
        self.setFixedHeight(page.sizeHint().height() + (self.horizontalScrollBar().height() if overflow else 0))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_height()

    def wheelEvent(self, event) -> None:
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            bar.setValue(bar.value() - event.angleDelta().y() - event.angleDelta().x())
            event.accept()
        else:
            super().wheelEvent(event)


class Ribbon(QtWidgets.QWidget):
    """Tabs of button groups, with the application title on the left."""

    def __init__(self, title: str = "Modalytics", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Ribbon")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.tabbar = QtWidgets.QTabBar()
        self.tabbar.setObjectName("RibbonTabs")
        self.tabbar.setDrawBase(False)
        self.tabbar.setExpanding(False)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("RibbonBody")
        self.tabbar.currentChanged.connect(self.stack.setCurrentIndex)
        self._pages: dict[str, QtWidgets.QHBoxLayout] = {}

        app_title = QtWidgets.QLabel(title)
        app_title.setObjectName("AppTitle")
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(6, 4, 6, 0)
        top.addWidget(app_title)
        top.addWidget(self.tabbar)
        top.addStretch(1)
        self.corner = QtWidgets.QHBoxLayout()      # right side of the tab row: small icon buttons
        self.corner.setSpacing(2)
        top.addLayout(self.corner)
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addLayout(top)
        col.addWidget(self.stack)

    def add_tab(self, name: str) -> None:
        """Add an empty tab."""
        page = QtWidgets.QWidget()
        page.setObjectName("RibbonPage")
        row = QtWidgets.QHBoxLayout(page)
        row.setContentsMargins(6, 4, 6, 2)
        row.setSpacing(6)
        row.addStretch(1)
        self.stack.addWidget(RibbonScroll(page))
        self.tabbar.addTab(name)
        self._pages[name] = row

    def add_group(self, tab: str, title: str, actions: list[QtGui.QAction]) -> RibbonGroup:
        """Add a group of buttons to a tab (a separator line follows each group)."""
        row = self._pages[tab]
        group = RibbonGroup(title)
        for a in actions:
            group.add(a)
        sep = QtWidgets.QFrame()
        sep.setObjectName("RibbonSeparator")
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        row.insertWidget(row.count() - 1, group)
        row.insertWidget(row.count() - 1, sep)
        group.separator = sep
        return group

    def remove_group(self, group: RibbonGroup) -> None:
        """Take a group (and its separator) off its tab (e.g. the Process tab rebuilt for another profile)."""
        for w in (group, getattr(group, "separator", None)):
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def add_corner(self, action: QtGui.QAction) -> QtWidgets.QToolButton:
        """A small icon-only button at the right end of the tab row (its name in the tooltip)."""
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        btn.setIconSize(QtCore.QSize(18, 18))
        btn.setObjectName("RibbonCorner")
        btn.setAutoRaise(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.corner.addWidget(btn)
        return btn

    def add_corner_separator(self) -> None:
        """A short vertical line between corner buttons (view toggles | help)."""
        sep = QtWidgets.QFrame()
        sep.setObjectName("RibbonCornerSeparator")
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFixedHeight(18)
        self.corner.addSpacing(4)
        self.corner.addWidget(sep)
        self.corner.addSpacing(4)

    def show_tab(self, name: str) -> None:
        """Switch to a tab by name."""
        for i in range(self.tabbar.count()):
            if self.tabbar.tabText(i) == name:
                self.tabbar.setCurrentIndex(i)
