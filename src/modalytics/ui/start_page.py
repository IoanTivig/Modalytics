"""
Start screen: shown when no project is open.

    ┌──────────────────────────────────────────────────────────────────┐
    │ Modalytics                                                        │
    │ ┌──────────────┐  Tissue   Cells   Cell kinetics                  │
    │ │ ▶ Analyze    │  ┌──────────────────────────────────────────┐    │
    │ │   Training   │  │ Colonic tissue                            │    │
    │ │   Settings   │  │ Colon biopsies … · Whole slide, Phase     │    │
    │ │              │  └──────────────────────────────────────────┘    │
    │ │ Open project…│  [New project]                                  │
    │ └──────────────┘  RECENT PROJECTS                                 │
    └──────────────────────────────────────────────────────────────────┘

Analyze lists the profiles of each kind; "New project" (or a double-click)
starts an empty project with a copy of that profile. Training opens the
Training section (classifiers from parameters; segmentation models later).
Settings opens the Settings window.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from modalytics import app_settings
from modalytics.core.model import IMAGE_TYPES
from modalytics.profiles.library import ProfileLibrary
from modalytics.profiles.model import KINDS

UID = QtCore.Qt.ItemDataRole.UserRole


class StartPage(QtWidgets.QWidget):
    """The start screen.

    Signals:
        new_project(object, str): (a copy of the chosen profile, its library id).
        open_requested(str): a project path, or "" to show the Open dialog.
        settings_requested(str): the Settings page to show ("General", "Profiles"…).
    """

    new_project = QtCore.Signal(object, str)
    open_requested = QtCore.Signal(str)
    settings_requested = QtCore.Signal(str)
    training_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StartPage")
        self.library = ProfileLibrary()
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(48, 36, 48, 36)
        outer.setSpacing(36)

        # 1. Left column: what to do
        left = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Modalytics")
        title.setObjectName("AppTitle")
        title.setStyleSheet("font-size: 22pt; padding: 0 0 18px 0;")
        left.addWidget(title)
        self.analyze_btn = self._big_button("Analyze", "Start a project from a profile")
        self.training_btn = self._big_button("Training", "Learn from graded projects: which parameters "
                                                         "separate the grades, classifiers from parameters")
        self.training_btn.setCheckable(False)
        self.training_btn.clicked.connect(self.training_requested)
        self.settings_btn = self._big_button("Settings", "General, parameters and profiles")
        self.analyze_btn.setChecked(True)
        self.settings_btn.clicked.connect(lambda: self.settings_requested.emit("General"))
        for b in (self.analyze_btn, self.training_btn, self.settings_btn):
            left.addWidget(b)
        left.addSpacing(18)
        open_btn = QtWidgets.QPushButton("Open project…")
        open_btn.clicked.connect(lambda: self.open_requested.emit(""))
        left.addWidget(open_btn)
        left.addStretch(1)
        outer.addLayout(left)

        # 2. Right column: profiles of the chosen kind, then recent projects
        right = QtWidgets.QVBoxLayout()
        self.kinds = QtWidgets.QTabBar()
        self.kinds.setObjectName("RibbonTabs")
        self.kinds.setExpanding(False)
        for name in KINDS.values():
            self.kinds.addTab(name)
        self.kinds.currentChanged.connect(lambda _i: self.refresh())
        right.addWidget(self.kinds)
        self.profiles = QtWidgets.QListWidget()
        self.profiles.setObjectName("ProfileCards")
        self.profiles.setSpacing(4)
        self.profiles.setWordWrap(True)
        self.profiles.itemDoubleClicked.connect(lambda _it: self._start())
        self.profiles.currentItemChanged.connect(lambda *_: self._update_buttons())
        right.addWidget(self.profiles, 2)
        row = QtWidgets.QHBoxLayout()
        self.empty_hint = QtWidgets.QLabel()
        self.empty_hint.setObjectName("Muted")
        self.empty_hint.setWordWrap(True)
        row.addWidget(self.empty_hint, 1)
        self.manage_btn = QtWidgets.QPushButton("Manage profiles…")
        self.manage_btn.clicked.connect(lambda: self.settings_requested.emit("Profiles"))
        self.start_btn = QtWidgets.QPushButton("New project")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start)
        row.addWidget(self.manage_btn)
        row.addWidget(self.start_btn)
        right.addLayout(row)
        recent_title = QtWidgets.QLabel("RECENT PROJECTS")
        recent_title.setObjectName("SectionTitle")
        right.addWidget(recent_title)
        self.recent = QtWidgets.QListWidget()
        self.recent.itemDoubleClicked.connect(lambda it: self.open_requested.emit(it.data(UID)))
        right.addWidget(self.recent, 1)
        outer.addLayout(right, 1)
        self.refresh()

    def _big_button(self, text: str, tip: str) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(text == "Analyze")
        b.setObjectName("StartNav")
        b.setMinimumSize(200, 46)
        return b

    # ------------------------------------------------------------ content
    def refresh(self) -> None:
        """Re-read profiles and recent projects (after Settings changed them)."""
        kind = list(KINDS)[max(self.kinds.currentIndex(), 0)]
        self.profiles.clear()
        for e in self.library.entries(kind=kind):
            p = e.profile
            types = ", ".join(IMAGE_TYPES.get(t, t).split(" (")[0] for t in p.images)
            it = QtWidgets.QListWidgetItem(f"{p.name}\n{p.description or 'No description.'}\nImages: {types}")
            it.setData(UID, e.id)
            it.setToolTip(p.description)
            self.profiles.addItem(it)
        if self.profiles.count():
            self.profiles.setCurrentRow(0)
            self.empty_hint.setText("")
        else:
            self.empty_hint.setText(f"No {KINDS[kind].lower()} profiles yet. Create one in Settings › Profiles.")
        self.recent.clear()
        for path in app_settings.recent_projects():
            it = QtWidgets.QListWidgetItem(f"{os.path.basename(path)}    {os.path.dirname(path)}")
            it.setData(UID, path)
            it.setToolTip(path)
            self.recent.addItem(it)
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.start_btn.setEnabled(self.profiles.currentItem() is not None)

    def _start(self) -> None:
        it = self.profiles.currentItem()
        if it is None:
            return
        pid = it.data(UID)
        try:
            profile = self.library.get(pid)
        except Exception as exc:                     # removed or broken since the list was built
            QtWidgets.QMessageBox.warning(self, "New project", str(exc))
            self.refresh()
            return
        self.new_project.emit(profile, pid)
