"""
The Settings window: General, Parameters, Importers, Processing, Registration, Architectures, Scripts, Profiles.

    ┌ Settings ──────────────────────────────────────────────────────┐
    │ General     │                                                   │
    │ Parameters  │   the selected page                               │
    │ Importers   │                                                   │
    │ Processing  │   (two tabs: Steps, Chains)                       │
    │ Scripts     │   external scripts (tasks beside the analysis)   │
    │ Profiles    │                                                   │
    └─────────────┴───────────────────────────────────────────────────┘

Opened from the start screen or File › Settings (Ctrl+,). The pages share one
parameter library, so a parameter added on the Parameters page can be ticked
in a profile right away.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.parameters.registry import ParameterLibrary
from modalytics.ui.settings.architectures import ArchitecturesPage
from modalytics.ui.settings.general import GeneralPage
from modalytics.ui.settings.importers import ImportersPage
from modalytics.ui.settings.chains import ChainsPage
from modalytics.ui.settings.parameters import ParametersPage
from modalytics.ui.settings.processing import ProcessingPage
from modalytics.ui.settings.registration import RegistrationPage
from modalytics.ui.settings.profiles import ProfilesPage
from modalytics.ui.settings.scripts import ScriptsPage

PAGES = ("General", "Parameters", "Importers", "Processing", "Registration", "Architectures", "Scripts", "Profiles")


class SettingsWindow(QtWidgets.QDialog):
    """The Settings window.

    Args:
        page: Page to show first (one of ``PAGES``).

    Signals:
        theme_changed(str): forwarded from General.
        profiles_changed(): a profile was saved, added, imported or removed.
    """

    theme_changed = QtCore.Signal(str)
    profiles_changed = QtCore.Signal()

    def __init__(self, page: str = "General", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings — Modalytics")
        self.resize(1200, 820)
        self.parameters = ParameterLibrary()
        self.general = GeneralPage()
        self.parameters_page = ParametersPage(self.parameters)
        self.importers = ImportersPage()
        self.processing = ProcessingPage()                 # Processing › Steps
        self.chains = ChainsPage(self.processing.library)   # Processing › Chains
        self.processing_tabs = QtWidgets.QTabWidget()
        self.processing_tabs.addTab(self.processing, "Steps")
        self.processing_tabs.addTab(self.chains, "Chains")
        self.processing_tabs.confirm_leave = self.processing.confirm_leave   # unsaved step edits are asked about
        self.processing_tabs.currentChanged.connect(lambda i: i == 1 and self.chains.reload())
        self.registration = RegistrationPage()
        self.architectures = ArchitecturesPage()
        # A script's files can be imported into the project of the window that opened Settings
        self.scripts = ScriptsPage(on_import=getattr(parent, "_import_images", None))
        self.profiles = ProfilesPage(parameters=self.parameters)

        # 1. Navigation + pages
        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("SettingsNav")
        self.nav.addItems(PAGES)
        self.nav.setFixedWidth(160)
        self.stack = QtWidgets.QStackedWidget()
        for w in (self.general, self.parameters_page, self.importers, self.processing_tabs, self.registration,
                  self.architectures, self.scripts, self.profiles):
            self.stack.addWidget(w)
        self.nav.currentRowChanged.connect(self._page_chosen)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.nav)
        row.addWidget(self.stack, 1)
        col = QtWidgets.QVBoxLayout(self)
        col.addLayout(row, 1)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close
                                           | QtWidgets.QDialogButtonBox.StandardButton.Help)
        close.rejected.connect(self.reject)
        close.helpRequested.connect(self.show_help)
        col.addWidget(close)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F1), self, activated=self.show_help)

        # 2. The pages talk through the shared parameter library
        self.parameters_page.library_changed.connect(self.profiles.parameters_changed)
        self.general.theme_changed.connect(self.theme_changed)
        self.profiles.profiles_changed.connect(self.profiles_changed)
        self.scripts.library_changed.connect(lambda: (self.profiles.reload_current(), self.profiles_changed.emit()))
        self.registration.library_changed.connect(self.profiles.reload_current)   # its default-method list
        self.chains.profiles_changed.connect(lambda: (self.profiles.refresh(self.profiles.pid),
                                                      self.profiles.reload_current(), self.profiles_changed.emit()))
        self.nav.setCurrentRow(PAGES.index(page) if page in PAGES else 0)

    def show_help(self) -> None:
        """The help on the page shown (F1)."""
        from modalytics import app_settings
        from modalytics import help as hp
        from modalytics.ui.help_window import show_help
        page = PAGES[max(0, self.nav.currentRow())]
        show_help(self, hp.page_for(f"settings:{page}"), app_settings.get("theme") or "dark")

    def _page_chosen(self, row: int) -> None:
        current = self.stack.currentIndex()
        if row != current and not self._leave(current):
            self.nav.blockSignals(True)
            self.nav.setCurrentRow(current)
            self.nav.blockSignals(False)
            return
        if row == PAGES.index("Parameters"):
            self.parameters_page.reload()            # an import may have added parameters
        elif row == PAGES.index("Importers"):
            self.importers.reload()
        elif row == PAGES.index("Processing"):
            self.processing.reload()
            self.chains.reload()
        elif row == PAGES.index("Registration"):
            self.registration.reload()
        elif row == PAGES.index("Architectures"):
            self.architectures.reload()
        elif row == PAGES.index("Scripts"):
            self.scripts.reload()
        elif row == PAGES.index("Profiles"):                 # steps may have changed: the profile lists them
            self.profiles.reload_current()
        self.stack.setCurrentIndex(row)

    def _leave(self, index: int) -> bool:
        """Ask about unsaved edits on a page before leaving it. Returns False to stay."""
        page = self.stack.widget(index)
        return page.confirm_leave() if hasattr(page, "confirm_leave") else True

    def reject(self) -> None:
        """Close (Esc, the Close button): first ask about unsaved edits."""
        if self._leave(self.stack.currentIndex()):
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        event.ignore()                            # the window's ✕ goes through reject() too
        self.reject()
