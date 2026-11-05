"""
Settings › General: per-computer preferences (GPU, theme, folders).

Analysis choices are not here: they belong to profiles.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from modalytics import app_settings, paths
from modalytics.ui.settings.widgets import muted, open_in_system, section


class GeneralPage(QtWidgets.QWidget):
    """General settings.

    Signals:
        theme_changed(str): "dark" or "light", chosen here.
    """

    theme_changed = QtCore.Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = app_settings.load()
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        col = QtWidgets.QVBoxLayout(self)
        col.addWidget(section("Graphics and models"))
        col.addLayout(form)

        # 1. Which GPU draws, which runs the models (effective at the next start)
        self.render = QtWidgets.QComboBox()
        self.render.addItem("Automatic (best found)", "")
        for name in self._render_choices():
            self.render.addItem(name, name)
        self.render.addItem("Software renderer (no GPU)", "software")
        self._select(self.render, self.settings["render_gpu"])
        self.render.currentIndexChanged.connect(lambda _i: self._store("render_gpu", self.render.currentData()))
        form.addRow("Draw the canvas on", self.render)

        self.models = QtWidgets.QComboBox()
        self.models.addItem("Automatic (best found)", "")
        from modalytics.services import compute
        for dev in compute.known_devices() or []:
            self.models.addItem(dev.label(), dev.key)
        self._select(self.models, self.settings["inference_device"])
        self.models.currentIndexChanged.connect(lambda _i: self._store("inference_device", self.models.currentData()))
        form.addRow("Run the models on", self.models)
        form.addRow(muted("These two take effect the next time Modalytics starts. The environment variables "
                          "MODALYTICS_GPU / MODALYTICS_INFERENCE override them."))
        details = QtWidgets.QPushButton("Hardware details…")
        details.clicked.connect(self._show_hardware)
        form.addRow("", details)

        # 2. Appearance
        col.addWidget(section("Appearance"))
        look = QtWidgets.QFormLayout()
        look.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.theme = QtWidgets.QComboBox()
        self.theme.addItem("Dark", "dark")
        self.theme.addItem("Light", "light")
        self._select(self.theme, self.settings["theme"])
        self.theme.currentIndexChanged.connect(self._theme_chosen)
        look.addRow("Theme", self.theme)
        col.addLayout(look)

        # 3. Where things are kept
        col.addWidget(section("Your library"))
        folders = QtWidgets.QFormLayout()
        for label, path in (("Library", paths.user_home()), ("Profiles", paths.user_dir("profiles")),
                            ("Parameters", paths.user_dir("parameters")), ("Models", paths.user_dir("models"))):
            row = QtWidgets.QHBoxLayout()
            text = muted(str(path), wrap=False)
            row.addWidget(text, 1)
            btn = QtWidgets.QPushButton("Open")
            btn.clicked.connect(lambda _c=False, p=path: open_in_system(p))
            row.addWidget(btn)
            folders.addRow(label, row)
        col.addLayout(folders)

        # 4. Diagnostics: a log of what the program does, for testing
        from modalytics import diagnostics
        col.addWidget(section("Diagnostics"))
        self.diagnostics = QtWidgets.QCheckBox("Keep a diagnostic log")
        self.diagnostics.setChecked(diagnostics.wanted())
        self.diagnostics.toggled.connect(diagnostics.set_wanted)
        col.addWidget(self.diagnostics)
        col.addWidget(muted("For testing: records what Modalytics does — actions, background tasks and how long they "
                            "took, projects opened and saved, every message and every error with its details, slow "
                            "screen updates. It stays on this computer (nothing is sent); look at it, or send it with "
                            "a bug report. Takes effect at once."))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(muted(str(diagnostics.log_file()), wrap=False), 1)
        logs = QtWidgets.QPushButton("Open log folder")
        logs.clicked.connect(lambda: open_in_system(diagnostics.log_dir()))
        row.addWidget(logs)
        col.addLayout(row)
        col.addStretch(1)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _render_choices() -> list[str]:
        """Hardware adapters found by the GPU probes (names, each once)."""
        from modalytics.render import gpu
        if not gpu.probes_done():
            return [gpu.choice.adapter.name] if gpu.choice and gpu.choice.adapter else []
        names = []
        for adapters, _msg in gpu.probe_results().values():
            names += [a.name for a in adapters if not a.software and a.name not in names]
        return names

    @staticmethod
    def _select(combo: QtWidgets.QComboBox, value) -> None:
        i = combo.findData(value)
        if i < 0 and value:                          # a saved choice not on this list (e.g. a removed GPU)
            combo.addItem(f"{value} (not found now)", value)
            i = combo.count() - 1
        combo.setCurrentIndex(max(i, 0))

    def _store(self, key: str, value) -> None:
        app_settings.put(key, value)

    def _theme_chosen(self, _i) -> None:
        name = self.theme.currentData()
        self._store("theme", name)
        self.theme_changed.emit(name)

    def _show_hardware(self) -> None:
        from modalytics.render import gpu
        from modalytics.services import compute
        text = gpu.report() + "\n\n" + compute.report()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Hardware")
        box.setText("What Modalytics found on this computer and what it uses:")
        box.setInformativeText(text)
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()
