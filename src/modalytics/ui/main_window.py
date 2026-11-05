"""
Main window: ribbon on top, Layers on the left, canvas in the centre,
Tool Settings and Properties on the right, live status at the bottom.

    ┌ Modalytics │ View  Arrange  Process  Annotate  Measure  Report ────────────┐
    │ [Select][Polygon][Freehand][Brush][Eraser] │ [Segment…]                    │  ribbon: tools only
    ├──────────┬──────────────────────────────────────────────┬─────────────────┤
    │ LAYERS   │                                              │ TOOL SETTINGS   │  settings of the
    │ ▣ WSI    │                 canvas (µm)                  │ class, size…    │  active tool
    │  ▣ DHM   │                                   ┌───────┐  ├─────────────────┤
    │   ● mask │                                   │minimap│  │ PROPERTIES      │  the selection
    ├──────────┴──────────────────────────────────────────────┴─────────────────┤
    │ x, y µm · µm/px · tiles · frame time · GPU                                 │
    └────────────────────────────────────────────────────────────────────────────┘

Tools are chosen in the ribbon and configured in Tool Settings (never mixed).

Classification is part of annotating: **Classify selected** (Annotate › Classification) grades
the selected objects with the profile's classifier, and the Segment tool's **Classify when
segmenting** grades what it finds — both as suggestions to review, graded from the profile's
minimum confidence (Segment's Tool Settings, or Settings › Profiles), else left as the plain
group (``services.suggest``). Measure applies the classifiers to the results, and first offers
a newer classifier saved in your library.
Classifiers are made in Training (start screen, or File › Training…).

Without a project the window shows the start screen instead (``StartPage``:
Analyze / Training / Settings, recent projects); ribbon, docks and status bar
come back when a project is started or opened. Every project has a profile
(``profiles``), shown in the title.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.core.commands import UndoStack, set_opacity_command, set_placement_command
from modalytics.core.geometry import Placement, bounds
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io import importers, project_file
from modalytics import app_settings
from modalytics.profiles.library import default_profile
from modalytics.render.canvas import Canvas
from modalytics.services import compute
from modalytics.services.classes import rename_to_profile
from modalytics.tools.annotate import all_tools
from modalytics.tools.manager import ToolManager
from modalytics.ui.layer_panel import LayerPanel
from modalytics.ui.properties_panel import PropertiesPanel
from modalytics.ui.ribbon import Ribbon, icon
from modalytics.ui.theme import apply_theme
from modalytics.ui.start_page import StartPage
from modalytics.ui.tool_settings import ToolSettingsPanel

TABS = ["Arrange", "Process", "Annotate", "Measure", "Report"]
FILE_FILTER = "Modalytics project (*.modalytics);;HoloPath project (*.hpro)"


class MainWindow(QtWidgets.QMainWindow):
    """Application window.

    Args:
        project: Project to show (a new empty one if None).
        theme: "dark" or "light".
        vsync: Passed to the canvas (False for benchmarks).
        present: Canvas present method ("bitmap" or "screen", see ``Canvas``).
    """

    def __init__(self, project: Project | None = None, theme: str = "dark", vsync: bool = True,
                 present: str | None = None) -> None:
        super().__init__()
        self.resize(1600, 960)
        start_screen = project is None                   # no project given: begin on the start screen
        self.project = project or Project()
        if self.project.profile is None:                 # no profile chosen: Colonic tissue (HoloPath's)
            self.project.profile, self.project.profile_id = default_profile()
        self.undo = UndoStack()
        self.theme_name = theme
        self.tokens = apply_theme(QtWidgets.QApplication.instance(), theme)
        self._dirty = False

        # 1. Canvas and tools (the devices the models can use are listed in the background meanwhile)
        compute.warm_up()
        self.canvas = Canvas(self.project, self.undo, self.tokens, vsync=vsync, present=present)
        self.start_page = StartPage()
        self.central = QtWidgets.QStackedWidget()
        self.central.addWidget(self.start_page)
        self.central.addWidget(self.canvas)
        self.setCentralWidget(self.central)
        self.tools = ToolManager()
        for cls in all_tools():
            self.tools.register(cls)
        self.canvas.set_tools(self.tools)

        # 2. Docks: Layers left; Tool Settings above Properties on the right
        self.layers = LayerPanel(self.project, fit_requested=self.frame_node, extend_menu=self._layers_menu,
                                 undo=self.undo)
        self.tool_settings = ToolSettingsPanel(self.tools, self.project)
        self.properties = PropertiesPanel(self.project, self.undo)
        self.properties.on_review = self._review
        ly = self._add_dock("Layers", self.layers, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, 280)
        ts = self._add_dock("Tool Settings", self.tool_settings, QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 300)
        pr = self._add_dock("Properties", self.properties, QtCore.Qt.DockWidgetArea.RightDockWidgetArea, 300)
        self.splitDockWidget(ts, pr, QtCore.Qt.Orientation.Vertical)
        from modalytics.ui.results_panel import ResultsPanel
        self.results = ResultsPanel(self.project)
        self.results_dock = QtWidgets.QDockWidget("Results", self)
        self.results_dock.setObjectName("Results")
        self.results_dock.setWidget(self.results)                 # a table scrolls by itself: no scroll area
        self.results_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                                      | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                                      | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)
        self.resizeDocks([self.results_dock], [340], QtCore.Qt.Orientation.Vertical)
        self.results_dock.hide()
        self.docks = [ly, ts, pr]

        # 3. Ribbon, menus, status bar
        self._menubar = QtWidgets.QMenuBar()
        self._build_actions()
        self._zoom_shortcuts()
        self._build_menus()
        self._build_ribbon()
        self._build_status()
        from modalytics.ui.tasks import TaskRunner
        self.tasks = TaskRunner(self.statusBar())
        self.canvas.tool_ctx.tasks = self.tasks
        self.tools.activate("select")

        # 4. Signals
        self.canvas.context_requested.connect(self._canvas_menu)
        self._cursor_xy = (0.0, 0.0)
        self._cursor_timer = QtCore.QTimer(self, singleShot=True, interval=33, timeout=self._paint_cursor)
        self.canvas.cursor_moved.connect(self._show_cursor)
        self._status_timer = QtCore.QTimer(self, interval=500, timeout=self._refresh_status)
        self._status_timer.start()
        self.tools.active_changed.connect(self._tool_changed)
        self.tools.actions_changed.connect(lambda _tool: self._update_tool_commands())
        for sig in (self.project.structure_changed, self.undo.changed):
            sig.connect(self._mark_dirty)
        self.start_page.new_project.connect(self.start_project)
        self.start_page.open_requested.connect(lambda p: self.open_path(p) if p else self.open_dialog())
        self.start_page.settings_requested.connect(self.open_settings)
        self.start_page.training_requested.connect(self.open_training)
        self.results.measure_requested.connect(self.measure)
        self.results.export_requested.connect(self.export_results)
        self.results.object_activated.connect(self.show_object)
        self.results.image_activated.connect(self._show_result_image)
        self.results_dock.visibilityChanged.connect(self.act["results"].setChecked)
        self.project.results_changed.connect(lambda: self.canvas.preview.set_highlight(None))
        if start_screen:
            self.show_start()
        else:
            self.show_workspace()

    # ------------------------------------------------------------ building
    def _add_dock(self, title: str, widget: QtWidgets.QWidget, area, width: int) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title)
        # Panels keep their natural height and scroll when the dock is shorter
        # (otherwise Qt squeezes every row and clips buttons and combo boxes)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        dock.setWidget(scroll)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        self.resizeDocks([dock], [width], QtCore.Qt.Orientation.Horizontal)
        return dock

    def _action(self, text: str, icon_name: str, slot=None, shortcut: str = "", tip: str = "",
                checkable: bool = False, enabled: bool = True) -> QtGui.QAction:
        act = QtGui.QAction(icon(icon_name, self.tokens["text"]), text, self)
        act.setProperty("icon_name", icon_name)        # re-coloured when the theme changes
        act.setCheckable(checkable)
        act.setEnabled(enabled)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
            act.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        act.setToolTip(f"{tip or text}" + (f"  ({shortcut})" if shortcut else ""))
        if slot is not None:
            act.triggered.connect(slot)
        from modalytics import diagnostics
        act.triggered.connect(lambda *_a, t=text: diagnostics.event("action", name=t.replace("&", "")))
        self.addAction(act)          # shortcuts work even when the ribbon tab is hidden
        return act

    def _build_actions(self) -> None:
        """All commands and tools as QActions (shared by the ribbon and the menus)."""
        a = self._action
        self.act = {
            "fit": a("Fit all", "mdi6.fit-to-screen-outline", self.canvas.fit_all, "F"),
            "one": a("1:1", "mdi6.magnify-scan", lambda: self.canvas.zoom_to_pixels(), "1",
                     "One image pixel per screen pixel"),
            "zoom_in": a("Zoom in", "mdi6.magnify-plus-outline", lambda: self.canvas.zoom_step(1), "Ctrl++",
                         "Zoom in (about the cursor when it is on the canvas)"),
            "zoom_out": a("Zoom out", "mdi6.magnify-minus-outline", lambda: self.canvas.zoom_step(-1), "Ctrl+-",
                          "Zoom out (about the cursor when it is on the canvas)"),
            "minimap": a("Minimap", "mdi6.map-outline", self._toggle_minimap, checkable=True,
                         tip="Show or hide the minimap (bottom right of the canvas)"),
            "theme": a("Theme", self._theme_icon(), self.toggle_theme, tip=self._theme_tip()),
            "bug": a("Report a bug…", "mdi6.bug-outline", self._report_bug,
                     tip="Report a bug: what went wrong, with the diagnostic log attached"),
            "help": a("Help", "mdi6.help-circle-outline", self._show_help, "F1",
                      tip="The help: the page for what you are doing (F1)"),
            "reset": a("Reset", "mdi6.restore", self._reset_selected, tip="Reset the selected image's position"),
            "flip": a("Flip", "mdi6.flip-horizontal", self._flip_selected,
                      tip="Mirror the selected image left↔right in place (its masks follow)"),
            "front": a("To front", "mdi6.arrange-bring-to-front", lambda: self._restack(True)),
            "back": a("To back", "mdi6.arrange-send-to-back", lambda: self._restack(False)),
            "chains": a("Chain editor…", "mdi6.transit-connection-variant", self._chain_editor,
                        tip="Make a chain of processing steps for the selected image, with a preview; apply it, "
                            "save it to your chains"),
            "quick_step": a("Quick step", "mdi6.lightning-bolt-outline", self._quick_step_menu,
                            tip="Every processing step the selected image(s) can take, by category: one step makes "
                                "a view (the original stays)"),
            "apply_chain": a("Apply chain", "mdi6.link-variant", self._chain_menu,
                             tip="Apply one of your saved chains to the selected image(s)"),
            "show_view": a("Show…", "mdi6.layers-outline", self._show_menu,
                           tip="Draw the selected image's original or one of its views"),
            "segment_image": a("Segment image", "mdi6.image-filter-center-focus", self._segment_image,
                               tip="Segment the whole selected image with the Segment tool's settings"),
            "measure": a("Measure", "mdi6.ruler-square", lambda: self.measure("project"), "Ctrl+M",
                         "Measure every image with the profile's parameters: each object, each class, each image"),
            "measure_image": a("Measure image", "mdi6.ruler", lambda: self.measure("image"),
                               tip="Measure only the selected image (the others keep their results)"),
            "include_suggestions": a("Include suggestions", "mdi6.checkbox-multiple-blank-outline", None,
                                     checkable=True,
                                     tip="Also measure the unreviewed suggestions, as if accepted (their rows are "
                                         "marked); off: only the annotation is measured"),
            "results": a("Results", "mdi6.table-large", self._toggle_results, checkable=True,
                         tip="Show or hide the Results panel"),
            "export_results": a("Export CSV…", "mdi6.file-delimited-outline",
                                lambda: self.export_results(self.results.current_scope()),
                                tip="Save the results table shown as a CSV file"),
            "classify_selected": a("Classify selected", "mdi6.shape-plus", self.classify_selected,
                                   tip="Grade the selected objects with the profile's classifier: where a grade "
                                       "is sure enough (the minimum confidence: Segment's Tool Settings) and "
                                       "differs from the object's, it is suggested — Review, Enter accepts. Unsure "
                                       "ones stay as they are."),
            "report": a("Report…", "mdi6.file-chart-outline", self.make_report,
                        tip="The project's report: one HTML file (summary, classification, images with numbered "
                            "objects, objects, distributions, methods), opened in your browser"),
            "report_pdf": a("PDF…", "mdi6.file-pdf-box", self.make_report_pdf,
                            tip="The report as a PDF summary (A4): the parameters ticked In PDF in the profile (else "
                                "the classifier's, then key ones, at most six), the objects with their predicted "
                                "grade and the reason as text — made here and opened in your PDF viewer"),
        }
        self.act["minimap"].setChecked(True)
        # The Process tab's step and chain buttons: made from the project's profile (rebuild_process_tab)
        self.step_actions: dict[str, QtGui.QAction] = {}   # id ("filters:median", "chain:Phase denoising") → button
        self.step_specs: dict = {}                       # id → StepSpec or Chain
        self._process_groups: list = []
        self.project.selection_changed.connect(lambda _n: self._update_step_actions())
        self.project.structure_changed.connect(self._update_step_actions)
        # One checkable action per tool, mutually exclusive (a tool can appear on several tabs)
        self.tool_actions: dict[str, QtGui.QAction] = {}
        group = QtGui.QActionGroup(self)
        for tool in self.tools.tools.values():
            act = a(tool.label, tool.icon, lambda _c=False, t=tool.id: self.tools.activate(t), tool.shortcut,
                    tool.hint.split(".")[0], checkable=True)
            group.addAction(act)
            self.tool_actions[tool.id] = act
        # Each tool's own commands (e.g. Register's Start / Previous / Next / Save), greyed out unless the
        # tool is active and the command applies (``Tool.action_enabled``)
        self.tool_commands: dict[tuple[str, str], QtGui.QAction] = {}
        for tool in self.tools.tools.values():
            for ta in tool.actions:
                self.tool_commands[(tool.id, ta.key)] = a(
                    ta.label, ta.icon, lambda _c=False, t=tool, k=ta.key: self.tools.trigger(t, k),
                    tip=ta.tooltip, enabled=False)

    def _zoom_shortcuts(self) -> None:
        """Zoom in / out also on Ctrl + "=" and on the keypad's + / − with Ctrl (what tablet dials send)."""
        K, M = QtCore.Qt.Key, QtCore.Qt.KeyboardModifier
        for key, keys in (("zoom_in", (K.Key_Plus, K.Key_Equal)), ("zoom_out", (K.Key_Minus,))):
            seqs = [QtGui.QKeySequence(QtCore.QKeyCombination(mods, k)) for k in keys
                    for mods in (M.ControlModifier, M.ControlModifier | M.KeypadModifier)]
            self.act[key].setShortcuts(seqs)

    def _build_ribbon(self) -> None:
        self.ribbon = Ribbon("Modalytics")
        for t in TABS:
            self.ribbon.add_tab(t)
        def tools_on(tab: str, grp: str) -> list[QtGui.QAction]:
            """The tools of one ribbon group, each followed by its own commands."""
            out = []
            for t in self.tools.tools.values():
                if tab in t.tabs and t.group == grp:
                    out += [self.tool_actions[t.id]] + [self.tool_commands[(t.id, x.key)] for x in t.actions]
            return out

        A = self.act
        for key in ("zoom_in", "zoom_out", "fit"):         # zoom (a pen alone can) | view toggles | bug, help
            self.ribbon.add_corner(A[key])
        self.ribbon.add_corner_separator()
        for key in ("minimap", "theme"):
            self.ribbon.add_corner(A[key])
        self.ribbon.add_corner_separator()
        for key in ("bug", "help"):
            self.ribbon.add_corner(A[key])
        self.ribbon.add_group("Arrange", "Tools", tools_on("Arrange", "Tools"))
        self.ribbon.add_group("Arrange", "Placement", [A["reset"], A["flip"], A["front"], A["back"]])
        self.ribbon.add_group("Arrange", "Registration", tools_on("Arrange", "Registration"))
        self.ribbon.add_group("Process", "Process", [A["quick_step"], A["apply_chain"], A["chains"], A["show_view"]])
        self.rebuild_process_tab()
        self.ribbon.add_group("Annotate", "Tools", tools_on("Annotate", "Tools"))
        self.ribbon.add_group("Annotate", "Draw", tools_on("Annotate", "Draw"))
        self.ribbon.add_group("Annotate", "Automatic", tools_on("Annotate", "Automatic") + [A["segment_image"]])
        self.ribbon.add_group("Annotate", "Classification", [A["classify_selected"]])
        self.ribbon.add_group("Measure", "Parameters", [A["measure"], A["measure_image"], A["include_suggestions"]])
        self.ribbon.add_group("Measure", "Results", [A["results"], A["export_results"]])
        self.ribbon.add_group("Report", "Report", [A["report"], A["report_pdf"]])
        self.ribbon.show_tab("Annotate")
        # Menu bar + ribbon share the window's menu-widget slot, so the ribbon spans the full width
        top = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(top)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._menubar)
        col.addWidget(self.ribbon)
        self.setMenuWidget(top)

    def _build_menus(self) -> None:
        mb = self._menubar
        f = mb.addMenu("&File")
        f.addAction("New project…", self.go_to_start, QtGui.QKeySequence.StandardKey.New)
        f.addAction("Open…", self.open_dialog, QtGui.QKeySequence.StandardKey.Open)
        f.addAction("Save", self.save, QtGui.QKeySequence.StandardKey.Save)
        f.addAction("Save as…", self.save_as, QtGui.QKeySequence("Ctrl+Shift+S"))
        imp = f.addMenu("Import")
        imp.addAction("Images (slides, phase maps, microscope images)…", self._import_images,
                      QtGui.QKeySequence("Ctrl+I"))
        imp.addAction("HoloPath project (.hpro) into this project…", self._import_hpro)
        imp.addSeparator()
        imp.addAction("Annotations into the selected image (GeoJSON, Leica/Aperio XML)…", self._import_annotations)
        exp = f.addMenu("Export")
        exp.addAction("Masks of the selected image as GeoJSON (QuPath)…", self._export_geojson)
        f.addSeparator()
        f.addAction("Start screen", self.go_to_start)
        f.addAction("Training…", self.open_training)
        f.addAction("Settings…", lambda: self.open_settings("General"), QtGui.QKeySequence("Ctrl+,"))
        f.addSeparator()
        f.addAction("Quit", self.close, QtGui.QKeySequence.StandardKey.Quit)
        e = mb.addMenu("&Edit")
        self.undo_action = e.addAction("Undo", self.undo.undo, QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action = e.addAction("Redo", self.undo.redo, QtGui.QKeySequence.StandardKey.Redo)
        e.addAction("Redo (Ctrl+Y)", self.undo.redo, QtGui.QKeySequence("Ctrl+Y")).setVisible(False)
        e.addSeparator()
        e.addAction("Delete", self._on_delete, QtGui.QKeySequence.StandardKey.Delete)
        self.undo.changed.connect(self._update_undo_text)
        self._update_undo_text()
        v = mb.addMenu("&View")
        for key in ("zoom_in", "zoom_out", "fit", "one", "minimap", "theme", "results"):
            v.addAction(self.act[key])
        t = mb.addMenu("&Tools")
        for act in self.tool_actions.values():
            t.addAction(act)
        self.scripts_menu = mb.addMenu("&Scripts")        # the profile's external scripts (hidden if none)
        self.scripts_menu.menuAction().setVisible(False)
        h = mb.addMenu("&Help")
        h.addAction(self.act["help"])
        h.addAction("Contents", lambda: self._show_help("user/index"))
        h.addAction("Controls", lambda: self._show_help("user/08-controls"))
        h.addSeparator()
        h.addAction(self.act["bug"])

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.st_cursor = QtWidgets.QLabel("x —  y —")
        self.st_zoom = QtWidgets.QLabel("")
        self.st_tiles = QtWidgets.QLabel("")
        self.st_fps = QtWidgets.QLabel("")
        self.st_gpu = QtWidgets.QLabel("")
        for w in (self.st_cursor, self.st_zoom, self.st_tiles):
            sb.addWidget(w)
        sb.addPermanentWidget(self.st_fps)
        sb.addPermanentWidget(self.st_gpu)
        self._show_hardware()

    def _show_hardware(self) -> None:
        """Status bar: what draws the canvas and where the models run (the model device list is
        built in the background, so this re-polls until it is known)."""
        from modalytics.render import gpu
        text = f"Display: {self.canvas.gpu_name}"
        # The slower GPU probes may still be running: the full list follows when they are done
        tip = gpu.report() if gpu.probes_done() else f"Rendering: {self.canvas.gpu_name}"
        tip += f"\n  Present mode: {self.canvas.present}"
        devs = compute.known_devices()
        if devs is None:
            text += " · Models: …"
        else:
            text += f" · Models: {devs[0].label()}"
            tip += "\n\n" + compute.report()
        if devs is None or not gpu.probes_done():
            QtCore.QTimer.singleShot(250, self._show_hardware)
        self.st_gpu.setText(text)
        self.st_gpu.setToolTip(tip + "\n\n(python -m modalytics --gpu-info prints this; "
                               "Help › Settings and hardware explains the choice)")

    # ------------------------------------------------------------ project files
    def show_start(self) -> None:
        """Show the start screen (no ribbon, docks or status bar)."""
        self.start_page.refresh()
        self.central.setCurrentWidget(self.start_page)
        self.ribbon.hide()
        for d in self.docks:
            d.hide()
        self.statusBar().hide()
        self._update_title()

    def fill_scripts_menu(self) -> None:
        """The Scripts menu: the external scripts the project's profile offers, by category (hidden if none).

        The list is read from the profile in your library (by the project's profile id), so ticking a script in
        Settings › Profiles shows it at once; a project without a library profile uses its own copy.
        """
        from modalytics.profiles.library import ProfileLibrary
        from modalytics.scripts.registry import ScriptLibrary
        menu = self.scripts_menu
        menu.clear()
        prof = self.project.profile
        try:
            prof = ProfileLibrary().get(self.project.profile_id) if self.project.profile_id else prof
        except Exception:                                  # noqa: BLE001 — removed from the library: the copy
            pass
        wanted = list(prof.scripts) if prof is not None else []
        library = ScriptLibrary() if wanted else None
        specs = [library.get(u) for u in wanted if library.get(u) is not None] if library else []
        missing = [u for u in wanted if library is not None and library.get(u) is None]
        for category in dict.fromkeys(sp.group for sp in specs):
            menu.addSection(category)
            for sp in [x for x in specs if x.group == category]:
                act = menu.addAction(sp.name, lambda _c=False, x=sp: self.run_script(x))
                act.setToolTip(sp.description)
        for uid in missing:
            menu.addAction(f"{uid} (not installed here)").setEnabled(False)
        if wanted:
            menu.addSeparator()
            menu.addAction("Manage scripts…", lambda: self.open_settings("Scripts"))
        menu.setToolTipsVisible(True)
        menu.menuAction().setVisible(bool(wanted))

    def run_script(self, spec) -> None:
        """The run window of an external script; files it hands over can be imported into this project."""
        from modalytics.ui.scripts import RunScriptDialog
        self._script_window = RunScriptDialog(spec, self, self._import_images)
        self._script_window.show()

    def show_workspace(self) -> None:
        """Show the project: canvas, ribbon, docks and status bar."""
        self.fill_scripts_menu()
        self.central.setCurrentWidget(self.canvas)
        self.ribbon.show()
        for d in self.docks:
            d.show()
        self.statusBar().show()
        self._update_title()

    def go_to_start(self) -> None:
        """Leave the current project for the start screen (asks to save first)."""
        if self._confirm_discard():
            self._dirty = False
            self.show_start()

    def start_project(self, profile, profile_id: str) -> None:
        """Start an empty project with a profile chosen on the start screen."""
        self._load_contents([], None, None, profile=profile, profile_id=profile_id)
        self.show_workspace()
        self.statusBar().showMessage(f"🆕 New {profile.name} project — import images from File › Import", 8000)

    def open_training(self) -> None:
        """The Training section (one window, brought to the front if already open)."""
        from modalytics.ui.training.window import TrainingWindow
        if getattr(self, "training", None) is None:
            self.training = TrainingWindow(self)
            self.training.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.training.destroyed.connect(lambda *_a: setattr(self, "training", None))
        self.training.show()
        self.training.raise_()
        self.training.activateWindow()

    def open_settings(self, page: str = "General") -> None:
        """Open the Settings window on a page ("General", "Parameters", "Profiles")."""
        from modalytics.ui.settings import SettingsWindow
        dlg = SettingsWindow(page, self)
        dlg.theme_changed.connect(self.set_theme)
        dlg.profiles_changed.connect(self.start_page.refresh)
        dlg.exec()
        self.start_page.refresh()
        self.fill_scripts_menu()                          # a profile may offer other scripts now

    def new_project(self, profile=None, profile_id: str = "") -> None:
        """Start an empty project with a profile (default: Colonic tissue)."""
        if self._confirm_discard():
            if profile is None:
                profile, profile_id = default_profile()
            self._load_contents([], None, None, profile=profile, profile_id=profile_id)

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", self._start_dir(), FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        """Open a .modalytics project (or a HoloPath .hpro, which opens as an unsaved new project)."""
        try:
            if path.lower().endswith(".hpro"):
                profile, pid = default_profile()                  # HoloPath projects are colonic tissue
                self._load_contents(importers.import_hpro(path), None, None, profile=profile, profile_id=pid)
                self._dirty = True
            else:
                data = project_file.load_project(path)
                if data.profile is None:                          # saved before profiles existed
                    data.profile, data.profile_id = default_profile()
                self._load_contents(data.images, data.labels, path, data.view, data.profile, data.profile_id,
                                    data.results)
                if data.warnings:
                    QtWidgets.QMessageBox.warning(self, "Problems opening the project", "\n".join(data.warnings))
                app_settings.add_recent(path)
        except Exception as exc:                         # show, don't crash
            from modalytics import diagnostics
            diagnostics.error(f"could not open {path}", exc)
            QtWidgets.QMessageBox.critical(self, "Could not open", f"{path}\n\n{exc}")
            return
        from modalytics import diagnostics
        diagnostics.event("project opened", path=path, images=len(self.project.images),
                          profile=self.project.profile.name if self.project.profile else "")
        self.show_workspace()

    def save(self) -> bool:
        if self.project.path is None:
            return self.save_as()
        v = self.canvas.view
        project_file.save_project(self.project, self.project.path, {"cx": v.cx, "cy": v.cy, "um_per_px": v.um_per_px})
        from modalytics import diagnostics
        diagnostics.event("project saved", path=self.project.path)
        app_settings.add_recent(self.project.path)
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"✅ Saved {os.path.basename(self.project.path)}", 4000)
        return True

    def save_as(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", self._start_dir(),
                                                        "Modalytics project (*.modalytics)")
        if not path:
            return False
        self.project.path = path if path.endswith(project_file.EXTENSION) else path + project_file.EXTENSION
        return self.save()

    def _load_contents(self, images, labels, path, view=None, profile=None, profile_id: str = "",
                       results=None) -> None:
        if profile is not None and profile.aliases:      # e.g. a HoloPath project's Normal glands → Low grade
            for img in images:
                rename_to_profile(img, profile)
        self.project.reset(images, labels, path, profile, profile_id, results)
        self.rebuild_process_tab()                        # the profile's step and chain buttons
        self.results_dock.setVisible(results is not None)
        self.undo.clear()
        if view:
            self.canvas.view.cx, self.canvas.view.cy = view["cx"], view["cy"]
            self.canvas.view.um_per_px = view["um_per_px"]
            self.canvas.request_draw()
        else:
            self.canvas.fit_all()
        self._dirty = False

    def _start_dir(self) -> str:
        return os.path.dirname(self.project.path) if self.project.path else ""

    def _mark_dirty(self, *_args) -> None:
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _update_title(self) -> None:
        if self.central.currentWidget() is self.start_page:
            self.setWindowTitle("Modalytics")
            return
        name = os.path.basename(self.project.path) if self.project.path else "Untitled"
        profile = f" — {self.project.profile.name}" if self.project.profile is not None else ""
        self.setWindowTitle(f"{name}{' *' if self._dirty else ''}{profile} — Modalytics")

    def _confirm_discard(self) -> bool:
        """Ask to save unsaved changes; returns False if the user cancels."""
        if not self._dirty or not self.project.images:
            return True
        btn = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Save changes to the current project?",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if btn == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return btn == QtWidgets.QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------ actions
    def toggle_theme(self) -> None:
        """Switch between dark and light (and remember it)."""
        self.set_theme("light" if self.theme_name == "dark" else "dark")
        app_settings.put("theme", self.theme_name)

    def set_theme(self, name: str) -> None:
        """Apply the "dark" or "light" theme to the window, canvas and icons."""
        self.theme_name = name
        self.tokens = apply_theme(QtWidgets.QApplication.instance(), self.theme_name)
        self.canvas.set_theme(self.tokens)
        # Re-colour icons for the new theme (the theme button shows the theme it switches to)
        if hasattr(self, "act"):
            self.act["theme"].setProperty("icon_name", self._theme_icon())
            self.act["theme"].setToolTip(self._theme_tip())
            for act in self.act.values():
                if act.property("icon_name"):
                    act.setIcon(icon(act.property("icon_name"), self.tokens["text"]))
        for tool in self.tools.tools.values():
            self.tool_actions[tool.id].setIcon(icon(tool.icon, self.tokens["text"]))
            for ta in tool.actions:
                self.tool_commands[(tool.id, ta.key)].setIcon(icon(ta.icon, self.tokens["text"]))

    def _theme_icon(self) -> str:
        """The theme button shows where a click goes: the sun in the dark theme, the moon in the light one."""
        return "mdi6.white-balance-sunny" if self.theme_name == "dark" else "mdi6.weather-night"

    def _theme_tip(self) -> str:
        return "Light theme" if self.theme_name == "dark" else "Dark theme"

    def _report_bug(self) -> None:
        """Report a bug (placeholder until the bug reporter is made)."""
        from modalytics import diagnostics
        QtWidgets.QMessageBox.information(
            self, "Report a bug",
            "The bug reporter is not made yet.\n\nMeanwhile: turn on the diagnostic log (Settings › General), "
            f"do again what went wrong, and send the log file:\n{diagnostics.log_file()}")

    def help_context(self) -> str:
        """Where the user is, for F1 (``help.CONTEXTS``): the start screen, a tool other than Select, else the
        ribbon tab."""
        if self.central.currentWidget() is self.start_page:
            return "start"
        tool = self.tools.active
        if tool is not None and tool.id not in ("select", ""):
            return f"tool:{tool.id}"
        return f"tab:{self.ribbon.tabbar.tabText(self.ribbon.tabbar.currentIndex())}"

    def _show_help(self, page: str | None = None) -> None:
        """The Help window, on ``page`` or on the page for what the user is doing."""
        from modalytics import help as hp
        from modalytics.ui.help_window import show_help
        show_help(self, page or hp.page_for(self.help_context()), self.theme_name)

    def frame_node(self, node) -> None:
        """Fit the canvas on an image, or on a mask's extent (a list: on all of them)."""
        if isinstance(node, (list, tuple)):
            from modalytics.core import geometry as geo
            corners = []
            for n in node:
                if isinstance(n, MaskNode) and n.parent is not None and not n.geometry.is_empty:
                    x0, y0, x1, y1 = n.geometry.bounds
                    corners.append(geo.apply(n.parent.local_to_world(), [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
                else:
                    img = n if isinstance(n, ImageNode) else getattr(n, "parent", None)
                    if img is not None:
                        corners.append(img.world_corners())
            if corners:
                self.canvas.view.fit(bounds(np.vstack(corners)), margin=0.05)
                self.canvas._view_changed()
            return
        if isinstance(node, MaskNode) and node.parent is not None and not node.geometry.is_empty:
            x0, y0, x1, y1 = node.geometry.bounds
            from modalytics.core import geometry as geo
            corners = geo.apply(node.parent.local_to_world(), [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self.canvas.view.fit(bounds(corners), margin=0.15)
        else:
            img = node if isinstance(node, ImageNode) else getattr(node, "parent", None)
            if img is None:
                return
            self.canvas.view.fit(bounds(img.world_corners()))
        self.canvas._view_changed()

    def _update_tool_commands(self) -> None:
        """Enable a tool's ribbon commands only while it is active and they apply."""
        for (tool_id, key), act in self.tool_commands.items():
            tool = self.tools.tools[tool_id]
            act.setEnabled(tool is self.tools.active and tool.action_enabled(key))

    def _tool_changed(self, tool) -> None:
        from modalytics import diagnostics
        diagnostics.event("tool", name=tool.label)
        self.tool_actions[tool.id].setChecked(True)
        self._update_tool_commands()
        # Show a ribbon tab that holds the tool (e.g. after its keyboard shortcut)
        current = self.ribbon.tabbar.tabText(self.ribbon.tabbar.currentIndex())
        if tool.tabs and current not in tool.tabs:
            self.ribbon.show_tab(tool.tabs[0])

    def _toggle_minimap(self) -> None:
        self.canvas.minimap_visible = self.act["minimap"].isChecked()
        self.canvas.request_draw()

    def _selected_image(self) -> ImageNode | None:
        sel = self.project.selected
        return sel if isinstance(sel, ImageNode) else getattr(sel, "parent", None)

    def _reset_selected(self) -> None:
        img = self._selected_image()
        if img is not None and img.placement != Placement():
            self.undo.push(set_placement_command(self.project, img, img.placement, Placement()))

    def _flip_selected(self) -> None:
        from modalytics.services.registration import mirrored_in_place
        img = self._selected_image()
        if img is not None:
            new = mirrored_in_place(img.placement, img.size_um[0])
            self.undo.push(set_placement_command(self.project, img, img.placement, new))

    def _restack(self, front: bool) -> None:
        img = self._selected_image()
        if img is not None:
            self.project.move_image_in_stack(img, len(self.project.images) if front else 0)

    def _on_delete(self) -> None:
        """Delete key: the active tool first (e.g. selected points in Edit points), else the selected mask."""
        tool = self.tools.active
        if tool is not None and tool.on_key(self.canvas.tool_ctx, {"key": "Delete", "modifiers": ()}):
            return
        if self.project.objects:                         # single objects selected: only them, not their layers
            self._delete_objects()
            return
        self._delete_selected_mask()

    def fill_class_menu(self, menu: QtWidgets.QMenu, objects: list | None = None, layer=None) -> None:
        """The profile's classes, with their colours; choosing one gives it to the objects (or the whole layer).
        Grades are numbered as their keys (1–9 with objects selected)."""
        from modalytics.services import classes as cls
        from modalytics.tools.review import grades_of
        current = {(m.group, m.subgroup) for m, _p in objects} if objects else \
            {(layer.group, layer.subgroup)} if layer is not None else set()
        for group, sub in cls.class_choices(self.project):
            r, g, b, _a = cls.class_color(self.project, group, sub)
            pix = QtGui.QPixmap(12, 12)
            pix.fill(QtGui.QColor.fromRgbF(r, g, b))
            grades = grades_of(self.project, group)
            key = f"\t{grades.index(sub) + 1}" if objects and sub in grades and grades.index(sub) < 9 else ""
            act = menu.addAction(QtGui.QIcon(pix), (f"{group} › {sub}" if sub else group) + key,
                                 lambda g_=group, s_=sub: self.set_class(g_, s_, objects, layer))
            act.setCheckable(True)
            act.setChecked(current == {(group, sub)})

    def set_class(self, group: str, subgroup: str, objects: list | None = None, layer=None) -> None:
        """Give the objects (or a whole layer) the class (group, subgroup): one undo step."""
        from modalytics.services import classes as cls
        if layer is not None:
            cmd = cls.change_layer_class(self.project, layer, group, subgroup)
            if cmd is not None:
                self.undo.push(cmd)
            return
        cmd, moved = cls.change_class(self.project, objects or [], group, subgroup)
        if cmd is not None:
            self.undo.push(cmd)
            self.project.select_objects(moved)

    def _add_partial_action(self, menu, objects: list) -> None:
        """"Partial: shape incomplete" for the selected objects (those cut by the image edge are so by themselves)."""
        from modalytics.core.model import EDGE_SHARE, PARTIAL_EDGE
        share = (self.project.profile.edge_share if self.project.profile is not None else EDGE_SHARE)
        reasons = [m.parent.partial_reason(p, share) for m, p in objects if m.parent is not None]
        act = menu.addAction("Partial: shape incomplete")
        act.setCheckable(True)
        act.setChecked(bool(reasons) and all(reasons))
        act.setToolTip("Its shape parameters are left out of classifier training and the shape statistics "
                       "(cut by the section plane, torn, folded, out of focus…)")
        if reasons and all(r == PARTIAL_EDGE for r in reasons):
            act.setText("Partial: cut by the image edge (found by itself)")
            act.setEnabled(False)
        else:
            act.toggled.connect(lambda on: self.set_partial(objects, on))

    def set_partial(self, objects: list, on: bool) -> None:
        """Mark (or unmark) objects partial — one undo step."""
        from modalytics.services.partial import set_partial
        cmd = set_partial(self.project, objects, on)
        if cmd is not None:
            self.undo.push(cmd)

    def _delete_objects(self) -> None:
        """Delete the selected objects (single glands; their layers stay) — one undo step."""
        from modalytics.services.classes import delete_objects
        cmd = delete_objects(self.project, list(self.project.objects))
        if cmd is not None:
            self.project.select(None)
            self.undo.push(cmd)

    def _delete_selected_mask(self) -> None:
        """Delete the selected mask(s) as one undo step (selected images: Layers asks first)."""
        from modalytics.core.commands import Command
        nodes = self.project.selection or ([self.project.selected] if self.project.selected else [])
        masks = [(m, m.parent) for m in nodes if isinstance(m, MaskNode) and m.parent is not None]
        images = [n for n in nodes if isinstance(n, ImageNode)]
        if images:
            self.layers.remove(nodes)
            return
        if not masks:
            return

        def do():
            for m, _p in masks:
                self.project.remove(m)

        def undo():
            for m, p in masks:
                self.project.add_mask(p, m)
        self.undo.push(Command(f"Delete {masks[0][0].name}" if len(masks) == 1 else f"Delete {len(masks)} masks",
                               do, undo))

    def _update_undo_text(self) -> None:
        self.undo_action.setText(f"Undo {self.undo.undo_text}".strip())
        self.redo_action.setText(f"Redo {self.undo.redo_text}".strip())
        self.undo_action.setEnabled(bool(self.undo.undo_text))
        self.redo_action.setEnabled(bool(self.undo.redo_text))

    def _canvas_menu(self, node, pos: QtCore.QPoint) -> None:
        """Right-click menu: settings of the image or mask under the cursor."""
        menu = QtWidgets.QMenu(self)
        if node is None:
            menu.addAction(self.act["fit"])
            menu.exec(pos)
            return
        objects = list(self.project.objects) if isinstance(node, MaskNode) else []
        if not objects:
            self.project.select(node)
        title = menu.addAction(node.name)
        title.setEnabled(False)
        if objects:
            # 0. Selected objects (single glands): their class
            n = len(objects)
            head = menu.addAction(f"{n} object{'s' if n != 1 else ''} selected")
            head.setEnabled(False)
            self.fill_class_menu(menu.addMenu("Class"), objects=objects)
            self._add_partial_action(menu, objects)
            menu.addAction(f"Delete object{'s' if n != 1 else ''}", self._delete_objects)
            menu.addSeparator()
        # 1. Opacity slider inside the menu
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(round(node.opacity * 100))
        slider.setMinimumWidth(180)
        start = node.opacity
        slider.valueChanged.connect(lambda v: self.project.set_opacity(node, v / 100))
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.addWidget(QtWidgets.QLabel("Opacity"))
        lay.addWidget(slider)
        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(box)
        menu.addAction(wa)
        menu.addSeparator()
        menu.addAction("Hide", lambda: self.project.set_visible(node, False))
        menu.addAction("Frame in view", lambda: self.frame_node(node))
        if isinstance(node, ImageNode):
            # 2. Image: order, scale, position
            menu.addAction("Bring to front", lambda: self.project.move_image_in_stack(node, len(self.project.images)))
            menu.addAction("Send to back", lambda: self.project.move_image_in_stack(node, 0))
            menu.addAction("1:1 pixels", lambda: self.canvas.zoom_to_pixels(node))
            from modalytics.ui.processing import fill_process_menu, fill_show_menu
            menu.addSeparator()
            fill_show_menu(menu.addMenu("Show"), self.project, self.undo, node, self)
            fill_process_menu(menu.addMenu("Process"), self, self.project, self.undo, node)
            menu.addSeparator()
            old = node.placement
            menu.addAction("Reset position", lambda: self.undo.push(
                set_placement_command(self.project, node, old, Placement())))
        else:
            # 3. Mask (the whole class layer): its class, delete
            self.fill_class_menu(menu.addMenu("Change class of the whole layer"), layer=node)
            menu.addAction("Delete mask (the whole layer)", self._delete_selected_mask)
        menu.exec(pos)
        if node.opacity != start:
            self.undo.push(set_opacity_command(self.project, node, start, node.opacity), already_done=True)

    def _show_cursor(self, x: float, y: float) -> None:
        # Pointer events arrive at up to several hundred Hz; repaint the label at ≤ 30 Hz
        self._cursor_xy = (x, y)
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _paint_cursor(self) -> None:
        x, y = self._cursor_xy
        self.st_cursor.setText(f"x {x:,.1f} µm   y {y:,.1f} µm")

    def _refresh_status(self) -> None:
        v = self.canvas.view
        self.st_zoom.setText(f"{v.um_per_px:.3g} µm/px")
        mb = self.canvas.budget.used / 1024 ** 2
        self.st_tiles.setText(f"tiles {self.canvas.visible_tile_count()} · {mb:,.0f} MB GPU · "
                              f"{self.canvas.pending_tiles()} loading")
        frames = list(self.canvas.stats.cpu_ms)[-60:]
        # Only meaningful once a few frames exist (the first ones include shader compilation)
        self.st_fps.setText(f"frame {sorted(frames)[len(frames) // 2]:.1f} ms" if len(frames) >= 10 else "frame —")

    # ------------------------------------------------------------ imports
    def add_images(self, images: list[ImageNode]) -> int:
        """Add imported images, keeping only the image types the project's profile accepts, and give them
        the views the profile's processing makes (e.g. a slide's OD map).

        Returns:
            int: How many were added.
        """
        from modalytics.core.model import IMAGE_TYPES
        from modalytics.processing.registry import ProcessingLibrary
        from modalytics.processing.run import profile_views
        profile = self.project.profile
        refused = [i for i in images if profile is not None and not profile.accepts(i.modality)]
        library, problems = None, []
        for img in images:
            if img not in refused:
                if profile is not None and profile.aliases:
                    rename_to_profile(img, profile)          # older class names (e.g. HoloPath's) → current ones
                self.project.add_image(img)
                # The profile's processing for this image type (e.g. a slide's OD map), as views
                if profile is not None and profile.chains_for(img.modality):
                    library = library or ProcessingLibrary()
                    views, why = profile_views(profile, img, library)
                    for view in views:
                        self.project.add_view(img, view)
                    problems += why
        if problems:
            QtWidgets.QMessageBox.warning(self, "Processing at import", "\n".join(problems))
        if refused:
            kinds = sorted({IMAGE_TYPES.get(i.modality, i.modality) for i in refused})
            accepted = ", ".join(IMAGE_TYPES.get(t, t) for t in profile.images)
            QtWidgets.QMessageBox.information(
                self, "Not added",
                f"The profile “{profile.name}” does not use {' or '.join(kinds).lower()} images, so "
                f"{len(refused)} image(s) were not added.\n\nIt accepts: {accepted}.\n"
                f"(Settings › Profiles can change that.)")
        return len(images) - len(refused)

    # ------------------------------------------------------------ measuring
    def measure(self, what: str = "project") -> None:
        """Measure the project (or the selected image) with its profile's parameters, in the background.

        Args:
            what: "project" or "image".
        """
        from modalytics.parameters import results as res
        from modalytics.services.segmentation import Cancelled
        if not self.project.images:
            self.statusBar().showMessage("ℹ️ Nothing to measure: import images first", 6000)
            return
        self._library_classifiers("Measure")                 # a newer classifier saved from Training?
        images = list(self.project.images)
        if what == "image":
            image = self._selected_image()
            if image is None:
                self.statusBar().showMessage("ℹ️ Select an image to measure it", 6000)
                return
            images = [image]
        project = self.project
        include = self.act["include_suggestions"].isChecked()
        n_objects = sum(len(res.engine.objects_of(m.geometry)) for img in images for m in img.masks) + \
            (sum(len([s for s in img.suggestions.items if not s.regrade]) for img in images
                 if img.suggestions is not None) if include else 0)

        def job(progress, cancelled):
            try:
                return res.measure_project(project, progress=progress, cancelled=cancelled, images=images,
                                           include_suggestions=include)
            except res.Cancelled:
                raise Cancelled() from None

        def done(results) -> None:
            if what == "image" and project.results is not None:
                results = project.results.merged(results, [i.id for i in images], [i.id for i in project.images])
            results, _counts, _problems = self._classified(results)        # the profile's classifiers, if any
            project.set_results(results)
            self._mark_dirty()
            self.results_dock.show()
            self.statusBar().showMessage(
                f"✅ Measured {n_objects} object{'s' if n_objects != 1 else ''} on {len(images)} image"
                f"{'s' if len(images) != 1 else ''} ({results.seconds:.1f} s)", 10000)

        def failed(exc) -> None:
            QtWidgets.QMessageBox.warning(self, "Measure", f"Measuring stopped:\n\n{exc}")

        self.tasks.run(f"Measuring {n_objects} objects on {len(images)} image{'s' if len(images) != 1 else ''}…",
                       job, done, failed)

    def _report_ready(self, title: str) -> bool:
        """Measured (else said), and up to date or made anyway (asked)."""
        project = self.project
        if project.results is None:
            QtWidgets.QMessageBox.information(self, title, "Measure the project first (Measure tab): the report "
                                              "shows its results.")
            return False
        if project.results.outdated(project):
            ask = QtWidgets.QMessageBox.question(
                self, title, "The results are out of date (the project changed since it was measured). Make the "
                "report anyway? It says so at the top.\n\nNo: measure first.")
            if ask != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        return True

    def _report_name(self, suffix: str) -> str:
        project = self.project
        base = os.path.splitext(os.path.basename(project.path))[0] if project.path else "Untitled project"
        return os.path.join(self._start_dir(), f"{base} - report{suffix}")

    def make_report_pdf(self) -> None:
        """The project's report as a PDF: the HTML report made in the background (in a temporary folder), then
        printed to PDF here (``ui.pdf_export``: A4, the report's print rules) and opened."""
        import shutil
        import tempfile

        from modalytics.report import html as rep
        if not self._report_ready("Report as PDF"):
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the report as PDF", self._report_name(".pdf"),
                                                        "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        project, folder = self.project, tempfile.mkdtemp(prefix="modalytics_report_")
        html_path = os.path.join(folder, "report.html")

        def printed(written: str) -> None:
            shutil.rmtree(folder, ignore_errors=True)
            self.last_report = written
            self.statusBar().showMessage(f"✅ Report saved as PDF: {written}", 10000)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(written))

        def not_printed(why: str) -> None:
            shutil.rmtree(folder, ignore_errors=True)
            QtWidgets.QMessageBox.warning(self, "Report as PDF", f"The PDF could not be made: {why}.")

        def made(_written) -> None:
            from modalytics.ui.pdf_export import print_pdf
            self.statusBar().showMessage("Printing the report to PDF…")
            try:
                print_pdf(html_path, path, printed, not_printed)
            except ImportError as exc:
                not_printed(f"Qt WebEngine is not available ({exc}); make the HTML report and print it from the "
                            f"browser")

        self.tasks.run("Making the report…",
                       lambda progress, _c: rep.write(project, html_path, progress=progress, pdf=True),
                       made, lambda exc: not_printed(f"the report could not be made: {exc}"))

    def make_report(self) -> None:
        """Write the project's report (``report.html``) in the background and open it in the browser."""
        from modalytics.report import html as rep
        project = self.project
        if not self._report_ready("Report"):
            return
        start = self._report_name(".html")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the report", start, "Report (*.html)")
        if not path:
            return
        if not path.lower().endswith((".html", ".htm")):
            path += ".html"

        def done(written) -> None:
            self.statusBar().showMessage(f"✅ Report saved: {written}", 10000)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(written))
            self.last_report = written

        self.tasks.run("Making the report…", lambda progress, _c: rep.write(project, path, progress=progress), done,
                       lambda exc: QtWidgets.QMessageBox.warning(self, "Report",
                                                                 f"The report could not be made:\n\n{exc}"))

    def _review(self, key: str, grade: str | None = None) -> None:
        """A review button in Properties: done by the Review tool, as its keys do."""
        tool, ctx = self.tools.tools["review"], self.canvas.tool_ctx
        if key == "accept":
            tool.accept(ctx, grade)
        elif key == "accept_all":
            tool.accept(ctx, every=True)
        elif key == "delete":
            tool.delete(ctx)
        elif key == "delete_all":
            tool.delete_all(ctx)
        elif key in ("next", "previous"):
            tool.step(ctx, 1 if key == "next" else -1)
        self.canvas._view_changed()

    def _classified(self, results):
        """Results with the profile's classifiers applied: (results, {grade: objects}, problems)."""
        from modalytics.parameters import results as res
        prof = self.project.profile
        counts, problems = {}, []
        for g in (prof.classes.groups if prof is not None else []):
            clf = prof.classifier_for(g.name)
            if clf is None:
                continue
            results, p = res.classify(results, clf)
            problems += [f"{g.name}: {x}" for x in p]
            for r in results.of_scope("object"):
                if r.group == g.name and r.prediction:
                    grade = r.prediction["grade"] or "not graded (shape incomplete)"
                    counts[grade] = counts.get(grade, 0) + 1
        return results, counts, problems

    def _library_classifiers(self, title: str) -> None:
        """If the profile in your library classifies differently from the project's copy (e.g. a newer classifier
        saved from Training, another minimum confidence), offer to take its classifiers."""
        from modalytics.profiles.library import ProfileLibrary
        project, prof = self.project, self.project.profile
        if prof is None:
            return
        try:
            lib = ProfileLibrary().get(project.profile_id) if project.profile_id else None
        except Exception:                                            # noqa: BLE001 — removed or unreadable
            lib = None
        if lib is None or (lib.classification == prof.classification and lib.min_confidence == prof.min_confidence):
            return
        changed = [g for g in set(lib.classification) | set(prof.classification)
                   if lib.classification.get(g) != prof.classification.get(g)]
        lines = []
        from modalytics.training.classifiers import describe
        for g in sorted(changed):
            c = lib.classifier_for(g)
            lines.append(f"• {g}: " + (f"{describe(c)} ({c.note})" if c else "no classifier"))
        for g in sorted(set(lib.min_confidence) | set(prof.min_confidence)):
            if lib.confidence_for(g) != prof.confidence_for(g):
                lines.append(f"• {g}: minimum confidence {lib.confidence_for(g):.2f} (this project: "
                             f"{prof.confidence_for(g):.2f})")
        ask = (f"The profile '{lib.name}' in your library classifies differently from this project's copy:\n\n"
               + "\n".join(lines) + "\n\nUse the library's classifiers for this project?")
        if QtWidgets.QMessageBox.question(self, title, ask) == QtWidgets.QMessageBox.StandardButton.Yes:
            prof.classification = dict(lib.classification)
            prof.min_confidence = dict(lib.min_confidence)
            self._mark_dirty()

    # ------------------------------------------------------------ classification (Annotate)
    def classify_selected(self) -> None:
        """Grade the selected objects with the profile's classifiers, in the background: a sure grade that differs
        from an object's becomes a suggestion to review (``services.suggest.classify_objects``); the Review tool
        then starts on the first."""
        from modalytics.services import suggest as sg
        project, prof = self.project, self.project.profile
        selected = list(project.objects)
        if not selected:
            self.statusBar().showMessage("ℹ️ Select objects first: Select tool, click a gland (Shift adds more)",
                                         8000)
            return
        self._library_classifiers("Classify selected")
        objects = [(m, p) for m, p in selected if prof is not None and prof.classifier_for(m.group) is not None]
        if not objects:
            groups = ", ".join(sorted({m.group for m, _p in selected}))
            QtWidgets.QMessageBox.information(
                self, "Classify selected", f"The profile '{prof.name if prof else ''}' has no classifier for "
                f"{groups}. Make one in Training (File › Training…: Build, then Save to profile).")
            return

        def job(_progress, cancelled):
            return sg.classify_objects(project, objects, cancelled=cancelled)

        def done(got) -> None:
            made, said = got
            first = None
            for image, items in made.items():
                cmd = sg.add_command(project, image, items, {"classifier": "Classify selected"})
                if cmd is not None:
                    self.undo.push(cmd)
                    first = first or (image.suggestions, items[0])
            sure = [x for x in said if x[2]]
            agree = [x for x in sure if x[2] == x[1]]
            unsure = [x for x in said if not x[2]]
            parts = [f"{len(sure) - len(agree)} grade(s) suggested"] if len(sure) > len(agree) else []
            parts += [f"{len(agree)} agree with their grade"] if agree else []
            parts += [f"{len(unsure)} unsure, left as they are (confidence "
                      + ", ".join(f"{s:.2f}" for _g, _c, _n, s, _w in unsure[:5]) + ")"] if unsure else []
            if first is not None:
                node, item = first
                project.set_current_suggestion(node, item.id)
                self.tools.activate("review")
            self.statusBar().showMessage("✅ Classified " + str(len(said)) + " object(s): " + "; ".join(parts)
                                         + (" — Enter accepts, Delete drops" if first is not None else ""), 15000)

        self.tasks.run(f"Classifying {len(objects)} object(s)…", job, done,
                       lambda exc: QtWidgets.QMessageBox.warning(self, "Classify selected", str(exc)))

    def _toggle_results(self) -> None:
        self.results_dock.setVisible(not self.results_dock.isVisible())

    def show_object(self, image_id: str, mask_id: str, index: int) -> None:
        """Select an object of the results on the canvas: its mask selected, the view on it, its outline drawn."""
        from modalytics.core import geometry as geo
        from modalytics.parameters.engine import objects_of
        mask = self.project.find(mask_id)
        if not isinstance(mask, MaskNode):
            self.statusBar().showMessage("ℹ️ That object's mask no longer exists (measure again)", 6000)
            return
        objects = objects_of(mask.geometry)
        if not 0 <= index < len(objects):
            self.statusBar().showMessage("ℹ️ That object is no longer in its mask (measure again)", 6000)
            return
        poly = objects[index]
        to_world = mask.parent.local_to_world()
        rings = [geo.apply(to_world, np.asarray(poly.exterior.coords))] + \
            [geo.apply(to_world, np.asarray(r.coords)) for r in poly.interiors]
        self.project.select(mask)
        self.canvas.preview.set_highlight(rings)
        self.canvas.view.fit(bounds(rings[0]), margin=0.6)
        self.canvas._view_changed()

    def show_graded_object(self, path: str, image_name: str, group: str, x: float, y: float,
                           outline: list | None = None) -> None:
        """Open a project (if it is not the one open) and show the object of ``group`` at (x, y) µm of an image.

        Used by Training (Build › Mistakes, Segmentation models › Evaluate). The point finds the object even
        after older class names were merged. ``outline`` (an exterior in the image's µm): an outline that is not
        in the project, e.g. one a model found, is drawn instead.
        """
        from modalytics.parameters.engine import objects_of
        from shapely.geometry import Point
        same = self.project.path and os.path.normcase(os.path.abspath(self.project.path)) == \
            os.path.normcase(os.path.abspath(path))
        if not same:
            if not self._confirm_discard():
                return
            self.open_path(path)
        self.raise_()
        self.activateWindow()
        image = next((i for i in self.project.images if i.name == image_name), None)
        if image is None:
            self.statusBar().showMessage(f"ℹ️ {image_name} is not in this project any more", 8000)
            return
        if outline:
            from modalytics.core import geometry as geo
            ring = geo.apply(image.local_to_world(), np.asarray(outline, float))
            self.project.select(image)
            self.canvas.preview.set_highlight([ring])
            self.canvas.view.fit(bounds(ring), margin=0.6)
            self.canvas._view_changed()
            return
        pt = Point(x, y)
        for m in image.masks:
            if m.group != group:
                continue
            for k, poly in enumerate(objects_of(m.geometry)):
                if poly.buffer(0.5).contains(pt):
                    self.show_object(image.id, m.id, k)
                    return
        self.statusBar().showMessage(f"ℹ️ That object is no longer on {image_name} (its outline changed)", 8000)

    def _show_result_image(self, image_id: str) -> None:
        image = self.project.find(image_id)
        if isinstance(image, ImageNode):
            self.canvas.preview.set_highlight(None)
            self.project.select(image)
            self.frame_node(image)

    def export_results(self, scope: str) -> None:
        """Save one table of the results as CSV."""
        from modalytics.parameters.results import export_csv
        results = self.project.results
        if results is None:
            self.statusBar().showMessage("ℹ️ Nothing to export: measure first", 6000)
            return
        names = {"object": "objects", "class": "classes", "image": "images"}
        base = os.path.splitext(os.path.basename(self.project.path))[0] if self.project.path else "results"
        start = os.path.join(self._start_dir(), f"{base} - {names[scope]}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Export the {names[scope]} table", start,
                                                        "CSV (*.csv)")
        if not path:
            return
        n = export_csv(results, scope, path)
        self.statusBar().showMessage(f"✅ {n} rows saved to {path}", 10000)

    def _import_hpro(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import HoloPath project", "", "HoloPath (*.hpro)")
        if path and self.add_images(importers.import_hpro(path)):
            self.canvas.fit_all()

    def _import_images(self, paths: list[str] | None = None) -> None:
        """Import slides (read in place), phase maps and microscope images (the importer library).

        Args:
            paths: Files to import (default: ask with an Open dialog).
        """
        from modalytics.image_import.registry import ImporterLibrary
        from modalytics.ui.image_import import file_filter, import_images
        library = ImporterLibrary()
        if paths is None:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Import images", self._start_dir(),
                                                              file_filter(library))
        if not paths:
            return
        nodes, errors, notes = import_images(self, paths, library)
        if nodes and (not self.large_ok(nodes) or not self.place_imported(nodes)):
            self.statusBar().showMessage("ℹ️ Import cancelled: nothing was added", 6000)
            return
        added = self.add_images(nodes) if nodes else 0
        if errors:
            QtWidgets.QMessageBox.warning(self, "Import images", "\n\n".join(errors))
        if added:
            if added == 1:
                self.frame_node(nodes[-1])
            else:                                            # the new images, wherever they were put
                corners = np.vstack([n.world_corners() for n in nodes])
                self.canvas.view.fit(bounds(corners), margin=0.05)
                self.canvas._view_changed()
            self.statusBar().showMessage(f"✅ Imported {added} image(s)" + "".join(f" · {n}" for n in notes),
                                         12000 if notes else 5000)

    LARGE_IMAGES = 3            # more whole-slide images than this in one project: a word of warning first
    LARGE_GIGAPIXELS = 10.0     # … or more pixels than this, all images together

    def large_ok(self, nodes: list) -> bool:
        """With many large images (whole-slide scans) in one project, say it may become slow; False to cancel."""
        def gpx(img):
            h, w = img.source.shape[:2]
            return h * w / 1e9
        every = list(self.project.images) + list(nodes)
        slides = [i for i in every if i.modality == "wsi" or gpx(i) > 0.2]
        new_large = [i for i in nodes if i.modality == "wsi" or gpx(i) > 0.2]
        total = sum(gpx(i) for i in every)
        if not new_large or (len(slides) <= self.LARGE_IMAGES and total <= self.LARGE_GIGAPIXELS):
            return True
        ask = QtWidgets.QMessageBox.question(
            self, "Many large images",
            f"With these, the project would hold {len(slides)} whole-slide or other large image(s), "
            f"{total:,.1f} gigapixels in all.\n\nModalytics reads large images in tiles, so it works with them, but "
            f"many in one project can make it slow: more tiles kept in memory, slower saving and measuring. For "
            f"whole-slide scans one (or a few) per project is usually best.\n\nImport them anyway?")
        return ask == QtWidgets.QMessageBox.StandardButton.Yes

    def place_imported(self, nodes: list) -> bool:
        """Where new images go: one is centred in the view; several are arranged as chosen (``ArrangeDialog``).

        Returns:
            bool: False if the arrangement was cancelled.
        """
        from modalytics.core import arrange as ar
        cx, cy = self.canvas.view.cx, self.canvas.view.cy
        sizes = [n.size_um for n in nodes]
        if len(nodes) == 1:
            w, h = sizes[0]
            nodes[0].placement = Placement(cx - w / 2, cy - h / 2)
            return True
        from modalytics.ui.arrange_dialog import ArrangeDialog
        others = self.project.world_bounds() if self.project.images else None
        dlg = ArrangeDialog([n.name for n in nodes], sizes, others is not None, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        layout = dlg.layout_chosen()
        spots = ar.positions(sizes, [n.name for n in nodes], layout)
        bw, bh = ar.bounds(sizes, spots)
        if dlg.placement() == "next" and others is not None:
            gap = max(w for w, _h in sizes) * max(layout.gap_pct, 10.0) / 100.0
            x0, y0 = others[2] + gap, others[1]                     # to the right of what is there, top-aligned
        else:
            x0, y0 = cx - bw / 2, cy - bh / 2
        for n, (x, y) in zip(nodes, spots):
            n.placement = Placement(x0 + x, y0 + y)
        return True

    def _image_for_menu(self) -> ImageNode | None:
        """The selected image (the only one if there is one), else a hint in the status bar."""
        img = self._selected_image()
        if img is None and len(self.project.images) == 1:
            img = self.project.images[0]
        if img is None:
            self.tasks.message("ℹ️ Select an image first (in Layers or on the canvas)")
        return img

    def _chain_editor(self) -> None:
        """Ribbon › Processing chains…: the chain editor on the selected image."""
        from modalytics.ui.processing import open_chain_editor
        img = self._image_for_menu()
        if img is not None:
            open_chain_editor(self, self.project, self.undo, img)

    def _images_to_process(self, hint: bool = True) -> list:
        """The selected images (a selected mask counts as its image); the only image when nothing is selected."""
        images = self.project.selected_images()
        if not images and len(self.project.images) == 1:
            images = list(self.project.images)
        if not images and hint:
            self.tasks.message("ℹ️ Select an image first (in Layers or on the canvas)")
        return images

    def _quick_step_menu(self) -> None:
        """Ribbon › Quick step: every step the selected image(s) can take, as a menu at the cursor."""
        from modalytics.ui.processing import fill_step_menu
        images = self._images_to_process()
        if images:
            menu = QtWidgets.QMenu(self)
            if len(images) > 1:
                menu.addSection(f"{len(images)} images")
            fill_step_menu(menu, self, self.project, self.undo, images)
            menu.exec(QtGui.QCursor.pos())

    def _chain_menu(self) -> None:
        """Ribbon › Apply chain: your saved chains that fit the selected image(s)."""
        from modalytics.ui.processing import fill_chain_menu
        images = self._images_to_process()
        if images:
            menu = QtWidgets.QMenu(self)
            if len(images) > 1:
                menu.addSection(f"{len(images)} images")
            fill_chain_menu(menu, self, self.project, self.undo, images)
            menu.exec(QtGui.QCursor.pos())

    def rebuild_process_tab(self) -> None:
        """The Process tab's step and chain buttons, as the project's profile chooses (its toolbar steps and chains
        that fit its image types; without a profile, the default steps), grouped by step category, chains last."""
        from modalytics.processing.chains import ChainLibrary
        from modalytics.processing.registry import DEFAULT_TOOLBAR, WORKFLOW_ORDER, ProcessingLibrary
        if not hasattr(self, "ribbon"):
            return
        for group in self._process_groups:
            self.ribbon.remove_group(group)
        for act in self.step_actions.values():
            self.removeAction(act)
            act.deleteLater()
        self._process_groups, self.step_actions, self.step_specs = [], {}, {}
        profile = self.project.profile
        library, chains = ProcessingLibrary(), ChainLibrary()
        types = list(profile.images) if profile is not None else None
        by_group: dict[str, list] = {}
        for item_id in (profile.toolbar() if profile is not None else DEFAULT_TOOLBAR):
            if item_id.startswith("chain:"):
                item = chains.get(item_id[len("chain:"):])
                fits = item is not None and (types is None or any(not item.problems(t, library) for t in types))
                group, tip = "My chains", (f"{item.name}: {item.summary(library)}" if item else "")
            else:
                item = library.get(item_id)
                fits = item is not None and (types is None or any(item.accepts(t) for t in types))
                group, tip = (item.group, f"{item.name}: {item.description}") if item else ("", "")
            if not fits:
                continue
            self.step_specs[item_id] = item
            self.step_actions[item_id] = self._action(item.button_label, item.button_icon,
                                                      lambda _c=False, k=item_id: self._run_step(k), tip=tip)
            by_group.setdefault(group, []).append(self.step_actions[item_id])
        order = WORKFLOW_ORDER
        for title in [*order, *[g for g in by_group if g not in order and g != "My chains"], "My chains"]:
            if by_group.get(title):
                self._process_groups.append(self.ribbon.add_group("Process", title, by_group[title]))
        self._update_step_actions()

    def _run_step(self, item_id: str) -> None:
        """A step or chain button: on the selected image(s) that can take it (a step's options asked once)."""
        from modalytics.processing.registry import ProcessingLibrary
        from modalytics.ui.processing import apply_chain, apply_chain_to_images, run_step
        item = self.step_specs.get(item_id)
        if item is None:
            return
        if item_id.startswith("chain:"):
            library = ProcessingLibrary()
            images = [i for i in self._images_to_process() if not item.problems(i.modality, library)]
            if len(images) == 1:
                apply_chain(self, self.project, self.undo, images[0], item)
            elif images:
                apply_chain_to_images(self, self.project, self.undo, images, item)
            return
        images = [i for i in self._images_to_process() if item.accepts(i.modality)]
        if images:
            run_step(self, self.project, self.undo, images, item)

    def _update_step_actions(self) -> None:
        """A button is on when one of the images it would process can take its step (or chain)."""
        types = {i.modality for i in self._images_to_process(hint=False)}
        library = None
        for item_id, act in self.step_actions.items():
            item = self.step_specs[item_id]
            if item_id.startswith("chain:"):
                if library is None:
                    from modalytics.processing.registry import ProcessingLibrary
                    library = ProcessingLibrary()
                act.setEnabled(any(not item.problems(t, library) for t in types))
            else:
                act.setEnabled(any(item.accepts(t) for t in types))

    def _show_menu(self) -> None:
        """Ribbon › Show…: original or one of the selected image's views."""
        from modalytics.ui.processing import fill_show_menu
        img = self._image_for_menu()
        if img is not None:
            menu = QtWidgets.QMenu(self)
            fill_show_menu(menu, self.project, self.undo, img, self)
            menu.exec(QtGui.QCursor.pos())

    def _segment_image(self) -> None:
        """Ribbon button: segment the whole selected image(s) (several: one after the other; large slides ask)."""
        images = self.project.selected_images()
        if not images and len(self.project.images) == 1:
            images = list(self.project.images)
        self.segment_images(images)

    def segment_images(self, images: list) -> None:
        """Segment whole images with the Segment tool's settings, in one background task."""
        if not images:
            self.tasks.message("ℹ️ Select the image(s) to segment (in Layers — Ctrl/Shift-click for several — or "
                               "on the canvas)")
            return
        if self.tasks.busy:
            self.tasks.message("⚠️ A segmentation is already running")
            return
        self.tools.activate("segment")          # its settings (model, threshold…) are the ones used
        self.tools.tools["segment"].run_whole_images(self.canvas.tool_ctx, images)

    def _layers_menu(self, menu, nodes: list) -> None:
        """What the Layers right-click menu offers for the selected layers' images: process, segment, measure."""
        from modalytics.ui.processing import fill_process_menu, fill_process_menu_many
        masks = [n for n in nodes if isinstance(n, MaskNode)]
        if len(masks) == 1:                                # a class layer: all its objects get another class
            self.fill_class_menu(menu.addMenu("Change class"), layer=masks[0])
        images = []
        for n in nodes:
            img = n if isinstance(n, ImageNode) else getattr(n, "parent", None)
            if isinstance(img, ImageNode) and img not in images:
                images.append(img)
        if not images:
            return
        if len(images) == 1:
            fill_process_menu(menu.addMenu("Process"), self, self.project, self.undo, images[0])
        else:                                            # each image that can take the step or chain
            fill_process_menu_many(menu.addMenu(f"Process {len(images)} images"), self, self.project, self.undo,
                                   images)
        menu.addAction(f"Segment {len(images)} images" if len(images) > 1 else "Segment image",
                       lambda: self.segment_images(images))

    def _import_annotations(self) -> None:
        """Merge GeoJSON or Leica/Aperio XML annotations into the selected image (one undo step)."""
        from modalytics.io import annotation_files as af
        img = self._selected_image()
        if img is None:
            QtWidgets.QMessageBox.information(self, "Import annotations", "Select the image the annotations belong to.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Annotations for {img.name}", self._start_dir(),
            "Annotations (*.geojson *.json *.xml);;GeoJSON (*.geojson *.json);;Leica/Aperio XML (*.xml)")
        if not path:
            return
        try:
            if path.lower().endswith(".xml"):
                classes, info = af.import_aperio_xml(path, img)
                if info["mpp_mismatch"]:
                    QtWidgets.QMessageBox.warning(
                        self, "Calibration differs",
                        f"The file says {info['microns_per_pixel']} µm/px, the image {img.pixel_size_um[0]} µm/px. "
                        "Vertices are image pixels, so the image calibration is used.")
            else:
                classes = af.import_geojson(path, img)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Import annotations", f"{path}\n\n{exc}")
            return
        n = self.merge_masks(img, classes, f"Import {os.path.basename(path)}")
        self.statusBar().showMessage(f"✅ Imported {n} object(s) into {img.name}", 5000)

    def merge_masks(self, img, classes: dict, text: str) -> int:
        """Union {(group, subgroup): (geometry, rgba)} into an image's class masks as one undo step.

        Returns:
            int: Number of objects imported.
        """
        from modalytics.core import annotations as ann
        from modalytics.core.commands import Command
        profile = self.project.profile
        if profile is not None and profile.aliases:          # older class names (e.g. HoloPath's) → current ones
            renamed: dict = {}
            for (group, sub), (geom, color) in classes.items():
                key = profile.label_for(group, sub)
                renamed[key] = (renamed[key][0].union(geom), renamed[key][1]) if key in renamed else (geom, color)
            classes = renamed
        changes, created = [], []
        for (group, sub), (geom, color) in classes.items():
            self.project.labels.ensure(group, sub, color)
            mask = self.project.mask_for(img, group, sub, create=False)
            if mask is None:
                mask = MaskNode(geometry=None, color=self.project.labels.color(group, sub), group=group, subgroup=sub)
                created.append(mask)
            changes.append((mask, mask.geometry, ann.as_multipolygon(mask.geometry.union(geom))))

        def do():
            for m in created:
                if m.parent is None:
                    self.project.add_mask(img, m)
            for m, _o, n in changes:
                self.project.set_mask_geometry(m, n)

        def undo():
            for m, o, _n in changes:
                self.project.set_mask_geometry(m, o)
            for m in created:
                self.project.remove(m)

        self.undo.push(Command(text, do, undo))
        return sum(len(g.geoms) for g, _c in classes.values())

    def _export_geojson(self) -> None:
        from modalytics.io import annotation_files as af
        img = self._selected_image()
        if img is None or not img.masks:
            QtWidgets.QMessageBox.information(self, "Export GeoJSON", "Select an image that has masks.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export masks", self._start_dir(), "GeoJSON (*.geojson)")
        if path:
            n = af.export_geojson(img, path if path.endswith(".geojson") else path + ".geojson")
            self.statusBar().showMessage(f"✅ Exported {n} object(s)", 5000)

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self.canvas.shutdown()
        super().closeEvent(event)

