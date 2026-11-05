"""
The canvas widget: draws the project in µm with pygfx and handles the
mouse/keyboard scheme.

Controls (Autodesk-style):

    Wheel                 zoom about the cursor
    + / − (Ctrl or not)   zoom about the cursor (a tablet's dial sends Ctrl + "+/−")
    Middle drag           pan
    Right drag            pan (a pen's side buttons send the right button)
    Ctrl + middle drag    move the image under the cursor (or the selected one)
    Left click            select the top-most image under the cursor
    Right click           context menu (built by the UI layer), on release when the pointer did not move
    F                     fit everything        1   1:1 pixels of the selected image
    Esc                   clear the selection
    Minimap               click or drag to jump there

The canvas observes the ``Project``: it never stores geometry of its own,
it only mirrors it into GPU objects (see ``visuals.py``).
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
import pygfx as gfx
from PySide6 import QtCore, QtGui, QtWidgets
from rendercanvas.qt import QRenderWidget

from modalytics.core import geometry as geo
from modalytics.core.commands import UndoStack, set_placement_command
from modalytics.core.model import ImageNode, MaskNode, Project, SuggestionNode
from modalytics.render import tiling
from modalytics.render.gpu import init_gpu
from modalytics.render.tile_loader import TileLoader
from modalytics.render.view import View, format_length, nice_length
from modalytics.core import annotations as ann
from modalytics.render import minimap
from modalytics.render.visuals import COLORMAPS, ImageVisual, TileBudget, prepare_tile
from modalytics.tools.base import ToolContext
from modalytics.tools.select import RectSelect, node_at, object_at

MIDDLE, LEFT, RIGHT = 3, 1, 2           # rendercanvas button ids
WHEEL_STEP = 1.2                        # zoom factor per wheel notch (120 units), per + / − key
PAN_SLOP_PX = 4.0                       # a right press moving further pans; less: a right click (context menu)
CURSORS = {"default": QtCore.Qt.CursorShape.ArrowCursor, "crosshair": QtCore.Qt.CursorShape.CrossCursor,
           "none": QtCore.Qt.CursorShape.BlankCursor}       # a tool's ``cursor`` → the system pointer
MINIMAP_BORDER = 2                      # minimap frame width (logical px)
MINIMAP_MIN_VIEW = 12                   # smallest drawn view rectangle in the minimap (px)


def _make_line(capacity: int, material: gfx.LineMaterial) -> gfx.Line:
    """A line with a fixed-size position buffer (all NaN = nothing drawn).

    Replacing a line's geometry every frame makes pygfx rebuild its pipeline
    (≈ 2 ms per line on the dev machine); writing into a fixed buffer is free.
    """
    return gfx.Line(gfx.Geometry(positions=np.full((capacity, 3), np.nan, np.float32)), material)


def _set_line(line: gfx.Line, points: np.ndarray) -> None:
    """Write (N, 2) points into a fixed-capacity line or point set (NaN rows break a line).

    The buffer doubles when a path needs more room (rare: long freehand
    outlines, large selected masks), so steady-state updates never rebuild.
    """
    if len(points) > line.geometry.positions.data.shape[0]:
        cap = 1 << int(np.ceil(np.log2(len(points))))
        line.geometry = gfx.Geometry(positions=np.full((cap, 3), np.nan, np.float32))
    buf = line.geometry.positions
    n = min(len(points), buf.data.shape[0])
    buf.data[:] = np.nan
    buf.data[:n, :2] = points[:n]
    buf.data[:n, 2] = 0
    buf.update_full()


def _closed(c: np.ndarray) -> np.ndarray:
    """Close a ring of corners and end it with a NaN break."""
    return np.vstack([c, c[:1], np.full((1, 2), np.nan)])


PEN_ERASES = ("brush", "freehand", "polygon", "eraser")     # tools whose stroke the pen's eraser end turns to erasing


class Preview:
    """What tools draw while a gesture is in progress (world µm, overlay scene).

    Args:
        scene: Overlay scene to add the preview objects to.
        color: Accent colour.
    """

    def __init__(self, scene: gfx.Scene, color: str) -> None:
        self._scene = scene
        self.dabs = None                       # a pen stroke of varying width (``set_dabs``)
        self.path = _make_line(1024, gfx.LineMaterial(thickness=1.5, color=color))
        # Brush stroke: a line whose thickness is in *world* units = the brush diameter
        self.stroke = _make_line(1024, gfx.LineMaterial(thickness=1.0, thickness_space="world",
                                                        color=(1, 1, 1, 0.35)))
        self.circle = _make_line(100, gfx.LineMaterial(thickness=1.5, color=color))
        self.circle_halo = _make_line(100, gfx.LineMaterial(thickness=3.5, color="#000000"))
        # Vertex handles (Edit points): white squares with a dark edge, constant screen size
        self.handles = gfx.Points(gfx.Geometry(positions=np.full((256, 3), np.nan, np.float32)),
                                  gfx.PointsMarkerMaterial(marker="square", size=7, color="#ffffff",
                                                           edge_width=1.5, edge_color="#000000"))
        self.handle_hot = gfx.Points(gfx.Geometry(positions=np.full((1, 3), np.nan, np.float32)),
                                     gfx.PointsMarkerMaterial(marker="square", size=11, color=color,
                                                              edge_width=1.5, edge_color="#000000"))
        # Selected handles (accent) and the rubber-band selection box (dashed)
        self.handle_sel = gfx.Points(gfx.Geometry(positions=np.full((64, 3), np.nan, np.float32)),
                                     gfx.PointsMarkerMaterial(marker="square", size=10, color=color,
                                                              edge_width=1.5, edge_color="#000000"))
        self.box = _make_line(6, gfx.LineMaterial(thickness=1.25, color=color, dash_pattern=(5, 4)))
        # An object picked in the results: a thick outline with a dark halo (tools do not clear it)
        self.highlight_halo = _make_line(1024, gfx.LineMaterial(thickness=5.0, color="#000000"))
        self.highlight = _make_line(1024, gfx.LineMaterial(thickness=2.5, color=color))
        for i, ob in enumerate((self.highlight_halo, self.highlight, self.stroke, self.path, self.circle_halo,
                                self.circle, self.handles, self.handle_sel, self.handle_hot, self.box)):
            ob.material.depth_test = False
            ob.material.alpha_mode = "blend"
            ob.render_order = 10 + i
            scene.add(ob)

    def set_path(self, points: np.ndarray, closed: bool = False) -> None:
        """An open or closed polyline (polygon / freehand in progress)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        _set_line(self.path, np.vstack([pts, pts[:1]]) if closed and len(pts) > 2 else pts)

    def set_stroke(self, points: np.ndarray, radius: float) -> None:
        """A brush stroke of the given radius (µm) along points."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 1:
            pts = np.vstack([pts, pts + (radius * 1e-3, 0)])     # a single dab still shows
        self.stroke.material.thickness = max(2 * radius, 1e-6)
        _set_line(self.stroke, pts)

    def set_dabs(self, points, radii) -> None:
        """A brush stroke whose width varies (pen pressure): one round dab per point, sized in world µm.

        Dabs are depth-tested against each other, so where they overlap each pixel is drawn once (no darker
        overlaps). Empty ``points`` clears it.
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if self.dabs is not None:
            self._scene.remove(self.dabs)
            self.dabs = None
        if not len(pts):
            return
        pos = np.full((len(pts), 3), 100.0, np.float32)      # above the images (their tiles write depth)
        pos[:, :2] = pts
        sizes = np.asarray(radii, np.float32).reshape(-1) * 2
        mat = gfx.PointsMaterial(size_space="world", size_mode="vertex", color=(1, 1, 1, 0.35))
        mat.alpha_mode = "blend"
        mat.depth_test, mat.depth_write, mat.depth_compare = True, True, "<"
        self.dabs = gfx.Points(gfx.Geometry(positions=pos, sizes=sizes), mat)
        self.dabs.render_order = 5
        self._scene.add(self.dabs)

    def set_circle(self, center, radius: float) -> None:
        """Brush outline at the cursor."""
        t = np.linspace(0, 2 * np.pi, 97)
        ring = np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])
        _set_line(self.circle, ring)
        _set_line(self.circle_halo, ring)

    def set_handles(self, points: np.ndarray, hot=None, selected=None) -> None:
        """Vertex handles (world µm), the selected ones, and an optional hovered one."""
        _set_line(self.handles, np.asarray(points, dtype=float).reshape(-1, 2))
        sel = np.zeros((0, 2)) if selected is None else np.asarray(selected, dtype=float).reshape(-1, 2)
        _set_line(self.handle_sel, sel)
        _set_line(self.handle_hot, np.zeros((0, 2)) if hot is None else np.asarray([hot], dtype=float))

    def set_box(self, a=None, b=None) -> None:
        """Rubber-band rectangle between world points a and b (None hides it)."""
        if a is None:
            _set_line(self.box, np.zeros((0, 2)))
            return
        (x0, y0), (x1, y1) = a, b
        _set_line(self.box, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]))

    def set_highlight(self, rings: list[np.ndarray] | None) -> None:
        """Outline these closed rings (world µm; None or [] hides it). Kept until replaced."""
        parts = []
        for ring in rings or []:
            ring = np.asarray(ring, dtype=float).reshape(-1, 2)
            parts += [ring, ring[:1], np.full((1, 2), np.nan)]           # closed, then a break
        pts = np.vstack(parts) if parts else np.zeros((0, 2))
        _set_line(self.highlight, pts)
        _set_line(self.highlight_halo, pts)

    def clear(self) -> None:
        for ob in (self.path, self.stroke, self.circle, self.circle_halo, self.handles, self.handle_sel,
                   self.handle_hot, self.box):
            _set_line(ob, np.zeros((0, 2)))
        self.set_dabs([], [])


class FrameStats:
    """Rolling frame-time statistics for the status bar and the benchmarks."""

    def __init__(self, size: int = 240) -> None:
        self.cpu_ms: deque[float] = deque(maxlen=size)       # time spent in our draw function
        self.interval_ms: deque[float] = deque(maxlen=size)  # time between frame starts
        self._last = None

    def frame(self, start: float, end: float) -> None:
        if self._last is not None:
            self.interval_ms.append((start - self._last) * 1e3)
        self._last = start
        self.cpu_ms.append((end - start) * 1e3)

    def summary(self) -> dict:
        cpu = np.array(self.cpu_ms) if self.cpu_ms else np.zeros(1)
        itv = np.array(self.interval_ms) if self.interval_ms else np.zeros(1)
        return {"cpu_ms_mean": float(cpu.mean()), "cpu_ms_median": float(np.median(cpu)),
                "cpu_ms_p95": float(np.percentile(cpu, 95)),
                "interval_ms_mean": float(itv.mean()),
                "fps": float(1000.0 / itv.mean()) if itv.mean() > 0 else 0.0}


class Canvas(QtWidgets.QWidget):
    """GPU canvas showing a ``Project`` in world µm.

    Args:
        project: The domain model to display.
        undo: Undo stack for interactive edits.
        theme: Dict of colours ("canvas_bg", "accent", "minimap_bg", "minimap_border", "text").
        vsync: Sync presentation to the display (off for benchmarks).
        present: "bitmap" (default) renders offscreen and paints each frame with
            Qt: ~145 fps on the dev machine, and docks resize smoothly. "screen"
            renders into a native swapchain: ~1.9× faster, but every dock-separator
            step reconfigures the swapchain (18–64 ms), so the separator lags and
            jumps behind the cursor. Default from MODALYTICS_PRESENT, else "bitmap".

    Signals:
        context_requested(object, QPoint): right-click on a node (or None) at a global position.
        cursor_moved(float, float): world µm under the cursor.
        view_changed(): pan/zoom changed.
    """

    context_requested = QtCore.Signal(object, QtCore.QPoint)
    cursor_moved = QtCore.Signal(float, float)
    view_changed = QtCore.Signal()
    _tile_ready = QtCore.Signal()

    def __init__(self, project: Project, undo: UndoStack, theme: dict, vsync: bool = True,
                 present: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.gpu_name = init_gpu()
        self.project = project
        self.undo = undo
        self.view = View()
        self.stats = FrameStats()
        self.max_magnification = 1.5
        self.minimap_visible = True
        self.detail = (0, 0)                   # (wanted, missing) detail tiles in the last frame
        self.uploads, self.upload_ms = 0, 0.0  # tiles uploaded in the last frame and the time it took
        self._theme = dict(theme)

        # 1. Render widget + renderer
        self.present = present or os.environ.get("MODALYTICS_PRESENT", "bitmap")
        # Bitmap presentation is not vsync-locked: cap at the display refresh rate so
        # the GPU does not render frames nobody can see
        screen = QtWidgets.QApplication.primaryScreen()
        refresh = screen.refreshRate() if screen is not None else 60.0
        max_fps = max(refresh, 30.0) if vsync else 1000
        self.rc = QRenderWidget(self, update_mode="ondemand", max_fps=max_fps, vsync=vsync,
                                present_method=self.present)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.rc)
        from modalytics.render.pen import PenInput
        self.pen = PenInput(self.rc)          # graphics tablets (pressure, eraser end) and every point of a stroke
        self._pen_return: str | None = None   # the tool to go back to after an eraser-end stroke
        self._pressed = False                   # a button is down (a release without a press is ignored: some
                                                # pen drivers report a side button's release twice)
        self.renderer = gfx.WgpuRenderer(self.rc, ppaa="none", pixel_filter="linear")

        # 2. Scenes: main (images + masks), overlay (selection, scale bar), minimap
        self.scene = gfx.Scene()
        self.background = gfx.Background.from_color(theme["canvas_bg"])
        self.scene.add(self.background)
        self.camera = gfx.OrthographicCamera(1, 1, maintain_aspect=False, depth_range=(1, 10000))
        self.camera.local.scale_y = -1          # world y points down, like images
        self.overlay = gfx.Scene()
        self._build_overlay()
        self._build_minimap()
        self.preview = Preview(self.overlay, theme["accent"])
        self.tools = None
        self._cursor_name: str | None = None    # the system pointer set (a tool's ``cursor``, or "drag")
        self.tool_ctx = ToolContext(project, undo, self.view, self.preview, self.request_draw)

        # 3. Visuals and tile loading
        self.visuals: dict[str, ImageVisual] = {}
        self.budget = TileBudget()
        self.loader = TileLoader(workers=6, on_ready=self._tile_ready.emit)
        self._tile_ready.connect(self.request_draw)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.shutdown)

        # 4. Interaction state
        self._drag: dict | None = None
        self._minimap_drag = False
        self._last_cursor = (0.0, 0.0)
        self.rc.add_event_handler(self._on_event, "pointer_down", "pointer_move", "pointer_up",
                                  "wheel", "key_down", "resize", "double_click")

        # 5. Observe the model
        project.structure_changed.connect(self._on_structure)
        project.node_changed.connect(self._on_node_changed)
        project.selection_changed.connect(lambda _n: (self._update_selection(), self._refresh_minimap_target()))
        self._on_structure()
        self.rc.request_draw(self._draw)

    # ------------------------------------------------------------------ public
    def request_draw(self) -> None:
        """Schedule a redraw (cheap; coalesced by the render loop)."""
        self.rc.request_draw()

    def fit_all(self) -> None:
        """Zoom to show every visible image."""
        b = self.project.world_bounds()
        if b:
            self._sync_size()
            self.view.fit(b)
            self._view_changed()

    def zoom_step(self, steps: float) -> None:
        """Zoom in (``steps`` > 0) or out by wheel notches, about the cursor when it is over the canvas (else the
        centre): the + / − keys, a tablet's dial, the Zoom in / out buttons."""
        self._sync_size()
        pos = self.rc.mapFromGlobal(QtGui.QCursor.pos())
        inside = self.rc.rect().contains(pos)
        x, y = (pos.x(), pos.y()) if inside else (self.view.width / 2, self.view.height / 2)
        self.view.zoom_at(WHEEL_STEP ** steps, x, y)
        self._view_changed()

    def zoom_to_pixels(self, image: ImageNode | None = None) -> None:
        """Zoom so one level-0 pixel of ``image`` (default: selection) is one screen pixel."""
        image = image or (self.project.selected if isinstance(self.project.selected, ImageNode) else None)
        if image is not None:
            self.view.um_per_px = image.pixel_size_um[0] * image.placement.scale
            self._view_changed()

    def set_theme(self, theme: dict) -> None:
        """Apply canvas colours from the theme tokens."""
        self._theme = dict(theme)
        self.scene.remove(self.background)
        self.background = gfx.Background.from_color(theme["canvas_bg"])
        self.scene.add(self.background)
        self._minimap_bg.material.set_colors(theme["minimap_bg"])
        self._minimap_border_bg.material.set_colors(theme["minimap_border"])
        self._scalebar_text.material.color = theme["text"]
        self._scalebar_text.material.outline_color = theme["canvas_bg"]
        self._scalebar_line.material.color = theme["text"]
        self.request_draw()

    def set_tools(self, manager) -> None:
        """Route left/right clicks, drags and keys to the manager's active tool."""
        self.tools = manager
        manager.bind(self.tool_ctx)
        manager.active_changed.connect(lambda _t: self._tool_cursor())
        self._tool_cursor()

    def _tool_cursor(self, over_minimap: bool = False) -> None:
        """The system pointer the active tool wants (its ``pointer``: none where it draws its own, as Brush's ring);
        the arrow over the minimap. Set only when it changes."""
        name = "default" if over_minimap or self.tool is None else getattr(self.tool, "pointer", "default")
        if name != self._cursor_name:
            self._cursor_name = name
            self.rc.setCursor(CURSORS.get(name, QtCore.Qt.CursorShape.ArrowCursor))

    @property
    def tool(self):
        return self.tools.active if self.tools is not None else None

    def snapshot(self):
        """Return the last rendered canvas frame as an (H, W, 4) uint8 array.

        Works in both present modes (OS screenshots cannot see a native
        swapchain in "screen" mode).
        """
        return self.renderer.snapshot()

    def visible_tile_count(self) -> int:
        """Number of tile objects currently in the scene (for diagnostics)."""
        return sum(len(v.tiles) for v in self.visuals.values())

    def pending_tiles(self) -> int:
        """Tiles queued or being read."""
        return self.loader.busy

    def shutdown(self) -> None:
        """Stop background tile reading (no callbacks into a deleted widget)."""
        self.loader.on_ready = None
        self.loader.shutdown()

    # ---------------------------------------------------------------- overlay
    def _build_overlay(self) -> None:
        accent = self._theme["accent"]
        self._selection = _make_line(64, gfx.LineMaterial(thickness=2.0, color=accent))
        self.overlay.add(self._selection)
        self._scalebar_line = _make_line(2, gfx.LineMaterial(thickness=3.0, color=self._theme["text"]))
        self._scalebar_label = ""
        self._scalebar_text = gfx.Text(text="", font_size=13, screen_space=True, anchor="bottom-left",
                                       material=gfx.TextMaterial(color=self._theme["text"],
                                                                 outline_color=self._theme["canvas_bg"],
                                                                 outline_thickness=0.3))
        self.overlay.add(self._scalebar_line, self._scalebar_text)
        for ob in (self._selection, self._scalebar_line, self._scalebar_text):
            ob.material.depth_test = False

    def _build_minimap(self) -> None:
        self.minimap_scene = gfx.Scene()
        self._minimap_bg = gfx.Background.from_color(self._theme["minimap_bg"])
        self.minimap_scene.add(self._minimap_bg)
        self.minimap_camera = gfx.OrthographicCamera(1, 1, maintain_aspect=False, depth_range=(1, 10000))
        self.minimap_camera.local.scale_y = -1
        self._minimap_target: ImageNode | None = None
        self._minimap_mask: MaskNode | None = None
        self._minimap_key = None
        self._minimap_geom = None
        self._thumbs: dict[str, tuple] = {}
        self._minimap_timer = QtCore.QTimer(self, singleShot=True, interval=150,
                                            timeout=self._refresh_minimap_target)
        self._minimap_image: gfx.Image | None = None
        # View rectangle: a white line over a wider black one, readable on tissue,
        # on white slide background and on the dark canvas alike
        self._minimap_frame_halo = _make_line(6, gfx.LineMaterial(thickness=4.0, color="#000000"))
        self._minimap_frame = _make_line(6, gfx.LineMaterial(thickness=2.0, color="#ffffff"))
        self._minimap_frame_halo.local.z = 49
        self._minimap_frame.local.z = 50
        self._minimap_frame_halo.render_order = 10
        self._minimap_frame.render_order = 11
        # Footprints of the other images (e.g. phase maps), in the phase-modality colour
        self._minimap_outlines = _make_line(6 * 256, gfx.LineMaterial(thickness=1.0, color="#4cc9f0"))
        self._minimap_outlines.local.z = 40
        self.minimap_scene.add(self._minimap_frame_halo, self._minimap_frame, self._minimap_outlines)
        # Frame around the minimap: drawn as a slightly larger viewport filled with the border colour
        self._minimap_border_scene = gfx.Scene()
        self._minimap_border_bg = gfx.Background.from_color(self._theme["minimap_border"])
        self._minimap_border_scene.add(self._minimap_border_bg)
        # The minimap is drawn after the main scene, whose tiles have already written
        # depth there: its backgrounds must ignore depth or they vanish over tissue
        for bg in (self._minimap_bg, self._minimap_border_bg):
            bg.material.depth_test = False
        self.minimap_rect = (0, 0, 0, 0)

    def _minimap_choice(self):
        """(image, mask) the minimap shows, following the selection.

        A selected image (WSI, phase map…) is shown as is; a selected mask shows
        its image with the masked area(s) bright and the rest dimmed; with
        nothing selected, the largest visible image.
        """
        sel = self.project.selected
        if isinstance(sel, MaskNode) and sel.parent is not None and sel.parent.visible:
            return sel.parent, sel
        if isinstance(sel, ImageNode) and sel.visible:
            return sel, None
        images = [i for i in self.project.images if i.visible]
        return max(images, key=lambda i: i.size_um[0] * i.size_um[1], default=None), None

    def _thumbnail(self, image):
        """Cached minimap thumbnail of an image (rebuilt when its display or calibration changes)."""
        key = (image.view_source.uid, image.view_display, image.pixel_size_um)
        hit = self._thumbs.get(image.id)
        if hit is None or hit[0] != key:
            hit = (key, minimap.thumbnail(image, COLORMAPS))
            self._thumbs[image.id] = hit
        return hit[1]

    def _refresh_minimap_target(self) -> None:
        """Rebuild the minimap picture if what it should show has changed."""
        target, mask = self._minimap_choice()
        key = None if target is None else (target.id, target.view_source.uid, target.view_display, target.pixel_size_um,
                                           mask.id if mask else None, mask.visible if mask else None)
        geom = mask.geometry if mask is not None else None
        if key == self._minimap_key and geom is self._minimap_geom:
            return
        self._minimap_key, self._minimap_geom = key, geom
        if self._minimap_image is not None:
            self.minimap_scene.remove(self._minimap_image)
            self._minimap_image = None
        self._minimap_target, self._minimap_mask = target, mask
        if target is None:
            self.request_draw()
            return
        # 1. Thumbnail (coarsest level, colormapped like the canvas); masked areas bright in mask mode
        rgba, px, py = self._thumbnail(target)
        if mask is not None:
            rgba = minimap.highlight(rgba, mask.geometry if mask.visible else None, px, py)
        mat = gfx.ImageBasicMaterial(clim=(0, 255), interpolation="linear")
        mat.depth_test = False
        img = gfx.Image(gfx.Geometry(grid=gfx.Texture(rgba, dim=2, colorspace="srgb")), mat)
        img.local.scale = (px, py, 1)
        img.local.position = (0.5 * px, 0.5 * py, 0)
        # 2. Placed with the image's own geometry (rotation, flip, scale)
        holder = gfx.Group()
        holder.local.matrix = geo.to_4x4(target.local_to_world())
        holder.add(img)
        self._minimap_image = holder
        self.minimap_scene.add(holder)
        self.request_draw()

    def _layout_minimap(self) -> None:
        """Bottom-right rectangle sized to the target's aspect ratio."""
        t = self._minimap_target
        if t is None or not self.minimap_visible:
            self.minimap_rect = (0, 0, 0, 0)
            return
        x0, y0, x1, y1 = geo.bounds(t.world_corners())
        aspect = (y1 - y0) / max(x1 - x0, 1e-9)
        w = 260.0
        h = min(w * aspect, 220.0)
        w = h / aspect if aspect > 0 else w
        m = 12.0
        self.minimap_rect = (self.view.width - w - m, self.view.height - h - m, w, h)
        self.minimap_camera.width, self.minimap_camera.height = x1 - x0, y1 - y0
        self.minimap_camera.local.position = ((x0 + x1) / 2, (y0 + y1) / 2, 5000)

    def _minimap_to_world(self, x: float, y: float) -> tuple[float, float]:
        rx, ry, rw, rh = self.minimap_rect
        cam = self.minimap_camera
        cx, cy = cam.local.position[:2]
        return (cx + ((x - rx) / rw - 0.5) * cam.width, cy + ((y - ry) / rh - 0.5) * cam.height)

    def _in_minimap(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.minimap_rect
        return rw > 0 and rx <= x <= rx + rw and ry <= y <= ry + rh

    # ------------------------------------------------------------ model sync
    def _on_structure(self) -> None:
        """Create/remove visuals to match the project, then restack."""
        ids = {img.id for img in self.project.images}
        for vid in list(self.visuals):
            if vid not in ids:
                vis = self.visuals.pop(vid)
                for key in list(vis.tiles):
                    self.budget.forget((vid,) + key)
                self.scene.remove(vis.group)
        for index, img in enumerate(self.project.images):
            vis = self.visuals.get(img.id)
            if vis is None:
                vis = ImageVisual(img)
                self.visuals[img.id] = vis
                self.scene.add(vis.group)
            vis.set_stack_index(index + 1)
            vis.sync_masks()
            if vis.suggestions is not None:
                vis.suggestions.rebuild(self._class_colors())
            vis.update_style()
        for iid in [i for i in self._thumbs if i not in {img.id for img in self.project.images}]:
            del self._thumbs[iid]
        self._refresh_minimap_target()
        self._update_selection()
        self.request_draw()

    def _class_colors(self) -> dict:
        """{(group, subgroup): rgba} of the project's classes (suggestions are drawn in their class's colour)."""
        # The colour of the mask a suggestion would join; for a class with no mask yet, the profile's colour
        labels = self.project.labels
        out = {(g, s): labels.color(g, s) for g, s in labels.choices()}
        prof = self.project.profile
        if prof is not None:
            out.update({(g, s): prof.classes.color(g, s) for g, s in prof.classes.choices()})
        out.update({(m.group, m.subgroup): m.color for img in self.project.images for m in img.masks})
        return out

    def _on_node_changed(self, node, field: str) -> None:
        if isinstance(node, SuggestionNode):
            vis = self.visuals.get(node.parent.id) if node.parent else None
            if vis is not None and vis.suggestions is not None:
                vis.suggestions.rebuild(self._class_colors()) if field == "suggestions" \
                    else vis.suggestions.update_style()
            self._update_selection()
            self.request_draw()
            return
        if isinstance(node, MaskNode):
            vis = self.visuals.get(node.parent.id) if node.parent else None
            mv = vis.masks.get(node.id) if vis else None
            if mv is not None:
                mv.rebuild() if field == "geometry" else mv.update_style()
            if node is self._minimap_mask:
                self._minimap_timer.start()             # the highlighted area changed: refresh shortly
        else:
            vis = self.visuals.get(node.id)
            if vis is None:
                return
            if field == "source":                       # other pixels (a relinked file): rebuild the visual
                for key in list(vis.tiles):
                    self.budget.forget((node.id,) + key)
                self.scene.remove(vis.group)
                del self.visuals[node.id]
                self._thumbs.pop(node.id, None)
                self._on_structure()
                return
            if field == "placement":
                vis.update_placement()          # GPU-only: no tile is touched
                if node is self._minimap_target and self._minimap_image is not None:
                    self._minimap_image.local.matrix = geo.to_4x4(node.local_to_world())
            elif field == "pixel_size":
                vis.update_tile_transforms()
                for mv in vis.masks.values():
                    mv.rebuild()
            else:
                vis.update_style()
            if field in ("visible", "display", "pixel_size"):
                self._refresh_minimap_target()
        self._update_selection()
        self.request_draw()

    def _update_selection(self) -> None:
        """Outline every selected layer (several, after a Ctrl/Shift-click in Layers)."""
        nodes = self.project.selection or ([self.project.selected] if self.project.selected else [])
        parts = []
        for mask, poly in self.project.objects:                  # single objects: only their own outline
            if mask.parent is not None and mask.parent.visible and mask.visible:
                m = mask.parent.local_to_world()
                parts += [_closed(geo.apply(m, np.asarray(r.coords))) for r in [poly.exterior, *poly.interiors]]
        if self.project.objects:
            nodes = []
        for node in nodes:
            if isinstance(node, MaskNode) and node.parent is not None and node.parent.visible and node.visible:
                # The mask's own outline, in world µm
                m = node.parent.local_to_world()
                parts += [_closed(geo.apply(m, r)) for ext, holes in ann.rings(node.geometry) for r in [ext, *holes]]
            elif isinstance(node, ImageNode) and node.visible:
                parts.append(_closed(node.world_corners()))
            elif isinstance(node, SuggestionNode) and node.visible and node.current_item is not None:
                # The suggestion under review
                m = node.parent.local_to_world()
                g = node.current_item.geometry
                parts += [_closed(geo.apply(m, np.asarray(r.coords))) for r in [g.exterior, *g.interiors]]
        _set_line(self._selection, np.vstack(parts) if parts else np.zeros((0, 2)))
        self.request_draw()

    # ------------------------------------------------------------- drawing
    def _sync_size(self) -> None:
        w, h = self.renderer.logical_size
        self.view.width, self.view.height = max(int(w), 1), max(int(h), 1)

    def _view_changed(self) -> None:
        self.view_changed.emit()
        self.request_draw()

    def _draw(self) -> None:
        """Per-frame: update camera, choose/upload tiles, render all scenes."""
        t0 = time.perf_counter()
        self._sync_size()
        v = self.view

        # 1. Camera from the view
        self.camera.width, self.camera.height = v.width * v.um_per_px, v.height * v.um_per_px
        self.camera.local.position = (v.cx, v.cy, 5000)

        # 2. Upload tiles finished by the loader, then decide what is needed
        t_up = time.perf_counter()
        self.uploads = 0
        for key, done in self.loader.take_ready():
            if done is None:                                     # the read failed (reported by the loader)
                continue
            vid, tkey = key[0], key[1:]
            uid, data = done
            vis = self.visuals.get(vid)
            # A tile read before the image switched to other pixels (another view, a relinked file) is dropped
            if vis is not None and vis.node.view_source.uid == uid and tkey not in vis.tiles:
                self.budget.add(key, vis.add_tile(tkey, data))
                self.uploads += 1
        self.upload_ms = (time.perf_counter() - t_up) * 1e3
        self._update_tiles()

        # 3. Overlay: scale bar
        self._update_scalebar()

        # 4. Render: main scene, overlay, then the minimap in its corner
        self.renderer.render(self.scene, self.camera, flush=False)
        self.renderer.render(self.overlay, self.camera, flush=False)
        self._layout_minimap()
        if self.minimap_rect[2] > 0:
            self._update_minimap_overlay()
            rx, ry, rw, rh = self.minimap_rect
            b = MINIMAP_BORDER
            self.renderer.render(self._minimap_border_scene, self.minimap_camera,
                                 rect=(rx - b, ry - b, rw + 2 * b, rh + 2 * b), flush=False)
            self.renderer.render(self.minimap_scene, self.minimap_camera, rect=self.minimap_rect, flush=False)
        self.renderer.flush()
        t1 = time.perf_counter()
        self.stats.frame(t0, t1)
        if t1 - t0 > 0.5:                               # a screen update this slow is worth a line in the log
            from modalytics import diagnostics
            diagnostics.event("slow screen update", ms=round((t1 - t0) * 1e3), upload_ms=round(self.upload_ms))

    def _update_tiles(self) -> None:
        """Choose levels and tiles per image; queue missing ones; evict over budget."""
        corners = self.view.corners()
        requests, protected = [], set()
        wanted_detail = missing_detail = 0
        for vid, vis in self.visuals.items():
            node = vis.node
            if not node.visible:
                continue
            # 1. View in this image's level-0 pixel grid
            to_px = np.linalg.inv(node.pixel_to_world())
            view_px = geo.apply(to_px, corners)
            levels = node.view_source.levels
            level = tiling.choose_level(self.view.um_per_px, vis.level_um_per_px(), self.max_magnification)
            coarsest = len(levels) - 1
            # 2. Wanted tiles: the chosen level, plus the whole coarsest level as fallback
            wanted = [(level, ty, tx) for ty, tx in tiling.visible_tiles(levels[level], view_px)]
            if level != coarsest:
                wanted += [(coarsest, ty, tx) for ty, tx in tiling.visible_tiles(levels[coarsest], view_px, 0)]
            show = set(wanted)
            detail = [k for k in wanted if k[0] == level]
            wanted_detail += len(detail)
            missing_detail += sum(1 for k in detail if k not in vis.tiles)
            # 3. Already-cached tiles of nearby levels stay visible while detail loads
            for key in vis.tiles:
                lv = key[0]
                if lv != level and lv != coarsest and abs(lv - level) <= 1:
                    show.add(key)
            for key, obj in vis.tiles.items():
                obj.visible = key in show
                if key in show:
                    self.budget.touch((vid,) + key)
                    protected.add((vid,) + key)
            # 4. Queue the missing ones (chosen level first, centre first)
            src = node.view_source
            for key in wanted:
                if key not in vis.tiles:
                    requests.append(((vid,) + key,
                                     lambda s=src, k=key: (s.uid, prepare_tile(s.read_tile(*k)))))
        self.loader.set_wanted(requests)
        # Diagnostics: how much of the needed detail is on screen this frame
        self.detail = (wanted_detail, missing_detail)
        # 5. Evict least-recently-used tiles beyond the GPU budget
        for key in self.budget.over_budget(protected):
            vis = self.visuals.get(key[0])
            if vis is not None:
                vis.remove_tile(key[1:])
            self.budget.forget(key)

    def _update_scalebar(self) -> None:
        v = self.view
        length = nice_length(v.um_per_px * 120)
        x0, y0 = v.screen_to_world(20, v.height - 24)
        _set_line(self._scalebar_line, np.array([[x0, y0], [x0 + length, y0]]))
        tx, ty = v.screen_to_world(20, v.height - 31)
        self._scalebar_text.local.position = (tx, ty, 0)
        label = format_length(length)
        if label != self._scalebar_label:        # text layout is costly: only when it changes
            self._scalebar_text.set_text(label)
            self._scalebar_label = label

    def _update_minimap_overlay(self) -> None:
        # 1. View rectangle, never smaller than MINIMAP_MIN_VIEW px so it stays visible when zoomed in
        c = self.view.corners()
        um_per_mpx = self.minimap_camera.width / max(self.minimap_rect[2], 1)
        half = MINIMAP_MIN_VIEW / 2 * um_per_mpx
        cx, cy = c.mean(axis=0)
        x0, y0 = min(c[:, 0].min(), cx - half), min(c[:, 1].min(), cy - half)
        x1, y1 = max(c[:, 0].max(), cx + half), max(c[:, 1].max(), cy + half)
        ring = _closed(geo.rect_corners(x1 - x0, y1 - y0) + (x0, y0))
        _set_line(self._minimap_frame_halo, ring)
        _set_line(self._minimap_frame, ring)
        # 2. Footprints of the other images
        rings = [_closed(img.world_corners()) for img in self.project.images
                 if img.visible and img is not self._minimap_target]
        _set_line(self._minimap_outlines, np.vstack(rings) if rings else np.zeros((0, 2)))

    # ------------------------------------------------------- Qt geometry
    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._reframe()

    def moveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().moveEvent(event)
        self._reframe()

    def _reframe(self) -> None:
        """Keep the content still on the monitor while docks resize the canvas.

        Without this the view keeps its centre, so the image slides by half of
        every resize step (``View.reframe``). The redraw is only *requested*:
        forcing a synchronous draw per step was measured to slow separator
        dragging (34 ms per step instead of 8 ms).
        """
        top_left = self.mapToGlobal(QtCore.QPoint(0, 0))
        geom = (top_left.x(), top_left.y(), self.width(), self.height())
        last, self._last_geom = getattr(self, "_last_geom", None), geom
        if last is None or last == geom or not self.isVisible():
            return
        self.view.reframe(geom[0] - last[0], geom[1] - last[1], geom[2] - last[2], geom[3] - last[3])
        self.view.width, self.view.height = max(geom[2], 1), max(geom[3], 1)
        self.view_changed.emit()
        self.rc.request_draw()

    # ------------------------------------------------------------ interaction
    def _on_event(self, ev: dict) -> None:
        etype = ev["event_type"]
        if etype in ("pointer_down", "pointer_move", "pointer_up", "double_click"):
            # The pen's state (tablets): which end touches, how hard (None for a mouse)
            pointer, pressure = self.pen.press if etype == "pointer_down" else self.pen.current()
            ev.setdefault("pointer_type", pointer)
            ev.setdefault("pressure", pressure)
        if etype == "resize":
            self._sync_size()
            self._view_changed()
        elif etype == "wheel":
            self._on_wheel(ev)
        elif etype == "pointer_down":
            self._on_down(ev)
        elif etype == "pointer_move":
            self._on_move(ev)
        elif etype == "pointer_up":
            self._on_up(ev)
        elif etype == "key_down":
            self._on_key(ev)
        elif etype == "double_click" and self.tool is not None:
            wx, wy = self.view.screen_to_world(ev["x"], ev["y"])
            self.tool.on_double_click(self.tool_ctx, ev, wx, wy)

    def _on_wheel(self, ev: dict) -> None:
        if self._in_minimap(ev["x"], ev["y"]):
            return
        self.view.zoom_at(WHEEL_STEP ** (-ev["dy"] / 120.0), ev["x"], ev["y"])
        self._view_changed()

    def _on_down(self, ev: dict) -> None:
        self._pressed = True
        x, y, button = ev["x"], ev["y"], ev["button"]
        wx, wy = self.view.screen_to_world(x, y)
        ctrl = "Control" in ev.get("modifiers", ())
        if (button == LEFT and ev.get("pointer_type") == "eraser" and self.tools is not None
                and self.tool is not None and self.tool.id in PEN_ERASES):
            # The pen turned over: this stroke erases, then the drawing tool comes back
            self._pen_return = self.tool.id
            self.tools.activate("eraser")
        if button == LEFT and self._in_minimap(x, y):
            # 1. Minimap: jump and keep following the drag
            self._minimap_drag = True
            self.view.cx, self.view.cy = self._minimap_to_world(x, y)
            self._view_changed()
        elif button == MIDDLE and ctrl:
            # 2. Move tool: the selected image if under the cursor, else the top-most one
            sel = self.project.selected
            target = sel if isinstance(sel, ImageNode) and sel.contains_world(wx, wy) else self.project.image_at(wx, wy)
            if target is not None:
                self.project.select(target)
                self._drag = {"mode": "move", "image": target, "start": target.placement,
                              "anchor": (wx, wy)}
                self._set_drag_cursor(QtCore.Qt.CursorShape.SizeAllCursor)
        elif button == MIDDLE:
            # 3. Pan
            self._drag = {"mode": "pan", "last": (x, y)}
            self._set_drag_cursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif (button == LEFT and "Shift" in ev.get("modifiers", ()) and self.tool is not None
              and not self.tool.uses_shift_click and not self.tool.busy()):
            # 4. Shift + click / drag: add to the selection, whatever the tool (it stays active)
            self._drag = {"mode": "select", "pick": RectSelect(self.tool_ctx, ev, wx, wy, add=True)}
        elif self.tool is not None and self.tool.on_pointer_down(self.tool_ctx, ev, wx, wy):
            # 5. The active tool consumed it (draw, select, move…)
            pass
        elif button == LEFT:
            # 6. Default: select what is under the cursor
            self.project.select(node_at(self.project, wx, wy))
        elif button == RIGHT:
            # 7. Right: a drag pans (a pen's side button), a click opens the context menu (on release)
            self._drag = {"mode": "right", "start": (x, y), "last": (x, y), "panning": False}

    def _set_drag_cursor(self, shape) -> None:
        """The pointer during a pan or a move (the tool's comes back on release)."""
        self._cursor_name = "drag"
        self.rc.setCursor(shape)

    def _context_menu(self, x: float, y: float) -> None:
        """The context menu of what is under (x, y) (built by the UI layer)."""
        wx, wy = self.view.screen_to_world(x, y)
        node = node_at(self.project, wx, wy)
        obj = object_at(self.project, wx, wy) if isinstance(node, MaskNode) else None
        if obj is not None and not any(m is obj[0] and p.equals(obj[1]) for m, p in self.project.objects):
            self.project.select_objects([obj])                   # the object under the cursor (unless selected)
        pos = self.rc.mapToGlobal(QtCore.QPoint(int(x), int(y)))
        self.context_requested.emit(node, pos)

    def _replay(self, trail: list, ev: dict) -> None:
        """Give the tool the pointer positions the event system merged away (all but the event's own, the last)."""
        if not trail or not ev.get("buttons"):
            return
        for x, y, pressure in trail[:-1]:
            if (x, y) == (ev["x"], ev["y"]):
                continue
            e = dict(ev, x=x, y=y, pressure=pressure if pressure is not None else ev.get("pressure"))
            wx, wy = self.view.screen_to_world(x, y)
            self.tool.on_pointer_move(self.tool_ctx, e, wx, wy)

    def _on_move(self, ev: dict) -> None:
        x, y = ev["x"], ev["y"]
        wx, wy = self.view.screen_to_world(x, y)
        self._last_cursor = (wx, wy)
        self.cursor_moved.emit(wx, wy)
        if self._minimap_drag:
            self.view.cx, self.view.cy = self._minimap_to_world(x, y)
            self._view_changed()
            return
        trail = self.pen.take_trail()
        if self._drag is None:
            self._tool_cursor(self._in_minimap(x, y))
            if self.tool is not None:
                self._replay(trail, ev)
                self.tool.on_pointer_move(self.tool_ctx, ev, wx, wy)
            return
        if self._drag["mode"] == "select":
            self._drag["pick"].move(self.tool_ctx, ev, wx, wy)
        elif self._drag["mode"] == "right" and not self._drag["panning"]:
            sx, sy = self._drag["start"]
            if (x - sx) ** 2 + (y - sy) ** 2 > PAN_SLOP_PX ** 2:   # moved: a pan, not a click
                self._drag["panning"] = True
                self._set_drag_cursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        if self._drag["mode"] == "pan" or (self._drag["mode"] == "right" and self._drag["panning"]):
            lx, ly = self._drag["last"]
            self.view.pan_pixels(x - lx, y - ly)
            self._drag["last"] = (x, y)
            self._view_changed()
        elif self._drag["mode"] == "move":
            ax, ay = self._drag["anchor"]
            # Live update of the domain model; the canvas mirrors it on the GPU only
            self.project.set_placement(self._drag["image"], self._drag["start"].moved_by(wx - ax, wy - ay))

    def _on_up(self, ev: dict) -> None:
        self._minimap_drag = False
        if not self._pressed:                              # a release without a press (a pen driver's echo)
            return
        self._pressed = False
        drag, self._drag = self._drag, None
        if drag is None and self.tool is not None:
            self._replay(self.pen.take_trail(), ev)
            wx, wy = self.view.screen_to_world(ev["x"], ev["y"])
            self.tool.on_pointer_up(self.tool_ctx, ev, wx, wy)
            if self._pen_return is not None:               # after an eraser-end stroke: the drawing tool again
                back, self._pen_return = self._pen_return, None
                self.tools.activate(back)
            return
        if self._cursor_name == "drag":                    # after a pan or a move: the tool's pointer again
            self._tool_cursor()
        if drag and drag["mode"] == "right" and not drag["panning"]:
            self._context_menu(*drag["start"])
        if drag and drag["mode"] == "select":
            drag["pick"].finish(self.tool_ctx, *self.view.screen_to_world(ev["x"], ev["y"]))
        if drag and drag["mode"] == "move":
            img = drag["image"]
            if img.placement != drag["start"]:
                # One undo step for the whole drag
                self.undo.push(set_placement_command(self.project, img, drag["start"], img.placement),
                               already_done=True)

    def _on_key(self, ev: dict) -> None:
        key = ev.get("key", "")
        if len(key) == 1 and ord(key) < 32 and "Control" in ev.get("modifiers", ()):
            # Qt reports Ctrl+letter as a control character (Ctrl+A → "\x01"): restore the letter
            key = chr(ord(key) + 96)
            ev = {**ev, "key": key}
        if self.tool is not None and self.tool.on_key(self.tool_ctx, ev):
            return
        if key in ("+", "="):
            self.zoom_step(1)
        elif key in ("-", "_"):
            self.zoom_step(-1)
        elif key in ("f", "F"):
            self.fit_all()
        elif key == "1":
            self.zoom_to_pixels()
        elif key == "Escape":
            self.project.select(None)
