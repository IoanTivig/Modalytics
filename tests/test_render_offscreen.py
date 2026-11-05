"""Headless GPU test: draw an image through ImageVisual offscreen and check pixels.

wgpu renders without a window, so visual checks can run in CI without a
display (Qt's offscreen platform has no OpenGL; this path does not need it).
"""
import numpy as np
import pytest

gfx = pytest.importorskip("pygfx")


def test_image_visual_orientation_and_placement():
    from rendercanvas.offscreen import RenderCanvas

    from modalytics.core.geometry import Placement
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.render.gpu import init_gpu
    from modalytics.render.visuals import ImageVisual, prepare_tile

    init_gpu()
    # 20×10 µm RGB image: top half red, bottom half blue, placed at (100, 50) µm
    data = np.zeros((10, 20, 3), np.uint8)
    data[:5, :, 0] = 255
    data[5:, :, 2] = 255
    node = ImageNode("t", ArraySource(data), (1.0, 1.0), "brightfield", Placement(100, 50))
    vis = ImageVisual(node)
    vis.add_tile((0, 0, 0), prepare_tile(node.source.read_tile(0, 0, 0)))
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"), vis.group)

    canvas = RenderCanvas(size=(40, 20))
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)
    cam = gfx.OrthographicCamera(40, 20, maintain_aspect=False, depth_range=(1, 10000))
    cam.local.scale_y = -1
    cam.local.position = (110, 55, 5000)          # view = world [90, 130] × [45, 65]
    canvas.request_draw(lambda: renderer.render(scene, cam))
    img = np.asarray(canvas.draw())

    assert tuple(img[7, 20, :3]) == (255, 0, 0)    # world (110, 52): top half → red
    assert tuple(img[13, 20, :3]) == (0, 0, 255)   # world (110, 58): bottom half → blue
    assert tuple(img[10, 5, :3]) == (0, 0, 0)      # world (95, 55): left of the image → background


def _render_phase_over(bottom_rgb, cmap: str) -> np.ndarray:
    """Render a 0→1 phase ramp (20×20 µm at (10, 10)) over a 40×40 µm solid-colour image."""
    from rendercanvas.offscreen import RenderCanvas

    from modalytics.core.geometry import Placement
    from modalytics.core.model import Display, ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.render.gpu import init_gpu
    from modalytics.render.visuals import ImageVisual, prepare_tile

    init_gpu()
    below = np.zeros((40, 40, 3), np.uint8)
    below[...] = bottom_rgb
    ramp = np.tile(np.linspace(0, 1, 20, dtype=np.float32), (20, 1))
    bottom = ImageNode("wsi", ArraySource(below), (1, 1), "wsi")
    top = ImageNode("phase", ArraySource(ramp, robust_range=False), (1, 1), "phase", Placement(10, 10))
    top.display = Display(clim=(0.0, 1.0), colormap=cmap)
    scene = gfx.Scene()
    for i, n in enumerate((bottom, top)):
        vis = ImageVisual(n)
        vis.set_stack_index(i + 1)
        vis.add_tile((0, 0, 0), prepare_tile(n.source.read_tile(0, 0, 0)))
        scene.add(vis.group)
    canvas = RenderCanvas(size=(40, 40))
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1, ppaa="none")  # as in the app
    cam = gfx.OrthographicCamera(40, 40, maintain_aspect=False, depth_range=(1, 10000))
    cam.local.scale_y = -1
    cam.local.position = (20, 20, 5000)
    canvas.request_draw(lambda: renderer.render(scene, cam))
    return np.asarray(canvas.draw()).astype(int)


def test_phase_map_at_full_opacity_hides_what_is_below():
    """Regression: pygfx's built-in gray colormap ramps alpha, so phase maps looked ~20 % opaque at 100 %.

    At opacity 1 the phase pixels must not depend on the image underneath.
    """
    from modalytics.render.visuals import COLORMAP_NAMES

    for cmap in COLORMAP_NAMES:
        over_red = _render_phase_over((255, 0, 0), cmap)
        over_black = _render_phase_over((0, 0, 0), cmap)
        inner = (slice(12, 28), slice(12, 28))           # inside the phase map, away from its edge
        diff = np.abs(over_red[inner][..., :3] - over_black[inner][..., :3]).max()
        assert diff <= 2, f"{cmap}: what is below shows through (max diff {diff})"
