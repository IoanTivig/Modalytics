"""Segmentation service: region reading, vectorisation, block-wise runs (synthetic model-free parts)."""
import numpy as np
import pytest

from modalytics.core.model import ImageNode
from modalytics.io.sources import ArraySource
from modalytics.services import segmentation as seg


def test_read_region_resamples_to_model_resolution():
    data = np.arange(400 * 600, dtype=np.float32).reshape(400, 600)
    img = ImageNode("p", ArraySource(data), (0.5, 0.5), "phase")
    pixels, (x0, y0, px, py) = seg.read_region(img, (10, 20, 110, 70), 1.0)    # 100 × 50 µm at 1 µm/px
    assert pixels.shape == (50, 100) and np.isclose(px, 1.0) and (x0, y0) == (10.0, 20.0)


def test_vectorize_square_with_hole_gives_exact_areas():
    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True           # 40 × 40 px
    mask[25:35, 25:35] = False          # 10 × 10 px hole
    geom = seg.vectorize(mask, (100.0, 200.0, 0.5, 0.5), simplify_px=0)
    (poly,) = geom.geoms
    assert len(list(poly.interiors)) == 1
    # Pixel-edge outlines: the outer square is exact; hole corners are chamfered by OpenCV's
    # 8-connected tracing (−0.25 µm² here), hence the 0.2 % tolerance
    assert np.isclose(geom.area, (40 * 40 - 10 * 10) * 0.25, rtol=2e-3)
    assert np.allclose(geom.bounds, (105, 205, 125, 225))


def test_blockwise_equals_single_pass(monkeypatch):
    """A fake 'model' (threshold on intensity) shows blocks merge seamlessly into the single-pass result."""
    rng = np.random.default_rng(0)
    data = np.zeros((900, 900), np.float32)
    for _ in range(40):
        cy, cx, r = rng.integers(50, 850), rng.integers(50, 850), rng.integers(10, 40)
        yy, xx = np.ogrid[:900, :900]
        data[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1.0
    img = ImageNode("p", ArraySource(data), (1.0, 1.0), "phase")
    pkg = seg.ModelPackage("fake", {"name": "fake", "modality": "phase", "input_um_per_px": 1.0, "in_channels": 1,
                                    "tile": 128, "overlap": 32, "pad_multiple": 32,
                                    "classes": [{"group": "Gland"}]})
    monkeypatch.setattr(seg, "predict", lambda pkg, image, device="auto", **kw: (image > 0.5).astype(np.float32)
                        if kw.get("stats") is None else (image > 0.5).astype(np.float32))
    one = seg.segment(img, (0, 0, 900, 900), pkg, block_px=4096)
    blocks = seg.segment(img, (0, 0, 900, 900), pkg, block_px=256)        # 4 × 4 blocks
    # Objects cut by block edges are re-merged into the same objects. Outlines differ only by
    # corner chamfers at seams and by where simplification places vertices (measured: 0.19 %)
    assert len(one.geoms) == len(blocks.geoms)
    assert np.isclose(one.area, blocks.area, rtol=1e-3)
    assert one.symmetric_difference(blocks).area < 5e-3 * one.area


def test_background_objects_are_dropped_but_gland_lumens_stay():
    pixels = np.full((100, 200, 3), 240, np.uint8)            # bare glass everywhere…
    pixels[10:90, 10:90] = (150, 80, 160)                      # …except a stained "gland"
    pixels[40:60, 40:60] = 245                                 # with a white lumen inside
    mask = np.zeros((100, 200), bool)
    mask[10:90, 10:90] = True                                  # the gland, lumen included (as the model does)
    mask[20:80, 120:190] = True                                # a false object on bare glass
    out = seg.drop_background_objects(mask, pixels)
    assert out[50, 50] and out[15, 15]                         # gland and its lumen kept
    assert not out[50, 150]                                    # glass object dropped


def test_duration_wording():
    from modalytics.tools.segment import duration_text
    assert duration_text(20) == "under a minute"
    assert duration_text(700) == "about 12 minutes"
    assert duration_text(4300) == "about 1 hour 10 minutes"


def test_estimate_counts_only_tissue(monkeypatch):
    data = np.full((2000, 2000, 3), 240, np.uint8)              # bare glass…
    data[:, :500] = (150, 80, 160)                              # …with a quarter of stained tissue
    img = ImageNode("w", ArraySource(data), (1.0, 1.0), "wsi")
    pkg = seg.ModelPackage("fake", {"name": "f", "modality": "wsi", "input_um_per_px": 1.0, "in_channels": 3,
                                    "tile": 512, "overlap": 256, "pad_multiple": 32, "classes": [{"group": "G"}]})
    monkeypatch.setattr(seg, "tile_ms", lambda pkg, device="auto": 10.0)
    e = seg.estimate(img, (0, 0, 2000, 2000), pkg)
    assert np.isclose(e["tissue_mm2"], 1.0, rtol=0.02)          # 500 × 2000 µm = 1 mm²
    assert e["tiles"] == int(1e6 / 256 ** 2)
    assert np.isclose(e["seconds"], e["tiles"] * 10.0 * seg.END_TO_END / 1e3, rtol=0.05)


def test_probability_map_is_placed_exactly_on_the_output_grid(monkeypatch):
    """With an identity 'model', the map is the image itself, averaged onto the output grid."""
    data = np.zeros((600, 800), np.float32)
    data[100:300, 200:500] = 1.0                                   # a 300 × 200 px block at 0.5 µm/px
    img = ImageNode("p", ArraySource(data), (0.5, 0.5), "phase")
    pkg = seg.ModelPackage("fake", {"name": "f", "modality": "phase", "input_um_per_px": 0.5, "in_channels": 1,
                                    "tile": 128, "overlap": 32, "pad_multiple": 32, "classes": [{"group": "G"}]})
    monkeypatch.setattr(seg, "predict", lambda pkg, image, device="auto", **kw: image.astype(np.float32))
    for block in (4096, 256):                                      # single pass and 4 × 3 blocks
        prob, (x0, y0, ox, oy) = seg.probability_map(img, (0, 0, 400, 300), pkg, 2.0, block_px=block)
        assert prob.shape == (150, 200) and (x0, y0, ox) == (0, 0, 2.0)
        # The block spans local µm x 100–250, y 50–150 → output px x 50–125, y 25–75
        assert np.isclose(prob[30:70, 55:120].mean(), 1.0, atol=0.02) and prob[:20].max() < 0.05
        assert np.isclose(prob.sum() * 4.0, 150 * 100, rtol=0.03)   # area in µm² preserved
