"""A referenced image whose file is missing is kept (placeholder + masks), saved unchanged, and relinked."""
import os
import sqlite3
import json

import numpy as np
import pytest
from shapely.geometry import box

from modalytics.core.model import MaskNode, Project
from modalytics.io import importers
from modalytics.io import project_file as pf
from modalytics.io.convert import write_pyramid
from modalytics.io.sources import MissingSource, is_missing
from modalytics.io.tiff_source import TiffPyramidSource


def _project_with_slide(tmp_path):
    slide = tmp_path / "slides" / "s1.ome.tif"
    slide.parent.mkdir()
    write_pyramid(np.full((3000, 4000, 3), 128, np.uint8), str(slide), 0.5)
    p = Project()
    img = importers.import_tiff_wsi(str(slide))
    p.add_image(img)
    p.add_mask(img, MaskNode(geometry=box(100, 100, 600, 400), color=(1, 0, 0, 1), group="Gland", subgroup=""))
    pf.save_project(p, str(tmp_path / "case"))
    img.source.close()
    return slide, tmp_path / "case.modalytics"


def test_missing_file_keeps_image_and_masks_through_a_save(tmp_path):
    slide, case = _project_with_slide(tmp_path)
    moved = tmp_path / "elsewhere.ome.tif"
    os.replace(slide, moved)                                     # the drive is "disconnected"

    # 1. Opened as a placeholder, with its size and its masks
    data = pf.load_project(str(case))
    img = data.images[0]
    assert is_missing(img) and isinstance(img.source, MissingSource)
    assert img.source.shape == (3000, 4000) and img.pixel_size_um == (0.5, 0.5)
    assert img.masks and img.masks[0].area_um2 == pytest.approx(500 * 300)
    assert any("file not found" in w and "Relink" in w for w in data.warnings)

    # 2. Saved while missing: the reference is written back unchanged
    p = Project()
    for i in data.images:
        p.add_image(i)
    pf.save_project(p, str(case))
    con = sqlite3.connect(str(case))
    kind, spath = con.execute("SELECT source_kind, source_path FROM images").fetchone()
    con.close()
    assert kind == "tiff" and spath == str(slide)

    # 3. The file is back: it opens normally, masks intact
    os.replace(moved, slide)
    data = pf.load_project(str(case))
    assert isinstance(data.images[0].source, TiffPyramidSource) and not data.warnings
    assert data.images[0].masks[0].area_um2 == pytest.approx(500 * 300)
    data.images[0].source.close()


def test_relink_by_set_source(tmp_path):
    slide, case = _project_with_slide(tmp_path)
    moved = tmp_path / "elsewhere.ome.tif"
    os.replace(slide, moved)
    data = pf.load_project(str(case))
    p = Project()
    p.add_image(data.images[0])
    img = p.images[0]
    seen = []
    p.node_changed.connect(lambda n, f: seen.append(f))
    p.set_source(img, TiffPyramidSource(str(moved)), dict(img.metadata, path=str(moved)))
    assert not is_missing(img) and seen == ["source"] and img.masks
    pf.save_project(p, str(case))
    data = pf.load_project(str(case))
    assert isinstance(data.images[0].source, TiffPyramidSource) and data.images[0].source.path == str(moved)
    data.images[0].source.close()
    img.source.close()


def test_projects_saved_before_sizes_were_kept(tmp_path):
    slide, case = _project_with_slide(tmp_path)
    con = sqlite3.connect(str(case))                             # an older file: no source_shape
    meta = json.loads(con.execute("SELECT metadata FROM images").fetchone()[0])
    meta.pop("source_shape")
    con.execute("UPDATE images SET metadata = ?", (json.dumps(meta),))
    con.commit()
    con.close()
    os.remove(slide)
    img = pf.load_project(str(case)).images[0]
    assert is_missing(img) and img.source.shape == (1000, 1201)  # the masks' extent (600 µm / 0.5), at least 1000


def test_placeholder_pixels_and_refusals():
    src = MissingSource("tiff", "C:/gone.svs", None, (60373, 147407))
    assert src.levels[-1].shape[1] <= 1024 and len(src.levels) > 3      # a pyramid: quick to draw
    tile = src.read_tile(0, 0, 0)
    assert tile.shape == (1024, 1024, 3) and tile.dtype == np.uint8 and len(np.unique(tile)) == 2
    from modalytics.core.model import ImageNode
    from modalytics.parameters.context import read_native
    from modalytics.services.segmentation import read_region
    img = ImageNode("gone", src, (0.25, 0.25), "wsi")
    with pytest.raises(ValueError, match="not found"):
        read_region(img, (0, 0, 100, 100), 1.0)
    with pytest.raises(ValueError, match="not found"):
        read_native(img, 0, 10, 0, 10)
