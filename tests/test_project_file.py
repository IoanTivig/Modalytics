"""Project file: round trip, atomic save, change log, missing images."""
import os
import sqlite3

import numpy as np

from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io import project_file as pf
from modalytics.io.sources import ArraySource


def build():
    p = Project()
    data = np.random.default_rng(0).normal(size=(30, 40)).astype(np.float32)
    img = p.add_image(ImageNode("phase", ArraySource(data), (0.314, 0.314), "phase", Placement(10, -5, 12, 1.05, True)))
    holed = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)]
    m = p.add_mask(img, MaskNode(geometry=holed, color=(1, 0, 0, 1), group="Gland", subgroup="Normal"))
    p.set_mask_geometry(m, m.geometry.difference(m.geometry.buffer(-3)))      # ring with a hole
    p.labels.ensure("Crypt", "Branched", (0, 1, 1, 1))
    return p, img, m


def test_round_trip(tmp_path):
    p, img, m = build()
    path = str(tmp_path / "case")
    pf.save_project(p, path, view={"cx": 1.0, "cy": 2.0, "um_per_px": 0.5})
    assert p.path.endswith(".modalytics") and os.path.exists(p.path)
    data = pf.load_project(p.path)
    images, labels, view, warnings = data.images, data.labels, data.view, data.warnings
    assert not warnings and view["um_per_px"] == 0.5
    (img2,) = images
    assert img2.id == img.id and img2.placement == img.placement and img2.pixel_size_um == img.pixel_size_um
    assert np.array_equal(img2.source.data, img.source.data)
    (m2,) = img2.masks
    assert m2.geometry.equals(m.geometry) and len(list(m2.geometry.geoms[0].interiors)) == 1
    assert (m2.group, m2.subgroup) == ("Gland", "Normal")
    assert ("Crypt", "Branched") in labels.choices()


def test_save_is_atomic_and_keeps_the_change_log(tmp_path):
    p, _img, _m = build()
    path = str(tmp_path / "case.modalytics")
    pf.save_project(p, path)
    pf.save_project(p, path)
    assert not os.path.exists(path + ".tmp")
    log = pf.read_changelog(path)
    assert len(log) == 2 and "1 images, 1 masks" in log[-1][2]
    # The file is plain SQLite: readable by any tool
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT value FROM meta WHERE key='format'").fetchone()[0] == "modalytics-project"


def test_missing_zarr_is_reported_and_kept_as_a_placeholder(tmp_path):
    path = str(tmp_path / "case.modalytics")
    p, _img, _m = build()
    pf.save_project(p, path)
    with sqlite3.connect(path) as con:          # simulate a reference to a slide that moved
        con.execute("INSERT INTO images (id, ord, name, modality, source_kind, source_path, source_rel, "
                    "pixel_size_x, pixel_size_y, placement, display, visible, opacity, metadata) "
                    "VALUES ('x', 1, 'slide', 'wsi', 'zarr', 'Z:/gone.zarr', '../gone.zarr', 0.25, 0.25, "
                    "'{}', '{\"clim\": [0, 255], \"colormap\": \"gray\"}', 1, 1.0, '{}')")
    data = pf.load_project(path)
    images, warnings = data.images, data.warnings
    assert len(images) == 2 and warnings and "slide" in warnings[0] and "placeholder" in warnings[0]
    from modalytics.io.sources import is_missing
    slide = images[1]
    assert is_missing(slide) and slide.source.kind == "zarr" and slide.source.path == "Z:/gone.zarr"
    assert data.profile_id == p.profile_id                      # the project's own settings are not disturbed
