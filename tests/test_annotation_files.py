"""GeoJSON round trip and Leica/Aperio XML import (µm areas checked against Leica's own numbers)."""
import os

import numpy as np
import pytest

import testdata
from modalytics.core.model import ImageNode, MaskNode, Project
from modalytics.io import annotation_files as af
from modalytics.io.sources import ArraySource

LEICA_XML = testdata.LEICA_XML


def slide(ps=0.2512, shape=(60373, 147407)):
    """A fake slide of the right size (only shape and calibration matter here)."""
    class _Src(ArraySource):
        def __init__(self):
            super().__init__(np.zeros((2, 2, 3), np.uint8))
            from modalytics.io.sources import Level
            self.levels = [Level(shape=shape, tile=(1024, 1024), downsample=(1.0, 1.0))]
    return ImageNode("slide", _Src(), (ps, ps), "wsi")


def test_geojson_round_trip_keeps_classes_holes_and_um(tmp_path):
    p = Project()
    img = p.add_image(slide(ps=0.5))
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    m = p.add_mask(img, MaskNode(geometry=[sq], color=(1, 0, 0, 1), group="Gland", subgroup="Normal"))
    p.set_mask_geometry(m, m.geometry.difference(m.geometry.buffer(-20)))      # ring with a hole
    p.add_mask(img, MaskNode(geometry=[sq + 200], group="Region"))
    path = str(tmp_path / "a.geojson")
    assert af.export_geojson(img, path) == 2
    got = af.import_geojson(path, img)
    assert set(got) == {("Gland", "Normal"), ("Region", "")}
    geom, color = got[("Gland", "Normal")]
    assert geom.equals(m.geometry) and color[:3] == (1.0, 0.0, 0.0)


def test_qupath_style_pixel_coordinates_are_converted_to_um(tmp_path):
    img = slide(ps=0.25)
    path = tmp_path / "q.geojson"
    path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon",'
                    '"coordinates":[[[0,0],[400,0],[400,400],[0,400],[0,0]]]},"properties":{"objectType":'
                    '"annotation","classification":{"name":"Tumor","color":[200,0,0]}}}]}')
    (geom, _c), = af.import_geojson(str(path), img).values()
    assert np.isclose(geom.area, (400 * 0.25) ** 2)


@pytest.mark.skipif(not os.path.exists(LEICA_XML), reason="test data not found")
def test_leica_xml_areas_match_leica_reported_microns():
    img = slide()
    classes, info = af.import_aperio_xml(LEICA_XML, img)
    assert info["regions"] == 4 and not info["mpp_mismatch"]
    # The file's 4 regions are all "NonDisplasicGland"; its "LGG" annotation is empty and is skipped
    assert set(classes) == {("Gland", "Normal")}
    total = sum(g.area for g, _c in classes.values())
    # Leica reports each region's area in µm²; our µm geometry must agree
    assert np.isclose(total, info["reported_area_um2"], rtol=1e-6), (total, info["reported_area_um2"])
