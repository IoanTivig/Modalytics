"""Slides read in place (SVS, pyramidal TIFF), the pyramidal OME-TIFF conversion, TIFF calibration rules,
the brightfield importer, and slide files in project files (with the fingerprint check)."""
import os
import shutil

import numpy as np
import pytest
import tifffile

import testdata
from modalytics.image_import import import_file
from modalytics.image_import.registry import ImporterLibrary
from modalytics.image_import.run import ImportFailed
from modalytics.io.convert import converted_path, write_pyramid
from modalytics.io.tiff_source import TiffPyramidSource, image_size, is_slide, tiff_pixel_size

SVS = testdata.SLIDE_SVS
LICOR = testdata.LICOR_TIF
QPI_TIF = os.path.join(testdata.QPI, "tif", "PHA_ep_capture_0000.tif")
_need = testdata.need


# ---------------------------------------------------------------- HoloPath's two slide types, in place
def test_aperio_svs_in_place_matches_holopaths_conversion():
    src = TiffPyramidSource(_need(SVS))
    assert src.pixel_size_um == 0.2512 and src.channels == 3 and src.dtype == np.uint8
    assert [lv.shape for lv in src.levels] == [(60373, 147407), (15093, 36851), (3773, 9212), (943, 2303)]
    assert src.levels[0].tile == (960, 960)                         # 4 × the file's 240 px tiles
    assert src.metadata["format"] == "Aperio SVS" and src.metadata["magnification"] == "40"
    zarr_path = testdata.SLIDE_ZARR
    if os.path.isdir(zarr_path):                                     # the same pixels as HoloPath's zarr copy
        from modalytics.io.zarr_source import ZarrPyramidSource
        z = ZarrPyramidSource(zarr_path)
        tile = src.read_tile(0, 30, 70)
        y, x = 30 * 960, 70 * 960
        assert np.array_equal(tile, np.asarray(z._arrays[0][y:y + 960, x:x + 960]))
    assert is_slide(SVS)


@pytest.mark.parametrize("compression, colour", [("jpeg", True), ("zlib", True), ("zlib", False)])
def test_direct_tile_reads_equal_zarr(tmp_path, compression, colour):
    """Tiled pages are read directly (stored tiles decoded in the calling thread): the same pixels as
    tifffile's zarr view, at every level, edge tiles included."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (700, 900, 3) if colour else (700, 900), dtype=np.uint8)
    path = tmp_path / f"pyr_{compression}.tif"
    with tifffile.TiffWriter(path) as tif:
        tif.write(base, tile=(64, 64), compression=compression, subifds=1,
                  photometric="rgb" if colour else "minisblack")
        tif.write(base[::4, ::4], tile=(64, 64), compression=compression, subfiletype=1,
                  photometric="rgb" if colour else "minisblack")
    src = TiffPyramidSource(str(path))
    assert len(src.levels) == 2 and all(p is not None for p in src._direct)
    for level, lv in enumerate(src.levels):
        gy, gx = lv.grid
        for ty in range(gy):
            for tx in range(gx):
                th, tw = lv.tile
                y, x = ty * th, tx * tw
                want = np.asarray(src._arrays[level][y:min(y + th, lv.shape[0]), x:min(x + tw, lv.shape[1])])
                got = src.read_tile(level, ty, tx)
                assert got.shape == want.shape and got.dtype == want.dtype and np.array_equal(got, want)
    src.close()


def test_licor_tif_in_place():
    src = TiffPyramidSource(_need(LICOR))
    assert np.isclose(src.pixel_size_um, 5.04) and len(src.levels) == 8 and src.levels[0].tile == (1024, 1024)
    assert src.metadata["model"].startswith("Odyssey M")
    assert max(src.thumbnail().shape[:2]) >= 1000                    # the minimap is not 116 px wide
    assert is_slide(LICOR) and image_size(LICOR) == (4961, 14881)


def test_phasestudio_tif_is_refused_with_a_reason():
    with pytest.raises(ImportFailed, match="display picture without the scale"):
        import_file(_need(QPI_TIF))
    assert not is_slide(QPI_TIF)


# ---------------------------------------------------------------- conversion
def test_pyramidal_ome_tiff_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (3000, 4100, 3), dtype=np.uint8)
    out = write_pyramid(img, str(tmp_path / "big.pyramid.ome.tif"), 0.25)
    src = TiffPyramidSource(out)
    assert src.pixel_size_um == 0.25 and len(src.levels) >= 2 and is_slide(out)
    gy, gx = src.levels[0].grid
    back = np.concatenate([np.concatenate([src.read_tile(0, ty, tx) for tx in range(gx)], 1) for ty in range(gy)], 0)
    assert np.array_equal(back, img)                                 # lossless
    with tifffile.TiffFile(out) as t:
        assert t.is_ome and len(t.series[0].levels) == len(src.levels)
    assert converted_path("C:/a/mosaic.tif").endswith("mosaic.pyramid.ome.tif")
    assert not os.path.exists(out + ".tmp")


# ---------------------------------------------------------------- TIFF calibration rules
@pytest.mark.parametrize("kwargs, expected", [
    (dict(resolution=(1e4 / 0.65, 1e4 / 0.65), resolutionunit="CENTIMETER"), 0.65),     # microscope software
    (dict(resolution=(300, 300), resolutionunit="INCH"), None),                         # a print DPI
    (dict(resolution=(25400 / 0.5, 25400 / 0.5), resolutionunit="INCH"), 0.5),
    (dict(), None),
])
def test_tiff_pixel_size_rules(tmp_path, kwargs, expected):
    f = tmp_path / "a.tif"
    tifffile.imwrite(f, np.zeros((8, 8), np.uint8), **kwargs)
    with tifffile.TiffFile(f) as t:
        got = tiff_pixel_size(t)
    assert (got is None) if expected is None else np.isclose(got, expected)


def test_tiff_pixel_size_imagej_and_ome(tmp_path):
    f = tmp_path / "ij.tif"
    tifffile.imwrite(f, np.zeros((8, 8), np.uint8), imagej=True, resolution=(1 / 0.2, 1 / 0.2),
                     metadata={"unit": "micron"})
    with tifffile.TiffFile(f) as t:
        assert np.isclose(tiff_pixel_size(t), 0.2)
    f = tmp_path / "o.ome.tif"
    tifffile.imwrite(f, np.zeros((8, 8), np.uint8), metadata={"PhysicalSizeX": 0.33, "PhysicalSizeXUnit": "µm"})
    with tifffile.TiffFile(f) as t:
        assert np.isclose(tiff_pixel_size(t), 0.33)


# ---------------------------------------------------------------- brightfield importer
def test_brightfield_formats(tmp_path):
    import cv2
    lib = ImporterLibrary()
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0] = 200                                                # red
    cv2.imwrite(str(tmp_path / "a.png"), rgb[..., ::-1])            # OpenCV writes BGR
    cv2.imwrite(str(tmp_path / "a.bmp"), rgb[..., ::-1])
    cv2.imwrite(str(tmp_path / "a.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
    rgba = np.dstack([rgb, np.full((20, 30), 255, np.uint8)])
    cv2.imwrite(str(tmp_path / "rgba.png"), rgba[..., [2, 1, 0, 3]])
    cv2.imwrite(str(tmp_path / "g16.png"), np.full((20, 30), 40000, np.uint16))
    tifffile.imwrite(tmp_path / "cal.tif", rgb, resolution=(1e4 / 0.65, 1e4 / 0.65), resolutionunit="CENTIMETER")
    tifffile.imwrite(tmp_path / "stack.tif", np.zeros((5, 20, 30), np.uint16))
    for name, shape, red in (("a.png", (20, 30, 3), 200), ("a.bmp", (20, 30, 3), 200), ("a.jpg", (20, 30, 3), None),
                             ("rgba.png", (20, 30, 3), 200), ("g16.png", (20, 30), None),
                             ("cal.tif", (20, 30, 3), 200), ("stack.tif", (20, 30), None)):
        path = str(tmp_path / name)
        assert lib.candidates(path)[0][0].uid == "brightfield:image_file", name
        node = import_file(path, lib, pixel_size_um=None if name == "cal.tif" else 1.0)[0]
        px = node.source.read_tile(0, 0, 0)
        assert node.modality == "brightfield" and px.shape == shape, name
        if red is not None:
            assert tuple(px[0, 0]) == (red, 0, 0), name                # RGB order
    cal = import_file(str(tmp_path / "cal.tif"), lib)[0]
    assert np.isclose(cal.pixel_size_um[0], 0.65) and cal.metadata["pixel_size_source"] == "file"
    assert import_file(str(tmp_path / "stack.tif"), lib, pixel_size_um=1.0)[0].metadata["planes_in_file"] == 5
    with pytest.raises(ImportFailed, match="does not give the pixel size"):
        import_file(str(tmp_path / "a.png"), lib)                    # PNG: asked, never guessed


def test_unicode_path(tmp_path):
    import cv2
    d = tmp_path / "Măsurători µm"
    d.mkdir()
    ok, buf = cv2.imencode(".png", np.full((4, 5, 3), 9, np.uint8))
    (d / "imagine.png").write_bytes(buf.tobytes())
    assert import_file(str(d / "imagine.png"), pixel_size_um=1.0)[0].source.shape == (4, 5)


# ---------------------------------------------------------------- slide files in projects
def test_slide_file_in_a_project_and_the_fingerprint(tmp_path):
    from modalytics.core.model import Project
    from modalytics.io import importers
    from modalytics.io import project_file as pf
    slide = tmp_path / "slide.ome.tif"
    write_pyramid(np.full((2500, 2600, 3), 128, np.uint8), str(slide), 0.5)
    p = Project()
    p.add_image(importers.import_tiff_wsi(str(slide)))
    pf.save_project(p, str(tmp_path / "case"))
    data = pf.load_project(str(tmp_path / "case.modalytics"))
    assert not data.warnings and isinstance(data.images[0].source, TiffPyramidSource)
    assert data.images[0].pixel_size_um == (0.5, 0.5) and data.images[0].modality == "wsi"
    # Another file now at that path: opened, with a warning
    data.images[0].source.close()
    p.images[0].source.close()                                       # Windows: open files cannot be replaced
    shutil.copy(slide, tmp_path / "keep.tif")
    write_pyramid(np.full((2500, 2600, 3), 30, np.uint8), str(slide), 0.5)
    data = pf.load_project(str(tmp_path / "case.modalytics"))
    assert any("not the file this project was made with" in w for w in data.warnings)
