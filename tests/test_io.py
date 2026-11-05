"""Readers on the HoloPath test data (skipped when the data is absent)."""
import numpy as np

from modalytics.io import importers
from modalytics.io.dhm_legacy import read_phase_txt
from modalytics.io.zarr_source import ZarrPyramidSource


def test_phase_txt(phase_txt):
    pm = read_phase_txt(phase_txt)
    assert pm.values.shape == (750, 750) and pm.values.dtype == np.float32 and pm.unit == "rad"
    assert np.isclose(pm.pixel_size_um, 0.314006)
    assert np.isclose(pm.height_factor_nm, 211.517)
    node = importers.import_phase_txt(phase_txt)                   # through the importer + contract
    assert node.modality == "phase" and node.metadata["unit"] == "rad" and node.metadata["unit_read"] == "rad"
    assert np.isclose(node.metadata["phase_to_height_nm"], 211.517)
    assert np.array_equal(node.source.read_tile(0, 0, 0), pm.values)


def test_zarr_levels_and_calibration(wsi_zarr):
    src = ZarrPyramidSource(wsi_zarr)
    assert src.shape == (60373, 147407) and src.channels == 3
    assert len(src.levels) == 4 and np.isclose(src.pixel_size_um, 0.2512)
    assert np.isclose(src.levels[1].downsample[0], 4.0, atol=0.01)


def test_zarr_fast_path_matches_zarr(wsi_zarr):
    import zarr
    src = ZarrPyramidSource(wsi_zarr)
    arr = zarr.open_group(wsi_zarr, mode="r")["1"]
    gy, gx = src.levels[1].grid
    for ty, tx in [(3, 10), (gy - 1, gx - 1)]:   # an inner tile and the cropped corner tile
        fast = src.read_tile(1, ty, tx)
        ref = np.asarray(arr[ty * 1024:(ty + 1) * 1024, tx * 1024:(tx + 1) * 1024])
        assert fast.shape == ref.shape and np.array_equal(fast, ref)


def test_hpro_import_builds_hierarchy(hpro):
    images = importers.import_hpro(hpro)
    phase = [i for i in images if i.modality == "phase"]
    assert phase and phase[0].masks
    w, h = phase[0].size_um
    pts = np.vstack(phase[0].masks[0].polygons)
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= w and pts[:, 1].max() <= h


def test_hpro_pixel_masks_become_vector_objects(tmp_path):
    """HoloPath "Manual Mask" labels layers (painted pixel maps): each 8-connected patch is one object, in the
    mask's class, placed on its image (a mask stored at half resolution too)."""
    import json
    import zipfile
    import numpy as np
    from modalytics.io import importers
    img = np.zeros((40, 60), np.float32)
    full = np.zeros((40, 60), np.uint8)
    full[5:15, 5:15] = 1                                      # one object
    full[20:30, 20:30] = 2                                    # another (painted with value 2)
    full[30:35, 30:35] = 2                                    # touching it at a corner: the same object
    half = np.zeros((20, 30), np.uint8)
    half[2:8, 20:26] = 1                                      # stored downsampled ×2
    layers = [{"id": "I", "name": "map", "type": "image", "visible": True, "opacity": 1.0,
               "metadata": {"id": "M", "image_type": "Phase"}, "data_file": "img.npy",
               "transform": {"scale": [0.5, 0.5], "translate": [100.0, 200.0]}},
              {"id": "L1", "name": "map - Manual Mask", "type": "labels", "data_file": "l1.npy",
               "metadata": {"owner": "M", "group": "Glande", "subgroup": "Displasia High Grade"},
               "transform": {"scale": [0.5, 0.5], "translate": [100.0, 200.0]}},
              {"id": "L2", "name": "map - Manual Mask 2", "type": "labels", "data_file": "l2.npy",
               "metadata": {"owner": "M", "group": "Glande", "subgroup": "Displasia Low Grade"},
               "transform": {"scale": [1.0, 1.0], "translate": [100.0, 200.0]}}]
    path = tmp_path / "p.hpro"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("project.json", json.dumps({"version": "1.0", "layers": layers}))
        for name, arr in (("img.npy", img), ("l1.npy", full), ("l2.npy", half)):
            with z.open("layers/" + name, "w") as f:
                np.save(f, arr)
    image = importers.import_hpro(str(path))[0]
    assert (image.placement.x_um, image.placement.y_um) == (200.0, 100.0)
    high = next(m for m in image.masks if m.subgroup == "Dysplasia high grade")
    low = next(m for m in image.masks if m.subgroup == "Dysplasia low grade")
    assert high.group == "Gland" and len(high.geometry.geoms) == 2                 # 8-connected: two objects
    assert abs(high.geometry.area - (100 + 100 + 25) * 0.25) < 1.5                # pixel areas (0.5 µm px)
    assert len(low.geometry.geoms) == 1 and abs(low.geometry.area - 36) < 0.5      # 6 × 6 px of 1 µm
    x0, y0, x1, y1 = low.geometry.bounds
    assert abs(x0 - 20) < 0.01 and abs(y0 - 2) < 0.01                              # its own grid, on the image
