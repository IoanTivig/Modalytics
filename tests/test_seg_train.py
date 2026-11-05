"""Training segmentation models (full edition): data, a tiny training run, the package, fine-tuning it again."""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from modalytics.training import seg_train as st                       # noqa: E402


def test_prepare_targets_and_patches(tmp_path):
    from synthetic_projects import PX, graded_project
    path = graded_project(str(tmp_path / "p"), seed=1, n_low=3, n_high=2)
    samples, skipped = st.prepare([path], "Gland", "phase", PX, 1)
    assert len(samples) == 1 and not skipped
    s = samples[0]
    assert s.pixels.shape == (1, 400, 400) and abs(float(s.pixels.mean())) < 1e-4    # z-scored over the region
    assert s.target.dtype == bool and 0.05 < s.target.mean() < 0.5 and s.valid.all()
    # only around the hand outlines: pixels outside are not used
    hull, _ = st.prepare([path], "Gland", "phase", PX, 1, area="hull", margin_um=5)
    assert not hull[0].valid.all() and hull[0].target.sum() == s.target.sum()
    # patches: padded to 512, augmented, mostly on objects
    x, y, v = st.batch([st._pad(s, st.PATCH)], 6, np.random.default_rng(0))
    assert x.shape == (6, 1, 512, 512) and y.shape == v.shape == (6, 1, 512, 512)
    assert y.sum() > 0 and v.sum() < v.size                               # the padding is not used
    # nothing of another image type
    none, _ = st.prepare([path], "Gland", "wsi", 0.25, 3)
    assert none == []


def test_train_write_and_fine_tune_again(tmp_path):
    """A tiny run on the CPU: the package runs with ONNX Runtime and can be fine-tuned again."""
    from synthetic_projects import PX, graded_project
    from modalytics.services.segmentation import load_package, segment
    from modalytics.training.dataset import _open
    paths = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(3)]
    seen = []
    settings = st.Settings(start="scratch", modality="phase", um_per_px=PX, epochs=2, steps=2, batch_size=2,
                           device="cpu")
    res = st.train(settings, paths[:2], paths[2:], "Gland", str(tmp_path / "new.mmodel"),
                   note={"training_set": "test"}, progress=seen.append)
    assert [r["epoch"] for r in res.history] == [1, 2] and 1 <= res.best_epoch <= 2
    assert all(np.isfinite(r["loss"]) and 0 <= r["dice"] <= 1 for r in res.history)
    assert seen[-1]["stage"] == "Done" and any("epoch" in d for d in seen)
    # 1. an ordinary package: ONNX for running, the weights for training again
    with zipfile.ZipFile(res.path) as z:
        assert {"model.onnx", st.WEIGHTS, "manifest.json"} <= set(z.namelist())
    pkg = load_package(res.path)
    assert pkg.modality == "phase" and pkg.um_per_px == PX and pkg.output_class == ("Gland", "")
    assert pkg.manifest["source"]["training"]["training_set"] == "test" and st.trainable(pkg)
    image = _open(paths[2])[0]
    w, h = image.size_um
    segment(image, (0, 0, w, h), pkg, device="cpu")                      # runs where the models run
    # 2. ONNX and PyTorch agree
    model = st.build(pkg.manifest["source"]["architecture_id"], 1)
    model.load_state_dict(st.load_weights(pkg))
    model.eval()
    with torch.no_grad():
        out = model(torch.linspace(-2, 2, 64 * 64).reshape(1, 1, 64, 64)).numpy()
    assert abs(out.mean() - pkg.manifest["self_test"]["logits_mean"]) < 1e-4
    # 3. fine-tuned again from it: the parent is recorded
    again = st.train(st.Settings(start="model", epochs=1, steps=1, batch_size=2, device="cpu"), paths[:2], [],
                     "Gland", str(tmp_path / "again.mmodel"), base=pkg)
    child = load_package(again.path)
    assert child.manifest["source"]["parent"] == pkg.id and child.um_per_px == PX and again.best_epoch == 1
    with pytest.raises(ValueError):
        st.train(st.Settings(start="model", device="cpu"), paths, [], "Gland", str(tmp_path / "x.mmodel"), base=None)
    assert os.path.getsize(again.path) > 1e6


def test_devices_end_with_the_cpu():
    found = st.devices()
    assert found[-1].key == "cpu" and len({d.key for d in found}) == len(found)
    assert st.available()[0]


def test_marked_fully_outlined_and_unreviewed_suggestions(tmp_path):
    """"Where marked": the whole image only on images whose annotation is marked complete; unreviewed suggestions
    left out."""
    import shapely
    from synthetic_projects import PX, graded_project
    from modalytics.core.model import Project, Suggestion
    from modalytics.io.project_file import load_project, save_project
    from modalytics.training.segmentation import unreviewed
    path = graded_project(str(tmp_path / "p"), seed=1, n_low=3, n_high=2)
    unmarked, _ = st.prepare([path], "Gland", "phase", PX, 1, area="marked", margin_um=5)
    assert not unmarked[0].valid.all()                                     # not marked: around the outlines
    data = load_project(path)
    project = Project()
    project.reset(data.images)
    project.profile = data.profile
    img = project.images[0]
    project.set_fully_outlined(img, "Gland", True)
    project.set_suggestions(project.suggestions_for(img), [
        Suggestion(shapely.box(10, 10, 30, 30), "Gland"),                          # a new object: left out
        Suggestion(shapely.box(150, 150, 170, 170), "Gland", regrade=True)])       # only a grade: used
    save_project(project, path)
    assert load_project(path).images[0].fully_outlined("Gland")                    # kept in the file
    assert unreviewed([path], "Gland") == {"p": 1} and unreviewed([path], "Lumen") == {}
    marked, skipped = st.prepare([path], "Gland", "phase", PX, 1, area="marked", margin_um=5)
    v = marked[0].valid
    assert v.mean() > 0.97 and not v[int(20 / PX), int(20 / PX)]                 # the whole image but the suggestion
    assert v[int(160 / PX), int(160 / PX)] and any("1 unreviewed suggestion" in x for x in skipped)


def test_resolution_report_proposes_and_warns(tmp_path):
    """Images of several systems: the most common pixel size is proposed; enlarged images and outliers are said."""
    from synthetic_projects import PX, graded_project
    from modalytics.io.project_file import load_project, save_project
    paths = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(4)]
    for path, px in ((paths[2], 0.25), (paths[3], 5.0)):                    # another system; a wrong pixel size
        project = load_project(path)
        project.images[0].pixel_size_um = (px, px)
        save_project(project, path)
    scales, skipped = st.image_scales(paths, "Gland", "phase")
    assert sorted(sc.um_per_px for sc in scales) == [0.25, PX, PX, 5.0] and not skipped
    assert st.image_scales(paths, "Gland", "wsi")[0] == []                  # only the image type trained
    r = st.resolution_report(scales)
    assert r["proposal"] == PX and r["groups"][0] == (PX, 2, ["p0 › map 0", "p1 › map 1"])
    assert r["tile_um"] == st.PATCH * PX
    assert [(n, f) for n, _u, f in r["enlarged"]] == [("p3 › map 3", 10.0)]   # coarser: blurred when enlarged
    assert [n for n, _u, _r in r["outliers"]] == ["p3 › map 3"]
    finer = st.resolution_report(scales, works_at=0.25)
    assert {n for n, _u, _f in finer["enlarged"]} == {"p0 › map 0", "p1 › map 1", "p3 › map 3"}


def test_scale_jitter_resizes_patches_and_changes_nothing_when_off(tmp_path):
    from synthetic_projects import PX, graded_project
    path = graded_project(str(tmp_path / "p"), seed=1, n_low=3, n_high=2)
    samples, _ = st.prepare([path], "Gland", "phase", PX, 1)
    plain = st.batch([st._pad(samples[0], st.PATCH)], 4, np.random.default_rng(3))
    again = st.batch([st._pad(samples[0], st.PATCH)], 4, np.random.default_rng(3), scale_jitter=0.0)
    assert all(np.array_equal(a, b) for a, b in zip(plain, again))          # off: exactly as before
    padded = st._pad(samples[0], int(np.ceil(st.PATCH * 1.3)))
    x, y, v = st.batch([padded], 8, np.random.default_rng(3), scale_jitter=0.3)
    assert x.shape == (8, 1, 512, 512) and y.shape == v.shape == (8, 1, 512, 512)
    assert set(np.unique(y)) <= {0.0, 1.0} and y.sum() > 0
    assert not np.array_equal(x[:4], plain[0])                              # resized: other pixels


def test_only_a_power_of_ten_is_a_pixel_size_slip():
    """Other magnifications are normal; 10×, 100×… from a more common size looks like a decimal or unit slip."""
    from modalytics.training.dataset import unit_slips
    assert unit_slips({0.314: 120, 1.256: 18}) == {}                        # 20× and 5× (4×): normal
    assert unit_slips({0.314: 120, 1.256: 18, 3.14: 2}) == {3.14: 0.314}   # 3.14 for 0.314: asked about
    assert unit_slips({0.5: 10, 0.0005: 1}) == {0.0005: 0.5}               # nm read as µm (1000×)
    assert unit_slips({0.5: 10, 4.0: 3, 0.25: 2}) == {}                    # 8× and 2×: other systems
    assert unit_slips({0.5: 1, 5.0: 1}) == {}                              # as many of each: which is wrong is unknown
    scales = [st.ImageScale("p", f"map {i}", 0.314) for i in range(6)] + [st.ImageScale("q", "map", 1.256)]
    assert st.resolution_report(scales)["outliers"] == []


def test_augmentations_can_each_be_switched_off():
    from modalytics.training import seg_train as st_
    pixels = np.arange(32 * 32, dtype=np.float32).reshape(1, 32, 32)       # rows and columns both increasing
    sample = st_.Sample(pixels, np.zeros((32, 32), bool), np.ones((32, 32), bool), "s")
    plain, _y, _v = st_.batch([sample], 20, np.random.default_rng(0), size=8, flips=False, turns=False,
                              intensity=0.0)
    assert all((np.diff(x[0], axis=1) == 1).all() and (np.diff(x[0], axis=0) == 32).all() for x in plain)
    changed, _y, _v = st_.batch([sample], 20, np.random.default_rng(0), size=8)   # as before: all of them
    assert not all((np.diff(x[0], axis=1) == 1).all() for x in changed)
    assert all(st_.Settings().__getattribute__(k) == v for k, v in (("flips", True), ("turns", True),
                                                                     ("intensity", 0.15)))
