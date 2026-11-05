"""Training runs on disk: the folder layout (set / kind / date name), names Windows accepts, the list newest first,
the results read back (curves, counts, the models' paths relative to the run), a moved run still found, and a
renamed set taking its folder along."""
import datetime as dt
import json
import shutil

import pytest

from modalytics.training import compare as cmp
from modalytics.training import dataset as ds
from modalytics.training import runs as rn
from modalytics.training import segmentation as seg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MODALYTICS_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _tset(name="Colonic Glands Simplified"):
    from modalytics.profiles.model import Profile
    return ds.TrainingSet(name, Profile("p"))


def _counts(tp, fp, fn):
    return seg.Counts(hand=tp + fn, found=tp + fp, tp=tp, fp=fp, fn=fn, inter=10.0 * tp, hand_area=12.0 * (tp + fn),
                      found_area=11.0 * (tp + fp))


def _comparison(run):
    per = [(i % 2, f"img {i}", _counts(5 - i % 2, 1, 1)) for i in range(4)]
    ev = seg.Evaluation(["case_a", "case_b"], [], [], cmp.pooled([c for _p, _n, c in per]), [], ["x: skipped"], {},
                        per)
    history = [{"epoch": e, "loss": 1 / e, "dice": 0.5 + e / 10, "seconds": 10.0 * e, "best": min(e, 2)}
               for e in (1, 2, 3)]
    new = cmp.Entry("unet_resnet34", "U-Net (ResNet-34)", str(run.models / "unet_resnet34__imagenet.mmodel"),
                    history, 2, 30.0, 24.4, 98.0, 6.1, "GPU", evaluation=ev)
    ref = cmp.Entry("unet_resnet18", "Old — as it was", r"C:\elsewhere\old.mmodel", kind="reference", base="old",
                    evaluation=ev)
    return cmp.Comparison("Gland on phase map", "Gland", "phase", 0.314, {"epochs": 3}, {"training": ["t"],
                          "evaluation": ["case_a", "case_b"], "test": []}, "marked", 50.0, 0.5, "f1", [new, ref],
                          training_set="Colonic Glands Simplified")


def test_safe_names_have_no_forbidden_characters_and_are_short():
    assert rn.safe_name('a<b>c:d"e/f\\g|h?i*j') == "a b c d e f g h i j"
    assert rn.safe_name("x" * 80) == "x" * rn.NAME_CHARS
    assert rn.safe_name("  name. ") == "name" and rn.safe_name("::") == "Untitled"


def test_a_run_has_its_folder_report_results_and_models_under_its_set(home):
    tset = _tset()
    folder = rn.set_folder(tset, create=True)
    assert tset.runs_folder == "Colonic Glands Simplified" and folder.is_dir()
    run = rn.Run.new(folder, "segmentation", "Gland on phase map", dt.datetime(2026, 10, 25, 1, 14, 33))
    assert run.folder == folder / "segmentation" / "2026-10-25 01-14 Gland on phase map"
    assert run.report.name == "Report - Gland on phase map - 2026-10-25 01-14.html"
    assert run.results.name == "Results - Gland on phase map - 2026-10-25 01-14.json"
    assert run.models == run.folder / "models" and run.label == "2026-10-25 01:14 — Gland on phase map"
    other = ds.TrainingSet("Colonic Glands Simplified", tset.profile)       # another set of the same name
    assert rn.set_folder(other, create=True).name == "Colonic Glands Simplified (2)"


def test_results_are_read_back_with_paths_relative_to_the_run(home):
    tset = _tset()
    run = rn.Run.new(rn.set_folder(tset, create=True), "segmentation", "Gland on phase map")
    run.models.mkdir(parents=True)
    comp = _comparison(run)
    comp.save(run.results)
    stored = json.loads(run.results.read_text(encoding="utf-8"))
    assert stored["entries"][0]["path"] == "models/unet_resnet34__imagenet.mmodel"     # inside the run: relative
    assert stored["entries"][1]["path"] == r"C:\elsewhere\old.mmodel"                   # an installed model: as is
    back = rn.load(run)
    e = back.entries[0]
    assert e.path == str(run.models / "unet_resnet34__imagenet.mmodel") and e.history == comp.entries[0].history
    assert e.evaluation.total == comp.entries[0].evaluation.total and len(e.evaluation.per_image) == 4
    assert e.evaluation.per_image[1][:2] == (1, "img 1") and back.entries[1].kind == "reference"
    assert back.best().run_id == comp.best().run_id and back.training_set == "Colonic Glands Simplified"
    lo, hi = cmp.bootstrap([c for _p, _n, c in e.evaluation.per_image], cmp.METRICS["f1"][1], n=200)
    assert lo <= cmp.METRICS["f1"][1](e.evaluation.total) <= hi                         # intervals work again
    moved = run.folder.parent / "2026-10-25 02-00 moved"
    shutil.move(str(run.folder), str(moved))                                            # moved whole: still found
    assert rn.load(rn.Run.of(moved)).entries[0].path == str(moved / "models" / "unet_resnet34__imagenet.mmodel")


def test_runs_are_listed_newest_first_and_only_with_results(home):
    tset = _tset()
    folder = rn.set_folder(tset, create=True)
    for when, name in ((dt.datetime(2026, 10, 25, 1, 14), "first"), (dt.datetime(2026, 10, 26, 9, 0), "second"),
                       (dt.datetime(2026, 10, 25, 18, 30), "third")):
        run = rn.Run.new(folder, "segmentation", name, when)
        run.models.mkdir(parents=True)
        _comparison(run).save(run.results)
    rn.Run.new(folder, "segmentation", "unfinished", dt.datetime(2026, 10, 27)).models.mkdir(parents=True)
    assert [r.name for r in rn.list_runs(tset)] == ["second", "third", "first"]
    assert rn.list_runs(tset, "classification") == [] and rn.list_runs(_tset("Never trained")) == []


def test_old_runs_are_read_and_a_renamed_set_takes_its_folder(home):
    tset = _tset()
    old = rn.set_folder(tset, create=True) / "segmentation" / "gland_on_phase_map_20261025_0114"
    old.mkdir(parents=True)
    _comparison(rn.Run.of(old)).save(old / rn.LEGACY_RESULTS)
    (old / rn.LEGACY_REPORT).write_text("<html></html>", encoding="utf-8")
    run = rn.list_runs(tset)[0]
    assert run.results.name == "comparison.json" and run.report.name == "report.html" and run.when is None
    rn.rename_set(tset, "Colon: all grades")
    assert tset.runs_folder == "Colon all grades" and (rn.root() / "Colon all grades").is_dir()
    assert not (rn.root() / "Colonic Glands Simplified").exists() and len(rn.list_runs(tset)) == 1
    back = ds.TrainingSet.from_dict(tset.to_dict())
    assert back.runs_folder == "Colon all grades"                                      # kept with the set


def test_the_report_says_what_each_role_holds_and_how_the_models_were_trained(home):
    from modalytics.report import comparison as rep
    run = rn.Run.new(rn.set_folder(_tset(), create=True), "segmentation", "Counts")
    run.models.mkdir(parents=True)
    comp = _comparison(run)
    comp.settings = {"epochs": 30, "steps": 50, "batch_size": 8, "seed": 1, "scale_jitter": 0.1}
    comp.data = {"training": {"projects": 14, "images": 120, "complete": 3, "objects": 218},
                 "evaluation": {"projects": 2, "images": 15, "complete": 0, "objects": 38}, "test": {}}
    comp.save(run.results)
    assert rn.load(run).data == comp.data                                               # kept with the run
    text = rep.build(comp)
    block = text[text.index("Data and training"):text.index("Results on the")]
    for part in ("Images with Gland outlined", ">120<", ">135<", ">256<", "Gland objects drawn", ">85 %<", ">15 %<",
                 "12,000 per model", "Epochs</dt><dd>30", "Augmentation</dt><dd>mirrored, turned by quarter turns, brightness and contrast varied by up to ±15 %, resized by up to ±10 %", "chooses each model&#x27;s epoch"):
        assert part in block, part
    assert text.index("Data and training") > text.index("No test projects")              # under the warning
    comp.data = {}
    assert "not recorded by the version that made this run" in rep.build(comp)


def test_existing_models_judged_saved_reopened_and_reported(monkeypatch, tmp_path):
    """Evaluate: existing models judged without training; the judgement (items, project paths) kept whole."""
    import os
    from synthetic_projects import graded_project
    from modalytics.parameters.engine import objects_of
    from modalytics.report import comparison as rep
    from modalytics.services.segmentation import load_package
    from modalytics.training import dataset as ds
    from modalytics.training import segmentation as se
    model = os.path.join(os.path.dirname(__file__), "..", "models", "unet_efficientnet_b0__imagenet_micronet.mmodel")
    paths = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(2)]

    def fake_found(image, rect, pkg, path, cache_dir, progress, cancelled, device):
        hand = [p for m in image.masks if m.group == "Gland" for p in objects_of(m.geometry)]
        return hand[1:]                                                     # one missed per image
    monkeypatch.setattr(se, "_found", fake_found)
    monkeypatch.setattr(ds, "library_folder", lambda: tmp_path / "lib")
    monkeypatch.setattr(cmp, "speed", lambda pkg, device="auto": (5.0, "test device"))
    results = tmp_path / "Results - t.json"
    comp = cmp.evaluate_models([model, str(tmp_path / "gone.mmodel")], paths, "all", "Gland", "t", results=results,
                               training_set="Set")
    e, gone = comp.entries
    assert comp.purpose == "evaluation" and comp.judged_on == "all" and comp.done() == [e]
    assert e.kind == "existing" and e.evaluation.total.fn == 2 and e.ms_per_tile == 5.0 and e.test is None
    assert "could not be read" in gone.error and comp.data["all"]["images"] == 2
    assert e.origin["trained_on"] is None and e.origin["architecture"]                 # an older file: not said
    again = cmp.Comparison.load(results)                                                 # reopened from the list
    ev = again.entries[0].judged("all")
    assert [i.kind for i in ev.items] == ["missed", "missed"] and ev.paths == paths
    assert [c.fn for c in ev.per_project] == [1, 1] and again.purpose == "evaluation"
    text = rep.build(again)
    assert "model evaluation" in text and "Data and models" in text and "Judged on all the projects" in text
    assert "Training loss per epoch" not in text and "Epoch kept" not in text and "not said by its file" in text
    # A model whose file names its training projects: judged on one of them, it is said
    pkg = load_package(model)
    pkg.manifest = dict(pkg.manifest, source=dict(pkg.manifest["source"], training={"projects": ["p0"]}))
    assert cmp.origin(pkg, ["p0", "p1"])["learned_from_judged"] == ["p0"]
    e.origin = cmp.origin(pkg, ["p0", "p1"])
    assert "learned from 1 of the judged projects (p0)" in rep.build(comp)
