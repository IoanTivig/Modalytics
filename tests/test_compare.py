"""Training runs of several models: the bootstrap intervals and paired differences with known answers, the proposal
(criterion, then speed), the report built from a run (no training needed), the models a run can train, and a
fine-tuned model judged with its start as it was."""
import math

import numpy as np
import pytest
from shapely.geometry import box

from modalytics.report import comparison as rep
from modalytics.training import compare as cmp
from modalytics.training import segmentation as seg


def counts(tp, fp, fn):
    return seg.Counts(hand=tp + fn, found=tp + fp, tp=tp, fp=fp, fn=fn)


def test_bootstrap_interval_contains_the_pooled_value_and_needs_two_images():
    per = [counts(9, 1, 0), counts(8, 0, 2), counts(10, 1, 1), counts(7, 2, 1)]
    value = cmp.METRICS["f1"][1](cmp.pooled(per))
    lo, hi = cmp.bootstrap(per, cmp.METRICS["f1"][1], n=500, seed=0)
    assert lo <= value <= hi and 0.8 < lo < hi <= 1
    assert all(math.isnan(v) for v in cmp.bootstrap(per[:1], cmp.METRICS["f1"][1]))
    same = [counts(5, 0, 0)] * 4                                           # no spread: a point interval
    assert cmp.bootstrap(same, cmp.METRICS["f1"][1], n=200) == (1.0, 1.0)


def test_paired_difference_separates_a_clearly_better_model_and_not_a_twin():
    good = [counts(10, 0, 0), counts(9, 0, 1), counts(10, 1, 0), counts(8, 0, 0), counts(9, 1, 0)]
    poor = [counts(6, 3, 4), counts(5, 4, 5), counts(7, 2, 3), counts(4, 3, 4), counts(6, 4, 3)]
    d, lo, hi = cmp.paired_difference(good, poor, cmp.METRICS["f1"][1], n=500)
    assert d > 0.2 and lo > 0                                             # clearly better
    d, lo, hi = cmp.paired_difference(good, good, cmp.METRICS["f1"][1], n=200)
    assert d == 0 and lo == hi == 0                                       # identical: no difference
    assert all(math.isnan(v) for v in cmp.paired_difference(good, poor[:3], cmp.METRICS["f1"][1]))


def _evaluation(per_image, examples=()):
    total = cmp.pooled([c for _p, _n, c in per_image])
    ev = seg.Evaluation(["case_a", "case_b"], ["a", "b"], [cmp.pooled([c for p, _n, c in per_image if p == i])
                                                           for i in range(2)], total, [], [], {})
    ev.per_image, ev.examples = list(per_image), list(examples)
    return ev


def _comparison(test=True):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    entries = []
    for k, (aid, label, tp, ms) in enumerate((("unet_resnet34", "U-Net (ResNet-34)", 9, 17.0),
                                              ("unet_efficientnet_b0", "U-Net (EfficientNet-B0)", 7, 47.0),
                                              ("unet_convnext_tiny", "U-Net (ConvNeXt-Tiny)", 9, 31.0))):
        per = [(i % 2, f"img {i}", counts(tp - (i % 2), 10 - tp, 1 + (i % 2))) for i in range(6)]
        ex = seg.Example("case_a", "img 0", pixels, (0.0, 0.0, 1.0, 1.0), [box(10, 10, 40, 40)],
                         [box(12, 11, 41, 40)])
        ev = _evaluation(per, [ex])
        entries.append(cmp.Entry(aid, label, f"{aid}.mmodel", [{"epoch": e, "loss": 1 / e, "dice": 0.5 + e / 10,
                                                                "seconds": e * 10.0} for e in range(1, 5)],
                                 3, 40.0, 24.4, 98.0, ms, "GPU", evaluation=ev, test=ev if test else None))
    entries.append(cmp.Entry("deeplabv3p_resnet50", "DeepLabV3+ (ResNet-50)", error="RuntimeError: out of memory"))
    return cmp.Comparison("demo", "Gland", "phase", 0.5, {"epochs": 4, "steps": 8, "batch_size": 4, "seed": 0},
                          {"training": ["t1", "t2"], "evaluation": ["e1"], "test": ["x1", "x2"] if test else []},
                          "marked", 50.0, 0.5, "f1", entries, "2026-10-24T10:00:00", "2026-10-24T11:00:00",
                          {"modalytics": "0.1"}, "RTX A2000")


def test_the_proposal_is_the_best_by_the_criterion_then_the_faster():
    comp = _comparison()
    assert comp.judged_on == "test" and len(comp.done()) == 3             # the failed one is left out
    assert comp.best().arch_id == "unet_resnet34"                          # tied with ConvNeXt on F1: faster
    comp.criterion = "f1"
    comp.entries[0].ms_per_tile = 99.0
    assert comp.best().arch_id == "unet_convnext_tiny"


def test_report_has_the_results_differences_figures_examples_and_methods(tmp_path):
    comp = _comparison()
    text = rep.build(comp)
    for part in ("Results on the test projects", "Proposed: <b>U-Net (ResNet-34)</b>", "Against the proposal",
                 "not distinguishable", "the proposal is clearly better", "<svg", "Training loss per epoch",
                 "data:image/png;base64", "out of memory", "Methods", "doi:10.1016/j.media.2016.08.008", "Object F1 (GlaS)", "of the matched objects only"):
        assert part in text, part
    for doi in ("10.1214/aos/1176344552", "10.2307/1932409", "10.2196/26151"):   # references kept to a minimum
        assert doi not in text, doi
    assert "No test projects" not in text
    path = rep.write(comp, str(tmp_path / "report.html"))
    assert (tmp_path / "report.html").read_text(encoding="utf-8") == text and path.endswith("report.html")
    assert "No test projects" in rep.build(_comparison(test=False))        # judged on evaluation: said plainly
    comp.save(tmp_path / "comparison.json")                                 # plain data (no pictures)
    assert (tmp_path / "comparison.json").stat().st_size < 50_000


def test_plan_reads_rows_and_says_what_cannot_be_trained():
    pytest.importorskip("torch")
    p = cmp.Planned.of({"kind": "finetune", "base": "x"})
    assert p.run_id == "ft:x" and cmp.Planned.of("unet_resnet34").run_id == "unet_resnet34+imagenet"
    assert cmp.Planned.of(("unet_resnet18", "scratch")).weights == "scratch"
    runs, left = cmp.plan([("unet_resnet34", "micronet"), ("unet_convnext_tiny", "micronet"),
                           ("unet_resnet18", "scratch"), {"kind": "finetune", "base": "no_such_model"},
                           {"kind": "finetune", "base": "unet_efficientnet_b0__imagenet_micronet"}])
    assert [r.run_id for r in runs] == ["unet_resnet34+micronet", "unet_resnet18+scratch",
                                        "ft:unet_efficientnet_b0__imagenet_micronet"]
    assert len(left) == 2 and "ConvNeXt" in left[0] and "not installed" in left[1]
    assert [p.run_id for p in cmp.cross(["a", "b"], ["imagenet", "micronet"])] == [
        "a+imagenet", "a+micronet", "b+imagenet", "b+micronet"]


def test_a_fine_tuned_model_is_judged_with_its_start_as_it_was(tmp_path):
    pytest.importorskip("torch")
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from synthetic_projects import graded_project
    from modalytics.training import seg_train as st
    paths = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(2)]
    settings = st.Settings(modality="phase", um_per_px=0.5, epochs=1, steps=1, batch_size=2, device="cpu")
    comp = cmp.run([{"kind": "finetune", "base": "unet_efficientnet_b0__imagenet_micronet"}], settings, paths[:1], paths[1:], [],
                   "Gland", tmp_path / "run", "tiny", device="cpu")
    kinds = [(e.kind, e.run_id) for e in comp.entries]
    assert kinds == [("finetune", "ft:unet_efficientnet_b0__imagenet_micronet"), ("reference", "ref:unet_efficientnet_b0__imagenet_micronet")]
    tuned, ref = comp.entries
    assert not tuned.error and not ref.error and ref.evaluation is not None and tuned.evaluation is not None
    assert round(tuned.um_per_px, 4) == round(ref.um_per_px, 4) == 0.314                    # the base's, kept
    assert ref.history == [] and ref.path.endswith("unet_efficientnet_b0__imagenet_micronet.mmodel")
    page = rep.build(comp)
    assert "as it was" in page and "fine-tuned" in page and "Against the proposal" in page


def test_each_test_group_is_reported_apart(tmp_path):
    comp = _comparison()
    assert "Per test group" not in rep.build(comp)                          # one group (or none named): pooled only
    comp.test_groups = {"Naive": ["case_a"], "Hospital B": ["case_b"]}
    text = rep.build(comp)
    part = text.split("Per test group")[1].split("Per project")[0]
    first = comp.done()[0].test
    on_b = cmp.pooled([c for p, _n, c in first.per_image if p == 1])        # Hospital B's images only
    assert "Naive" in part and "Hospital B" in part and f"{on_b.f1:.3f}" in part
    assert part.count("<h3>") == 2                                          # then each group, every measure
    comp.save(tmp_path / "comparison.json")
    assert cmp.Comparison.load(tmp_path / "comparison.json").test_groups == comp.test_groups
