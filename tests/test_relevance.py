"""Parameter relevance: the AUC across grades with known answers, objects / images / projects compared, the
correction for many parameters, redundancy groups, the proposal, saving and the report."""
import math

import numpy as np
import pytest

from modalytics.training import relevance as rv
from modalytics.training.dataset import Dataset


def test_concordance_known_answers():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    assert rv.concordance(x, g, np.ones(4), 2)[0] == 1.0                      # every high above every low
    assert rv.concordance(-x, g, np.ones(4), 2)[0] == 0.0
    assert rv.concordance(np.ones(4), g, np.ones(4), 2)[0] == 0.5            # all tied: one half
    three = rv.concordance(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]), np.ones(3), 3)
    assert three[0] == 1.0 and three[1] == [1.0, 1.0]                        # the trend and each neighbouring pair
    w = rv.concordance(x, g, np.array([2.0, 1.0, 1.0, 1.0]), 2)              # a weight counts as copies
    assert w[0] == rv.concordance(np.r_[1.0, x], np.r_[0, g], np.ones(5), 2)[0]


def test_benjamini_hochberg():
    q = rv.benjamini_hochberg([0.01, 0.04, 0.03, math.nan, 0.2])
    assert q[0] == pytest.approx(0.04) and q[1] == pytest.approx(0.0533, abs=1e-3) and q[2] == pytest.approx(0.0533,
                                                                                                             abs=1e-3)
    assert math.isnan(q[3]) and q[4] == pytest.approx(0.2)


def _data(seed=1, n_proj=12, per=30):
    rng = np.random.default_rng(seed)
    proj = np.repeat(np.arange(n_proj), per)
    grade = rng.integers(0, 2, len(proj))
    shift = rng.normal(0, 1.0, n_proj)[proj]
    strong = grade * 1.5 + rng.normal(0, 1, len(proj))
    cols = {"strong": strong, "twin": strong * 3 + rng.normal(0, 0.05, len(proj)),
            "batch": shift * 6 + rng.normal(0, 0.3, len(proj)),                       # follows the projects only
            "few": np.where(np.arange(len(proj)) < 12, strong, np.nan)}               # values on 12 objects only
    cols.update({f"noise{i}": rng.normal(0, 1, len(proj)) + shift for i in range(10)})
    names = list(cols)
    return Dataset([(u, u.title(), "µm" if u == "strong" else "", f"help of {u}") for u in names],
                   np.column_stack([cols[u] for u in names]), grade, ["Low", "High"], proj,
                   [f"P{i}" for i in range(n_proj)])


def test_analysis_finds_proposes_and_notes_what_follows_the_projects():
    data = _data()
    a = rv.analyse(data, np.ones(len(data.grade), bool), "training", "Gland", "t", n_boot=300)
    strong, twin, batch, few = (a.row(u) for u in ("strong", "twin", "batch", "few"))
    assert strong.verdict == twin.verdict == "separates" and strong.auc > 0.8 and strong.direction == "higher"
    assert strong.ci[0] < strong.auc < strong.ci[1] and strong.q < 0.05
    assert strong.group == twin.group != 0 and [strong.proposed, twin.proposed].count(True) == 1    # one per group
    assert batch.verdict == "chance" and batch.follows_projects and any("between projects" in n for n in batch.notes())
    assert few.verdict == "too few"
    assert all(r.verdict == "chance" for r in a.rows if r.uid.startswith("noise"))
    assert a.rows[0].proposed and [r.uid for r in a.proposed()] in (["strong"], ["twin"])
    assert a.counts == {"Low": int(np.sum(data.grade == 0)), "High": int(np.sum(data.grade == 1))}
    assert strong.agree[0] >= strong.agree[1] - 1 and strong.agree[1] >= 10
    assert a.unit == "objects" and strong.n == a.counts                         # the default: each object a case


def test_saved_reopened_and_reported(tmp_path):
    from modalytics.report import relevance as rep
    data = _data()
    rows = np.ones(len(data.grade), bool)
    a = rv.analyse(data, rows, "all", "Gland", "On everything", n_boot=200, training_set="Set", profile="Colonic")
    a.save(tmp_path / "r.json")
    again = rv.Analysis.load(tmp_path / "r.json")
    assert [r.uid for r in again.rows] == [r.uid for r in a.rows] and again.groups == a.groups
    assert again.row("strong").ci == pytest.approx(a.row("strong").ci) and again.proposed()[0].uid == a.proposed()[0].uid
    text = rep.build(again, data, rows)
    assert "which parameters separate the classes" in text and "Proposed:" in text and "description of the data" in text
    assert "Redundancy groups" in text and "doi:10.1111/j.2517-6161.1995.tb02031.x" in text
    assert "per class (│ median)" in text and "<svg" in text and "Per class" in text


def test_holm_and_class_stats():
    assert rv.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    st = rv.class_stats(np.arange(1.0, 12.0))
    assert st[0] == 11 and st[1] == 6.0 and st[2] == 3.5 and st[3] == 8.5 and st[4] == 6.0 and st[6] > 0.05


def _categories(design, seed=3):
    rng = np.random.default_rng(seed)
    if design == "mixed":
        proj = np.repeat(np.arange(12), 40)
        grade = rng.integers(0, 4, len(proj))
    else:                                                       # one class per project, 6 projects each
        proj = np.repeat(np.arange(24), 20)
        grade = np.repeat(np.arange(24) % 4, 20)
    real = (grade == 2) * 2.5 + rng.normal(0, 1, len(proj))    # only C differs
    cols = np.column_stack([real] + [rng.normal(0, 1, len(proj)) for _ in range(5)])
    names = ["real"] + [f"n{i}" for i in range(5)]
    return Dataset([(u, u, "", "") for u in names], cols, grade, ["A", "B", "C", "D"], proj,
                   [f"P{i}" for i in range(int(proj.max()) + 1)], ordered=False)


def test_categories_compared_as_objects_images_or_projects():
    for design, unit in (("mixed", "objects"), ("per project", "projects"), ("per project", "images")):
        data = _categories(design)
        a = rv.analyse(data, np.ones(len(data.grade), bool), "training", "Lesion", n_boot=300, unit=unit)
        real = a.row("real")
        assert a.unit == unit and not a.ordered
        assert real.best == "C" and real.direction == "higher" and real.verdict == "separates", (design, real)
        assert len(real.pairs) == 6 and {(x, y) for x, y, _a, _p in real.pairs} >= {("A", "C"), ("C", "D")}
        by = {(x, y): (auc, p) for x, y, auc, p in real.pairs}
        assert by[("A", "C")][0] > 0.8 and by[("A", "C")][1] < 0.05                 # C above A, after Holm
        assert by[("A", "C")][1] < by[("A", "B")][1]                                 # A–B: no difference
        assert set(real.stats) == {"A", "B", "C", "D"} and real.stats["C"][1] > real.stats["A"][1]
        if unit != "objects":                                # no images named: each project's objects one image
            assert real.n == {"A": 6, "B": 6, "C": 6, "D": 6}                       # cases: projects (images)


def test_kappa_weighted_only_for_grades_and_one_against_the_rest():
    from modalytics.training import classifiers as cl
    from modalytics.training import ranges
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 2, 1, 1, 2, 0])                                       # two mistakes two classes apart
    graded = cl.agreement(true, pred, ["N", "L", "H"], ordered=True).kappa
    plain = cl.agreement(true, pred, ["A", "B", "C"], ordered=False).kappa
    assert plain == pytest.approx(0.5) and graded < plain                    # far mistakes cost more on a scale
    data = _categories("mixed")
    rest = ranges.between(data, None, [0, 1, 3], [2], "the rest", "C")
    real = next(s for s in rest if s.uid == "real")
    assert rest[0].uid == "real" and real.higher == "C" and real.lower == "the rest" and real.auc > 0.9


def test_a_perfect_separation_with_few_cases_is_promising_not_chance():
    """7 projects of each grade (one grade per project): perfect, but the correction for 51 parameters asks for
    more cases than 14 can give — said as promising, not as chance."""
    rng = np.random.default_rng(7)
    proj = np.repeat(np.arange(14), 15)
    grade = np.repeat(np.arange(14) % 2, 15)
    perfect = grade * 5.0 + rng.normal(0, 0.3, len(proj))
    cols = np.column_stack([perfect] + [rng.normal(0, 1, len(proj)) for _ in range(50)])
    names = ["perfect"] + [f"n{i}" for i in range(50)]
    data = Dataset([(u, u, "", "") for u in names], cols, grade, ["Low", "High"], proj,
                   [f"P{i}" for i in range(14)])
    a = rv.analyse(data, np.ones(len(grade), bool), "training", "Gland", n_boot=400, unit="projects")
    r = a.row("perfect")
    assert a.unit == "projects" and r.n == {"Low": 7, "High": 7} and r.auc == 1.0 and r.p < 0.05 and r.q >= 0.05 and r.verdict == "promising"
    assert rv.VERDICTS["promising"] == "Promising, not confirmed"



def test_the_same_data_as_objects_or_as_projects():
    """Few projects, many objects: as objects each one counts (confirmed); as projects, the cases are few."""
    rng = np.random.default_rng(5)
    proj = np.repeat(np.arange(8), 30)
    grade = np.repeat(np.arange(8) % 2, 30)                    # one grade per project, 4 projects each
    real = grade * 1.0 + rng.normal(0, 1, len(proj))
    cols = np.column_stack([real] + [rng.normal(0, 1, len(proj)) for _ in range(20)])
    names = ["real"] + [f"n{i}" for i in range(20)]
    data = Dataset([(u, u, "", "") for u in names], cols, grade, ["Low", "High"], proj, [f"P{i}" for i in range(8)])
    objects = rv.analyse(data, np.ones(len(grade), bool), "training", "Gland", n_boot=300, unit="objects")
    projects = rv.analyse(data, np.ones(len(grade), bool), "training", "Gland", n_boot=300, unit="projects")
    assert objects.row("real").verdict == "separates" and objects.row("real").n == {"Low": 120, "High": 120}
    assert projects.row("real").n == {"Low": 4, "High": 4} and projects.row("real").verdict == "too few"
