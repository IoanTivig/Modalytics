"""Training results as Excel workbooks: the sheets, the values (intervals as two columns), the native charts."""
from __future__ import annotations

import json
import re
import zipfile

import numpy as np
import pytest

pytest.importorskip("xlsxwriter")


def _book(path) -> tuple[list[str], int, str]:
    """(sheet names, charts, all the sheets' text) of a workbook, read from its parts."""
    with zipfile.ZipFile(path) as z:
        names = re.findall(r'<sheet name="([^"]+)"', z.read("xl/workbook.xml").decode("utf-8"))
        charts = sum(1 for n in z.namelist() if re.match(r"xl/charts/chart\d+\.xml$", n))
        text = z.read("xl/sharedStrings.xml").decode("utf-8") if "xl/sharedStrings.xml" in z.namelist() else ""
    return [n.replace("&amp;", "&") for n in names], charts, text


def test_parameters_workbook(tmp_path):
    from test_relevance import _data
    from modalytics.report import xlsx
    from modalytics.training import relevance as rv
    data = _data()
    rows = np.ones(len(data.grade), bool)
    a = rv.analyse(data, rows, "training", "Gland", "t", n_boot=200)
    path = xlsx.write_parameters(a, str(tmp_path / "p.xlsx"), data, rows)
    sheets, charts, text = _book(path)
    assert sheets == ["About", "Parameters", "Per class", "Pairs", "Redundancy groups", "Charts", "Values",
                      "Chart data"]
    shown = [r for r in a.rows if r.proposed or r.verdict in ("separates", "weak")]
    assert charts == 1 + len(shown) and "AUC 95% low" in text and "Strong" in text
    without = xlsx.write_parameters(a, str(tmp_path / "q.xlsx"))                 # no objects: the AUC chart only
    assert _book(without)[1] == 1 and "Values" not in _book(without)[0]


def test_segmentation_workbook(tmp_path):
    from test_compare import _comparison
    from modalytics.report import xlsx
    comp = _comparison()
    comp.test_groups = {"Naive": ["case_a"], "Hospital B": ["case_b"]}
    path = xlsx.write_segmentation(comp, str(tmp_path / "s.xlsx"))
    sheets, charts, text = _book(path)
    assert sheets == ["About", "Results", "Against the proposal", "Per project", "Per test group", "Per image",
                      "Training history", "Charts"]
    assert charts == 4 + 2 + 1 and "Object F1" in text and "Hospital B" in text and "Evaluation Dice" in text


def test_classifiers_workbook(tmp_path):
    from test_classifier_report import _data
    from modalytics.report import xlsx
    from modalytics.training import classifiers as cl
    from modalytics.training import metrics as mt
    data = _data(2, n=400, seed=3)
    train, held = data.project < 7, data.project >= 7
    built = {"logistic": cl.fit_logistic(data, train, ["a", "c"], "G"),
             "forest": cl.fit_forest(data, train, ["a", "b", "c"], "G", trees=50)}
    a = mt.assess(data, built, train, held, n_boot=100, finals={"Test": (data, data.project == 9)})
    results = json.loads(json.dumps(dict(a, training_set="Set", profile="P", group="G", grades=list(data.grades),
                                         ordered=True, classifiers={k: c.to_dict() for k, c in built.items()}),
                                    default=float))                           # as the Results file keeps it
    path = xlsx.write_classifiers(results, str(tmp_path / "c.xlsx"))
    sheets, charts, text = _book(path)
    assert sheets == ["About", "Results", "Against the best", "Final checks", "Per class", "Confusion matrices",
                      "Classifiers", "Charts"]
    assert charts == 2 and "Balanced accuracy" in text and "Logistic regression" in text
