"""The HTML report: sections, key parameters, classification, pictures, self-contained, printable."""

from __future__ import annotations

import re

import pytest

LOW, HIGH = "Low grade dysplasia", "High grade dysplasia"


def _measured(tmp_path, classify=True, keys=("object:any:area", "object:any:circularity")):
    from synthetic_projects import graded_project
    from modalytics.io.project_file import load_project
    from modalytics.core.model import Project
    from modalytics.parameters.results import classify as apply, measure_project
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.training.classifiers import Classifier, Condition, Rule
    path = graded_project(str(tmp_path / "case"), seed=3, n_low=4, n_high=3)
    data = load_project(path)
    prof = ProfileLibrary().get("colonic_tissue")
    prof.parameters = {u: {} for u in ("object:any:area", "object:any:circularity", "object:any:perimeter")}
    prof.report = {"parameters": list(keys)}
    if classify:
        prof.classification["Gland"] = Classifier("ranges", "Gland", [LOW, HIGH], rules=[
            Rule(HIGH, [Condition("object:any:circularity", "<=", 0.8, "Circularity")])], note="test").to_dict()
    project = Project()
    project.reset(data.images, profile=prof)
    project.path = path
    res = measure_project(project)
    if classify:
        res, _p = apply(res, prof.classifier_for("Gland"))
    project.set_results(res)
    return project


def test_report_sections_and_numbers(tmp_path):
    from modalytics.report.html import build
    project = _measured(tmp_path)
    page = build(project)
    titles = ("Summary", "Classification", "Distributions", "Objects", "Images", "Methods")
    places = [page.index(f">{title}</h2>") for title in titles]
    assert places == sorted(places)                                # the whole project first, then by image
    # summary: counts per grade, key parameters per class
    assert "<b>7</b>Gland object(s)" in page and f"<b>4</b>" in page and LOW in page
    summary = page[page.index(">Summary</h2>"):page.index(">Classification</h2>")]
    assert "<th class='param stick'>Area [µm²]</th>" in summary                       # a parameter per row
    assert all(f">{name}</th>" in summary for name in ("Mean", "SD", "Median", "IQR"))
    assert f"{LOW}<span class='muted'> · n = " in summary                      # a block of columns per class
    # classification: the classifier, agreement with the annotation, the confusion matrix, the shares
    assert "Ranges: High grade dysplasia when Circularity ≤ 0.8" in page and "Cohen's κ" in page
    assert "Share of High grade dysplasia" in page
    # pictures embedded, objects table with a row per object (numbered), box plots as SVG with the cut-off
    assert page.count("data:image/jpeg;base64,") == len(project.images)
    assert len(re.findall(r"<tr data-cls=", page)) == 7
    assert page.count("<svg") == 2 and "≤ 0.8" in page
    # methods: the key parameters' descriptions; the others listed
    assert "Also measured" in page and "Perimeter" in page
    # self-contained and printable
    assert "http://" not in page and "https://" not in page and "@media print" in page and "@page" in page


def test_report_without_key_parameters_or_classifier(tmp_path):
    from modalytics.report.html import build
    page = build(_measured(tmp_path, classify=False, keys=()))
    assert ">Classification</h2>" not in page and "No key parameters chosen" in page
    assert ">Distributions</h2>" not in page and "<b>7</b>Gland object(s)" in page


def test_report_needs_results(tmp_path):
    from modalytics.core.model import Project
    from modalytics.report.html import build
    with pytest.raises(ValueError, match="Measure"):
        build(Project())


def test_out_of_date_and_suggestions_are_said(tmp_path):
    from shapely.geometry import box
    from modalytics.core.model import Suggestion
    from modalytics.report.html import build
    project = _measured(tmp_path)
    img = project.images[0]
    node = project.suggestions_for(img)
    project.set_suggestions(node, [Suggestion(box(1, 1, 5, 5), "Gland", HIGH)])
    m = img.masks[0]
    project.set_mask_geometry(m, m.geometry.buffer(1.0))
    page = build(project)
    assert "out of date" in page and "1 suggestion(s) of a model are still waiting for review" in page


def test_profile_report_parameters_round_trip_and_old_files():
    from modalytics.profiles import Profile
    p = Profile("X", images=["phase"], parameters={"object:any:area": {}},
                report={"parameters": ["object:any:area", "object:any:circularity"]})
    assert p.report_parameters == ["object:any:area"]                         # only what is measured
    assert any("does not measure" in x for x in p.problems())
    again = Profile.from_yaml(p.to_yaml())
    assert again.report == p.report
    old = p.to_dict()
    old["report"] = ""                                                        # profiles from before the report
    assert Profile.from_dict(old).report == {} and Profile.from_dict(old).report_parameters == []


def test_the_pdf_summary_has_few_parameters_and_the_reasons_as_text(tmp_path):
    from modalytics.parameters.registry import ParameterLibrary
    from modalytics.profiles.model import Profile
    from modalytics.report.html import PDF_MAX, build, pdf_keys
    project = _measured(tmp_path, keys=("object:any:area", "object:any:perimeter"))
    lib = ParameterLibrary()
    # 1. By default: the classifier's parameters first, then the key parameters
    assert [k.uid for k in pdf_keys(project, lib)] == ["object:any:circularity", "object:any:area",
                                                       "object:any:perimeter"]
    page = build(project, pdf=True)
    assert "Summary for print" in page and "table class='objlist'" in page and "id='objects'" not in page
    reasons = re.findall(r"<td class='why'>([^<]*)</td>", page)
    assert reasons and all(reasons) and any("Circularity" in w or "no rule met" in w for w in reasons)
    assert re.findall(r"<h2[^>]*>([^<]+)</h2>", page) == ["Summary", "Classification", "Distributions", "Objects",
                                                          "Images", "Methods"]
    # 2. Ticked In PDF: only those (the screen report keeps its own); kept in the profile
    project.profile.report["pdf"] = ["object:any:perimeter"]
    assert [k.uid for k in pdf_keys(project, lib)] == ["object:any:perimeter"]
    assert "Perimeter" in build(project, pdf=True) and "Area [µm²]" in build(project)
    assert PDF_MAX == 6
    assert Profile.from_dict(project.profile.to_dict()).pdf_parameters == ["object:any:perimeter"]
