"""
The classifiers of a training set compared, as one self-contained HTML report (Qt-free).

Sections:

1. **Header** — the training set, profile, objects, the classes (ordered grades or
   categories), the objects of each class in the training, evaluation and test projects.
2. **Recommendation** — the classifier to use and why, and where it was chosen
   (cross-validation or the evaluation projects): the best by balanced accuracy, or a
   readable one (Ranges, Score, logistic regression) when it is not distinguishable from
   the best (``training.metrics.assess``).
3. **Results** — every classifier where it was chosen: balanced accuracy, accuracy, κ, MCC,
   macro-F1, ROC AUC with their 95 % intervals; training next to them.
4. **Against the best** — the balanced accuracy difference with its interval (and McNemar's
   test) on the same objects; the verdict.
5. **Final checks** — each group of projects that took no part in the choice (the evaluation
   projects when chosen by cross-validation; each test group), side by side and one by one.
6. **Per class** — sensitivity, specificity, PPV, NPV and F1 of each classifier.
7. **Confusion matrices** — counts and row percentages.
8. **The classifiers** — what each is: its rules, points, weights (odds ratios) or the
   parameters' importance.
9. **Methods** — the measures (formulas), the statistics, the classifiers.
"""

from __future__ import annotations

import datetime as dt
import math

from modalytics.report.comparison import CSS, _e
from modalytics.training import classifiers as cl
from modalytics.training import metrics as mt

EXTRA_CSS = """
.cm td.n, .cm th.n { min-width: 64px; } .cm td.hit { background: #e8f6f0; font-weight: 600; }
.verdict-best { color: #1f8a70; font-weight: 600; } .verdict-same { color: #0072B2; } .verdict-worse { color: #8b97a8; }
"""


def _f(v, digits: int = 3) -> str:
    return "—" if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else f"{v:.{digits}f}"


def _pct(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.0f} %"


def _ci(ci, pct: bool = False) -> str:
    if not ci or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci):
        return ""
    lo, hi = ci
    text = f"{100 * lo:.0f}–{100 * hi:.0f} %" if pct else f"{lo:.2f} – {hi:.2f}"
    return f"<br><span class='muted'>[{text}]</span>"


def _p(p) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


MEASURE_HEADS = ["Balanced accuracy", "Accuracy", "κ", "MCC", "Macro-F1", "ROC AUC"]


def _measure_cells(m: dict) -> str:
    """The six measures of ``MEASURE_HEADS`` as table cells, each with its interval."""
    if not m.get("n"):
        return "<td class='n' colspan='6'>not judged</td>"
    return (f"<td class='n'><b>{_pct(m['balanced_accuracy'])}</b>{_ci(m.get('balanced_accuracy_ci'), True)}</td>"
            f"<td class='n'>{_pct(m['accuracy'])}{_ci(m.get('accuracy_ci'), True)}</td>"
            f"<td class='n'>{_f(m['kappa'], 2)}{_ci(m.get('kappa_ci'))}</td>"
            f"<td class='n'>{_f(m['mcc'], 2)}{_ci(m.get('mcc_ci'))}</td>"
            f"<td class='n'>{_f(m['macro_f1'], 2)}{_ci(m.get('macro_f1_ci'))}</td>"
            f"<td class='n'>{_f(m['auc'], 2)}{_ci(m.get('auc_ci'))}</td>")


def _matrix(title: str, cm: list, grades: list) -> str:
    """A confusion matrix (rows: by hand; columns: the classifier) with counts and row shares."""
    rows_html = []
    for i, row in enumerate(cm):
        total = sum(row) or 1
        rows_html.append(f"<tr><th>{_e(grades[i])}</th>" + "".join(
            f"<td class='n{' hit' if i == j else ''}'>{v}<br><span class='muted'>{100 * v / total:.0f} %</span></td>"
            for j, v in enumerate(row)) + "</tr>")
    return (f"<div><b>{_e(title)}</b><table class='cm'><thead><tr><th></th>"
            + "".join(f"<th class='n'>{_e(g)}</th>" for g in grades) + "</tr></thead><tbody>"
            + "".join(rows_html) + "</tbody></table></div>")


def _few(groups: list[int]) -> str:
    """The warning for intervals over objects (fewer projects than ``classifiers.MIN_PROJECTS``)."""
    return (f"<div class='warn'>The objects judged come from only {' or '.join(map(str, groups))} project(s) — "
            f"fewer than {cl.MIN_PROJECTS}, too few to resample whole projects. The intervals treat the objects as "
            f"independent and are likely too narrow, since objects of one project resemble each other: read them as "
            f"a lower bound of the uncertainty.</div>")


def build(a: dict, built: dict, tset, data, train, held, finals: dict | None = None, tried: int = 0) -> str:
    """The report as one HTML string.

    Args:
        a: ``training.metrics.assess``.
        built: {kind: Classifier}.
        tset, data: The training set and its measured objects.
        train, held: The training and evaluation objects.
        finals: {test group: (its measured objects, its rows)} — for their counts.
        tried: How many parameter selections were tried on the training projects (Build), for the warning.
    """
    grades, ordered = list(data.grades), data.ordered
    kinds = [k for k in mt.NAMES if k in built]
    kind_text = ("ordered grades: " + " < ".join(grades)) if ordered else ("categories: " + ", ".join(grades))
    counts = lambda d, rows: " · ".join(f"{_e(g)} {int(((d.grade == i) & rows).sum())}"   # noqa: E731
                                        for i, g in enumerate(grades))
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>{_e(tset.name)} — classifiers compared</title>"
           f"<style>{CSS}{EXTRA_CSS}</style></head><body><main>"]
    # 1. Header
    tests = "".join(f" · test group {_e(g)}: {counts(d, rows)}" for g, (d, rows) in (finals or {}).items())
    out.append(f"<h1>{_e(tset.name)}: classifiers compared</h1><div class='muted'>{_e(tset.group)} objects · profile "
               f"{_e(tset.profile.name)} · {_e(kind_text)} · training projects: {counts(data, train)} · evaluation "
               f"projects: {counts(data, held)}{tests} · {len(kinds)} classifier(s) · "
               f"{dt.datetime.now():%Y-%m-%d %H:%M}</div>")
    # 2. Recommendation
    judged = a.get("judged_on")
    on = "the evaluation projects" if judged == "evaluation" else "cross-validation on the training projects"
    if a.get("recommended"):
        out.append(f"<div class='good'>Recommended: <b>{_e(mt.NAMES[a['recommended']])}</b> — {_e(a['reason'])} "
                   f"Chosen by {on}, by balanced accuracy. You choose what to save to the profile.</div>")
    else:
        out.append(f"<div class='warn'>No recommendation: {_e(a.get('reason', ''))}.</div>")
    if a.get("fallback"):
        out.append(f"<div class='warn'>Asked to choose by "
                   f"{'the evaluation projects' if judged != 'evaluation' else 'cross-validation'}, but "
                   f"{_e(a['fallback'])}.</div>")
    pool = a["evaluation"] if judged == "evaluation" else a["cv"]
    by_objects = sorted({m.get("projects", 0) for m in pool.values() if m.get("n") and m.get("resampled") == "objects"})
    if by_objects:
        out.append(_few(by_objects))
    total = int((train if judged == "cross-validation" else held).sum())
    short = [(k, m["n"]) for k, m in pool.items() if m.get("n") and m["n"] < 0.9 * total]
    if short:
        out.append(f"<div class='warn'>{_e(', '.join(f'{mt.NAMES[k]} graded {n}' for k, n in short))} of the {total} "
                   f"objects: the others have an incomplete shape and it reads parameters that need the whole object — "
                   f"they stay ungraded in projects too. Classifiers on pixel parameters only grade every object; "
                   f"compare them on the same objects with care.</div>")
    if tried >= 3:
        out.append(f"<div class='warn'>{tried} different parameter selections were tried on the training projects: each "
                   f"was judged on them, so the classifiers kept look better there (cross-validation included) than "
                   f"they will on new projects. The final checks below — projects not used to choose — are the honest "
                   f"figure.</div>")
    # 3. Results
    extra = "Cross-validation (balanced accuracy)" if judged == "evaluation" else ""
    heads = ["Classifier", *MEASURE_HEADS, *([extra] if extra else []), "Training (balanced accuracy)", "Objects"]
    out.append("<h2>Results</h2><p class='muted'>Where the classifier is chosen: " + ("on the evaluation projects"
                                                                                     if judged == "evaluation" else
                                                                                     "the training projects, "
                                                                                     "cross-validated")
               + ", with 95 % intervals" + (" (resampling whole projects)" if not by_objects else "")
               + ". Balanced accuracy: the mean of the classes' sensitivities (each class counts "
                 "the same). κ: agreement beyond chance" + (" (quadratic-weighted: a mistake two grades off counts more)"
                                                            if ordered and len(grades) > 2 else "")
               + ". MCC: the correlation of the grades given with the true ones (−1 … 1). Training: on the objects it "
                 "was made from (optimistic). ★: recommended.</p><div class='scroll'><table><thead><tr>"
               + "".join(f"<th{' class=n' if i else ''}>{_e(h)}</th>" for i, h in enumerate(heads)) + "</tr></thead><tbody>")
    for kind in kinds:
        m = pool.get(kind) or {}
        cv, tr = a["cv"].get(kind) or {}, a["training"].get(kind) or {}
        star = "★ " if kind == a.get("recommended") else ""
        cells = [f"<td>{star}{_e(mt.NAMES[kind])}</td>", _measure_cells(m)]
        if extra:
            cells.append(f"<td class='n'>{_pct(cv.get('balanced_accuracy', math.nan)) if cv else '—'}</td>")
        cells += [f"<td class='n'>{_pct(tr.get('balanced_accuracy', math.nan)) if tr.get('n') else '—'}</td>",
                  f"<td class='n'>{m.get('n', 0)}</td>"]
        out.append(f"<tr{' class=best' if kind == a.get('recommended') else ''}>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    # 4. Against the best
    if a.get("best") and a["against"]:
        out.append(f"<h2>Against the best ({_e(mt.NAMES[a['best']])})</h2><p class='muted'>On the objects both graded: "
                   f"the balanced accuracy difference (best − other) with its 95 % interval (paired bootstrap, "
                   f"{'resampling whole projects' if not by_objects else 'over objects'}), and McNemar's test on the "
                   f"objects only one of the two grades right. "
                   + ("Not distinguishable: the interval contains 0 (McNemar's test treats the objects as independent, "
                      "so it is shown but not used)." if not by_objects else
                      "Not distinguishable: the interval contains 0 and p ≥ 0.05.")
                   + "</p><table><thead><tr><th>Classifier</th>"
                   f"<th class='n'>Difference</th><th class='n'>Only the best right</th><th class='n'>Only this one "
                   f"right</th><th class='n'>McNemar p</th><th>Verdict</th></tr></thead><tbody>")
        for kind in kinds:
            c = a["against"].get(kind)
            if not c:
                continue
            lo, hi = c.get("difference_ci", (math.nan, math.nan))
            same = mt.alike(c)
            verdict = ("<span class='verdict-same'>not distinguishable</span>" if same else
                       "<span class='verdict-worse'>the best is clearly better</span>")
            out.append(f"<tr><td>{_e(mt.NAMES[kind])}</td><td class='n'>{c['difference']:+.3f}<br><span "
                       f"class='muted'>[{lo:+.3f}, {hi:+.3f}]</span></td><td class='n'>{c['only_a']}</td>"
                       f"<td class='n'>{c['only_b']}</td><td class='n'>{_p(c['p'])}</td><td>{verdict}</td></tr>")
        out.append("</tbody></table>")
    # 5. Final checks
    out.append("<h2>Final checks</h2><p class='muted'>Measured after the choice, on projects that took no part in it; "
               "they do not change it (choosing again on them would make them one more evaluation set). A group that "
               "differs on purpose — another microscope setup, another lab — shows how far the classifier carries "
               "over.</p>")
    fin = a.get("finals") or {}
    if not fin:
        out.append("<div class='warn'>No final check: " + ("choose by cross-validation (Build), so the evaluation "
                                                           "projects become one, or " if judged == "evaluation" else "")
                   + "mark projects as Test (Training › Data), in a named group per setup or lab if you have "
                     "several.</div>")
    else:
        choice = "Cross-validation" if judged == "cross-validation" else "Evaluation"
        out.append("<div class='scroll'><table><thead><tr><th>Classifier</th>"
                   f"<th class='n'>{choice} (the choice)</th>"
                   + "".join(f"<th class='n'>{_e(g)}</th>" for g in fin) + "</tr></thead><tbody>")
        for kind in kinds:
            m = pool.get(kind) or {}
            cells = [f"<td class='n'>{_pct(m.get('balanced_accuracy', math.nan)) if m.get('n') else '—'}</td>"]
            for g, per in fin.items():
                f = per.get(kind) or {}
                cells.append(f"<td class='n'>{_pct(f['balanced_accuracy']) if f.get('n') else '—'}"
                             f"{_ci(f.get('balanced_accuracy_ci'), True) if f.get('n') else ''}</td>")
            star = "★ " if kind == a.get("recommended") else ""
            out.append(f"<tr{' class=best' if star else ''}><td>{star}{_e(mt.NAMES[kind])}</td>" + "".join(cells)
                       + "</tr>")
        out.append("</tbody></table></div><p class='muted'>Balanced accuracy with its 95 % interval.</p>")
        for g, per in fin.items():
            any_m = next((m for m in per.values() if m.get("n")), {})
            what = "the evaluation projects" if g == mt.EVALUATION and g not in (finals or {}) else f"test group {g}"
            out.append(f"<h3>{_e(g)}</h3><p class='muted'>{_e(what)}: {any_m.get('projects', 0)} project(s), "
                       f"{any_m.get('n', 0)} objects.</p>")
            if any_m.get("n") and any_m.get("resampled") == "objects":
                out.append(_few([any_m.get("projects", 0)]))
            out.append("<div class='scroll'><table><thead><tr><th>Classifier</th>"
                       + "".join(f"<th class='n'>{_e(h)}</th>" for h in MEASURE_HEADS)
                       + "<th class='n'>Objects</th></tr></thead><tbody>")
            for kind in kinds:
                m = per.get(kind) or {}
                star = "★ " if kind == a.get("recommended") else ""
                out.append(f"<tr{' class=best' if star else ''}><td>{star}{_e(mt.NAMES[kind])}</td>"
                           f"{_measure_cells(m)}<td class='n'>{m.get('n', 0)}</td></tr>")
            out.append("</tbody></table></div>")
            rec = per.get(a.get("recommended")) or {}
            if rec.get("confusion"):
                out.append("<div class='figs'>" + _matrix(f"{mt.NAMES[a['recommended']]} on {g}", rec["confusion"],
                                                          grades) + "</div>")
    # 6. Per class
    out.append("<h2>Per class</h2><p class='muted'>Each class against the others: sensitivity (recall), specificity, "
               "PPV (precision), NPV and F1, with 95 % intervals" + (" (resampling whole projects)" if not by_objects
                                                                    else " (Wilson's)") + ".</p>"
               "<div class='scroll'><table><thead><tr>"
               "<th>Classifier</th><th>Class</th><th class='n'>n</th><th class='n'>Sensitivity</th><th class='n'>"
               "Specificity</th><th class='n'>PPV</th><th class='n'>NPV</th><th class='n'>F1</th></tr></thead><tbody>")
    for kind in kinds:
        m = pool.get(kind) or {}
        for i, pc in enumerate(m.get("per_class", [])):
            out.append(f"<tr><td>{_e(mt.NAMES[kind]) if i == 0 else ''}</td><td>{_e(pc['class'])}</td>"
                       f"<td class='n'>{pc['n']}</td>"
                       + "".join(f"<td class='n'>{_pct(pc[key])}{_ci(pc[key + '_ci'], True)}</td>"
                                 for key in ("sensitivity", "specificity", "ppv", "npv"))
                       + f"<td class='n'>{_f(pc['f1'], 2)}</td></tr>")
    out.append("</tbody></table></div>")
    # 7. Confusion matrices
    out.append("<h2>Confusion matrices</h2><p class='muted'>Rows: the grade given by hand; columns: the classifier's; "
               "counts and the share of each row.</p><div class='figs'>")
    for kind in kinds:
        m = pool.get(kind) or {}
        if m.get("confusion"):
            out.append(_matrix(mt.NAMES[kind], m["confusion"], grades))
    out.append("</div>")
    # 8. The classifiers
    out.append("<h2>The classifiers</h2><ul>")
    for kind in kinds:
        clf = built[kind]
        extra = ""
        if clf.learned:
            ranked = sorted(zip(clf.features, clf.importance), key=lambda t: -t[1])
            extra = "<br><span class='muted'>" + ("Weight share" if kind == "logistic" else "Importance") + ": " + \
                ", ".join(f"{_e(lab)} {100 * imp:.0f} %" for (_u, lab), imp in ranked) + "</span>"
        out.append(f"<li><b>{_e(mt.NAMES[kind])}</b> ({len(clf.parameters)} parameters): {_e(cl.describe(clf))}{extra}"
                   "</li>")
    out.append("</ul>")
    # 9. Methods
    from modalytics.training.prep import MIN_REFERENCE, Prep
    preps = {Prep.from_dict(c.prep).describe() for c in built.values()}
    prep = " / ".join(sorted(preps)) if preps else "none"
    prep_note = ("" if prep == "none" else
                 f" Normalisation per project: each parameter relative to the median (and interquartile range) of "
                 f"the project's complete objects of the group, whatever their grade (at least {MIN_REFERENCE}), "
                 f"applied the same way to every project classified. Augmented rows (from the training projects "
                 f"only; in cross-validation from the folds trained on) were used to make the classifiers, never to "
                 f"judge them.")
    chosen = ("on the evaluation projects (whole projects, never used to make it)" if judged == "evaluation" else
              "by cross-validation on the training projects: each training project graded by the classifier made "
              "again, the same way, without it")
    untouched = "the test groups" if judged == "evaluation" else "the evaluation projects and the test groups"
    out.append(f"""<h2>Methods</h2>
<p>The {_e(tset.group)} objects graded by hand, measured with the profile's parameters. Each classifier was made on the
training projects; a partial object (shape incomplete) was judged only by a classifier that reads none of the
parameters that need the whole object. The classifier was chosen {chosen}. The projects that took no part in the
choice — {untouched} — were each measured after it, apart, as final checks.</p><ul>
<li><b>Preparation</b> (the same for every classifier): {_e(prep)}.{prep_note}</li>
<li><b>Ranges</b>: conditions at cut-offs found together for each rule (the rule's Youden index, one cut-off
at a time); the highest grade whose conditions hold. <b>Score</b>:
points for criteria at cut-offs, fitted by logistic regression and rounded, the cut-offs then found together (the
training objects' balanced accuracy); the total's thresholds give the grade.
<b>Logistic regression</b>: the log-odds of each grade a weighted sum of the standardised parameters (L2-penalised).
<b>Random forest</b>: decision trees on bootstrap samples, random parameters at each split, the grades' votes.
<b>Gradient boosting</b>: shallow regression trees added one after another, each fitted to what the sum before got
wrong. Missing values count as the training median (learned classifiers).</li>
<li><b>Balanced accuracy</b> = mean over classes of TP / (TP + FN). <b>Accuracy</b> = graded right / all.
<b>Cohen's κ</b> = (p<sub>o</sub> − p<sub>e</sub>) / (1 − p<sub>e</sub>){', quadratic weights (i − j)² / (k − 1)²' if
ordered and len(grades) > 2 else ''}. <b>MCC</b> = (c·s − Σ p<sub>k</sub>t<sub>k</sub>) /
√((s² − Σ p<sub>k</sub>²)(s² − Σ t<sub>k</sub>²)). Per class: sensitivity TP / (TP + FN), specificity TN / (TN + FP),
PPV TP / (TP + FP), NPV TN / (TN + FN), F1 = 2·PPV·sensitivity / (PPV + sensitivity); macro-F1 their mean.
<b>ROC AUC</b> from each classifier's score (learned: probabilities; Score with two grades: points; Ranges: the
yes/no decision), {'of the higher grade' if len(grades) == 2 else 'the mean of the one-against-the-others AUCs'}.</li>
<li><b>Intervals</b>: objects of one project are not independent, so every interval resamples whole projects
(percentile bootstrap over projects, {mt.BOOTSTRAP} draws) when the objects come from {cl.MIN_PROJECTS} or more
projects; with fewer, proportions by Wilson's score interval and the other measures by bootstrap over the objects —
likely too narrow. Cross-validation: the out-of-fold grades and scores pooled over the folds.
<b>Comparison</b>: the paired bootstrap interval of the balanced accuracy difference (the same projects or objects
drawn for both), and McNemar's exact test on the objects one classifier grades right and the other wrong.</li>
<li><b>Recommendation</b>: the best by balanced accuracy; when it is a random forest or gradient boosting and a
readable classifier (Ranges, Score, logistic regression) is not distinguishable from it (the interval of the
difference contains 0; with intervals over objects, McNemar p ≥ 0.05 too), the best such readable one — one can see
how its parameters decide.</li></ul>
<p class='muted'>Report made by Modalytics. The results file (Results … .json) beside it holds every number and the
classifiers themselves.</p>""")
    out.append("</main></body></html>")
    return "".join(out)
