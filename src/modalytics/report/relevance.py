"""
A parameter relevance analysis as one self-contained HTML report (Qt-free).

Sections:

1. **Header** — the training set, profile, objects, the classes (ordered grades or
   categories), the projects used (training only, or training and evaluation: then said
   to be descriptive), the objects per class, and what was compared (each object a case,
   or each image's or project's class).
2. **Proposal** — the parameters proposed (the best of each redundancy group that
   separates the classes), or why none is.
3. **Parameters** — every parameter: its verdict, main AUC with its 95 % interval, how it
   goes (grades: larger or smaller with the grade; categories: the class that stands
   out), q, how many projects agree, its redundancy group and what to look at.
4. **Per class** — median (interquartile range), mean ± SD and the Shapiro–Wilk test of
   each parameter in each class.
5. **Pairs** — every pair of classes for the parameters that separate or weakly do: AUC
   and p corrected for the pairs (Holm).
6. **Figures** — the AUCs with their intervals; each separating parameter's values per
   class (each object a point).
7. **Redundancy groups**, then **Methods** (``training.relevance``).
"""

from __future__ import annotations

import math

import numpy as np

from modalytics.report.comparison import CSS, _ci, _e, _f, dot_ci_chart, strip_chart
from modalytics.training import relevance as rv

VERDICT_CSS = """
.v-separates { color: #1f8a70; font-weight: 600; } .v-weak { color: #b07a10; } .v-promising { color: #0072B2; }
.v-chance, .v-too { color: #8b97a8; }
tr.proposed td { background: #e8f6f0; }
"""
PALETTE = ("#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
# What was compared, as the report says it; and one case, in a sentence
COMPARED = {
    "objects": "Compared: <b>the objects</b> — each one a case (the classes as such, in this material). A project is "
               "one sample (a biopsy, a smear, a culture…); objects of one sample may resemble each other, so the "
               "p-values describe this material and may be optimistic about new samples (compare projects for that).",
    "images": "Compared: <b>the images</b> — each image's class one case, the median of its objects (for projects that "
              "hold images of several samples). The numbers count images, not objects.",
    "projects": "Compared: <b>the projects</b> — each project's class one case, the median of its objects (a project is "
                "one sample: a biopsy, a smear, a culture…): will it hold for new samples? The numbers count projects, "
                "not objects; with few projects per class, only large differences can be told from chance (see "
                "<i>promising</i>).",
    "within": "Compared: <b>the objects within their project</b> (an analysis made before the choice existed): only "
              "objects of the same project were compared with each other.",
}
CASE = {"objects": "object", "images": "image", "projects": "project", "within": "object"}


def _verdict(r: rv.Relevance) -> str:
    cls = "v-" + r.verdict.split()[0]
    return f"<span class='{cls}'>{_e(rv.VERDICTS[r.verdict])}</span>"


def _g(v) -> str:
    """A value in four significant digits ("—" if unknown)."""
    return "—" if v is None or not math.isfinite(v) else f"{v:.4g}"


def _p(p: float) -> str:
    if not math.isfinite(p):
        return "—"
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def _how(r: rv.Relevance, a: rv.Analysis) -> str:
    if not math.isfinite(r.auc):
        return ""
    if not a.ordered:
        return f"{r.best} {'higher' if r.direction == 'higher' else 'lower'}"
    return "larger with the grade" if r.direction == "higher" else "smaller with the grade"


def build(analysis: rv.Analysis, data=None, rows=None) -> str:
    """The report as one HTML string.

    Args:
        analysis: The analysis.
        data, rows: The measured objects and the rows used (for the distributions; without them, none are drawn).
    """
    a = analysis
    s = a.settings
    on = "the training projects" if a.on == "training" else "the training and evaluation projects"
    kind = ("ordered grades: " + " < ".join(a.grades)) if a.ordered else ("categories: " + ", ".join(a.grades))
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>{_e(a.name)} — parameter relevance</title>"
           f"<style>{CSS}{VERDICT_CSS}</style></head><body><main>"]
    # 1. Header
    counts = " · ".join(f"{_e(g)} {n}" for g, n in a.counts.items())
    out.append(f"<h1>{_e(a.name)}: which parameters separate the classes</h1>"
               f"<div class='muted'>{_e(a.group)} objects of training set {_e(a.training_set)} · profile "
               f"{_e(a.profile)} · {_e(kind)} · on {on} ({len(a.projects)} projects; {counts} objects) · "
               f"{len(a.rows)} parameters · {_e(a.made.replace('T', ' '))}</div>")
    out.append(f"<div class='{'good' if a.unit == 'objects' else 'warn'}'>{COMPARED[a.unit]}</div>")
    if a.on == "all":
        out.append("<div class='warn'>⚠ On the training and evaluation projects: a description of the data. "
                   "Parameters chosen here for a classifier would flatter it on the evaluation projects — choose them "
                   "on the training projects. (Test projects are never measured for classifiers.)</div>")
    # 2. Proposal
    proposed = a.proposed()
    if proposed:
        out.append("<div class='good'>Proposed: " + ", ".join(f"<b>{_e(r.title)}</b>" for r in proposed)
                   + " — each separates the classes (q &lt; " + f"{s.get('q_max', rv.Q_MAX):g}, AUC ≥ "
                   f"{s.get('separates_auc', rv.SEPARATES_AUC):g}) and is the best of its redundancy group. A "
                   "proposal, not a decision: look at what each measures and at the notes.</div>")
    else:
        weak = [r for r in a.rows if r.verdict == "weak"]
        out.append("<div class='warn'>No parameter separates the classes clearly on these projects"
                   + (f"; {len(weak)} do weakly (real, but small)" if weak else "") + ". More graded objects and "
                   "projects, or other parameters, may be needed.</div>")
    tally = {v: sum(1 for r in a.rows if r.verdict == v) for v in rv.VERDICTS}
    out.append("<p class='muted'>" + " · ".join(f"{rv.VERDICTS[v]}: {n}" for v, n in tally.items() if n) + "</p>")
    # 3. The table
    main = "AUC across the grades" if a.ordered else "AUC of the class that stands out"
    heads = ["Parameter", "Verdict", main, "How", "q", "Projects agreeing", "Group", "Notes"]
    out.append("<h2>Parameters</h2><p class='muted'>"
               + ("AUC: the probability that, of two objects of different grades, the higher-grade one has the larger "
                  "value (0.5: no relation; 1: perfect)" if a.ordered else
                  "AUC: the class that stands apart most, against all the others — the probability that its object "
                  "has the larger value (or the smaller: How)")
               + ", with its 95 % interval. q: the p-value ("
               + ("trend test" if a.ordered else "does any class differ") + ") corrected for the number of "
               "parameters. ★: proposed.</p><div class='scroll'><table><thead><tr>"
               + "".join(f"<th{' class=n' if i in (2, 4, 5, 6) else ''}>{_e(h)}</th>" for i, h in enumerate(heads))
               + "</tr></thead><tbody>")
    for r in a.rows:
        agree = f"{r.agree[0]} / {r.agree[1]}" if r.agree[1] else "—"
        cells = [f"<td>{'★ ' if r.proposed else ''}{_e(r.title)}</td>", f"<td>{_verdict(r)}</td>",
                 f"<td class='n'>{_f(r.auc)}{_ci(*r.ci)}</td>", f"<td>{_e(_how(r, a))}</td>",
                 f"<td class='n'>{_p(r.q)}</td>", f"<td class='n'>{agree}</td>",
                 f"<td class='n'>{r.group or ''}</td>", f"<td class='muted'>{_e('; '.join(r.notes()))}</td>"]
        out.append(f"<tr{' class=proposed' if r.proposed else ''}>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    # 4. Per class
    out.append("<h2>Per class</h2><p class='muted'>Median (interquartile range) and mean ± SD of the objects; "
               "Shapiro–Wilk: normal (p ≥ 0.05) or not, for information — the tests compare ranks and do not assume "
               "normality.</p><div class='scroll'><table><thead><tr><th>Parameter</th>"
               + "".join(f"<th class='n'>{_e(c)}: median (IQR)</th><th class='n'>mean ± SD</th>" for c in a.grades)
               + "<th>Normal?</th></tr></thead><tbody>")
    for r in a.rows:
        cells = []
        for c in a.grades:
            st = r.stats.get(c) or []
            if len(st) < 7 or not st[0]:
                cells += ["<td class='n'>—</td>", "<td class='n'>—</td>"]
                continue
            _n, med, q1, q3, mean, sd, _sw = st
            cells += [f"<td class='n'>{_g(med)} ({_g(q1)}–{_g(q3)})</td>", f"<td class='n'>{_g(mean)} ± {_g(sd)}</td>"]
        out.append(f"<tr><td>{_e(r.title)}</td>" + "".join(cells) + f"<td class='muted'>{_e(r.normal())}</td></tr>")
    out.append("</tbody></table></div>")
    # 5. Pairs
    shown = [r for r in a.rows if r.verdict in ("separates", "weak") and r.pairs]
    if shown and len(a.grades) > 2:
        names = [f"{x} vs {y}" for x, y, _a, _p in shown[0].pairs]
        unit = CASE[a.unit]
        out.append(f"<h2>Pairs</h2><p class='muted'>Every pair of classes, for the parameters that separate or weakly "
                   f"do: AUC — the probability that the second class's {unit} has the larger value — and p, corrected "
                   f"for the {len(names)} pairs (Holm).</p><div class='scroll'><table><thead><tr><th>Parameter</th>"
                   + "".join(f"<th class='n'>{_e(n)}</th>" for n in names) + "</tr></thead><tbody>")
        for r in shown:
            by = {f"{x} vs {y}": (auc, p) for x, y, auc, p in r.pairs}
            out.append(f"<tr><td>{_e(r.title)}</td>" + "".join(
                f"<td class='n'>{_f(by[n][0])}<br><span class='muted'>p {_p(by[n][1])}</span></td>" if n in by
                else "<td class='n'>—</td>" for n in names) + "</tr>")
        out.append("</tbody></table></div>")
    # 6. Figures
    out.append("<h2>Figures</h2><div class='figs'>")
    first = [r for r in a.rows if math.isfinite(r.auc)][:20]
    colour = {"separates": "#1f8a70", "weak": "#E69F00", "promising": "#0072B2", "chance": "#8b97a8",
              "too few": "#c0c6cf"}
    out.append(dot_ci_chart([(r.label, colour[r.verdict], r.auc, *r.ci) for r in first],
                            f"{main} (95 % interval), the first {len(first)}", width=520))
    if data is not None and rows is not None:
        columns = [c[0] for c in data.columns]
        for r in [x for x in a.rows if x.verdict == "separates"][:12]:
            j = columns.index(r.uid) if r.uid in columns else -1
            if j < 0:
                continue
            series = []
            for k, g in enumerate(data.grades):
                v = data.values[np.asarray(rows, bool) & (data.grade == k), j]
                series.append((g, PALETTE[k % len(PALETTE)], [float(x) for x in v if math.isfinite(x)]))
            out.append(strip_chart(series, f"{r.title} per class (│ median)"))
    out.append("</div>")
    # 7. Redundancy groups
    if a.groups:
        out.append(f"<h2>Redundancy groups</h2><p class='muted'>Parameters whose values follow each other (|Spearman ρ| "
                   f"≥ {s.get('redundant_rho', rv.REDUNDANT_RHO):g}): they measure one thing, and one of each group is "
                   f"enough.</p><ol>")
        by_uid = {r.uid: r for r in a.rows}
        for members in a.groups:
            names_ = [(by_uid[u].proposed, by_uid[u].title) if u in by_uid else (False, u) for u in members]
            out.append("<li>" + ", ".join(f"<b>{_e(t)}</b> ★" if p else _e(t) for p, t in names_) + "</li>")
        out.append("</ol>")
    # 8. Methods
    draws = s.get("bootstrap", rv.BOOTSTRAP)
    design = COMPARED[a.unit]
    main_m = ("<li><b>AUC across the grades</b>: over every pair of cases of different grades, the share in which "
              "the higher-grade one has the larger value (ties: one half) — the Jonckheere–Terpstra statistic as a "
              "share; with two grades the Mann–Whitney AUC. Given as ≥ 0.5 with its direction.</li>"
              if a.ordered else
              "<li><b>AUC of the class that stands out</b>: each class against all the others (one-vs-rest "
              "Mann–Whitney AUC); the class furthest from 0.5 is given, as ≥ 0.5 with its direction.</li>")
    test_m = ("the trend test — the concordance over pairs of cases of different grades, two-sided, against a "
              "normal curve with the shuffles' spread" if a.ordered else
              "the omnibus test (does any class differ? — the Kruskal–Wallis idea): the pairs of classes' squared "
              "departures of their concordance from 0.5, weighted by their numbers of pairs, against a gamma curve "
              "fitted to the shuffles")
    shuffle_m = ("the classes shuffled among each project's own objects" if a.unit == "within" else
                 "the classes shuffled among the cases")
    out.append(f"""<h2>Methods</h2>
<p>The {_e(a.group)} objects graded by hand on {on}, each measured with the profile's parameters; for each
parameter, the objects with a value. Classes: {_e(kind)}. {design}</p><ul>
{main_m}
<li><b>95 % interval</b>: bootstrap of the cases ({draws} draws with replacement), the 2.5th and 97.5th
percentiles.</li>
<li><b>p-value</b>: permutation ({draws} shuffles: {shuffle_m}) of {test_m}. <b>q</b>: the p-values corrected for the
number of parameters (Benjamini–Hochberg false discovery rate).</li>
<li><b>Pairs</b>: every pair of classes, its AUC and its p-value from the same shuffles, corrected for the number of
pairs (Holm's step-down), as a post-hoc comparison.</li>
<li><b>Per class</b>: median and interquartile range, mean and standard deviation; the Shapiro–Wilk test of normality
for information (the tests are rank-based and do not assume normality).</li>
<li><b>Verdict</b>: separates — q &lt; {s.get('q_max', rv.Q_MAX):g} and AUC ≥ {s.get('separates_auc', rv.SEPARATES_AUC):g};
weak — q &lt; {s.get('q_max', rv.Q_MAX):g}, a smaller AUC; promising, not confirmed — AUC ≥
{s.get('separates_auc', rv.SEPARATES_AUC):g} and p &lt; {s.get('q_max', rv.Q_MAX):g} before the correction for the
number of parameters but not after it (too few cases to confirm: more would tell); not distinguishable from chance —
otherwise; too few — fewer than {s.get('min_objects', rv.MIN_OBJECTS)} objects ({s.get('min_cases', rv.MIN_CASES)}
images or projects, when those are compared) of a class.</li>
<li><b>Projects agreeing</b>: of the projects with two classes of at least {rv.MIN_PER_PROJECT} objects each, those
where the parameter goes the same way. A note when it varies more between projects within a class than between
classes (Kruskal–Wallis ε² = H / (n − 1); between projects ≥ {rv.PROJECT_EFFECT:g}): it may follow the acquisition
or the sample rather than the class.</li>
<li><b>Redundancy groups</b>: Spearman correlation of every two parameters, average-linkage clustering of 1 − |ρ|,
cut at |ρ| = {s.get('redundant_rho', rv.REDUNDANT_RHO):g}. <b>Proposed</b>: the best of each group that separates
(alone in no group: itself).</li></ul>
<ol class='refs'><li>Benjamini Y, Hochberg Y (1995). Controlling the False Discovery Rate: A Practical and Powerful
Approach to Multiple Testing. J R Stat Soc B 57(1): 289–300. doi:10.1111/j.2517-6161.1995.tb02031.x</li></ol>
<p class='muted'>Report made by Modalytics. The results file (Results … .json) beside it holds every number.</p>""")
    out.append("</main></body></html>")
    return "".join(out)


def write(analysis: rv.Analysis, path: str, data=None, rows=None) -> str:
    from modalytics import fileio
    fileio.write_text(path, build(analysis, data, rows))
    return path
