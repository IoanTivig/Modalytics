"""
The report of an architecture comparison: one self-contained HTML file (Qt-free).

Like the project report (``report.html``): pictures embedded, plots as inline SVG, opens
offline and prints to PDF. Sections:

1. **Header** — what was compared, on which data, judged on which projects, by which
   measure; the proposal (the best by that measure) and a warning when there were no
   test projects (choosing on a set and reporting it flatters the winner).
2. **Results** — per architecture: the measures the gland-segmentation community
   reports (object F1 with precision and recall, panoptic quality, GlaS object Dice
   and object Hausdorff, pixel Dice and IoU, HD95 and NSD of the matched objects), each
   with its 95 % interval; merges and splits; speed, size and training time.
3. **Against the proposal** — the paired bootstrap difference of each architecture
   from the proposal, with its interval: clearly worse, or not distinguishable.
4. **Figures** — the main measures with their intervals; the learning curves (loss
   and evaluation Dice per epoch, the epoch kept marked); accuracy against speed; the
   measure per image (each image a point); what the errors are (found, missed,
   invented).
5. **Per test group** — with several test groups (e.g. one per microscope setup or lab), each a
   final check apart: the criterion per group side by side, then every measure per group.
6. **Per project** — the criterion per project and architecture.
7. **Examples** — a few judged images with the hand outlines and each model's.
An **evaluation** (existing models judged, Training › Evaluate: ``compare.evaluate_models``) has the
same parts without the training: a "Data and models" block (the projects judged, and where each
model comes from — its file, architecture, starting weights, training — with a warning when it is
judged on projects it learned from), no curves, and no training methods.

8. **Methods** — the data, the training, how the models were judged (definitions and
   formulas), the statistics, the architectures, the software and hardware. References are
   kept to a minimum (this is a report on a training run, not a paper): each measure is named
   with its formula; only the GlaS object measures, whose definition a name does not give,
   are cited; architectures and starting weights list their own (what each model is).
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import math

import numpy as np

from modalytics.training import compare as cmp
from modalytics.training import segmentation as seg

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")   # Okabe–Ito (colour-blind safe)
HAND, MODEL = (60, 200, 90), (230, 60, 200)                                      # outline colours (RGB)

# The few references the report cites, by key (see _cite). Standard measures, statistics and training steps
# are named with their formulas, not cited; the methods' references are in the code that computes them
# (training.segmentation, training.compare). Architectures and starting weights carry their own references
# (what each model is), shown in their own tables.
CITATIONS = {
    "glas": "Sirinukunwattana K, Pluim JPW, Chen H, et al. (2017). Gland segmentation in colon histology images: The glas "
            "challenge contest. Medical Image Analysis 35: 489–502. doi:10.1016/j.media.2016.08.008",
}


def _cite(*keys: str) -> str:
    """A short numbered list of the references of the given methods, placed under the section that uses them.

    Args:
        *keys: Keys of ``CITATIONS``, in the order they should be listed.

    Returns:
        HTML for the list (empty when no key is given).
    """
    if not keys:
        return ""
    return "<ol class='refs'>" + "".join(f"<li>{_e(CITATIONS[k])}</li>" for k in keys) + "</ol>"

SUMMARY = ("f1", "glas_f1", "pq", "object_dice", "object_hausdorff", "dice", "iou", "hd95", "nsd")
FIGURE = ("f1", "pq", "object_dice", "dice")


def _e(text) -> str:
    return html.escape(str(text))


def _f(v, fmt="{:.3f}") -> str:
    return "—" if v is None or not math.isfinite(v) else fmt.format(v)


def _ci(lo, hi, fmt="{:.3f}") -> str:
    return "" if not (math.isfinite(lo) and math.isfinite(hi)) else f"<br><span class='muted'>[{fmt.format(lo)}, " \
                                                                    f"{fmt.format(hi)}]</span>"


# ---------------------------------------------------------------- figures (inline SVG)
def _axis_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo] if math.isfinite(lo) else []
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


def _value_range(vals: list) -> tuple[float, float]:
    """The axis range of values: the data with a margin (shares stay within 0–1), so small differences show."""
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.15, 0.01 if hi <= 1 else 0.5)
    lo, hi = lo - pad, hi + pad
    if max(vals) <= 1 and min(vals) >= 0:
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return lo, hi


def _label_space(labels, width: int) -> tuple[int, int]:
    """(left margin, width): room for the longest label (≈ 6.3 px a character at the labels' size, at most 360 px),
    the chart widened by as much, so its plotting area stays as it was."""
    left = int(min(360, max(120, 6.3 * max((len(str(lab)) for lab in labels), default=0) + 16)))
    return left, width + max(0, left - 170)


def dot_ci_chart(rows: list, title: str, width: int = 460, row_h: int = 26) -> str:
    """One measure: a point and its 95 % interval per architecture (rows: (label, colour, value, lo, hi))."""
    vals = [v for _l, _c, v, lo, hi in rows for v in (v, lo, hi) if math.isfinite(v)]
    if not vals:
        return ""
    lo_v, hi_v = _value_range(vals)
    left, width = _label_space([r[0] for r in rows], width)
    right, top = 20, 26
    h = top + row_h * len(rows) + 30
    x = lambda v: left + (v - lo_v) / max(hi_v - lo_v, 1e-12) * (width - left - right)
    out = [f"<svg width='{width}' height='{h}' role='img' aria-label='{_e(title)}'>",
           f"<text x='{left}' y='16' class='ttl'>{_e(title)}</text>"]
    for t in _axis_ticks(lo_v, hi_v):
        out.append(f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='{top - 4}' y2='{h - 26}' class='grid'/>"
                   f"<text x='{x(t):.1f}' y='{h - 10}' class='ax' text-anchor='middle'>{t:g}</text>")
    for i, (label, colour, v, lo, hi) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(f"<text x='{left - 8}' y='{y + 4:.1f}' class='lbl' text-anchor='end'>{_e(label)}</text>")
        if math.isfinite(lo) and math.isfinite(hi):
            out.append(f"<line x1='{x(lo):.1f}' x2='{x(hi):.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='{colour}' "
                       f"stroke-width='2'/>")
        if math.isfinite(v):
            out.append(f"<circle cx='{x(v):.1f}' cy='{y:.1f}' r='5' fill='{colour}'><title>{_e(label)}: {v:.3f}"
                       f"</title></circle>")
    out.append("</svg>")
    return "".join(out)


def line_chart(series: list, title: str, ylabel: str, width: int = 460, height: int = 240, marks=None) -> str:
    """Lines over epochs (series: (label, colour, [(epoch, value)])), the kept epoch of each marked."""
    pts = [(e, v) for _l, _c, s in series for e, v in s if math.isfinite(v)]
    if not pts:
        return ""
    x0, x1 = min(e for e, _v in pts), max(e for e, _v in pts)
    y0, y1 = min(v for _e2, v in pts), max(v for _e2, v in pts)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    left, right, top, bottom = 52, 14, 26, 36
    X = lambda e: left + (e - x0) / max(x1 - x0, 1) * (width - left - right)
    Y = lambda v: height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)
    out = [f"<svg width='{width}' height='{height}' role='img' aria-label='{_e(title)}'>",
           f"<text x='{left}' y='16' class='ttl'>{_e(title)}</text>"]
    for t in _axis_ticks(y0, y1):
        out.append(f"<line x1='{left}' x2='{width - right}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' class='grid'/>"
                   f"<text x='{left - 6}' y='{Y(t) + 4:.1f}' class='ax' text-anchor='end'>{t:g}</text>")
    for t in _axis_ticks(x0, x1):
        out.append(f"<text x='{X(t):.1f}' y='{height - 18}' class='ax' text-anchor='middle'>{t:g}</text>")
    out.append(f"<text x='{(left + width - right) / 2:.0f}' y='{height - 3}' class='ax' text-anchor='middle'>epoch"
               f"</text><text x='12' y='{(top + height - bottom) / 2:.0f}' class='ax' text-anchor='middle' "
               f"transform='rotate(-90 12 {(top + height - bottom) / 2:.0f})'>{_e(ylabel)}</text>")
    for label, colour, s in series:
        s = [(e, v) for e, v in s if math.isfinite(v)]
        if len(s) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(e):.1f},{Y(v):.1f}" for i, (e, v) in enumerate(s))
            out.append(f"<path d='{d}' fill='none' stroke='{colour}' stroke-width='2'><title>{_e(label)}</title></path>")
        mark = (marks or {}).get(label)
        hit = next(((e, v) for e, v in s if e == mark), None)
        if hit:
            out.append(f"<circle cx='{X(hit[0]):.1f}' cy='{Y(hit[1]):.1f}' r='5' fill='white' stroke='{colour}' "
                       f"stroke-width='2'><title>{_e(label)}: epoch {hit[0]} kept</title></circle>")
    out.append("</svg>")
    return "".join(out)


def scatter_chart(points: list, title: str, xlabel: str, ylabel: str, width: int = 460, height: int = 260) -> str:
    """Points (label, colour, x, y, size)."""
    pts = [p for p in points if math.isfinite(p[2]) and math.isfinite(p[3])]
    if not pts:
        return ""
    xs, ys = [p[2] for p in pts], [p[3] for p in pts]
    x0, x1 = 0.0, max(xs) * 1.15
    y0, y1 = min(ys), max(ys)
    pad = max((y1 - y0) * 0.25, 0.02)
    y0, y1 = y0 - pad, y1 + pad
    left, right, top, bottom = 52, 20, 26, 36
    X = lambda v: left + (v - x0) / max(x1 - x0, 1e-9) * (width - left - right)
    Y = lambda v: height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)
    out = [f"<div><svg width='{width}' height='{height}' role='img' aria-label='{_e(title)}'>",
           f"<text x='{left}' y='16' class='ttl'>{_e(title)}</text>"]
    for t in _axis_ticks(y0, y1):
        out.append(f"<line x1='{left}' x2='{width - right}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' class='grid'/>"
                   f"<text x='{left - 6}' y='{Y(t) + 4:.1f}' class='ax' text-anchor='end'>{t:g}</text>")
    for t in _axis_ticks(x0, x1):
        out.append(f"<text x='{X(t):.1f}' y='{height - 18}' class='ax' text-anchor='middle'>{t:g}</text>")
    out.append(f"<text x='{(left + width - right) / 2:.0f}' y='{height - 3}' class='ax' text-anchor='middle'>"
               f"{_e(xlabel)}</text><text x='12' y='{(top + height - bottom) / 2:.0f}' class='ax' "
               f"text-anchor='middle' transform='rotate(-90 12 {(top + height - bottom) / 2:.0f})'>{_e(ylabel)}</text>")
    for label, colour, xv, yv, size in pts:
        r = 4 + 2 * math.sqrt(max(size, 0.1))
        out.append(f"<circle cx='{X(xv):.1f}' cy='{Y(yv):.1f}' r='{r:.1f}' fill='{colour}' fill-opacity='0.75' "
                   f"stroke='white'><title>{_e(label)}: {xv:.1f} ms, {yv:.3f}, {size:.1f} M parameters</title>"
                   f"</circle>")
    out.append("</svg>")
    # A legend below (labels beside the points collide when two score the same)
    out.append("<div class='legend'>" + "".join(f"<span><span class='swatch' style='background:{c}'></span>"
                                                f"{_e(label)}</span>" for label, c, *_r in pts) + "</div></div>")
    return "".join(out)


def strip_chart(rows: list, title: str, width: int = 460, row_h: int = 28) -> str:
    """Per-image values per architecture (rows: (label, colour, [values])), each image a point, the median a bar."""
    vals = [v for _l, _c, vs in rows for v in vs if math.isfinite(v)]
    if not vals:
        return ""
    lo_v, hi_v = _value_range(vals)
    left, width = _label_space([r[0] for r in rows], width)
    right, top = 20, 26
    h = top + row_h * len(rows) + 30
    X = lambda v: left + (v - lo_v) / max(hi_v - lo_v, 1e-12) * (width - left - right)
    out = [f"<svg width='{width}' height='{h}' role='img' aria-label='{_e(title)}'>",
           f"<text x='{left}' y='16' class='ttl'>{_e(title)}</text>"]
    for t in _axis_ticks(lo_v, hi_v):
        out.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 4}' y2='{h - 26}' class='grid'/>"
                   f"<text x='{X(t):.1f}' y='{h - 10}' class='ax' text-anchor='middle'>{t:g}</text>")
    rng = np.random.default_rng(0)
    for i, (label, colour, vs) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(f"<text x='{left - 8}' y='{y + 4:.1f}' class='lbl' text-anchor='end'>{_e(label)}</text>")
        vs = [v for v in vs if math.isfinite(v)]
        for v in vs:
            out.append(f"<circle cx='{X(v):.1f}' cy='{y + rng.uniform(-6, 6):.1f}' r='3' fill='{colour}' "
                       f"fill-opacity='0.6'/>")
        if vs:
            m = float(np.median(vs))
            out.append(f"<line x1='{X(m):.1f}' x2='{X(m):.1f}' y1='{y - 10:.1f}' y2='{y + 10:.1f}' stroke='#222' "
                       f"stroke-width='2'><title>median {m:.3f}</title></line>")
    out.append("</svg>")
    return "".join(out)


def error_bars(rows: list, title: str, width: int = 460, row_h: int = 26) -> str:
    """Stacked bars per architecture: found (TP), missed (FN), invented (FP), as shares of all three."""
    left, width = _label_space([r[0] for r in rows], width)
    right, top = 20, 26
    h = top + row_h * len(rows) + 46
    out = [f"<svg width='{width}' height='{h}' role='img' aria-label='{_e(title)}'>",
           f"<text x='{left}' y='16' class='ttl'>{_e(title)}</text>"]
    colours = {"found": "#009E73", "missed": "#E69F00", "invented": "#D55E00"}
    span = width - left - right
    for i, (label, c) in enumerate(rows):
        y = top + row_h * i + 4
        total = c.tp + c.fn + c.fp
        out.append(f"<text x='{left - 8}' y='{y + 13}' class='lbl' text-anchor='end'>{_e(label)}</text>")
        x = left
        for name, n in (("found", c.tp), ("missed", c.fn), ("invented", c.fp)):
            w = span * n / total if total else 0
            if w > 0:
                out.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{row_h - 8}' fill='{colours[name]}'>"
                           f"<title>{name}: {n}</title></rect>")
            x += w
    ly = top + row_h * len(rows) + 16
    lx = left
    for name, col in colours.items():
        out.append(f"<rect x='{lx}' y='{ly}' width='12' height='12' fill='{col}'/>"
                   f"<text x='{lx + 16}' y='{ly + 11}' class='lbl'>{name}</text>")
        lx += 90
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- example pictures
def example_picture(ex: seg.Example, found: list, width: int = 300) -> str:
    """A judged region: the hand outlines (green) and a model's (magenta), as an embedded PNG."""
    import cv2
    p = ex.pixels
    img = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB) if p.ndim == 2 else np.ascontiguousarray(p[..., :3])
    x0, y0, px, py = ex.frame
    h, w = img.shape[:2]
    scale = min(1.0, width / w)                             # shrink first, then draw: thin lines stay visible
    if scale < 1:
        img = cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img)

    def draw(polys, colour, thickness):
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)
                pts = (np.column_stack([(xy[:, 0] - x0) / px, (xy[:, 1] - y0) / py]) * scale).round().astype(np.int32)
                cv2.polylines(img, [pts], True, colour, thickness, cv2.LINE_AA)
    draw(ex.hand, HAND, 5)                                  # wide underneath: visible where the two agree
    draw(found, MODEL, 2)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return f"<img src='data:image/png;base64,{base64.b64encode(png.tobytes()).decode()}' alt='{_e(ex.image)}'>"


# ---------------------------------------------------------------- the report
CSS = """
:root { --fg:#1d232b; --muted:#5f6b7a; --line:#d9dee5; --accent:#1f8a70; --soft:#f4f6f8; --bg:#ffffff; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0; font-size: 14px; line-height: 1.45; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 24px 60px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 34px 0 10px; border-bottom: 2px solid
var(--accent); padding-bottom: 4px; } h3 { font-size: 15px; margin: 20px 0 6px; }
.muted { color: var(--muted); font-size: 12px; } .warn { background: #fff4e0; border-left: 4px solid #e0a040;
padding: 8px 12px; margin: 10px 0; } .good { background: #e8f6f0; border-left: 4px solid var(--accent);
padding: 8px 12px; margin: 10px 0; }
table { border-collapse: collapse; margin: 6px 0 12px; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--soft); font-weight: 600; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #e8f6f0; } .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
margin-right: 6px; }
.scroll { overflow-x: auto; max-width: 100%; border: 1px solid var(--line); border-radius: 4px; }
.scroll table { margin: 0; width: max-content; min-width: 100%; }
.figs { display: flex; flex-wrap: wrap; gap: 18px; }
svg text.ttl { font-size: 13px; font-weight: 600; fill: var(--fg); } svg text.ax { font-size: 11px; fill: var(--muted); }
svg text.lbl { font-size: 12px; fill: var(--fg); } svg line.grid { stroke: #e3e7ec; }
.legend { font-size: 12px; padding-left: 52px; } .legend > span { margin-right: 14px; white-space: nowrap; }
.examples { display: flex; flex-wrap: wrap; gap: 10px; } .examples figure { margin: 0; }
.examples figcaption { font-size: 12px; color: var(--muted); }
ol.refs li { margin-bottom: 4px; } code { font-size: 12px; }
@page { size: A4 landscape; margin: 12mm; }
@media print { main { max-width: none; padding: 0; } body { font-size: 11px; } h2 { break-after: avoid; }
  svg, table, figure { break-inside: avoid; } .scroll { overflow: visible; border: none; }
  table { font-size: 9.5px; } th, td { white-space: normal !important; padding: 3px 5px; }
  .scroll table { width: 100%; } td .muted, th .muted { font-size: 8.5px; } }
"""
# The "Data and training" block (``data_section``): kept apart so it can be added to a report written before it
DATA_CSS = """
.facts { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 18px; align-items: start;
         margin: 14px 0 6px; }
.facts .card { border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; background: var(--bg); }
.facts .card h3 { margin: 0 0 8px; font-size: 14px; }
.facts table { width: 100%; margin: 0; } .facts td.k, .facts dt { color: var(--muted); }
.facts td.k, .facts td.n, .facts th { white-space: nowrap; } .facts tr.use td { white-space: normal; font-size: 12px; }
.facts tr.share td { border-bottom: none; } .bar { height: 6px; background: var(--soft); border-radius: 3px;
         margin-top: 3px; } .bar > span { display: block; height: 100%; border-radius: 3px; background: var(--accent); }
.facts dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 0; font-size: 13px; }
.facts dd { margin: 0; font-variant-numeric: tabular-nums; }
@media (max-width: 760px) { .facts { grid-template-columns: 1fr; } }
"""
CSS += DATA_CSS


def data_section(comp: cmp.Comparison) -> str:
    """The "Data and training" block at the top: projects, images and objects of each role (and the objects'
    shares), and how the models were trained and judged."""
    roles = (("training", "Training", "learns the weights"),
             ("evaluation", "Evaluation", "chooses each model's epoch"
              + (" · and judges" if comp.judged_on == "evaluation" else "")),
             ("test", "Test", "judges (never used for a choice)"))
    data = comp.data or {}
    counts = {r: data.get(r) or {} for r, _t, _w in roles}
    known = bool(data)
    total = {k: sum(int(c.get(k, 0)) for c in counts.values()) for k in ("images", "complete", "objects")}
    num = lambda v: f"{v:,}" if known else "—"
    out = ["<h2>Data and training</h2><div class='facts'><div class='card'><h3>Data</h3><table><thead><tr><th></th>"]
    out += [f"<th class='n'>{t}</th>" for _r, t, _w in roles] + ["<th class='n'>Total</th></tr></thead><tbody>"]
    out.append("<tr class='use'><td class='k'>Used to</td>" + "".join(f"<td class='n muted'>{_e(w)}</td>"
                                                                     for _r, _t, w in roles)
               + "<td></td></tr>")
    out.append("<tr><td class='k'>Projects</td>" + "".join(f"<td class='n'>{len(comp.projects.get(r, []))}</td>"
                                                           for r, _t, _w in roles)
               + f"<td class='n'>{sum(len(comp.projects.get(r, [])) for r, _t, _w in roles)}</td></tr>")
    for key, label in (("images", f"Images with {_e(comp.group)} outlined"), ("complete", "… of them fully outlined"),
                       ("objects", f"{_e(comp.group)} objects drawn")):
        out.append(f"<tr><td class='k'>{label}</td>" + "".join(f"<td class='n'>{num(int(counts[r].get(key, 0)))}</td>"
                                                               for r, _t, _w in roles)
                   + f"<td class='n'><b>{num(total[key])}</b></td></tr>")
    if known and total["objects"]:
        cells = []
        for r, _t, _w in roles:
            share = 100 * int(counts[r].get("objects", 0)) / total["objects"]
            cells.append(f"<td class='n'><b>{share:.0f} %</b><div class='bar'><span style='width:{share:.1f}%'>"
                         f"</span></div></td>")
        out.append("<tr class='share'><td class='k'>Share of the objects</td>" + "".join(cells)
                   + "<td class='n'>100 %</td></tr>")
    out.append("</tbody></table>")
    if not known:
        out.append("<p class='muted'>Images and objects: not recorded by the version that made this run.</p>")
    out.append("</div>")
    s = comp.settings
    epochs, steps, batch = int(s.get("epochs", 0)), int(s.get("steps", 0)), int(s.get("batch_size", 0))
    tuned = any(e.kind == "finetune" for e in comp.entries)
    rows = [("Epochs", f"{epochs}"), ("Steps per epoch", f"{steps}"),
            ("Patches per step", f"{batch} (512 × 512 px)"),
            ("Patches seen", f"{epochs * steps * batch:,} per model"),
            ("Works at", f"{comp.um_per_px:g} µm/px" + (" (new models; fine-tuned: their own)" if tuned else "")),
            ("Augmentation", _e(augmentation(s))), ("Seed", f"{s.get('seed', '—')}"),
            ("Judged on", f"the {comp.judged_on} projects"),
            ("Proposed by", _e(cmp.METRICS[comp.criterion][0])),
            ("Compared where", _e(seg.AREAS.get(comp.area, comp.area)) + f" (margin {comp.margin_um:g} µm)"),
            ("Trained on", _e(comp.training_device or "—"))]
    out.append("<div class='card'><h3>Training</h3><dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
               + "</dl></div></div>")
    return "".join(out)


def augmentation(s: dict) -> str:
    """A training run's augmentations in words (runs made before they were settable: all of them, as then)."""
    parts = []
    if s.get("flips", True):
        parts.append("mirrored")
    if s.get("turns", True):
        parts.append("turned by quarter turns")
    intensity = float(s.get("intensity", 0.15) or 0)
    if intensity:
        parts.append(f"brightness and contrast varied by up to ±{100 * intensity:.0f} %")
    jitter = float(s.get("scale_jitter") or 0)
    if jitter:
        parts.append(f"resized by up to ±{100 * jitter:.0f} %")
    return ", ".join(parts) if parts else "none"


PROJECTS = {"test": "test projects", "evaluation": "evaluation projects", "all": "projects (all of them)"}


def projects_judged(comp: cmp.Comparison) -> str:
    """The projects an evaluation judged, in words: e.g. "test projects of group Hospital B"."""
    on = comp.judged_on
    return PROJECTS[on] + (f" of group {comp.test_group}" if on == "test" and comp.test_group else "")


def evaluation_section(comp: cmp.Comparison) -> str:
    """An evaluation's block at the top: the projects judged (images, objects), how, and the models' origins."""
    on = comp.judged_on
    c = (comp.data or {}).get(on) or {}
    num = lambda k: f"{int(c[k]):,}" if k in c else "—"
    rows = [("Projects", f"{len(comp.projects.get(on, []))}"), (f"Images with {_e(comp.group)} outlined", num("images")),
            ("… of them fully outlined", num("complete")), (f"{_e(comp.group)} objects drawn", num("objects"))]
    out = [f"<h2>Data and models</h2><div class='facts'><div class='card'><h3>Judged on the {_e(projects_judged(comp))}</h3>"
           "<table><tbody>"]
    out += [f"<tr><td class='k'>{k}</td><td class='n'>{v}</td></tr>" for k, v in rows]
    out.append("</tbody></table></div>")
    how = [("Proposed by", _e(cmp.METRICS[comp.criterion][0])),
           ("Compared where", _e(seg.AREAS.get(comp.area, comp.area)) + f" (margin {comp.margin_um:g} µm)"),
           ("Match at", f"IoU ≥ {comp.iou_min:g}"), ("Image type", _e(comp.modality or "—"))]
    out.append("<div class='card'><h3>How</h3><dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in how)
               + "</dl></div></div>")
    out.append("<h3>The models</h3><div class='scroll'><table><thead><tr><th>Model</th><th>Architecture</th>"
               "<th>Started from</th><th class='n'>Works at</th><th>Trained</th><th class='n'>Training images</th>"
               "<th>Learned from the judged projects?</th></tr></thead><tbody>")
    for e in comp.entries:
        o = e.origin or {}
        run = f" (run '{o['run']}')" if o.get("run") else ""
        learned = ("not said by its file" if o.get("trained_on") is None else
                   ", ".join(o.get("learned_from_judged") or []) or "no")
        out.append(f"<tr><td>{_e(e.label)}<br><span class='muted'>{_e(o.get('file', e.path))}</span></td>"
                   f"<td>{_e(o.get('architecture') or e.arch_id or '—')}</td><td>{_e(o.get('weights') or '—')}</td>"
                   f"<td class='n'>{_f(e.um_per_px, '{:.4g}')} µm/px</td>"
                   f"<td>{_e((o.get('trained') or '—')[:10])}{_e(run)}</td>"
                   f"<td class='n'>{_e(o['images'] if o.get('images') is not None else '—')}</td>"
                   f"<td>{'<b>' + _e(learned) + '</b>' if o.get('learned_from_judged') else _e(learned)}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def per_group_section(comp: cmp.Comparison, done: list, groups: dict, colours: dict) -> str:
    """Each test group apart: the criterion side by side (with its interval), then every measure per group.

    Args:
        groups: {test group: [project names]}.
    """
    crit = comp.criterion
    label, fn, _higher, fmt = cmp.METRICS[crit]

    def images(e, names) -> list:
        ev = e.judged("test")
        return [c for p, _n, c in ev.per_image if 0 <= p < len(ev.projects) and ev.projects[p] in names]

    out = [f"<h2>Per test group</h2><p class='muted'>Each test group is a final check of its own — most telling when "
           f"it differs on purpose (another microscope setup, another lab): how far the models carry over. "
           f"{_e(label)} with its 95 % interval (bootstrap over the group's images).</p>"
           "<div class='scroll'><table><thead><tr><th>Model</th><th class='n'>All test projects</th>"
           + "".join(f"<th class='n'>{_e(g)}<br><span class='muted'>{len(n)} project(s)</span></th>"
                     for g, n in groups.items()) + "</tr></thead><tbody>"]
    for e in done:
        cells = [f"<td class='n'>{_f(comp.score(e), fmt)}</td>"]
        for names in groups.values():
            per = images(e, set(names))
            v = fn(cmp.pooled(per)) if per else math.nan
            cells.append(f"<td class='n'>{_f(v, fmt)}{_ci(*cmp.bootstrap(per, fn, seed=1), fmt)}</td>")
        out.append(f"<tr><td><span class='swatch' style='background:{colours[e.run_id]}'></span>{_e(e.label)}</td>"
                   + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    for g, names in groups.items():
        out.append(f"<h3>{_e(g)}</h3><p class='muted'>{_e(', '.join(names))}</p><div class='scroll'><table><thead>"
                   "<tr><th>Model</th>" + "".join(f"<th class='n'>{_e(cmp.METRICS[k][0])}</th>" for k in SUMMARY)
                   + "<th class='n'>Hand / found objects</th></tr></thead><tbody>")
        for e in done:
            per = images(e, set(names))
            total = cmp.pooled(per)
            cells = []
            for k in SUMMARY:
                f_k, fmt_k = cmp.METRICS[k][1], cmp.METRICS[k][3]
                cells.append(f"<td class='n'>{_f(f_k(total), fmt_k) if per else '—'}"
                             f"{_ci(*cmp.bootstrap(per, f_k, seed=1), fmt_k) if per else ''}</td>")
            out.append(f"<tr><td><span class='swatch' style='background:{colours[e.run_id]}'></span>{_e(e.label)}"
                       f"</td>" + "".join(cells) + f"<td class='n'>{total.hand} / {total.found}</td></tr>")
        out.append("</tbody></table></div>")
    return "".join(out)


def build(comp: cmp.Comparison) -> str:
    """The comparison report as one HTML string (see the module docstring)."""
    from modalytics.training import architectures as ar
    from modalytics.training import encoders as en
    archs, weights_lib = ar.ArchitectureLibrary(), en.EncoderLibrary()
    on = comp.judged_on
    done = comp.done()
    colours = {e.run_id: PALETTE[i % len(PALETTE)] for i, e in enumerate(comp.entries)}
    best = comp.best()
    crit_label = cmp.METRICS[comp.criterion][0]
    evaluation = comp.purpose == "evaluation"                # existing models judged: no training to report
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>{_e(comp.name)} — "
           f"{'model evaluation' if evaluation else 'architecture comparison'}</title>"
           f"<style>{CSS}</style></head><body><main>"]

    # 1. Header
    roles = comp.projects
    trained = [e for e in comp.entries if e.kind != "reference"]
    several = len(comp.entries) > 1
    if evaluation:
        out.append(f"<h1>{_e(comp.name)}: segmentation model{'s' if several else ''} judged</h1>"
                   f"<div class='muted'>{_e(comp.group)} on {_e(comp.modality)} images · "
                   + (f"training set {_e(comp.training_set)} · " if comp.training_set else "")
                   + f"{len(comp.entries)} existing model{'s' if several else ''}, none trained here · on the "
                   f"{_e(projects_judged(comp))} · started {_e(comp.started.replace('T', ' '))}, finished "
                   f"{_e(comp.finished.replace('T', ' '))} · Modalytics {_e(comp.versions.get('modalytics', ''))}"
                   f"</div>")
        if on != "test":
            out.append("<div class='warn'>⚠ " + ("Judged on all the projects: a model trained on some of them is "
                       "judged partly on its own training images, and looks better than it is. "
                       if on == "all" else "Judged on the evaluation projects: a model trained with this set chose "
                       "its best epoch on them, and looks somewhat better than it is. ")
                       + "Mark a few projects as Test (Training › Data) for numbers fit to compare.</div>")
        for e in comp.entries:
            learned = (e.origin or {}).get("learned_from_judged") or []
            if learned:
                out.append(f"<div class='warn'>⚠ {_e(e.label)} learned from {len(learned)} of the judged projects "
                           f"({_e(', '.join(learned))}): its numbers there are flattering.</div>")
    else:
        out.append(f"<h1>{_e(comp.name)}: segmentation model{'s compared' if several else ' trained'}</h1>"
                   f"<div class='muted'>{_e(comp.group)} on {_e(comp.modality)} images at {comp.um_per_px:g} µm/px · "
                   + (f"training set {_e(comp.training_set)} · " if comp.training_set else "")
                   + f"{len(trained)} model{'s' if len(trained) != 1 else ''} trained"
                   + (f" (+ {len(comp.entries) - len(trained)} as it was, for reference)" if several and
                      len(comp.entries) > len(trained) else "")
                   + f" · started {_e(comp.started.replace('T', ' '))}, finished "
                   f"{_e(comp.finished.replace('T', ' '))} · Modalytics {_e(comp.versions.get('modalytics', ''))}</div>")
        if on == "evaluation":
            out.append("<div class='warn'>⚠ No test projects: the numbers below come from the evaluation projects, which "
                       "also chose each model's best epoch. They are somewhat optimistic; mark a few projects as Test "
                       "(Training › Data) for numbers fit to report.</div>")
    if best is not None and len(done) > 1:
        out.append(f"<div class='good'>Proposed: <b>{_e(best.label)}</b> — {_e(crit_label)} "
                   f"{_f(comp.score(best))}. The proposal is the best by that measure; the table and the "
                   f"differences below say whether it is clearly better, and what it costs in speed and size. "
                   f"You choose which model to keep.</div>")
    failed = [e for e in comp.entries if e.error]
    for e in failed:
        out.append(f"<div class='warn'>✖ {_e(e.label)}: {_e(e.error)}</div>")
    out.append(evaluation_section(comp) if evaluation else data_section(comp))

    # 2. Results
    out.append(f"<h2>Results on the {on} projects</h2>")
    out.append("<p class='muted'>95 % intervals: bootstrap over images (precision and recall: Wilson). Higher is "
               "better, except the Hausdorff distance and HD95 (µm). Bold: the best value of a column.</p>")
    stats = {}
    for e in done:
        ev = e.judged(on)
        per = [c for _p, _n, c in ev.per_image]
        stats[e.run_id] = {k: (cmp.METRICS[k][1](ev.total), *cmp.bootstrap(per, cmp.METRICS[k][1], seed=1))
                            for k in cmp.METRICS}
    resolutions = evaluation or len({round(e.um_per_px, 4) for e in done if math.isfinite(e.um_per_px)}) > 1
    heads = ["Model"] + [cmp.METRICS[k][0] for k in SUMMARY] + \
        ["Precision", "Recall", "Hand / found objects", "Merges / splits", "ms per tile", "Size", "Parameters"] + \
        ([] if evaluation else ["Training", "Epoch kept"]) + (["Works at"] if resolutions else [])
    out.append("<div class='scroll'><table><thead><tr>" + "".join(f"<th{' class=n' if i else ''}>{_e(h)}</th>"
                                                                    for i, h in enumerate(heads)) + "</tr></thead><tbody>")
    bests = {}
    for k in SUMMARY:
        vals = [(stats[e.run_id][k][0], e.run_id) for e in done if math.isfinite(stats[e.run_id][k][0])]
        if vals:
            bests[k] = (max if cmp.METRICS[k][2] else min)(vals)[1]
    for e in done:
        ev = e.judged(on)
        c = ev.total
        row = [f"<td><span class='swatch' style='background:{colours[e.run_id]}'></span>{_e(e.label)}</td>"]
        for k in SUMMARY:
            v, lo, hi = stats[e.run_id][k]
            fmt = cmp.METRICS[k][3]
            cell = _f(v, fmt) + _ci(lo, hi, fmt)
            row.append(f"<td class='n'>{'<b>' + cell + '</b>' if bests.get(k) == e.run_id else cell}</td>")
        (plo, phi), (rlo, rhi) = c.precision_ci, c.recall_ci
        row += [f"<td class='n'>{_f(c.precision)}{_ci(plo, phi)}</td>", f"<td class='n'>{_f(c.recall)}{_ci(rlo, rhi)}</td>",
                f"<td class='n'>{c.hand} / {c.found}</td>", f"<td class='n'>{c.merges} / {c.splits}</td>",
                f"<td class='n'>{_f(e.ms_per_tile, '{:.0f}')}</td>", f"<td class='n'>{_f(e.size_mb, '{:.0f}')} MB</td>",
                f"<td class='n'>{_f(e.params_m, '{:.1f}')} M</td>"]
        if not evaluation:
            row += [f"<td class='n'>{'—' if e.kind == 'reference' else f'{e.seconds / 60:.0f} min'}</td>",
                    f"<td class='n'>{'not retrained' if e.kind == 'reference' else f'{e.best_epoch} / {len(e.history)}'}"
                    f"</td>"]
        row += [f"<td class='n'>{_f(e.um_per_px, '{:.4g}')} µm/px</td>"] if resolutions else []
        out.append(f"<tr{' class=best' if best is e else ''}>" + "".join(row) + "</tr>")
    out.append("</tbody></table></div>")
    if done:
        out.append(f"<p class='muted'>Speed: ONNX Runtime on {_e(done[0].device)}, one 512 × 512 tile, as Segment "
                   f"runs the model" + ("" if evaluation else f" (training ran on {_e(comp.training_device)})")
                   + ".</p>")

    # 3. Against the proposal
    if best is not None and len(done) > 1:
        out.append(f"<h2>Against the proposal ({_e(best.label)})</h2>")
        out.append("<p class='muted'>Paired bootstrap over the same images: the difference (proposal − other) and its "
                   "95 % interval. An interval that contains 0: the two are not distinguishable on these images.</p>")
        keys = [comp.criterion] + [k for k in FIGURE if k != comp.criterion]
        out.append("<table><thead><tr><th>Architecture</th>" + "".join(f"<th class='n'>{_e(cmp.METRICS[k][0])}</th>"
                                                                      for k in keys) + "<th>Verdict</th></tr></thead><tbody>")
        bper = {(p, n): c for p, n, c in best.judged(on).per_image}
        for e in done:
            if e is best:
                continue
            other = {(p, n): c for p, n, c in e.judged(on).per_image}
            both = [key for key in bper if key in other]          # paired: the images both judged
            cells, verdict = [], ""
            for k in keys:
                d, lo, hi = cmp.paired_difference([bper[x] for x in both], [other[x] for x in both],
                                                  cmp.METRICS[k][1], seed=2)
                cells.append(f"<td class='n'>{_f(d, '{:+.3f}')}{_ci(lo, hi, '{:+.3f}')}</td>")
                if k == comp.criterion:
                    higher = cmp.METRICS[k][2]                  # proposal − other: > 0 good when higher is better
                    if not (math.isfinite(lo) and math.isfinite(hi)):
                        verdict = "too few images to tell"
                    elif (lo > 0) if higher else (hi < 0):
                        verdict = "the proposal is clearly better"
                    elif (hi < 0) if higher else (lo > 0):
                        verdict = "this one is clearly better"
                    else:
                        verdict = "not distinguishable"
            out.append(f"<tr><td><span class='swatch' style='background:{colours[e.run_id]}'></span>{_e(e.label)}</td>"
                       + "".join(cells) + f"<td>{_e(verdict)}</td></tr>")
        out.append("</tbody></table>")

    # 4. Figures
    out.append("<h2>Figures</h2><div class='figs'>")
    for k in FIGURE:
        out.append(dot_ci_chart([(e.label, colours[e.run_id], *stats[e.run_id][k]) for e in done],
                                f"{cmp.METRICS[k][0]} (95 % interval)"))
    marks = {e.label: e.best_epoch for e in comp.entries if e.history}
    out.append(line_chart([(e.label, colours[e.run_id], [(r["epoch"], r["loss"]) for r in e.history])
                           for e in comp.entries if e.history], "Training loss per epoch", "loss", marks=marks))
    out.append(line_chart([(e.label, colours[e.run_id], [(r["epoch"], r["dice"]) for r in e.history])
                           for e in comp.entries if e.history], "Evaluation Dice per epoch (○ the epoch kept)",
                          "Dice", marks=marks))
    out.append(scatter_chart([(e.label, colours[e.run_id], e.ms_per_tile, comp.score(e), e.params_m) for e in done],
                             f"{crit_label} against speed (point size: parameters)", "ms per 512 × 512 tile",
                             crit_label))
    out.append(strip_chart([(e.label, colours[e.run_id], [cmp.METRICS[comp.criterion][1](c)
                                                           for _p, _n, c in e.judged(on).per_image]) for e in done],
                           f"{crit_label} per image (│ median)"))
    out.append(error_bars([(e.label, e.judged(on).total) for e in done], "Objects: found, missed, invented"))
    out.append("</div>")

    # 5. Per test group: each a final check apart
    groups = {g: names for g, names in (comp.test_groups or {}).items() if names}
    if on == "test" and len(groups) > 1 and done:
        out.append(per_group_section(comp, done, groups, colours))

    # 6. Per project
    if done:
        projects = done[0].judged(on).projects
        out.append(f"<h2>Per project ({_e(crit_label)})</h2><div class='scroll'><table><thead><tr><th>Project</th>"
                   + "".join(f"<th class='n'>{_e(e.label)}</th>" for e in done) + "</tr></thead><tbody>")
        for pi, name in enumerate(projects):
            cells = []
            for e in done:
                per = e.judged(on).per_project
                v = cmp.METRICS[comp.criterion][1](per[pi]) if pi < len(per) else math.nan
                cells.append(f"<td class='n'>{_f(v)}</td>")
            out.append(f"<tr><td>{_e(name)}</td>" + "".join(cells) + "</tr>")
        out.append("</tbody></table></div>")

    # 7. Examples
    if done and done[0].judged(on).examples:
        out.append("<h2>Examples</h2><p class='muted'>Green (wide): the hand outlines; magenta (thin): the model's.</p>")
        for i, ex in enumerate(done[0].judged(on).examples):
            out.append(f"<h3>{_e(ex.project)} › {_e(ex.image)}</h3><div class='examples'>")
            for e in done:
                exs = e.judged(on).examples
                if i < len(exs):
                    out.append(f"<figure>{example_picture(exs[i], exs[i].found)}<figcaption>{_e(e.label)}"
                               f"</figcaption></figure>")
            out.append("</div>")

    # 8. Methods
    s = comp.settings
    out.append("<h2>Methods</h2>")
    if evaluation:
        out.append(f"<h3>Data</h3><p>Projects with {_e(comp.group)} outlined by hand, judged on: "
                   f"{_e(', '.join(roles.get(on, [])) or '—')} (the {_e(PROJECTS[on])} of the training set"
                   + (f" {_e(comp.training_set)}" if comp.training_set else "") + "). The models were not trained "
                   f"here: each runs as it is, at its own resolution (the table above). Where the outlines are "
                   f"complete: {_e(seg.AREAS.get(comp.area, comp.area))} (margin {comp.margin_um:g} µm); unreviewed "
                   f"suggestions are left out.</p>")
    else:
        out.append(f"<h3>Data</h3><p>Projects with {_e(comp.group)} outlined by hand. Training: "
                   f"{_e(', '.join(roles['training']) or '—')}. Evaluation: "
                   f"{_e(', '.join(roles['evaluation']) or '—')}. Test: {_e(', '.join(roles['test']) or '—')}. "
                   f"Whole projects (one sample each) belong to one role only. Where the outlines are complete: "
                   f"{_e(seg.AREAS.get(comp.area, comp.area))} (margin {comp.margin_um:g} µm); unreviewed suggestions "
                   f"are left out.</p>")
    if comp.skipped:
        out.append(f"<p class='muted'>Left out: {_e('; '.join(comp.skipped[:20]))}"
                   f"{' …' if len(comp.skipped) > 20 else ''}</p>")
    if not evaluation:
        used = list(dict.fromkeys(e.weights_name for e in comp.entries))
        new = [e for e in comp.entries if e.kind == "new"]
        tuned = [e for e in comp.entries if e.kind == "finetune"]
        used = list(dict.fromkeys(e.weights_name for e in new))
        start = ((f"new models: encoder initialised with {_e(' or '.join(used))} weights (see Starting weights), "
                  f"decoder at random" if new else "")
                 + ("; " if new and tuned else "")
                 + ("fine-tuned models: every weight starts from the existing model (its architecture, image type and "
                    "µm/px kept), learning rate 0.0001; its starting point is judged as it was, for reference"
                    if tuned else ""))
        out.append(f"<h3>Training</h3><p>Every model trained the same way — {start}; {s.get('epochs')} epochs "
                   f"of {s.get('steps')} steps of "
                   f"{s.get('batch_size')} patches of 512 × 512 px at {comp.um_per_px:g} µm/px (70 % centred on an "
                   f"object), augmented: {_e(augmentation(s))}"
                   f"; every image resampled to {comp.um_per_px:g} µm/px from its own pixel size; loss: binary "
                   f"cross-entropy + soft Dice; AdamW, learning "
                   f"rate {_e(s.get('lr') or (0.001 if new else 0.0001))}{' (new models)' if new and tuned else ''} "
                   f"falling as a cosine; mixed precision on CUDA; seed {s.get('seed')}. "
                   f"After each epoch the evaluation projects were segmented and the epoch with the best pixel Dice "
                   f"kept.</p>")
    counts = (comp.data or {}).get(on) or {}
    where = {"image": "<p><b>Where</b>: the whole of every image (everything was outlined).</p>",
             "hull": f"<p><b>Where</b>: only around the hand outlines (their convex hull widened by "
                     f"{comp.margin_um:g} µm). A model object counts only when at least half of it lies there, so "
                     f"glands found farther away are not counted as invented: precision is higher than a whole-image "
                     f"judgement would give.</p>",
             "marked": f"<p><b>Where</b>: the whole image when its annotation is marked complete; otherwise only "
                       f"around the hand outlines (their convex hull widened by {comp.margin_um:g} µm), where a model "
                       f"object counts only when at least half of it lies there — glands found farther away are not "
                       f"counted as invented, so precision there is higher than a whole-image judgement would give"
                       + (f". Here {counts.get('complete', 0)} of the {counts['images']} {_e(on)} images were judged whole"
                          if counts.get("images") else "") + ".</p>"}.get(comp.area, "")
    out.append("<h3>How the models were judged</h3><p>Each model segments the images as Segment does (overlapping "
               "tiles, blended), probability ≥ 0.5, outlines as vectors; compared with the hand outlines on the "
               "vectors themselves (µm²). A model object matches a hand object when their IoU ≥ "
               f"{comp.iou_min:g} (one to one, best first).</p>" + where + "<ul>"
               "<li><b>Object precision, recall, F1</b>: TP / (TP + FP), TP / (TP + FN), 2TP / (2TP + FP + FN), TP "
               "the matches; Wilson intervals.</li>"
               "<li><b>Object F1 (GlaS)</b>: the challenge's rule — a model object is found when it covers ≥ 50 % of "
               "the hand object it overlaps most (measured on the hand object only, so an outline much larger than "
               "the gland still counts); 2TP / (hand objects + model objects). More lenient than the F1 above; the "
               "one to set beside published gland results.</li>"
               "<li><b>Panoptic quality</b> PQ = SQ · RQ, SQ the matches' mean IoU, RQ = F1.</li>"
               "<li><b>Object Dice and object Hausdorff</b> (GlaS, Sirinukunwattana et al. 2017): each object against "
               "the one overlapping it most, weighted by area, both directions averaged; pooled over all objects.</li>"
               "<li><b>Pixel Dice and IoU</b> of all hand against all model outlines.</li>"
               f"<li><b>HD95 and NSD</b> of the matched objects only: the 95th percentile of the "
               f"distances between their outlines, and the share of both outlines within τ = "
               f"{seg.NSD_TOLERANCE_UM:g} µm of the other. They describe how well a found gland is outlined, not "
               f"how many are found (missed and invented ones do not enter): read them with the F1.</li>"
               "<li><b>Merges</b>: a model object covering ≥ 50 % of two hand objects; <b>splits</b>: a hand object "
               "with ≥ 50 % of two model objects inside.</li></ul>"
               + _cite("glas"))
    out.append(f"<h3>Statistics</h3><p>95 % intervals by bootstrap over images ({cmp.BOOTSTRAP} draws), "
               f"each draw pooling its images' counts; differences between models by the paired bootstrap "
               f"(the same draws for both). The proposal is the best by {_e(crit_label)} (ties: the faster).</p>")
    if not evaluation:
        out.append("<h3>Architectures</h3><table><thead><tr><th>Architecture</th><th>Decoder · encoder</th><th>What</th>"
                   "<th>References</th></tr></thead><tbody>")
        for aid in dict.fromkeys(e.arch_id for e in comp.entries):
            spec = archs.get(aid)
            if spec is None:
                out.append(f"<tr><td>{_e(aid)}</td><td></td><td></td><td></td></tr>")
                continue
            parts = "own network" if spec.custom else f"{spec.decoder} · {spec.encoder}"
            refs = "<br>".join(_e(line) for line in spec.references_text().splitlines())
            out.append(f"<tr><td>{_e(spec.name)}</td><td>{_e(parts)}</td><td>{_e(spec.description)}</td>"
                       f"<td class='muted'>{refs}</td></tr>")
        out.append("</tbody></table>")
        out.append("<h3>Starting weights</h3><table><thead><tr><th>Weights</th><th>What</th><th>Licence</th>"
                   "<th>References</th></tr></thead><tbody>")
        for wkey in dict.fromkeys(e.weights for e in comp.entries):
            ws = weights_lib.get(wkey)
            if wkey.startswith("model:"):
                name_ = next(e.weights_name for e in comp.entries if e.weights == wkey)
                out.append(f"<tr><td>{_e(name_)}</td><td>An existing model: the fine-tuned model starts from all of its "
                           f"weights (encoder and decoder)</td><td>as the model's own</td><td></td></tr>")
                continue
            if wkey == "scratch":
                out.append("<tr><td>Random weights</td><td>No starting point: every weight drawn at random</td><td>—</td>"
                           "<td></td></tr>")
                continue
            if ws is None:
                out.append(f"<tr><td>{_e(wkey)}</td><td></td><td></td><td></td></tr>")
                continue
            refs = "<br>".join(_e(line) for line in ws.references_text().splitlines())
            out.append(f"<tr><td>{_e(ws.name)}</td><td>{_e(ws.description)}</td><td>{_e(ws.licence)}</td>"
                       f"<td class='muted'>{refs}</td></tr>")
        out.append("</tbody></table>")
    out.append("<h3>Software and hardware</h3><p>" + " · ".join(f"{_e(k)} {_e(v)}" for k, v in comp.versions.items())
               + ("" if evaluation else f" · training on {_e(comp.training_device)}")
               + f"; judging and timing with ONNX Runtime on "
                 f"{_e(done[0].device if done else '—')}.</p>")
    out.append(f"<p class='muted'>Report made {dt.datetime.now():%Y-%m-%d %H:%M} by Modalytics. This report"
               + (" and the results file (every count, per image; Results … .json) are in the evaluation's folder; "
                  "the models stay where they are (their files, in the table above).</p>" if evaluation else
                  ", the results file (every count, per image; Results … .json) and the models (models/) are in the "
                  "run's folder.</p>"))
    out.append("</main></body></html>")
    return "".join(out)


def write(comp: cmp.Comparison, path: str) -> str:
    from modalytics import fileio
    fileio.write_text(path, build(comp))
    return path
