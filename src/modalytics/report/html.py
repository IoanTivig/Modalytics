"""
A project's report: one self-contained HTML file (Qt-free).

Everything is inside the file — pictures as embedded JPEG, plots as inline SVG, a
few lines of script for sorting and filtering — so it opens offline in any browser,
can be sent as it is, and prints to PDF (the browser's Print › Save as PDF) on
A4 pages without the interactive controls.

Sections:

1. **Header** — project, date, profile (name, revision), Modalytics version; warnings
   when the results are out of date, suggestions are still waiting for review, or
   unreviewed suggestions were measured.
2. **Summary** — objects per class; per class (each grade of a graded group) the
   report's key parameters (``Profile.report``): n, mean ± SD, median and
   interquartile range.
3. **Classification** (when the objects were classified) — predicted grades against
   the annotated ones (confusion matrix, agreement with its 95 % interval, Cohen's κ)
   and the share of the highest grade per image and in the project.
4. **Images** — each image with its outlines in their class colours and every object
   numbered as in the objects table (a red number: predicted grade ≠ annotated), a
   scale bar; the image's class rows and image parameters.
5. **Objects** — one row per object: image, number, class, predicted grade and score,
   key parameters; sortable and filterable on screen.
6. **Distributions** — each key parameter per class (box plot with the objects as
   points), with the classifier's cut-off where it has one.
7. **Methods** — the profile's processing, the segmentation models (name, origin,
   how they were trained and evaluated), the classifier and its validation, and each
   key parameter's description with its formula and references.

**The PDF summary** (``build(pdf=True)``, Report › PDF…): paper cannot scroll sideways, so
the same sections with fewer columns — the parameters ticked *In PDF* in the profile, or by
default the classifier's then the key ones, at most ``PDF_MAX`` (``pdf_keys``) — and an
objects list without parameter columns: image, number, class, predicted grade, confidence
and the classifier's reason as text (on screen a tooltip). Every value stays in the HTML
report and the Results CSV.
"""

from __future__ import annotations

import base64
import datetime as dt
import getpass
import html
import math
import os
from dataclasses import dataclass

import numpy as np

OVERVIEW_PX = 1400            # longest side of an image's picture
MAX_NUMBERS = 400             # above this many objects on one image, the numbers are left out of its picture
PDF_MAX = 6                   # parameters in the PDF summary, when the profile ticks none


def _e(text) -> str:
    return html.escape(str(text))


def _num(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}g}"         # 22,830 rather than 2.283e+04


def _pct(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{100 * v:.0f} %"


# ---------------------------------------------------------------- what the report is made from
@dataclass
class Key:
    """A key parameter: its id, label, unit, scope and help text (formula, references); ``shape``: a shape
    parameter (left out for partial objects, whose shape is incomplete)."""

    uid: str
    label: str
    unit: str
    scope: str
    help: str
    shape: bool = False

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def key_parameters(project, library, uids: list[str] | None = None) -> list[Key]:
    """The profile's report parameters (or ``uids``) that were measured (in that order)."""
    prof = project.profile
    measured = {c.uid: (scope, c) for scope, cols in project.results.columns.items() for c in cols}
    out = []
    for uid in (uids if uids is not None else prof.report_parameters if prof is not None else []):
        if uid in measured:
            scope, col = measured[uid]
            spec = library.get(uid) if library is not None else None
            out.append(Key(uid, col.label, col.unit, scope, spec.help_text() if spec is not None else col.help,
                           shape=spec is not None and spec.category == "shape"))
    return out


def pdf_keys(project, library) -> list[Key]:
    """The PDF summary's parameters: those ticked *In PDF* in the profile; else the classifiers' parameters, then
    the key parameters, at most ``PDF_MAX``."""
    prof = project.profile
    if prof is not None and prof.pdf_parameters:
        return key_parameters(project, library, prof.pdf_parameters)
    chosen: list[str] = []
    for g in (prof.classification if prof is not None else {}):
        clf = prof.classifier_for(g)
        chosen += [u for u in (clf.parameters if clf is not None else []) if u not in chosen]
    chosen += [u for u in (prof.report_parameters if prof is not None else []) if u not in chosen]
    return key_parameters(project, library, chosen)[:PDF_MAX]


def object_numbers(results) -> dict:
    """{(image id, mask id, index): number}: objects numbered 1, 2, … per image, in the results' order."""
    out, counter = {}, {}
    for r in results.of_scope("object"):
        counter[r.image_id] = counter.get(r.image_id, 0) + 1
        out[(r.image_id, r.mask_id, r.index)] = counter[r.image_id]
    return out


def _class_order(project, rows) -> list[tuple[str, str]]:
    """Classes in the profile's order (group, then its subgroups), then any others found."""
    out = []
    prof = project.profile
    for g in (prof.classes.groups if prof is not None else []):
        out.append((g.name, ""))
        out += [(g.name, s.name) for s in g.subgroups]
    for r in rows:
        if (r.group, r.subgroup) not in out:
            out.append((r.group, r.subgroup))
    return out


def _label(group: str, sub: str) -> str:
    return f"{group} › {sub}" if sub else group


def _stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else math.nan,
            "median": float(med), "q1": float(q1), "q3": float(q3), "min": float(v.min()), "max": float(v.max())}


STATS = (("Mean", "mean"), ("SD", "sd"), ("Median", "median"), ("IQR", None))


def summary_table(keys: list[Key], objs: list, order: list, colours: dict) -> str:
    """The key parameters (rows) of each class (a block of columns: mean, SD, median, IQR).

    A group whose objects all have one subgroup is left out (its subgroup's block says the same); a group with
    several subgroups also gets an "(all)" block. Returns the HTML table (scrolls sideways on a narrow screen).
    """
    blocks = []
    for c in order:
        rows = [r for r in objs if (r.group, r.subgroup) == c or (not c[1] and r.group == c[0])]
        if not rows or (not c[1] and all(r.subgroup for r in rows) and len({r.subgroup for r in rows}) == 1):
            continue
        blocks.append((c, rows))
    head1 = "<tr><th rowspan='2' class='param stick'>Parameter</th>" + "".join(
        f"<th colspan='{len(STATS)}' class='cls{' alt' if b % 2 else ''}'><span class='swatch' "
        f"style='background:{colours.get(c, '#999')}'></span>{_e(_label(*c))}{'' if c[1] else ' (all)'}"
        f"<span class='muted'> · n = {len(rows)}</span></th>" for b, (c, rows) in enumerate(blocks)) + "</tr>"
    head2 = "<tr>" + "".join(
        f"<th class='n{' first' if i == 0 else ''}{' alt' if b % 2 else ''}'>{name}</th>"
        for b in range(len(blocks)) for i, (name, _k) in enumerate(STATS)) + "</tr>"
    body = []
    for k in keys:
        cells = []
        for b, (_c, rows) in enumerate(blocks):
            st = _stats([r.values.get(k.uid) for r in rows if not (k.shape and r.partial)])
            for i, (_name, key) in enumerate(STATS):
                cls = f"n{' first' if i == 0 else ''}{' alt' if b % 2 else ''}"
                if not st["n"]:
                    text = "—"
                elif key is None:
                    text = f"{_num(st['q1'])}–{_num(st['q3'])}"
                else:
                    text = _num(st[key])
                cells.append(f"<td class='{cls}'>{text}</td>")
        body.append(f"<tr><th class='param stick'>{_e(k.title)}</th>" + "".join(cells) + "</tr>")
    return ("<div class='scroll'><table class='stats'><thead>" + head1 + head2 + "</thead><tbody>" + "".join(body)
            + "</tbody></table></div><p class='muted'>For the objects of each class: mean, standard deviation (SD), "
              "median and interquartile range (IQR: 25th–75th percentile).</p>")


def _colour(project, group: str, sub: str) -> str:
    labels = project.labels
    r, g, b, _a = labels.color(group, sub)
    if project.profile is not None and (group, sub) in project.profile.classes.choices():
        r, g, b, _a = project.profile.classes.color(group, sub)
    for img in project.images:
        for m in img.masks:
            if m.group == group and m.subgroup == sub:
                r, g, b, _a = m.color
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------- pictures
def overview(image, numbers: dict, results_rows: list, colours: dict) -> tuple[str, float]:
    """An image's picture with its outlines and object numbers, as a base64 JPEG; and its µm per pixel."""
    import cv2

    from modalytics.parameters.engine import objects_of
    from modalytics.services.segmentation import read_region
    w, h = image.size_um
    um = max(w, h) / OVERVIEW_PX
    um = max(um, min(image.pixel_size_um))
    pixels, _frame = read_region(image, (0, 0, w, h), um)
    pixels = np.asarray(pixels)
    # 1. The image: RGB as it is, single-channel with its display contrast (else robust percentiles)
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        rgb = pixels[..., :3]
        if rgb.dtype != np.uint8:
            lo, hi = np.nanpercentile(rgb, [0.5, 99.5])
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9) * 255, 0, 255)
        canvas = np.ascontiguousarray(rgb.astype(np.uint8)[..., ::-1])               # BGR for OpenCV
    else:
        g = pixels.squeeze().astype(np.float32)
        lo, hi = image.view_display.clim if hasattr(image, "view_display") else (math.nan, math.nan)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            lo, hi = np.nanpercentile(g, [0.5, 99.5])
        g8 = np.clip((g - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    H, W = canvas.shape[:2]
    scale = 1.0 / um
    thick = max(1, int(round(max(W, H) / 700)))
    # 2. Outlines in their class colours
    by_object = {(r.mask_id, r.index): r for r in results_rows}
    labels = []
    for m in image.masks:
        hexc = colours.get((m.group, m.subgroup), "#33cc66")
        bgr = tuple(int(hexc[i:i + 2], 16) for i in (5, 3, 1))
        for k, poly in enumerate(objects_of(m.geometry)):
            for ring in [poly.exterior, *poly.interiors]:
                pts = np.round(np.asarray(ring.coords) * scale).astype(np.int32)
                cv2.polylines(canvas, [pts], True, bgr, thick, cv2.LINE_AA)
            n = numbers.get((image.id, m.id, k))
            row = by_object.get((m.id, k))
            if n is not None:
                differs = bool(row is not None and row.prediction.get("grade") and row.subgroup
                               and not row.partial and row.prediction["grade"] != row.subgroup)
                labels.append((poly.representative_point().coords[0], n, differs))
    # 3. Numbers (a red badge: predicted grade ≠ annotated)
    if len(labels) <= MAX_NUMBERS:
        size = max(0.55, min(1.1, max(W, H) / 1200))
        for (x, y), n, differs in labels:
            text = str(n)
            (tw, th), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
            cx, cy = int(x * scale), int(y * scale)
            cv2.rectangle(canvas, (cx - tw // 2 - 3, cy - th // 2 - 3), (cx + tw // 2 + 3, cy + th // 2 + 3),
                          (40, 40, 220) if differs else (30, 30, 30), -1)
            cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255),
                        2 if size > 0.8 else 1, cv2.LINE_AA)
    # 4. Scale bar (1-2-5 × 10ⁿ µm, about a fifth of the width)
    from modalytics.render.view import nice_length
    bar_um = nice_length(W * um / 5)
    bar_px = int(bar_um / um)
    x0, y0 = 20, H - 24
    cv2.rectangle(canvas, (x0 - 6, y0 - 24), (x0 + bar_px + 6, y0 + 10), (255, 255, 255), -1)
    cv2.line(canvas, (x0, y0), (x0 + bar_px, y0), (0, 0, 0), 4)
    text = f"{bar_um:g} um" if bar_um < 1000 else f"{bar_um / 1000:g} mm"
    cv2.putText(canvas, text, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])   # a sixth of a PNG's size
    return base64.b64encode(jpg.tobytes()).decode("ascii"), um


def _boxplot(key: Key, groups: list, cutoffs: list, width: int = 640) -> str:
    """An SVG box plot per class with the objects as points; dashed lines at the classifier's cut-offs."""
    every = [x for _n, _c, vals in groups for x in vals]
    if not every:
        return ""
    lo, hi = min(every + [c for c, _t in cutoffs]), max(every + [c for c, _t in cutoffs])
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    row_h, left, right = 34, 210, 20
    height = row_h * len(groups) + 40
    x = lambda v: left + (v - lo) / (hi - lo) * (width - left - right)   # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(key.title)}">']
    rng = np.random.default_rng(0)
    for i, (name, colour, vals) in enumerate(groups):
        y = 16 + i * row_h + row_h / 2
        parts.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="ax">{_e(name)} '
                     f'(n {len(vals)})</text>')
        if not vals:
            continue
        s = _stats(vals)
        parts.append(f'<line x1="{x(s["min"]):.1f}" x2="{x(s["max"]):.1f}" y1="{y}" y2="{y}" stroke="#888"/>')
        parts.append(f'<rect x="{x(s["q1"]):.1f}" y="{y - 9}" width="{max(1, x(s["q3"]) - x(s["q1"])):.1f}" '
                     f'height="18" fill="{colour}" fill-opacity="0.25" stroke="{colour}"/>')
        parts.append(f'<line x1="{x(s["median"]):.1f}" x2="{x(s["median"]):.1f}" y1="{y - 9}" y2="{y + 9}" '
                     f'stroke="{colour}" stroke-width="2.5"/>')
        for v in vals:
            parts.append(f'<circle cx="{x(v):.1f}" cy="{y + rng.uniform(-6, 6):.1f}" r="2.2" fill="{colour}" '
                         f'fill-opacity="0.8"/>')
    for c, text in cutoffs:
        parts.append(f'<line x1="{x(c):.1f}" x2="{x(c):.1f}" y1="8" y2="{height - 26}" stroke="#d33" '
                     f'stroke-dasharray="5 4"/><text x="{x(c) + 4:.1f}" y="14" class="cut">{_e(text)}</text>')
    for t in np.linspace(lo + pad, hi - pad, 5):
        parts.append(f'<text x="{x(t):.1f}" y="{height - 8}" text-anchor="middle" class="ax">{_num(t, 3)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- the report
CSS = """
:root { --fg:#1d232b; --muted:#5f6b7a; --line:#d9dee5; --accent:#1f8a70; --bad:#c23b3b; --bg:#ffffff;
        --soft:#f4f6f8; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0; font-size: 14px; line-height: 1.45; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 24px 60px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 34px 0 10px; border-bottom: 2px solid
var(--accent); padding-bottom: 4px; } h3 { font-size: 15px; margin: 20px 0 6px; }
.muted { color: var(--muted); } .warn { background: #fff4e0; border-left: 4px solid #e0a040; padding: 8px 12px;
margin: 10px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--soft); font-weight: 600; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.differs td { background: #fdeeee; }
/* A wide table keeps the page's width and scrolls sideways (a visible bar); its first column stays in view.
   The objects table also scrolls down inside its frame, with its titles kept at the top, so the sideways bar is
   always on screen. */
.scroll { overflow: auto; max-width: 100%; border: 1px solid var(--line); border-radius: 4px; margin: 6px 0 12px;
          scrollbar-width: auto; scrollbar-color: #9aa5b1 var(--soft); }
.scroll::-webkit-scrollbar { height: 12px; width: 12px; }
.scroll::-webkit-scrollbar-track { background: var(--soft); }
.scroll::-webkit-scrollbar-thumb { background: #9aa5b1; border-radius: 6px; border: 2px solid var(--soft); }
.scroll::-webkit-scrollbar-thumb:hover { background: #6f7b88; }
.scroll.tall { max-height: 70vh; }
.scroll table { width: max-content; min-width: 100%; margin: 0; }
.scroll td, .scroll th { white-space: nowrap; }
.scroll .stick { position: sticky; left: 0; z-index: 1; background: var(--bg); box-shadow: 1px 0 0 var(--line); }
.scroll th.stick { background: var(--soft); }
.scroll.tall thead th { position: sticky; top: 0; z-index: 2; box-shadow: 0 1px 0 var(--line); }
.scroll.tall thead th.stick { z-index: 3; }
table.stats { width: max-content; }
table.stats th.param { text-align: left; white-space: nowrap; }
table.stats th.cls { text-align: center; border-left: 2px solid var(--line); white-space: nowrap; }
table.stats .first { border-left: 2px solid var(--line); }
table.stats .alt { background: #f8f9fb; } table.stats th.alt { background: #eceff3; }
table.stats td { white-space: nowrap; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
figure { margin: 10px 0 18px; } figure img { max-width: 100%; border: 1px solid var(--line); }
figcaption { color: var(--muted); font-size: 12px; margin-top: 4px; }
.controls { display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
.controls input, .controls select { padding: 4px 8px; border: 1px solid var(--line); border-radius: 4px; }
th.sort { cursor: pointer; } th.sort:after { content: " ↕"; color: var(--muted); }
.help { white-space: pre-wrap; font-size: 12.5px; background: var(--soft); padding: 8px 10px; border-radius: 4px; }
svg text.ax { font-size: 11px; fill: var(--muted); } svg text.cut { font-size: 11px; fill: #d33; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
.card { background: var(--soft); border-radius: 6px; padding: 10px 14px; min-width: 150px; }
.card b { display: block; font-size: 20px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
@page { size: A4; margin: 14mm; }
table.objlist td.why { font-size: 10px; color: var(--muted); }
table.objlist tr.imgrow td { font-weight: 600; background: var(--head, #f3f4f6); }
.imggrid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
.imggrid section.image h3 { font-size: 12px; margin: 4px 0; } .imggrid figure { margin: 0 0 6px; }
.imggrid figcaption { font-size: 10px; }
@media print {
  .controls, .noprint { display: none !important; }
  main { max-width: none; padding: 0; } body { font-size: 11px; }
  h2 { break-after: avoid; } section.image, figure, table, svg { break-inside: avoid; }
  section.page, h2.page { break-before: page; }
  table.objlist { break-inside: auto; } table.objlist thead { display: table-header-group; }
  tr { break-inside: avoid; }
  .scroll, .scroll.tall { overflow: visible; max-height: none; border: none; }
  .scroll table { width: 100%; } .scroll td, .scroll th { white-space: normal; }
}
"""

SCRIPT = """
document.querySelectorAll('table.sortable').forEach(function (t) {
  t.querySelectorAll('th.sort').forEach(function (th, i) {
    th.addEventListener('click', function () {
      var body = t.tBodies[0], rows = Array.from(body.rows), asc = th.dataset.asc !== '1';
      rows.sort(function (a, b) {
        var x = a.cells[i].dataset.v ?? a.cells[i].textContent, y = b.cells[i].dataset.v ?? b.cells[i].textContent;
        var nx = parseFloat(x), ny = parseFloat(y);
        var c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : String(x).localeCompare(String(y));
        return asc ? c : -c;
      });
      th.dataset.asc = asc ? '1' : '0';
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
function filterObjects() {
  var q = document.getElementById('q').value.toLowerCase(), c = document.getElementById('cls').value;
  document.querySelectorAll('#objects tbody tr').forEach(function (r) {
    var ok = (!c || r.dataset.cls === c) && (!q || r.textContent.toLowerCase().indexOf(q) >= 0);
    r.style.display = ok ? '' : 'none';
  });
}
"""


def build(project, library=None, progress=None, pdf: bool = False) -> str:
    """The report of a measured project, as one HTML string (see the module docstring).

    Args:
        project: The Project (with ``results``).
        library: The parameter library (descriptions of the key parameters; default: the one here).
        progress: Called with 0–1 while the pictures are made.
        pdf: The PDF summary (see the module docstring): fewer parameters, a compact objects list.

    Raises:
        ValueError: If the project was not measured.
    """
    from modalytics import __version__
    from modalytics.parameters.registry import ParameterLibrary
    from modalytics.training import classifiers as cl
    res = project.results
    if res is None:
        raise ValueError("Measure the project first: the report shows its results")
    library = library or ParameterLibrary()
    prof = project.profile
    keys = pdf_keys(project, library) if pdf else key_parameters(project, library)
    obj_keys = [k for k in keys if k.scope == "object"]
    objs = res.of_scope("object")
    numbers = object_numbers(res)
    order = _class_order(project, objs)
    colours = {c: _colour(project, *c) for c in order}
    name = os.path.splitext(os.path.basename(project.path))[0] if project.path else "Untitled project"
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>{_e(name)} — report</title>"
           f"<style>{CSS}</style></head><body><main>"]

    # 1. Header and warnings
    out.append(f"<h1>{_e(name)}</h1><div class='muted'>Report made {dt.datetime.now():%d %B %Y, %H:%M} by "
               f"{_e(getpass.getuser())} with Modalytics {_e(__version__)} · profile "
               f"<b>{_e(prof.name if prof else '—')}</b>" + (f" (revision {prof.revision})" if prof else "")
               + f" · measured {_e(res.measured.replace('T', ' '))} · {len(project.images)} image(s)</div>")
    warnings = []
    outdated = res.outdated(project)
    if outdated:
        warnings.append("The results are out of date (" + "; ".join(outdated[:3]) + "): measure again for a report "
                        "of the project as it is now.")
    waiting = sum(len(i.suggestions.items) for i in project.images if i.suggestions is not None)
    if waiting:
        warnings.append(f"{waiting} suggestion(s) of a model are still waiting for review; they are "
                        + ("included in these results, marked (suggested)." if res.suggested
                           else "not part of these results."))
    elif res.suggested:
        warnings.append(f"{res.suggested} object(s) measured from unreviewed suggestions are included.")
    out += [f"<div class='warn'>⚠ {_e(w)}</div>" for w in warnings]
    if pdf:
        out.append(f"<p class='muted'>Summary for print: the parameters "
                   f"{_e(', '.join(k.title for k in keys)) or '— none —'} (Settings › Profiles › Parameters, "
                   f"<i>In PDF</i>), and the objects without their parameter values — every value is in the HTML "
                   f"report and the Results CSV export.</p>")

    # 2. Summary
    out.append("<h2>Summary</h2><div class='cards'>")
    counts = {}
    for r in objs:
        counts[(r.group, r.subgroup)] = counts.get((r.group, r.subgroup), 0) + 1
    groups = {}
    for (g, s), n in counts.items():
        groups[g] = groups.get(g, 0) + n
    for g, n in groups.items():
        out.append(f"<div class='card'><b>{n}</b>{_e(g)} object(s)</div>")
    for (g, s), n in counts.items():
        if s:
            out.append(f"<div class='card'><b>{n}</b><span class='swatch' style='background:{colours[(g, s)]}'>"
                       f"</span>{_e(s)}</div>")
    out.append("</div>")
    if obj_keys:
        partial = sum(1 for r in objs if r.partial)
        if partial and any(k.shape for k in obj_keys):
            out.append(f"<p class='muted'>{partial} object(s) with an incomplete shape (cut by the image edge, or "
                       f"marked partial) are left out of the shape parameters' statistics and distributions; they "
                       f"count everywhere else.</p>")
        out.append(summary_table(obj_keys, objs, order, colours))
    else:
        out.append("<p class='muted'>No key parameters chosen: Settings › Profiles › Parameters, column "
                   "<i>Report</i>. All values are in Results and its CSV export.</p>")

    # 3. Classification
    classified = [r for r in objs if r.prediction]
    clfs = {g: prof.classifier_for(g) for g in groups} if prof is not None else {}
    if classified:
        out.append("<h2>Classification</h2>")
        for g, clf in clfs.items():
            rows = [r for r in classified if r.group == g]
            if clf is None or not rows:
                continue
            grades = clf.grades
            out.append(f"<h3>{_e(g)}: {_e(cl.describe(clf))}</h3>")
            pred = {gr: sum(1 for r in rows if r.prediction.get("grade") == gr) for gr in grades}
            out.append("<div class='cards'>" + "".join(
                f"<div class='card'><b>{n}</b>predicted {_e(gr)}</div>" for gr, n in pred.items()) + "</div>")
            known = [r for r in rows if r.subgroup in grades and not r.partial]   # partial: grade less reliable
            left = sum(1 for r in rows if r.subgroup in grades and r.partial)
            if left:
                out.append(f"<p class='muted'>{left} object(s) graded by hand have an incomplete shape (cut by the "
                           f"image edge, or marked partial): their predicted grade is not compared with the "
                           f"annotation.</p>")
            if known:
                a = cl.agreement(np.array([grades.index(r.subgroup) for r in known]),
                                 np.array([grades.index(r.prediction["grade"]) for r in known]), grades,
                                 prof.classes.ordered(g))
                out.append(f"<p>Of the {a.n} objects graded by hand, the classifier gives the same grade to "
                           f"<b>{_pct(a.accuracy)}</b> (95 % CI {_pct(a.accuracy_ci[0])}–{_pct(a.accuracy_ci[1])}), "
                           f"Cohen's κ {a.kappa:.2f}.</p>")
                out.append("<table style='width:auto'><thead><tr><th>Annotated ↓ / predicted →</th>" + "".join(
                    f"<th class='n'>{_e(gr)}</th>" for gr in grades) + "</tr></thead><tbody>" + "".join(
                    f"<tr><th>{_e(gr)}</th>" + "".join(f"<td class='n'>{'<b>' if i == j else ''}{a.confusion[i, j]}"
                                                       f"{'</b>' if i == j else ''}</td>" for j in range(len(grades)))
                    + "</tr>" for i, gr in enumerate(grades)) + "</tbody></table>")
            top = grades[-1]
            out.append(f"<table style='width:auto'><thead><tr><th>Share of {_e(top)}</th><th class='n'>annotated</th>"
                       f"<th class='n'>predicted</th><th class='n'>objects</th></tr></thead><tbody>")
            for img in project.images + [None]:
                sel = [r for r in rows if img is None or r.image_id == img.id]
                if not sel:
                    continue
                ann = [r for r in sel if r.subgroup in grades]
                share_a = sum(r.subgroup == top for r in ann) / len(ann) if ann else math.nan
                share_p = sum(r.prediction.get("grade") == top for r in sel) / len(sel)
                out.append(f"<tr><td>{'<b>Project</b>' if img is None else _e(img.name)}</td><td class='n'>"
                           f"{_pct(share_a)}</td><td class='n'>{_pct(share_p)}</td><td class='n'>{len(sel)}</td></tr>")
            out.append("</tbody></table>")
            if clf.note:
                out.append(f"<p class='muted'>Classifier made from {_e(clf.note)}.</p>")

    # 4. Distributions (the whole project)
    if obj_keys:
        out.append("<h2 class='page'>Distributions</h2><p class='muted'>Each class: box (interquartile range) with "
                   "the median, whiskers to the extremes, every object as a point; dashed red: the classifier's "
                   "cut-off on that parameter.</p>")
        for k in obj_keys:
            series = []
            for c in order:
                if not c[1] and any((r.group == c[0] and r.subgroup) for r in objs):
                    continue                                       # graded group: shown by grade
                vals = [r.values.get(k.uid) for r in objs if (r.group, r.subgroup) == c
                        and not (k.shape and r.partial)]
                vals = [float(v) for v in vals if v is not None and math.isfinite(v)]
                if vals:
                    series.append((_label(*c), colours.get(c, "#999"), vals))
            cut = []
            for clf in clfs.values():
                if clf is None:
                    continue
                conds = [c for rule in clf.rules for c in rule.conditions] + [c.condition for c in clf.criteria]
                cut += [(c.value, f"{'≥' if c.op == '>=' else '≤'} {_num(c.value, 3)}") for c in conds
                        if c.uid == k.uid]
            svg = _boxplot(k, series, cut)
            if svg:
                out.append(f"<h3>{_e(k.title)}</h3>{svg}")

    # 5. Objects (the PDF: a compact list, the classifier's reason as text)
    if pdf:
        out.append(objects_list(objs, numbers, colours))
    else:
        out += _objects_table(objs, numbers, order, counts, colours, obj_keys)

    # 6. Images
    _images(out, project, res, objs, numbers, colours, keys, progress, pdf)

    # 7. Methods
    _methods(out, project, res, prof, keys, cl, __version__)
    return "".join(out)


def objects_list(objs, numbers: dict, colours: dict) -> str:
    """The PDF's objects, grouped by image (its name once): number, class (its grade), predicted grade, confidence and
    the reason as text — narrow, so long lists continue over the pages. Partial objects left ungraded are marked so,
    the reason said once above the table."""
    has_pred = any(r.prediction for r in objs)
    ungraded = [r for r in objs if r.partial and has_pred and not r.prediction.get("grade")]
    why_once = next((r.prediction.get("reason", "") for r in ungraded if r.prediction.get("reason")), "")
    cols = 5 if has_pred else 2
    out = ["<h2 class='page'>Objects</h2><p class='muted'>Numbered as in each image's picture. "
           + ("Predicted: the classifier's grade, its confidence (or Ranges' conditions met, Score's points) and "
              "why; red: it differs from the annotation." if has_pred else "")
           + (f" <b>{len(ungraded)} partial</b> object(s) (cut by the image edge, or marked partial) are not graded: "
              f"{_e(why_once.removeprefix('not graded: '))}." if ungraded else "") + "</p>"
           "<table class='objlist'><thead><tr><th class='n'>#</th><th>Class</th>"
           + ("<th>Predicted</th><th class='n'>Confidence</th><th>Why</th>" if has_pred else "")
           + "</tr></thead><tbody>"]
    image = None
    for r in objs:
        if r.image_name != image:
            image = r.image_name
            out.append(f"<tr class='imgrow'><td colspan='{cols}'>{_e(image)}</td></tr>")
        n = numbers.get((r.image_id, r.mask_id, r.index), "")
        grade = r.prediction.get("grade", "")
        differs = bool(grade and r.subgroup and not r.partial and grade != r.subgroup)
        score = r.prediction.get("score", math.nan)
        cls = (f"<span class='swatch' style='background:{colours.get((r.group, r.subgroup), '#999')}'></span>"
               f"{_e(r.subgroup or r.group)}{' (suggested)' if r.suggested else ''}"
               + (" <span class='muted'>· partial</span>" if r.partial else ""))
        row = f"<td class='n'>{n}</td><td>{cls}</td>"
        if has_pred:
            why = "not graded (partial)" if r in ungraded else r.prediction.get("reason", "")
            row += (f"<td>{_e(grade)}</td><td class='n'>{_num(score, 3) if grade else ''}</td>"
                    f"<td class='why'>{_e(why)}</td>")
        out.append(f"<tr{' class=differs' if differs else ''}>{row}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _objects_table(objs, numbers, order, counts, colours, obj_keys) -> list[str]:
    """The screen's objects table: every key parameter, sortable and filterable."""
    out = []
    out.append("<h2 class='page'>Objects</h2><div class='controls noprint'><input id='q' placeholder='Filter…' "
               "oninput='filterObjects()'><select id='cls' onchange='filterObjects()'><option value=''>All classes"
               "</option>" + "".join(f"<option>{_e(_label(*c))}</option>" for c in order if c in counts)
               + "</select><span class='muted'>Click a column title to sort.</span></div>")
    has_pred = any(r.prediction for r in objs)
    out.append("<div class='scroll tall'><table id='objects' class='sortable'><thead><tr><th class='sort stick'>Image"
               "</th><th class='sort n'>#</th>"
               "<th class='sort'>Class</th>" + ("<th class='sort'>Predicted</th><th class='sort n'>Score</th>"
                                                 if has_pred else "")
               + "".join(f"<th class='sort n'>{_e(k.title)}</th>" for k in obj_keys) + "</tr></thead><tbody>")
    for r in objs:
        n = numbers.get((r.image_id, r.mask_id, r.index), "")
        grade = r.prediction.get("grade", "")
        differs = bool(grade and r.subgroup and not r.partial and grade != r.subgroup)
        score = r.prediction.get("score", math.nan)
        cells = [f"<td class='stick'>{_e(r.image_name)}</td><td class='n' data-v='{n}'>{n}</td>"
                 f"<td><span class='swatch' style='background:{colours.get((r.group, r.subgroup), '#999')}'></span>"
                 f"{_e(r.class_name)}{' (suggested)' if r.suggested else ''}"
                 + (f" <span class='muted' title='{_e(r.partial)}: shape incomplete'>· partial</span>"
                    if r.partial else "") + "</td>"]
        if has_pred:
            cells.append(f"<td title='{_e(r.prediction.get('reason', ''))}'>{_e(grade)}</td>"
                         f"<td class='n' data-v='{score if isinstance(score, (int, float)) else ''}'>"
                         f"{_num(score, 3) if grade else ''}</td>")
        for k in obj_keys:
            v = r.values.get(k.uid)
            cells.append(f"<td class='n' data-v='{v if v is not None else ''}'>{_num(v)}</td>")
        out.append(f"<tr data-cls='{_e(_label(r.group, r.subgroup))}'{' class=differs' if differs else ''}>"
                   + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return out


def _images(out: list, project, res, objs, numbers, colours, keys, progress, pdf: bool = False) -> None:
    """Each image's picture (outlines, numbers, scale bar) with its class rows and image parameters; the PDF: two
    per row, a short caption."""
    out.append("<h2 class='page'>Images</h2>" + ("<p class='muted'>Outlines in their class colours, objects numbered as "
                                                 "in the objects list (red number: predicted grade differs from the "
                                                 "annotation).</p><div class='imggrid'>" if pdf else ""))
    for i, img in enumerate(project.images):
        rows = [r for r in objs if r.image_id == img.id]
        try:
            png, um = overview(img, numbers, rows, colours)
            if pdf:
                pic = (f"<figure><img alt='{_e(img.name)}' src='data:image/jpeg;base64,{png}'><figcaption>"
                       f"{img.size_um[0]:,.0f} × {img.size_um[1]:,.0f} µm · {len(rows)} objects"
                       + (" (too many to number)" if len(rows) > MAX_NUMBERS else "") + "</figcaption></figure>")
            else:
                pic = (f"<figure><img alt='{_e(img.name)}' src='data:image/jpeg;base64,{png}'><figcaption>"
                       f"{_e(img.name)} · {img.size_um[0]:,.0f} × {img.size_um[1]:,.0f} µm · shown at {um:.3g} µm/px · "
                       f"outlines in their class colours, objects numbered as in the objects table"
                       + (" (red number: predicted grade differs from the annotation)"
                          if any(r.prediction for r in rows) else "")
                       + (f" · {len(rows)} objects: too many to number at this size" if len(rows) > MAX_NUMBERS
                          else "") + "</figcaption></figure>")
        except Exception as exc:                                  # e.g. the image's file is not found
            pic = f"<p class='warn'>No picture of {_e(img.name)}: {_e(exc)}</p>"
        out.append(f"<section class='image'><h3>{_e(img.name)}</h3>{pic}")
        class_keys = [k for k in keys if k.scope == "class"]
        crow = [r for r in res.of_scope("class") if r.image_id == img.id]
        if crow and class_keys:
            out.append("<table><thead><tr><th>Class</th>" + "".join(f"<th class='n'>{_e(k.title)}</th>"
                                                                   for k in class_keys) + "</tr></thead><tbody>")
            for r in crow:
                out.append(f"<tr><td>{_e(r.class_name)}</td>" + "".join(
                    f"<td class='n'>{_num(r.values.get(k.uid))}</td>" for k in class_keys) + "</tr>")
            out.append("</tbody></table>")
        image_keys = [k for k in keys if k.scope == "image"]
        irow = next((r for r in res.of_scope("image") if r.image_id == img.id), None)
        if irow is not None and image_keys:
            out.append("<table style='width:auto'><tbody>" + "".join(
                f"<tr><th>{_e(k.title)}</th><td class='n'>{_num(irow.values.get(k.uid))}</td></tr>"
                for k in image_keys) + "</tbody></table>")
        out.append("</section>")
        if progress is not None:
            progress((i + 1) / max(len(project.images), 1))
    if pdf:
        out.append("</div>")


def _methods(out: list, project, res, prof, keys, cl, version: str) -> None:
    """Processing, segmentation, classification, the parameters shown (descriptions, formulas, references)."""
    out.append("<h2 class='page'>Methods</h2>")
    if prof is not None:
        if prof.description:
            out.append(f"<p>{_e(prof.description)}</p>")
        out.append("<h3>Processing</h3>")
        from modalytics.processing.registry import ProcessingLibrary
        plib = ProcessingLibrary()
        procs = [(t, ch) for t in prof.images for ch in prof.chains_for(t)]
        out.append("<ul>" + "".join(f"<li>{_e(t)}: <b>{_e(ch.name)}</b> — {_e(ch.summary(plib))}</li>"
                                    for t, ch in procs) + "</ul>" if procs else "<p class='muted'>None.</p>")
        out.append("<h3>Segmentation</h3>")
        from modalytics.services.segmentation import find_models
        models = {m.id: m for m in find_models()}
        items = []
        for g, per in prof.segmentation.items():
            for t, mid in per.items():
                m = models.get(mid)
                if m is None:
                    items.append(f"<li>{_e(g)} on {_e(t)}: {_e(mid)} (not installed here)</li>")
                    continue
                src = m.manifest.get("source", {})
                tr = src.get("training", {})
                bits = [f"{m.um_per_px:g} µm/px", src.get("architecture_id", "")]
                if src.get("trained"):
                    bits.append(f"trained {src['trained'][:10]}")
                if src.get("parent"):
                    bits.append(f"fine-tuned from {src['parent']}")
                if tr.get("training_set"):
                    bits.append(f"on training set '{tr['training_set']}'")
                if tr.get("evaluation_dice") is not None:
                    bits.append(f"evaluation Dice {tr['evaluation_dice']:.3f}")
                items.append(f"<li>{_e(g)} on {_e(t)}: <b>{_e(m.name)}</b> — {_e(', '.join(b for b in bits if b))}</li>")
        out.append("<ul>" + "".join(items) + "</ul>" if items else "<p class='muted'>None.</p>")
        out.append("<h3>Classification</h3>")
        cls_items = [f"<li>{_e(g)}: {_e(cl.describe(c))}" + (f" <span class='muted'>({_e(c.note)})</span>"
                                                               if c.note else "") + "</li>"
                     for g, c in ((g, prof.classifier_for(g)) for g in prof.classification) if c is not None]
        out.append("<ul>" + "".join(cls_items) + "</ul>" if cls_items else "<p class='muted'>None.</p>")
    out.append("<h3>Key parameters</h3>")
    if keys:
        for k in keys:
            out.append(f"<h4>{_e(k.title)}</h4><div class='help'>{_e(k.help)}</div>")
    else:
        out.append("<p class='muted'>None chosen.</p>")
    other = [c.label for s in ("object", "class", "image") for c in res.columns.get(s, [])
             if c.uid not in {k.uid for k in keys}]
    if other:
        out.append(f"<p class='muted'>Also measured (in Results and its CSV export): {_e(', '.join(other))}.</p>")
    out.append(f"<footer>Modalytics {_e(version)} · {dt.datetime.now():%Y-%m-%d %H:%M}</footer></main>"
               f"<script>{SCRIPT}</script></body></html>")


def write(project, path: str, library=None, progress=None, pdf: bool = False) -> str:
    """Build the report (``pdf``: its PDF summary) and write it (UTF-8); returns the path."""
    from modalytics import fileio
    fileio.write_text(path, build(project, library, progress, pdf))
    return path
