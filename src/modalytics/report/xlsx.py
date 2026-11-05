"""
Training results as Excel workbooks (.xlsx, Office Open XML — ISO/IEC 29500: Excel, LibreOffice, Google Sheets,
Python and R read it), with native charts drawn from the workbook's own data (Qt-free; needs XlsxWriter).

Three exports, each from what its run keeps, so any run can be exported again:

* **Parameters** (a relevance analysis, ``training.relevance``): About; Parameters (verdict, AUC with its 95 %
  interval, how it goes, p, q, projects agreeing, redundancy group, effects, notes); Per class (n, median, IQR,
  mean, SD, Shapiro–Wilk p); Pairs; Redundancy groups; Values (each object's value of the parameters shown, with
  its class and project); Charts — the AUCs with their intervals, and each separating parameter's values per class
  (each object a point; the classes along the axis, numbered as the chart says).
* **Segmentation** (a training run or an evaluation, ``training.compare``): About; Results (every measure with its
  95 % interval, counts, speed, size, training); Against the proposal; Per project; Per test group (several);
  Per image; Training history (loss and evaluation Dice per epoch); Charts — the main measures with their intervals,
  the learning curves, the criterion against speed.
* **Classifiers** (a classifier report's results, ``training.metrics.assess``): About; Results where it was chosen
  (every measure with its interval); Against the best; Final checks; Per class; Confusion matrices; The
  classifiers; Charts — balanced accuracy with its interval, and the final checks side by side.

A missing value is left empty. Intervals are two columns (low, high), so they can be used in formulas.
"""

from __future__ import annotations

import math

import numpy as np

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


class _Book:
    """An XlsxWriter workbook with the formats and helpers the three exports share."""

    def __init__(self, path: str) -> None:
        import xlsxwriter
        self.wb = xlsxwriter.Workbook(path, {"strings_to_numbers": False})
        add = self.wb.add_format
        self.title = add({"bold": True, "font_size": 14})
        self.head = add({"bold": True, "bg_color": "#E8EEF4", "border": 1, "border_color": "#C9D3DD",
                         "text_wrap": True, "valign": "top"})
        self.key = add({"bold": True, "valign": "top"})
        self.wrap = add({"text_wrap": True, "valign": "top"})
        self.num = add({"num_format": "0.000"})
        self.num1 = add({"num_format": "0.0"})
        self.int = add({"num_format": "0"})
        self.pct = add({"num_format": "0%"})
        self.sci = add({"num_format": "0.000E+00"})
        self.best = add({"bold": True, "bg_color": "#E8F6F0"})
        self._names: set = set()

    def sheet(self, name: str):
        """A new sheet (its name made valid and unique)."""
        clean = "".join("_" if ch in "[]:*?/\\" else ch for ch in name)[:31] or "Sheet"
        base, n = clean, 2
        while clean.lower() in self._names:
            clean = f"{base[:28]} {n}"
            n += 1
        self._names.add(clean.lower())
        return self.wb.add_worksheet(clean)

    def cell(self, ws, r: int, c: int, v, fmt=None) -> None:
        """A value: numbers as numbers (empty when missing), the rest as text."""
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            ws.write_blank(r, c, None, fmt)
        elif isinstance(v, (bool, np.bool_)):
            ws.write_string(r, c, "yes" if v else "", fmt)
        elif _finite(v):
            ws.write_number(r, c, float(v), fmt)
        else:
            ws.write_string(r, c, str(v), fmt)

    def table(self, ws, top: int, heads: list, rows: list, formats: dict | None = None, widths: dict | None = None,
              freeze: bool = True, filters: bool = True) -> int:
        """A table: a header row, then the rows (formats: {column: format}); returns the row after it."""
        for c, h in enumerate(heads):
            ws.write_string(top, c, h, self.head)
        for i, row in enumerate(rows):
            for c, v in enumerate(row):
                self.cell(ws, top + 1 + i, c, v, (formats or {}).get(c))
        for c, w in (widths or {}).items():
            ws.set_column(c, c, w)
        if freeze:
            ws.freeze_panes(top + 1, 1)
        if filters and rows:
            ws.autofilter(top, 0, top + len(rows), len(heads) - 1)
        return top + 1 + len(rows)

    def about(self, title: str, items: list) -> None:
        """The About sheet: a title, then (what, value) lines."""
        ws = self.sheet("About")
        ws.write_string(0, 0, title, self.title)
        for i, (k, v) in enumerate(items, start=2):
            ws.write_string(i, 0, str(k), self.key)
            self.cell(ws, i, 1, v, self.wrap)
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 110)

    def interval_chart(self, sheet, ws_name: str, first: int, label_col: int, value_col: int, triples: list,
                       aux_col: int, title: str, axis: str, lo=None, hi=None, kind: str = "bar",
                       colour: str = PALETTE[0]):
        """A chart of one value per row with its 95 % interval as error bars. The bars' lengths (to the interval's
        high and low ends) are written in two columns after the table (``aux_col``), named so.

        Args:
            first: The table's first data row (its header just above).
            triples: Per row (value, low, high).
        """
        last = first + len(triples) - 1
        plus_c, minus_c = aux_col, aux_col + 1
        sheet.write_string(first - 1, plus_c, f"{title}: + to high", self.head)
        sheet.write_string(first - 1, minus_c, f"{title}: − to low", self.head)
        plus, minus = [], []
        for i, (v, lo_v, hi_v) in enumerate(triples):
            ok = _finite(v)
            plus.append(float(hi_v) - float(v) if ok and _finite(hi_v) else 0.0)
            minus.append(float(v) - float(lo_v) if ok and _finite(lo_v) else 0.0)
            sheet.write_number(first + i, plus_c, plus[-1], self.num)
            sheet.write_number(first + i, minus_c, minus[-1], self.num)
        chart = self.wb.add_chart({"type": kind})
        bars = "x_error_bars" if kind == "bar" else "y_error_bars"
        chart.add_series({"name": title, "categories": [ws_name, first, label_col, last, label_col],
                          "values": [ws_name, first, value_col, last, value_col],
                          "fill": {"color": colour}, "border": {"none": True},
                          # the bars' lengths as values: Excel misreads custom error bars given as cell ranges
                          bars: {"type": "custom", "plus_values": plus, "minus_values": minus,
                                 "end_style": 1, "line": {"color": "#333333"}}})
        chart.set_title({"name": title, "name_font": {"size": 12}})
        chart.set_legend({"none": True})
        value_axis = {"name": axis, "major_gridlines": {"visible": True, "line": {"color": "#E5E8EC"}}}
        if lo is not None:
            value_axis["min"] = lo
        if hi is not None:
            value_axis["max"] = hi
        if kind == "bar":
            chart.set_x_axis(value_axis)
            chart.set_y_axis({"reverse": True})
        else:
            chart.set_y_axis(value_axis)
        n = len(triples)
        chart.set_size({"width": 760, "height": max(260, 26 * n + 110) if kind == "bar" else 360})
        return chart

    def close(self) -> None:
        self.wb.close()


# ---------------------------------------------------------------- parameters
def write_parameters(analysis, path: str, data=None, rows=None) -> str:
    """A relevance analysis as a workbook (see the module docstring).

    Args:
        analysis: The ``training.relevance.Analysis``.
        data, rows: The measured objects and the rows analysed (for the Values sheet and its charts; without them,
            the AUC chart only).
    """
    a = analysis
    b = _Book(path)
    b.about(f"{a.name} — parameter relevance",
            [("Training set", a.training_set), ("Profile", a.profile), ("Objects", a.group),
             ("Classes", (" < " if a.ordered else ", ").join(a.grades) + ("  (ordered grades)" if a.ordered
                                                                          else "  (categories)")),
             ("Projects", f"{len(a.projects)}: {', '.join(a.projects)}"),
             ("On", "the training projects" if a.on == "training" else "the training and evaluation projects"),
             ("Compared", a.unit), ("Objects per class", ", ".join(f"{c} {n}" for c, n in a.counts.items())),
             ("Made", a.made.replace("T", " ")),
             ("Settings", ", ".join(f"{k} {v}" for k, v in (a.settings or {}).items())),
             ("Proposed", ", ".join(r.title for r in a.proposed()) or "none")])
    # 1. Parameters
    ws, name = b.sheet("Parameters"), "Parameters"
    heads = ["Parameter", "Unit", "Verdict", "Proposed", "AUC", "AUC 95% low", "AUC 95% high",
             "Direction" if a.ordered else "Stands out", "p", "q", "Projects agreeing", "Projects with both",
             "Redundancy group", "Between classes ε²", "Between projects ε²", "Normality", "Notes", "Parameter id"]
    table = [[r.label, r.unit, r.verdict, bool(r.proposed), r.auc, r.ci[0], r.ci[1],
              (r.direction if a.ordered else r.best), r.p, r.q, r.agree[0], r.agree[1], r.group or None,
              r.grade_effect, r.project_effect, r.normal(), "; ".join(r.notes()), r.uid] for r in a.rows]
    b.table(ws, 0, heads, table, {4: b.num, 5: b.num, 6: b.num, 8: b.sci, 9: b.sci, 13: b.num, 14: b.num},
                  {0: 30, 2: 11, 4: 8, 5: 9, 6: 9, 7: 12, 16: 40, 17: 34})
    # 2. Per class
    ws2 = b.sheet("Per class")
    per = [[r.label, c, *(s + [math.nan] * 7)[:7]] for r in a.rows for c, s in r.stats.items()]
    b.table(ws2, 0, ["Parameter", "Class", "n", "Median", "Q1", "Q3", "Mean", "SD", "Shapiro–Wilk p"], per,
            {2: b.int, 3: b.num, 4: b.num, 5: b.num, 6: b.num, 7: b.num, 8: b.sci}, {0: 30, 1: 22})
    # 3. Pairs
    ws3 = b.sheet("Pairs")
    pairs = [[r.label, p[0], p[1], p[2], p[3] if len(p) > 3 else None] for r in a.rows for p in r.pairs]
    b.table(ws3, 0, ["Parameter", "Class A", "Class B", "AUC (B larger)", "p (Holm)"], pairs,
            {3: b.num, 4: b.sci}, {0: 30, 1: 22, 2: 22})
    # 4. Redundancy groups
    ws4 = b.sheet("Redundancy groups")
    labels = {r.uid: r.label for r in a.rows}
    b.table(ws4, 0, ["Group", "Parameters (follow each other: |Spearman ρ| ≥ 0.8)"],
            [[i + 1, ", ".join(labels.get(u, u) for u in g)] for i, g in enumerate(a.groups)], {0: b.int},
            {1: 100}, freeze=False)
    # 5. Charts: the AUCs with their intervals (helper columns beside the Parameters table)
    charts = b.sheet("Charts")
    if a.rows:
        chart = b.interval_chart(ws, name, 1, 0, 4, [(t[4], t[5], t[6]) for t in table], len(heads) + 1,
                                 "AUC with its 95 % interval", "AUC", 0.0, 1.0)
        charts.insert_chart(1, 1, chart)
    row = 4 + max(14, int((26 * len(a.rows) + 110) / 20))
    # 6. Values (each object), and each shown parameter's values per class
    if data is not None and rows is not None:
        shown = [r for r in a.rows if r.proposed or r.verdict in ("separates", "weak")][:12]
        col = {c[0]: i for i, c in enumerate(data.columns)}
        idx = np.flatnonzero(np.asarray(rows, bool))
        ws5 = b.sheet("Values")
        vals = [[data.projects[data.project[i]], data.grades[data.grade[i]]]
                + [data.values[i, col[r.uid]] if r.uid in col else None for r in shown] for i in idx]
        b.table(ws5, 0, ["Project", "Class"] + [r.title for r in shown], vals,
                {k: b.num for k in range(2, 2 + len(shown))}, {0: 16, 1: 22})
        # chart data: per parameter, per class, x (class number + jitter) and y
        cd = b.sheet("Chart data")
        rng = np.random.default_rng(0)
        c0 = 0
        for k, r in enumerate(shown):
            if r.uid not in col:
                continue
            chart = b.wb.add_chart({"type": "scatter"})
            for ci, cls in enumerate(a.grades):
                ys = [data.values[i, col[r.uid]] for i in idx if data.grade[i] == a.grades.index(cls)
                      if r.uid in col]
                ys = [float(y) for y in ys if _finite(y)]
                if not ys:
                    continue
                cd.write_string(0, c0, f"{r.label} · {cls} (x)", b.head)
                cd.write_string(0, c0 + 1, f"{r.label} · {cls}", b.head)
                for j, y in enumerate(ys):
                    cd.write_number(1 + j, c0, ci + 1 + float(rng.uniform(-0.18, 0.18)))
                    cd.write_number(1 + j, c0 + 1, y)
                chart.add_series({"name": cls, "categories": ["Chart data", 1, c0, len(ys), c0],
                                  "values": ["Chart data", 1, c0 + 1, len(ys), c0 + 1],
                                  "marker": {"type": "circle", "size": 4,
                                             "fill": {"color": PALETTE[ci % len(PALETTE)], "transparency": 35},
                                             "border": {"none": True}}})
                c0 += 2
            chart.set_title({"name": f"{r.title} — AUC {r.auc:.2f}" if _finite(r.auc) else r.title,
                             "name_font": {"size": 11}})
            k = len(a.grades)
            chart.set_x_axis({"min": 0, "max": k + 1, "major_unit": 1, "major_gridlines": {"visible": False},
                              "name": "   ".join(f"{i + 1}: {g}" for i, g in enumerate(a.grades))})
            chart.set_y_axis({"name": r.unit or r.label, "major_gridlines": {"visible": True,
                                                                              "line": {"color": "#E5E8EC"}}})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 480, "height": 320})
            charts.insert_chart(row + (k // 2) * 17, 1 + (k % 2) * 8, chart)
    b.close()
    return path


# ---------------------------------------------------------------- segmentation
def write_segmentation(comp, path: str) -> str:
    """A segmentation training run or evaluation (``training.compare.Comparison``) as a workbook."""
    from modalytics.report.comparison import augmentation
    from modalytics.training import compare as cmp
    on = comp.judged_on
    done = comp.done()
    best = comp.best()
    b = _Book(path)
    s = comp.settings or {}
    b.about(f"{comp.name} — " + ("segmentation models judged" if comp.purpose == "evaluation"
                                  else "segmentation models trained and compared"),
            [("Training set", comp.training_set), ("Objects", comp.group), ("Image type", comp.modality),
             ("Works at", f"{comp.um_per_px:g} µm/px" if _finite(comp.um_per_px) else "each model its own"),
             ("Judged on", f"the {on} projects" + (f" (group {comp.test_group})" if comp.test_group else "")),
             ("Proposed by", cmp.METRICS[comp.criterion][0]),
             ("Proposed", best.label if best is not None else "—"),
             ("Compared where", f"{comp.area} (margin {comp.margin_um:g} µm), matched from IoU {comp.iou_min:g}"),
             *([("Training", f"{s.get('epochs')} epochs × {s.get('steps')} steps × {s.get('batch_size')} patches; "
                             f"augmentation: {augmentation(s)}; seed {s.get('seed')}"),
                ("Trained on", comp.training_device)] if comp.purpose != "evaluation" else []),
             ("Projects", "; ".join(f"{role}: {', '.join(names)}" for role, names in comp.projects.items() if names)),
             ("Started", comp.started.replace("T", " ")), ("Finished", comp.finished.replace("T", " ")),
             ("Software", ", ".join(f"{k} {v}" for k, v in (comp.versions or {}).items()))])
    # 1. Results: every measure with its 95 % interval (bootstrap over images)
    keys = list(cmp.METRICS)
    heads = ["Model", "Proposed", "Kind", "Architecture", "Starting weights"]
    for k in keys:
        heads += [cmp.METRICS[k][0], "95% low", "95% high"]
    heads += ["Hand objects", "Found objects", "Found (TP)", "Missed (FN)", "Invented (FP)", "Merges", "Splits",
              "ms per tile", "Size (MB)", "Parameters (M)", "Device", "Training (min)", "Epoch kept", "Epochs",
              "Works at (µm/px)", "File"]
    rows, formats = [], {}
    for e in done:
        ev = e.judged(on)
        per = [c for _p, _n, c in ev.per_image]
        row = [e.label, e is best, e.kind, e.arch_id, e.weights_name]
        for k in keys:
            fn = cmp.METRICS[k][1]
            lo, hi = cmp.bootstrap(per, fn, seed=1)
            row += [fn(ev.total), lo, hi]
        c = ev.total
        row += [c.hand, c.found, c.tp, c.fn, c.fp, c.merges, c.splits, e.ms_per_tile, e.size_mb, e.params_m,
                e.device, e.seconds / 60 if e.kind != "reference" else None,
                e.best_epoch if e.history else None, len(e.history) or None, e.um_per_px, e.path]
        rows.append(row)
    for i in range(5, 5 + 3 * len(keys)):
        formats[i] = b.num
    ws = b.sheet("Results")
    b.table(ws, 0, heads, rows, formats, {0: 38, 3: 18, 4: 18})
    # 2. Against the proposal (paired bootstrap over the same images)
    if best is not None and len(done) > 1:
        ws2 = b.sheet("Against the proposal")
        bper = {(p, n): c for p, n, c in best.judged(on).per_image}
        heads2 = ["Model"]
        figure = ["f1", "pq", "object_dice", "dice"]
        for k in figure:
            heads2 += [f"{cmp.METRICS[k][0]}: proposal − this", "95% low", "95% high"]
        rows2 = []
        for e in done:
            if e is best:
                continue
            other = {(p, n): c for p, n, c in e.judged(on).per_image}
            both = [key for key in bper if key in other]
            row = [e.label]
            for k in figure:
                row += list(cmp.paired_difference([bper[x] for x in both], [other[x] for x in both],
                                                  cmp.METRICS[k][1], seed=2))
            rows2.append(row)
        b.table(ws2, 0, heads2, rows2, {i: b.num for i in range(1, len(heads2))}, {0: 38})
    # 3. Per project and per test group
    crit, fn = comp.criterion, cmp.METRICS[comp.criterion][1]
    if done:
        ws3 = b.sheet("Per project")
        projects = done[0].judged(on).projects
        rows3 = []
        for pi, pname in enumerate(projects):
            row = [pname]
            for e in done:
                pp = e.judged(on).per_project
                row.append(fn(pp[pi]) if pi < len(pp) else None)
            rows3.append(row)
        b.table(ws3, 0, ["Project"] + [f"{e.label}: {cmp.METRICS[crit][0]}" for e in done], rows3,
                {i: b.num for i in range(1, len(done) + 1)}, {0: 20})
        groups = {g: n for g, n in (comp.test_groups or {}).items() if n}
        if on == "test" and len(groups) > 1:
            ws4 = b.sheet("Per test group")
            rows4 = []
            for g, names in groups.items():
                for e in done:
                    ev = e.judged(on)
                    per = [c for p, _n, c in ev.per_image if 0 <= p < len(ev.projects) and ev.projects[p] in names]
                    lo, hi = cmp.bootstrap(per, fn, seed=1)
                    rows4.append([g, len(names), e.label, fn(cmp.pooled(per)) if per else None, lo, hi])
            b.table(ws4, 0, ["Test group", "Projects", "Model", cmp.METRICS[crit][0], "95% low", "95% high"], rows4,
                    {3: b.num, 4: b.num, 5: b.num}, {0: 20, 2: 38})
        # 4. Per image
        ws5 = b.sheet("Per image")
        rows5 = []
        for e in done:
            ev = e.judged(on)
            for p, n, c in ev.per_image:
                rows5.append([e.label, ev.projects[p] if 0 <= p < len(ev.projects) else "", n, c.hand, c.found,
                              c.tp, c.fn, c.fp, c.f1, c.dice, c.iou])
        b.table(ws5, 0, ["Model", "Project", "Image", "Hand", "Found", "TP", "FN", "FP", "Object F1", "Pixel Dice",
                         "Pixel IoU"], rows5, {8: b.num, 9: b.num, 10: b.num}, {0: 38, 1: 16, 2: 26})
    # 5. Training history
    hist = [(e.label, r) for e in comp.entries for r in e.history]
    if hist:
        ws6 = b.sheet("Training history")
        b.table(ws6, 0, ["Model", "Epoch", "Loss", "Evaluation Dice", "Seconds"],
                [[lab, r.get("epoch"), r.get("loss"), r.get("dice"), r.get("seconds")] for lab, r in hist],
                {2: b.num, 3: b.num, 4: b.num1}, {0: 38})
    # 6. Charts
    charts = b.sheet("Charts")
    at = 1
    if done:
        for i, k in enumerate(["f1", "pq", "object_dice", "dice"]):
            c = 5 + 3 * keys.index(k)
            chart = b.interval_chart(ws, "Results", 1, 0, c, [(t[c], t[c + 1], t[c + 2]) for t in rows],
                                     len(heads) + 1 + 2 * i, cmp.METRICS[k][0], cmp.METRICS[k][0], 0.0, 1.0,
                                     colour=PALETTE[i % len(PALETTE)])
            charts.insert_chart(at + (i // 2) * 18, 1 + (i % 2) * 12, chart)
        at += 38
    if hist:
        r0 = 1
        for metric, colnum, axis in (("loss", 2, "Loss"), ("dice", 3, "Evaluation Dice")):
            chart = b.wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
            for i, e in enumerate([e for e in comp.entries if e.history]):
                n = len(e.history)
                chart.add_series({"name": e.label, "categories": ["Training history", r0, 1, r0 + n - 1, 1],
                                  "values": ["Training history", r0, colnum, r0 + n - 1, colnum],
                                  "line": {"color": PALETTE[i % len(PALETTE)], "width": 1.5},
                                  "marker": {"type": "circle", "size": 3,
                                             "fill": {"color": PALETTE[i % len(PALETTE)]}, "border": {"none": True}}})
                r0 += n
            r0 = 1
            chart.set_title({"name": f"{axis} per epoch", "name_font": {"size": 12}})
            chart.set_x_axis({"name": "Epoch", "min": 1, "max": max(len(e.history) for e in comp.entries)})
            chart.set_y_axis({"name": axis, "major_gridlines": {"visible": True, "line": {"color": "#E5E8EC"}}})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 700, "height": 360})
            charts.insert_chart(at, 1 + (0 if metric == "loss" else 12), chart)
        at += 20
    if done:
        chart = b.wb.add_chart({"type": "scatter"})
        ms_col = 5 + 3 * len(keys) + 7
        crit_col = 5 + 3 * keys.index(crit)
        for i in range(len(done)):
            chart.add_series({"name": ["Results", 1 + i, 0], "categories": ["Results", 1 + i, ms_col, 1 + i, ms_col],
                              "values": ["Results", 1 + i, crit_col, 1 + i, crit_col],
                              "marker": {"type": "circle", "size": 9, "fill": {"color": PALETTE[i % len(PALETTE)]},
                                         "border": {"none": True}}})
        chart.set_title({"name": f"{cmp.METRICS[crit][0]} against speed", "name_font": {"size": 12}})
        chart.set_x_axis({"name": "ms per 512 × 512 tile", "min": 0})
        chart.set_y_axis({"name": cmp.METRICS[crit][0], "major_gridlines": {"visible": True,
                                                                             "line": {"color": "#E5E8EC"}}})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 700, "height": 380})
        charts.insert_chart(at, 1, chart)
    b.close()
    return path


# ---------------------------------------------------------------- classifiers
MEASURES = [("balanced_accuracy", "Balanced accuracy", "pct"), ("accuracy", "Accuracy", "pct"), ("kappa", "κ", "num"),
            ("mcc", "MCC", "num"), ("macro_f1", "Macro-F1", "num"), ("auc", "ROC AUC", "num")]


def _measure_row(b: _Book, m: dict) -> list:
    out = []
    for key, _label, _f in MEASURES:
        lo, hi = (m.get(key + "_ci") or (None, None))[:2] if m.get(key + "_ci") else (None, None)
        out += [m.get(key), lo, hi]
    return out


def _measure_heads() -> list:
    out = []
    for _key, label, _f in MEASURES:
        out += [label, "95% low", "95% high"]
    return out


def _measure_formats(b: _Book, first: int) -> dict:
    out = {}
    for i, (_k, _l, f) in enumerate(MEASURES):
        for j in range(3):
            out[first + 3 * i + j] = b.pct if f == "pct" else b.num
    return out


def write_classifiers(results: dict, path: str) -> str:
    """A classifier report's results (the Results … .json: ``metrics.assess`` and the classifiers) as a workbook."""
    from modalytics.training import classifiers as cl
    from modalytics.training import metrics as mt
    r = results
    judged = r.get("judged_on", "")
    pool = r.get("evaluation", {}) if judged == "evaluation" else r.get("cv", {})
    names = mt.NAMES
    kinds = [k for k in names if k in (r.get("classifiers") or {})]
    b = _Book(path)
    b.about(f"{r.get('training_set', '')} — classifiers compared",
            [("Training set", r.get("training_set")), ("Profile", r.get("profile")), ("Objects", r.get("group")),
             ("Classes", (" < " if r.get("ordered", True) else ", ").join(r.get("grades", []))),
             ("Chosen by", "cross-validation on the training projects" if judged == "cross-validation"
              else "the evaluation projects"),
             ("Recommended", names.get(r.get("recommended"), "—")), ("Why", r.get("reason", "")),
             ("Note", r.get("fallback", "")), ("Made", str(r.get("made", "")).replace("T", " "))])
    # 1. Results where it was chosen
    ws, name = b.sheet("Results"), "Results"
    heads = ["Classifier", "Recommended", "Objects", "Projects", "Intervals"] + _measure_heads() + \
        ["Cross-validation: balanced accuracy", "Training: balanced accuracy"]
    rows = []
    for k in kinds:
        m = pool.get(k) or {}
        rows.append([names[k], k == r.get("recommended"), m.get("n"), m.get("projects"),
                     "whole projects resampled" if m.get("resampled") == "projects" else "over objects",
                     *_measure_row(b, m), (r.get("cv", {}).get(k) or {}).get("balanced_accuracy"),
                     (r.get("training", {}).get(k) or {}).get("balanced_accuracy")])
    fmts = _measure_formats(b, 5)
    fmts.update({5 + 3 * len(MEASURES): b.pct, 6 + 3 * len(MEASURES): b.pct})
    b.table(ws, 0, heads, rows, fmts, {0: 22, 4: 22})
    # 2. Against the best
    against = r.get("against") or {}
    if against:
        ws2 = b.sheet("Against the best")
        rows2 = []
        for k in kinds:
            c = against.get(k)
            if not c:
                continue
            lo, hi = (c.get("difference_ci") or (None, None))[:2]
            rows2.append([names[k], c.get("difference"), lo, hi, c.get("only_a"), c.get("only_b"), c.get("p"),
                          "not distinguishable" if mt.alike(c) else "the best is clearly better"])
        b.table(ws2, 0, [f"Classifier (against {names.get(r.get('best'), '')})", "Balanced accuracy: best − this",
                         "95% low", "95% high", "Only the best right", "Only this one right", "McNemar p", "Verdict"],
                rows2, {1: b.num, 2: b.num, 3: b.num, 6: b.sci}, {0: 30, 7: 26})
    # 3. Final checks
    finals = r.get("finals") or {}
    if finals:
        ws3 = b.sheet("Final checks")
        rows3 = []
        for g, per in finals.items():
            for k in kinds:
                m = per.get(k) or {}
                rows3.append([g, names[k], m.get("n"), m.get("projects"), *_measure_row(b, m)])
        b.table(ws3, 0, ["Group", "Classifier", "Objects", "Projects"] + _measure_heads(), rows3,
                _measure_formats(b, 4), {0: 22, 1: 22})
    # 4. Per class
    ws4 = b.sheet("Per class")
    rows4 = []
    for k in kinds:
        for pc in (pool.get(k) or {}).get("per_class", []):
            row = [names[k], pc.get("class"), pc.get("n")]
            for key in ("sensitivity", "specificity", "ppv", "npv"):
                lo, hi = (pc.get(key + "_ci") or (None, None))[:2]
                row += [pc.get(key), lo, hi]
            rows4.append(row + [pc.get("f1")])
    heads4 = ["Classifier", "Class", "n"]
    for key in ("Sensitivity", "Specificity", "PPV", "NPV"):
        heads4 += [key, "95% low", "95% high"]
    b.table(ws4, 0, heads4 + ["F1"], rows4, {i: b.pct for i in range(3, 15)} | {15: b.num}, {0: 22, 1: 22})
    # 5. Confusion matrices
    ws5 = b.sheet("Confusion matrices")
    grades = r.get("grades", [])
    row = 0
    for k in kinds:
        cm = (pool.get(k) or {}).get("confusion")
        if not cm:
            continue
        ws5.write_string(row, 0, f"{names[k]} — rows: by hand, columns: the classifier's", b.key)
        ws5.write_string(row + 1, 0, "", b.head)
        for j, g in enumerate(grades):
            ws5.write_string(row + 1, 1 + j, g, b.head)
        for i, g in enumerate(grades):
            ws5.write_string(row + 2 + i, 0, g, b.head)
            for j in range(len(grades)):
                b.cell(ws5, row + 2 + i, 1 + j, cm[i][j] if i < len(cm) and j < len(cm[i]) else None,
                       b.best if i == j else b.int)
        row += len(grades) + 4
    ws5.set_column(0, len(grades), 24)
    # 6. The classifiers
    ws6 = b.sheet("Classifiers")
    rows6 = []
    for k in kinds:
        clf = cl.Classifier.from_dict(r["classifiers"][k])
        rows6.append([names[k], cl.describe(clf), ", ".join(lab for _u, lab in clf.features) or
                      ", ".join(clf.parameters), clf.note])
    b.table(ws6, 0, ["Classifier", "What it is", "Parameters", "Made from"], rows6, None,
            {0: 22, 1: 90, 2: 40, 3: 60}, freeze=False, filters=False)
    for i in range(len(rows6)):
        ws6.set_row(1 + i, 60, b.wrap)
    # 7. Charts
    charts = b.sheet("Charts")
    if rows:
        chart = b.interval_chart(ws, name, 1, 0, 5, [(t[5], t[6], t[7]) for t in rows], len(heads) + 1,
                                 "Balanced accuracy (95 % interval)", "Balanced accuracy", 0.0, 1.0)
        charts.insert_chart(1, 1, chart)
    if finals and rows:
        chart = b.wb.add_chart({"type": "column"})
        n = len(kinds)
        for gi, g in enumerate(finals):
            first = 1 + gi * n
            chart.add_series({"name": g, "categories": ["Final checks", first, 1, first + n - 1, 1],
                              "values": ["Final checks", first, 4, first + n - 1, 4],
                              "fill": {"color": PALETTE[gi % len(PALETTE)]}, "border": {"none": True}})
        chart.set_title({"name": "Final checks: balanced accuracy", "name_font": {"size": 12}})
        chart.set_y_axis({"min": 0, "max": 1, "num_format": "0%", "name": "Balanced accuracy",
                          "major_gridlines": {"visible": True, "line": {"color": "#E5E8EC"}}})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 760, "height": 380})
        charts.insert_chart(22, 1, chart)
    b.close()
    return path
