"""
Arranging imported images side by side (Qt-free).

When several images are imported they are laid out, in world µm, one of these ways
(``Layout.mode``):

* **grid** — in cells as large as the largest image plus a gap: ``gap_pct`` % of the
  largest width between columns and of the largest height between rows (100 % = one
  image's size). ``columns`` and ``rows`` (0 = worked out from the other, or a
  square-ish grid when both are 0); filled row by row or column by column; in name
  order (natural: ``…-2`` before ``…-10``) or in the order given.
* **names** — rows and columns read from parts of the names: a name is split at
  ``-``, ``_``, ``.`` and spaces, and field ``row_field`` gives the row, field
  ``column_field`` the column (e.g. ``sample01-1-2`` → fields ``sample01``, ``1``, ``2``:
  row = area 1, column = image 2, as the DHM phase map extractor names them). Rows
  and columns are in natural order; names without those fields go in a last row.
* **custom** — each image's row, place in its row and the gap after it (``custom``:
  {image index: (row, place, gap after in %)}); the rows are ``gap_pct`` apart.
* **stack** — all at the same place (e.g. each to be registered onto a slide).

``positions`` gives each image's top-left corner relative to the arrangement's
top-left corner; ``bounds`` its overall size.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

MODES = ("grid", "names", "custom", "stack")


def natural_key(text: str) -> list:
    """Sort key in which numbers sort as numbers: "map-2" before "map-10"."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.lower())
            for part in re.split(r"(\d+)", str(text)) if part != ""]


def name_fields(name: str) -> list[str]:
    """A name's parts, split at -, _, ., spaces (and the file extension dropped)."""
    stem = re.sub(r"\.(txt|tif|tiff|png|jpg|jpeg|bmp|npy|svs)$", "", str(name), flags=re.IGNORECASE)
    return [f for f in re.split(r"[-_.\s/\\]+", stem) if f]


@dataclass
class Layout:
    """How to arrange images (see the module docstring)."""

    mode: str = "grid"
    columns: int = 0
    rows: int = 0
    gap_pct: float = 10.0
    order: str = "name"          # "name" or "given"
    fill: str = "rows"           # "rows" (row by row) or "columns"
    row_field: int = 1
    column_field: int = 2
    custom: dict = field(default_factory=dict)


def grid_shape(n: int, columns: int = 0, rows: int = 0) -> tuple[int, int]:
    """(columns, rows) for n items: a given count is kept, the other worked out (both 0: square-ish)."""
    if n <= 0:
        return 0, 0
    if columns > 0:
        return columns, max(rows, math.ceil(n / columns)) if rows > 0 else math.ceil(n / columns)
    if rows > 0:
        return math.ceil(n / rows), rows
    c = math.ceil(math.sqrt(n))
    return c, math.ceil(n / c)


def cells(names: list[str], layout: Layout) -> list[tuple[int, int]]:
    """Each image's (row, column) cell for the grid and name modes."""
    n = len(names)
    order = sorted(range(n), key=lambda i: natural_key(names[i])) if layout.order == "name" else list(range(n))
    if layout.mode == "names":
        # 1. Row and column keys from the names' fields (natural order); names without them: a last row
        fields = [name_fields(nm) for nm in names]
        both = [len(f) > max(layout.row_field, layout.column_field) for f in fields]       # names that have both
        rk = [f[layout.row_field] if ok else None for f, ok in zip(fields, both)]
        ck = [f[layout.column_field] if ok else None for f, ok in zip(fields, both)]
        rows = sorted({k for k in rk if k is not None}, key=natural_key)
        cols = sorted({k for k in ck if k is not None}, key=natural_key)
        out: list = [None] * n
        taken: set = set()
        stray = 0
        for i in order:
            if rk[i] is None or ck[i] is None:
                cell = (len(rows), stray)
                stray += 1
            else:
                cell = (rows.index(rk[i]), cols.index(ck[i]))
                while cell in taken:                    # two images with the same fields: the next free place
                    cell = (cell[0], cell[1] + 1)
            taken.add(cell)
            out[i] = cell
        return out
    # 2. A grid, filled row by row or column by column
    columns, rows = grid_shape(n, layout.columns, layout.rows)
    out = [None] * n
    for k, i in enumerate(order):
        out[i] = (k // columns, k % columns) if layout.fill == "rows" else (k % rows, k // rows)
    return out


def positions(sizes: list[tuple[float, float]], names: list[str], layout: Layout) -> list[tuple[float, float]]:
    """Each image's top-left corner (µm), relative to the arrangement's top-left corner.

    Args:
        sizes: Each image's (width, height) in µm.
        names: Each image's name (for name order and the name mode).
        layout: How.
    """
    if not sizes:
        return []
    max_w = max(w for w, _h in sizes)
    max_h = max(h for _w, h in sizes)
    gx, gy = max_w * layout.gap_pct / 100.0, max_h * layout.gap_pct / 100.0
    if layout.mode == "stack":
        return [(0.0, 0.0)] * len(sizes)
    if layout.mode == "custom":
        # Each row: its images in their order, each followed by its own gap; rows one largest height apart
        rows: dict[int, list[int]] = {}
        for i in range(len(sizes)):
            row, place, _gap = layout.custom.get(i, (0, i, layout.gap_pct))
            rows.setdefault(int(row), []).append(i)
        out: list = [None] * len(sizes)
        for r, members in rows.items():
            members.sort(key=lambda i: (layout.custom.get(i, (0, i, 0))[1], i))
            x = 0.0
            for i in members:
                out[i] = (x, r * (max_h + gy))
                x += sizes[i][0] + max_w * float(layout.custom.get(i, (0, i, layout.gap_pct))[2]) / 100.0
        return out
    return [(c * (max_w + gx), r * (max_h + gy)) for r, c in cells(names, layout)]


def bounds(sizes: list[tuple[float, float]], spots: list[tuple[float, float]]) -> tuple[float, float]:
    """(width, height) the arranged images cover."""
    if not sizes:
        return 0.0, 0.0
    return (max(x + w for (x, _y), (w, _h) in zip(spots, sizes)),
            max(y + h for (_x, y), (_w, h) in zip(spots, sizes)))
