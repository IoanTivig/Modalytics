"""
Sorting the Training tables by a click on a column title.

``SortItem`` compares cells the way they read: numbers as numbers (also "94 %",
"0.95 (0.90–0.98)", "1,234 µm²", "12 ms"), text alphabetically; empty and "—" cells, and a
row marked ``pinned`` (e.g. "All"), stay at the bottom whichever way the column is sorted; a cell
holding a list (a combo box) sorts by the choice shown in it.
``sortable`` switches a tree to sorting by click, keeping its first order until a title
is clicked.
"""

from __future__ import annotations

import re

from PySide6 import QtCore, QtWidgets

_NUMBER = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
EMPTY = ("", "—", "-", "–")


def sort_key(text: str) -> tuple:
    """(0, number) for a cell that starts with a number, (1, text) for text, (2,) for an empty cell."""
    t = (text or "").strip().lstrip("★ ").strip()
    if t in EMPTY:
        return (2,)
    m = _NUMBER.match(t)
    if m:
        return (0, float(m.group().replace(",", "").replace("−", "-")))
    return (1, t.lower())


class SortItem(QtWidgets.QTreeWidgetItem):
    """A tree row that sorts by what its cells mean (see the module docstring)."""

    pinned = False

    def cell(self, col: int) -> str:
        """What the cell shows: its text, or the choice of a list placed in it."""
        tree = self.treeWidget()
        widget = tree.itemWidget(self, col) if tree is not None else None
        return widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else self.text(col)

    def __lt__(self, other) -> bool:
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        ascending = tree is None or tree.header().sortIndicatorOrder() == QtCore.Qt.SortOrder.AscendingOrder
        a = sort_key(self.cell(col))
        b = sort_key(other.cell(col) if isinstance(other, SortItem) else other.text(col))
        # Pinned rows, then empty cells, stay last either way (Qt reverses the comparison when descending)
        for last_a, last_b in ((self.pinned, getattr(other, "pinned", False)), (a[0] == 2, b[0] == 2)):
            if last_a != last_b:
                return last_b if ascending else last_a
        if a[0] != b[0]:                                           # numbers before text
            return a[0] < b[0]
        return a < b


def sortable(tree: QtWidgets.QTreeWidget) -> QtWidgets.QTreeWidget:
    """Let a tree be sorted by a click on a column title (its rows keep their order until then)."""
    tree.header().setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)
    tree.setSortingEnabled(True)
    tree.header().setSortIndicatorShown(True)
    tree.header().setSectionsClickable(True)
    return tree
