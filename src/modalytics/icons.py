"""
The icons a user can give a processing step or chain: Material Design Icons (Qt-free).

The set is Material Design Icons 6.9.96 (Pictogrammers, Apache License 2.0), drawn by
qtawesome as ``mdi6.<name>``. ``resources/icons/mdi6_index.json`` lists the icons with
their aliases and categories for searching; brand / logo and social-media icons are
left out (their licence does not cover them), so they can neither be found nor chosen.

    search("blur")      → ["mdi6.blur", "mdi6.blur-linear", "mdi6.blur-off", "mdi6.blur-radial"]
    search("microscope cell") — every word must match the name, an alias or the category
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

INDEX = Path(__file__).parent / "resources" / "icons" / "mdi6_index.json"
PREFIX = "mdi6."


@lru_cache(maxsize=1)
def _index() -> dict:
    with open(INDEX, encoding="utf-8") as fh:
        return json.load(fh)


def names() -> list[str]:
    """Every icon that may be chosen ("mdi6.<name>"), alphabetically."""
    return [PREFIX + n for n in sorted(_index()["icons"])]


def allowed(icon: str) -> bool:
    """Whether an icon name may be chosen (in the set, and not a brand or logo)."""
    return icon.startswith(PREFIX) and icon[len(PREFIX):] in _index()["icons"]


def search(query: str, limit: int | None = None) -> list[str]:
    """Icons whose name, aliases or category contain every word of ``query`` (best first: the name starting with
    the first word, then the name containing it, then aliases / category). An empty query gives every icon."""
    words = [w for w in query.lower().replace("-", " ").split() if w]
    icons = _index()["icons"]
    if not words:
        out = sorted(icons)
    else:
        ranked = []
        for name, (aliases, category) in icons.items():
            spaced = name.replace("-", " ")
            text = f"{spaced} {aliases} {category}".lower()
            if not all(w in text for w in words):
                continue
            first = words[0]
            rank = 0 if spaced.startswith(first) else 1 if first in spaced else 2
            ranked.append((rank, len(name), name))
        out = [n for _r, _l, n in sorted(ranked)]
    out = [PREFIX + n for n in out]
    return out[:limit] if limit else out
