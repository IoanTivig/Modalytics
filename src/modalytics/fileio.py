"""
Writing files safely (Qt-free).

Files are written whole to ``name.tmp`` first and then swapped in with
``os.replace``, so a crash never leaves a half-written file. On Windows the swap
can fail for a moment with "Access is denied" when another process has just
opened the target — typically the virus scanner or the search indexer looking at
a file that was just created — so it is retried for up to a second before giving up.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

RETRIES = 20
DELAY_S = 0.05


def replace(tmp: str | os.PathLike, target: str | os.PathLike) -> None:
    """``os.replace(tmp, target)``, retried while Windows briefly locks the target.

    Raises:
        PermissionError: If the target stays locked (after about a second).
    """
    for attempt in range(RETRIES):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(DELAY_S)


def write_text(path: str | os.PathLike, text: str) -> None:
    """Write a text file (UTF-8) atomically: to ``path.tmp``, then swapped in."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace(tmp, path)
