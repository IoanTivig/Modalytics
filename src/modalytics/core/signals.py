"""
Minimal signal/slot implementation for the Qt-free core.

The domain model must not import Qt (so it can be tested headless and reused
from scripts), but the UI still needs to be told when something changes.
A ``Signal`` keeps a list of callables and calls them in order on ``emit``.
"""

from __future__ import annotations

from typing import Callable


class Signal:
    """A list of callbacks called on ``emit``.

    Example:
        >>> changed = Signal()
        >>> changed.connect(lambda node, field: print(node, field))
        >>> changed.emit("image-1", "placement")
        image-1 placement
    """

    def __init__(self) -> None:
        self._slots: list[Callable] = []

    def connect(self, slot: Callable) -> None:
        """Register a callback (ignored if already connected).

        Args:
            slot: Any callable accepting the arguments passed to ``emit``.
        """
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot: Callable) -> None:
        """Remove a callback (no error if it was not connected)."""
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args) -> None:
        """Call every connected callback with ``args``."""
        # Iterate over a copy: a slot may disconnect itself while being called
        for slot in list(self._slots):
            slot(*args)
