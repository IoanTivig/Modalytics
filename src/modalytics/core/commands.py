"""
Undo/redo for user edits.

Every interactive edit (move an image, change opacity…) is a ``Command`` with
``do`` and ``undo``. Commands only call ``Project`` methods, so observers are
notified exactly as for any other change.

Live interactions (dragging, sliders) change the project continuously and
push **one** command at the end with the start and end values, so a whole
drag is one undo step.
"""

from __future__ import annotations

from typing import Any, Callable

from modalytics.core.signals import Signal


class Command:
    """An undoable edit.

    Args:
        text: Human-readable description shown in the Edit menu.
        do: Callable applying the edit.
        undo: Callable reverting it.
    """

    def __init__(self, text: str, do: Callable[[], Any], undo: Callable[[], Any]) -> None:
        self.text = text
        self._do = do
        self._undo = undo

    def do(self) -> None:
        self._do()

    def undo(self) -> None:
        self._undo()


class UndoStack:
    """Linear undo/redo history.

    Signals:
        changed(): the history or the current position changed.
    """

    def __init__(self, limit: int = 200) -> None:
        self._done: list[Command] = []
        self._undone: list[Command] = []
        self.limit = limit
        self.changed = Signal()

    def push(self, command: Command, already_done: bool = False) -> None:
        """Execute (unless ``already_done``) and record a command.

        Args:
            command: The command.
            already_done: True when the edit was applied live (e.g. a drag)
                and only needs recording.
        """
        if not already_done:
            command.do()
        self._done.append(command)
        del self._done[:-self.limit]
        self._undone.clear()
        self.changed.emit()

    def undo(self) -> None:
        """Revert the last command (no-op if none)."""
        if self._done:
            cmd = self._done.pop()
            cmd.undo()
            self._undone.append(cmd)
            self.changed.emit()

    def redo(self) -> None:
        """Re-apply the last undone command (no-op if none)."""
        if self._undone:
            cmd = self._undone.pop()
            cmd.do()
            self._done.append(cmd)
            self.changed.emit()

    def clear(self) -> None:
        """Forget the history (e.g. after opening another project)."""
        self._done.clear()
        self._undone.clear()
        self.changed.emit()

    @property
    def undo_text(self) -> str:
        return self._done[-1].text if self._done else ""

    @property
    def redo_text(self) -> str:
        return self._undone[-1].text if self._undone else ""


def set_placement_command(project, image, old, new) -> Command:
    """Command moving an image between two placements."""
    return Command(f"Move {image.name}",
                   lambda: project.set_placement(image, new),
                   lambda: project.set_placement(image, old))


def set_opacity_command(project, node, old: float, new: float) -> Command:
    """Command changing a node's opacity."""
    return Command(f"Opacity of {node.name}",
                   lambda: project.set_opacity(node, new),
                   lambda: project.set_opacity(node, old))


def set_geometry_command(project, mask, old, new, text: str = "") -> Command:
    """Command replacing a mask's geometry (draw, brush, erase…)."""
    return Command(text or f"Edit {mask.name}",
                   lambda: project.set_mask_geometry(mask, new),
                   lambda: project.set_mask_geometry(mask, old))
