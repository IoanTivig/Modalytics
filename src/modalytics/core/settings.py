"""
Declared settings: one description of a user-adjustable value, shared by
everything that has settings (tools, parameters, later classifiers).

Whoever owns settings declares them as a list of ``Setting``; the UI builds
the form from that list (Tool Settings panel, Settings › Parameters), so every
setting looks and behaves the same way and the owners never touch Qt.

Parameter scripts declare theirs (a parameter's *constants*) as plain dicts
(no import needed), turned into ``Setting`` objects by ``Setting.from_dict``::

    CONSTANTS = [
        {"key": "degree", "label": "Detrend degree", "kind": "int", "default": 2,
         "minimum": 0, "maximum": 4, "tooltip": "Polynomial removed before ..."},
    ]
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable

KINDS = ("float", "int", "choice", "bool", "label", "file", "folder", "text")


@dataclass
class Setting:
    """One user-adjustable setting.

    Attributes:
        key: Identifier used by the owner (e.g. ``tool.value(key)``).
        label: Text shown in the form.
        kind: "float", "int", "choice", "bool", "label" (an annotation class), "file" (a file's path), "folder" (a
            folder's path) or "text".
        default: Initial value.
        minimum, maximum, step: Ranges for numbers.
        suffix: Unit shown after numbers (e.g. " µm").
        choices: Options for "choice", or a function returning them (called each
            time the form is built, for lists that depend on the machine).
        shared: Shared by all tools (e.g. the active annotation class), so
            switching from Brush to Polygon keeps the same class.
        tooltip: Help text.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    suffix: str = ""
    choices: list[str] | Callable[[], list[str]] = field(default_factory=list)
    shared: bool = False
    tooltip: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Setting":
        """Build a setting from a plain dict (as parameter scripts declare them).

        Raises:
            ValueError: If a key is unknown, or ``key``/``label``/``kind`` is missing or wrong.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown setting field(s): {', '.join(sorted(unknown))}")
        if not d.get("key") or not d.get("label") or d.get("kind") not in KINDS:
            raise ValueError(f"a setting needs 'key', 'label' and a 'kind' in {KINDS}")
        return cls(**d)

    def clean(self, value):
        """``value`` converted to this setting's type and range (the default if unusable)."""
        try:
            if self.kind == "int":
                value = int(value)
            elif self.kind == "float":
                value = float(value)
            elif self.kind == "bool":
                value = bool(value)
            elif self.kind in ("file", "folder", "text"):
                value = "" if value is None else str(value)
            elif self.kind == "label":
                value = _label_value(value)
            elif self.kind == "choice" and not callable(self.choices) and value not in self.choices:
                return self.default
        except (TypeError, ValueError):
            return self.default
        if self.kind in ("int", "float"):
            if self.minimum is not None:
                value = max(self.minimum, value)
            if self.maximum is not None:
                value = min(self.maximum, value)
        return value


def _label_value(value) -> tuple[str, str]:
    """An annotation class as (group, subgroup) from a tuple, a list (YAML) or "Group › Subgroup" text."""
    if isinstance(value, (tuple, list)):
        group = str(value[0]) if value else ""
        sub = str(value[1]) if len(value) > 1 and value[1] is not None else ""
        return group.strip(), sub.strip()
    text = str(value or "")
    for sep in ("›", ">", "/"):
        if sep in text:
            group, sub = text.split(sep, 1)
            return group.strip(), sub.strip()
    return text.strip(), ""
