"""
Tool framework: what a tool is, and how its settings are declared.

Tools and their settings are separate on purpose:

* a ``Tool`` holds the *logic* (what happens on click, drag, key);
* its ``settings`` are a declarative list of ``Setting`` objects. The Tool
  Settings panel builds the form from that list, so tools never touch Qt
  and every tool's settings look and behave the same way;
* its ``actions`` are commands (e.g. Register's Start / Previous / Next /
  Save). The ribbon shows them next to the tool's button, greyed out unless
  the tool is active and ``action_enabled`` says they apply.

Navigation is never a tool's business: wheel zoom, middle-drag pan and
Ctrl+middle-drag move are handled by the canvas whatever tool is active.
Tools receive left/right clicks, drags, double-clicks, hover and keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modalytics.core.settings import Setting  # noqa: F401  (declared here for tools; shared with parameters)


@dataclass
class ToolAction:
    """A command of a tool, shown in the ribbon next to the tool.

    Attributes:
        key: Identifier passed to ``Tool.on_action``.
        label: Button text.
        icon: qtawesome name.
        tooltip: Help text (mention the keyboard equivalent, if any).
    """

    key: str
    label: str
    icon: str
    tooltip: str = ""


## The "Device" setting of tools that run models ##

AUTO_DEVICE = "Auto"


def _device_choices() -> list[str]:
    """Auto, every GPU the models can use here, CPU (just Auto/CPU while the list is being built)."""
    from modalytics.services import compute
    found = compute.known_devices() or []
    return [AUTO_DEVICE] + [d.label() for d in found if d.is_gpu] + ["CPU"]


def device_setting() -> Setting:
    """Where a tool's models run (see ``services.compute``)."""
    return Setting("device", "Device", "choice", AUTO_DEVICE, choices=_device_choices,
                   tooltip="Auto: the best device found on this machine (shown in the status bar)")


def device_key(choice: str) -> str:
    """Turn a Device setting value into a ``device`` argument for the services ("auto", "cpu", "dml:1", …)."""
    from modalytics.services import compute
    if choice == "CPU":
        return "cpu"
    return next((d.key for d in compute.known_devices() or [] if d.label() == choice), "auto")


class Tool:
    """Base class of interactive canvas tools.

    Class attributes describe how the tool appears in the ribbon; instances
    implement the event handlers. Handlers return True when they consumed
    the event (otherwise the canvas applies its default, e.g. the context
    menu on right-click).

    Event dicts are rendercanvas events ("x", "y" in screen px, "button",
    "buttons", "modifiers"); ``wx, wy`` are the matching world µm.
    """

    id = ""
    label = ""
    icon = ""             # qtawesome name, e.g. "mdi6.vector-polygon"
    shortcut = ""         # single key, e.g. "P"
    tabs: tuple[str, ...] = ()
    group = ""            # ribbon group title
    hint = ""             # one-line usage help shown in Tool Settings
    uses_shift_click = False   # True: Shift + click is the tool's own (else, in every tool, it selects layers)
    pointer = "default"   # the system pointer over the canvas: "default" (arrow), "crosshair", or "none" where the
                          # tool draws its own (Brush's ring)

    def busy(self) -> bool:
        """True in the middle of a gesture (e.g. a polygon being drawn): Shift + click then stays the tool's."""
        return bool(getattr(self, "points", None))
    settings: list[Setting] = []
    actions: list[ToolAction] = []

    def __init__(self, manager) -> None:
        self.manager = manager

    def value(self, key: str):
        """Current value of one of this tool's settings."""
        return self.manager.value(self, key)

    def set_info(self, text: str) -> None:
        """Show a live status line under the hint in Tool Settings (e.g. what will be processed)."""
        self.manager.set_info(self, text)

    def action_enabled(self, key: str) -> bool:
        """Whether one of this tool's actions applies right now (asked only while the tool is active)."""
        return True

    def actions_changed(self) -> None:
        """Tell the UI to ask ``action_enabled`` again (e.g. after a result arrived)."""
        self.manager.actions_changed.emit(self)

    # Lifecycle
    def activate(self, ctx) -> None:
        """Called when the tool becomes active."""

    def deactivate(self, ctx) -> None:
        """Called when another tool is chosen; must clear previews."""
        ctx.preview.clear()

    # Events (return True if consumed)
    def on_pointer_down(self, ctx, ev: dict, wx: float, wy: float) -> bool:
        return False

    def on_pointer_move(self, ctx, ev: dict, wx: float, wy: float) -> bool:
        return False

    def on_pointer_up(self, ctx, ev: dict, wx: float, wy: float) -> bool:
        return False

    def on_double_click(self, ctx, ev: dict, wx: float, wy: float) -> bool:
        return False

    def on_key(self, ctx, ev: dict) -> bool:
        return False

    def on_action(self, ctx, key: str) -> None:
        """One of this tool's ``actions`` was pressed (only happens while the tool is active)."""

    def on_setting_changed(self, ctx, key: str, value) -> None:
        """A setting of this tool changed while it is active (e.g. redraw a preview that depends on it)."""


@dataclass
class ToolContext:
    """What the canvas gives a tool to act on.

    Attributes:
        project: The domain model.
        undo: Undo stack.
        view: The canvas view (µm ↔ screen px).
        preview: Canvas preview drawing API (``set_path``, ``set_stroke``,
            ``set_circle``, ``clear``).
        request_draw: Schedule a redraw.
        tasks: Background task runner (``run``, ``cancel``, ``message``, ``busy``),
            provided by the UI; None in headless use.
    """

    project: Any
    undo: Any
    view: Any
    preview: Any
    request_draw: Any
    tasks: Any = None
