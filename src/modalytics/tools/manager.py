"""
Tool registry, active tool and setting values (Qt-free).
"""

from __future__ import annotations

from modalytics.core.signals import Signal
from modalytics.tools.base import Setting, Tool


class ToolManager:
    """Holds the tools, which one is active, and every setting's value.

    Signals:
        active_changed(tool): the active tool changed.
        value_changed(tool, key, value): a setting changed.
        info_changed(tool, text): a tool's live status line changed.
        actions_changed(tool): which of a tool's actions apply may have changed.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.active: Tool | None = None
        self._values: dict[tuple[str, str], object] = {}
        self.active_changed = Signal()
        self.value_changed = Signal()
        self.info_changed = Signal()
        self.actions_changed = Signal()
        self._info: dict[str, str] = {}
        self._ctx = None

    def register(self, tool_cls: type[Tool]) -> Tool:
        """Create and register a tool; its settings start at their defaults."""
        tool = tool_cls(self)
        self.tools[tool.id] = tool
        for s in tool.settings:
            self._values.setdefault(self._key(tool, s), s.default)
        return tool

    def bind(self, ctx) -> None:
        """Give the manager the canvas context (needed to (de)activate tools)."""
        self._ctx = ctx

    def activate(self, tool_id: str) -> None:
        """Make a tool active (no-op if it already is)."""
        tool = self.tools[tool_id]
        if tool is self.active:
            return
        if self.active is not None and self._ctx is not None:
            self.active.deactivate(self._ctx)
        self.active = tool
        if self._ctx is not None:
            tool.activate(self._ctx)
            self._ctx.request_draw()             # the previous tool's previews were cleared: show that
        self.active_changed.emit(tool)

    # Settings
    def setting(self, tool: Tool, key: str) -> Setting:
        return next(s for s in tool.settings if s.key == key)

    def _key(self, tool: Tool, setting: Setting) -> tuple[str, str]:
        return ("*" if setting.shared else tool.id, setting.key)

    def value(self, tool: Tool, key: str):
        return self._values[self._key(tool, self.setting(tool, key))]

    def profile(self):
        """The current project's profile (None before the canvas is bound or without a project)."""
        return getattr(getattr(self._ctx, "project", None), "profile", None)

    def trigger(self, tool: Tool, key: str) -> None:
        """Run one of a tool's actions (ignored unless that tool is active)."""
        if self._ctx is not None and tool is self.active:
            tool.on_action(self._ctx, key)

    def set_info(self, tool: Tool, text: str) -> None:
        """Set a tool's live status line (shown in Tool Settings)."""
        self._info[tool.id] = text
        self.info_changed.emit(tool, text)

    def info(self, tool: Tool) -> str:
        return self._info.get(tool.id, "")

    def set_value(self, tool: Tool, key: str, value) -> None:
        s = self.setting(tool, key)
        if s.kind in ("float", "int") and s.minimum is not None:
            value = max(s.minimum, min(s.maximum, value))
        self._values[self._key(tool, s)] = value
        self.value_changed.emit(tool, key, value)
        if tool is self.active and self._ctx is not None:
            tool.on_setting_changed(self._ctx, key, value)
