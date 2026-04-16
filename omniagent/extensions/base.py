"""Base extension class and API interface for OmniAgent plugins."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from omniagent.agents.events import EventBus
    from omniagent.agents.hooks import ToolHookManager
    from omniagent.config import OmniAgentConfig
    from omniagent.tools.base import Tool, ToolRegistry


class Extension:
    """Base class for OmniAgent extensions.

    Subclass this and implement on_load/on_unload to create a plugin.
    The extension will receive an ExtensionAPI instance with access to
    the agent's event bus, tool registry, and configuration.
    """

    name: str = ""
    version: str = "0.1.0"

    async def on_load(self, api: "ExtensionAPI") -> None:
        """Called when the extension is loaded. Register tools, hooks, event handlers here."""
        pass

    async def on_unload(self) -> None:
        """Called when the extension is unloaded. Clean up resources here."""
        pass


@dataclass
class ExtensionAPI:
    """API provided to extensions for interacting with the agent system.

    Extensions can use this to register custom tools, subscribe to events,
    add tool execution hooks, and access configuration.
    """

    event_bus: "EventBus"
    tool_registry: "ToolRegistry"
    tool_hook_manager: "ToolHookManager"
    config: "OmniAgentConfig"
    work_dir: Path

    _extra_tools: List["Tool"] = field(default_factory=list)

    def register_tool(self, tool: "Tool") -> None:
        """Register a custom tool with the agent."""
        self._extra_tools.append(tool)
        self.tool_registry.register(tool)

    def subscribe(self, event_type, handler) -> None:
        """Subscribe to an agent lifecycle event type."""
        from omniagent.agents.events import EventType
        self.event_bus.subscribe(event_type, handler)

    def add_before_tool_hook(self, hook) -> None:
        """Add a before-tool-call hook that can block execution."""
        self.tool_hook_manager.add_before_hook(hook)

    def add_after_tool_hook(self, hook) -> None:
        """Add an after-tool-call hook that can modify results."""
        self.tool_hook_manager.add_after_hook(hook)
