"""Agent event system for lifecycle observation and extension hooks."""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable


class EventType(str, Enum):
    """Agent lifecycle event types (4 granularity levels)."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # Turn lifecycle (one LLM call + tool execution)
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Message lifecycle (streaming)
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"

    # Tool execution lifecycle
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_END = "tool_execution_end"

    # Compaction lifecycle
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"

    # Approval lifecycle
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"


@dataclass
class AgentEvent:
    """A typed event emitted during agent execution."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# Type alias for async event handlers
EventHandler = Callable[[AgentEvent], Awaitable[None]]


class EventBus:
    """Async publish/subscribe event bus for agent lifecycle events."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._global_subscribers: List[EventHandler] = []

    def subscribe(
        self, event_type: EventType, handler: EventHandler
    ) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all event types."""
        self._global_subscribers.append(handler)

    def unsubscribe(
        self, event_type: EventType, handler: EventHandler
    ) -> None:
        """Unsubscribe from a specific event type."""
        handlers = self._subscribers.get(event_type)
        if handlers:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribers. Handlers run concurrently."""
        tasks: List[Awaitable[None]] = []

        # Type-specific subscribers
        for handler in self._subscribers.get(event.type, []):
            tasks.append(handler(event))

        # Global subscribers
        for handler in self._global_subscribers:
            tasks.append(handler(event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
