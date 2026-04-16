"""Agent state tracking dataclass."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    """Unified state object for the agent runtime."""

    is_streaming: bool = False
    error: Optional[str] = None
    iteration: int = 0
    total_tool_calls: int = 0
    pending_tool_calls: set = field(default_factory=set)
