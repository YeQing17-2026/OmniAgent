"""Base agent interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omniagent.gateway.router import IncomingMessage, OutgoingMessage
from .llm import LLMMessage


@dataclass
class AgentResult:
    """Agent execution result."""

    success: bool
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Agent(ABC):
    """Abstract agent interface."""

    @abstractmethod
    async def handle_message(self, message: IncomingMessage) -> OutgoingMessage:
        """
        Handle incoming message.

        Args:
            message: Incoming message

        Returns:
            Outgoing message
        """
        pass

    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task.

        Args:
            task: Task description
            context: Optional context

        Returns:
            Agent result
        """
        pass

    def clear_history(self) -> None:
        """Clear conversation history (optional override)."""
        pass

    def get_history(self) -> List[LLMMessage]:
        """Get conversation history (optional override)."""
        return []
