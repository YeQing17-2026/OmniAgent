"""Base channel interface for chat platforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger
from .bus import InboundMessage, MessageBus, OutboundMessage

logger = get_logger(__name__)


class BaseChannel(ABC):
    """Abstract base class for chat channel implementations.

    Subclasses must implement start(), stop(), and send().
    """

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the channel, connect, and listen for messages.

        This method should block until stop() is called.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through this channel. Raise on failure for retry."""
        ...

    def is_allowed(self, sender_id: str) -> bool:
        """Check if sender_id is in the allow_from list.

        Empty list = deny all, '*' = allow all.
        """
        allow_list: List[str] = getattr(self.config, "allow_from", [])
        if not allow_list:
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in [str(x) for x in allow_list]

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Check permissions and forward to the message bus."""
        if not self.is_allowed(sender_id):
            logger.warning("channel_access_denied", channel=self.name, sender_id=sender_id)
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
        )
        await self.bus.publish_inbound(msg)

    @property
    def is_running(self) -> bool:
        return self._running
