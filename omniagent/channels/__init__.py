"""Channel management system for OmniAgent.

Provides multi-channel support for connecting to chat platforms (Feishu, etc.)
via a message bus that decouples channels from the agent.
"""

from .base import BaseChannel
from .bus import InboundMessage, OutboundMessage, MessageBus
from .manager import ChannelManager

__all__ = [
    "BaseChannel",
    "ChannelManager",
    "InboundMessage",
    "OutboundMessage",
    "MessageBus",
]
