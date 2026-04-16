"""Async message bus for decoupled channel-agent communication.

Inbound:  Channel -> Agent (messages from users)
Outbound: Agent -> Channel (responses to users)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # e.g. "feishu"
    sender_id: str  # User identifier on the channel platform
    chat_id: str  # Chat/group ID on the platform
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: List[str] = field(default_factory=list)  # Local file paths
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str  # Target channel name
    chat_id: str  # Target chat ID on the platform
    content: str  # Message text
    reply_to: str = ""  # Optional reply-to message ID
    media: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Async message bus with inbound and outbound queues."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
