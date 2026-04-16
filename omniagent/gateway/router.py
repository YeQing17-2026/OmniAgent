"""Message router for OmniAgent gateway."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from omniagent.infra import get_logger

if TYPE_CHECKING:
    from omniagent.channels.bus import MessageBus

logger = get_logger(__name__)


@dataclass
class IncomingMessage:
    """Incoming message from channel."""

    session_id: Optional[str]
    user_id: str
    channel_id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class OutgoingMessage:
    """Outgoing message to channel."""

    session_id: str
    content: str
    metadata: Dict[str, Any]


class MessageRouter:
    """Routes messages between channels and agents.

    Supports two modes:
    1. Direct mode: agent_handler called directly (WebSocket/HTTP)
    2. Bus mode: consumes from MessageBus, bridges to agent, publishes responses
    """

    def __init__(self):
        """Initialize message router."""
        self.agent_handler: Optional[Callable] = None
        self._bus: Optional["MessageBus"] = None
        self._bridge_task: Optional[Callable] = None
        logger.info("message_router_initialized")

    def set_agent_handler(self, handler: Callable) -> None:
        """
        Set agent message handler.

        Args:
            handler: Async function that handles messages
                     Signature: async def handler(message: IncomingMessage) -> OutgoingMessage
        """
        self.agent_handler = handler
        logger.info("agent_handler_registered")

    def set_message_bus(self, bus: "MessageBus") -> None:
        """Set the message bus for channel integration."""
        self._bus = bus
        logger.info("message_bus_registered")

    async def start_bridge(self) -> None:
        """Start the inbound bridge loop that consumes from the bus and routes to agent."""
        if self._bus is None:
            return
        import asyncio
        self._bridge_task = asyncio.create_task(self._bridge_loop())

    async def stop_bridge(self) -> None:
        """Stop the bridge loop."""
        if self._bridge_task:
            import asyncio
            self._bridge_task.cancel()
            try:
                await self._bridge_task
            except asyncio.CancelledError:
                pass
            self._bridge_task = None

    async def _bridge_loop(self) -> None:
        """Consume inbound messages from bus and route to agent, then publish outbound."""
        import asyncio
        from omniagent.channels.bus import OutboundMessage

        while True:
            try:
                inbound = await asyncio.wait_for(
                    self._bus.consume_inbound(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # Convert InboundMessage -> IncomingMessage
                # Namespace user_id to avoid collision across channels
                incoming = IncomingMessage(
                    session_id=None,
                    user_id=f"{inbound.channel}:{inbound.sender_id}",
                    channel_id=inbound.channel,
                    content=inbound.content,
                    metadata={
                        "chat_id": inbound.chat_id,
                        "media": inbound.media,
                        **inbound.metadata,
                    },
                )

                # Route to agent
                outgoing = await self.route_to_agent(incoming)

                # Convert OutgoingMessage -> OutboundMessage for channel delivery
                chat_id = outgoing.metadata.get("chat_id", inbound.chat_id)
                outbound = OutboundMessage(
                    channel=inbound.channel,
                    chat_id=chat_id,
                    content=outgoing.content,
                    media=outgoing.metadata.get("media", []),
                    metadata=outgoing.metadata,
                )
                await self._bus.publish_outbound(outbound)

            except Exception as e:
                logger.error("bridge_loop_error", error=str(e))

    async def route_to_agent(self, message: IncomingMessage) -> OutgoingMessage:
        """
        Route message to agent.

        Args:
            message: Incoming message

        Returns:
            Agent response

        Raises:
            RuntimeError: If agent handler not set
        """
        if self.agent_handler is None:
            raise RuntimeError("Agent handler not set")

        logger.info(
            "routing_to_agent",
            session_id=message.session_id,
            user_id=message.user_id,
            channel_id=message.channel_id,
        )

        try:
            response = await self.agent_handler(message)
            logger.info(
                "agent_response_received",
                session_id=response.session_id,
            )
            return response
        except Exception as e:
            logger.error(
                "agent_error",
                session_id=message.session_id,
                error=str(e),
            )
            # Return error message
            return OutgoingMessage(
                session_id=message.session_id or "unknown",
                content=f"Error: {str(e)}",
                metadata={"error": True},
            )

    async def route_to_channel(self, message: OutgoingMessage, channel_id: str) -> None:
        """
        Route message to channel via the message bus.

        Args:
            message: Outgoing message
            channel_id: Target channel
        """
        if self._bus:
            from omniagent.channels.bus import OutboundMessage
            outbound = OutboundMessage(
                channel=channel_id,
                chat_id=message.metadata.get("chat_id", ""),
                content=message.content,
                metadata=message.metadata,
            )
            await self._bus.publish_outbound(outbound)

        logger.info(
            "routing_to_channel",
            session_id=message.session_id,
            channel_id=channel_id,
        )
