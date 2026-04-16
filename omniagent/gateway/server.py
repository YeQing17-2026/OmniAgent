"""Gateway WebSocket server for OmniAgent."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from aiohttp import web, WSMsgType

from omniagent.config import OmniAgentConfig
from omniagent.infra import get_logger
from .router import MessageRouter, IncomingMessage, OutgoingMessage
from .session import SessionManager

if TYPE_CHECKING:
    from omniagent.agents.reflexion import ReflexionAgent

logger = get_logger(__name__)


class GatewayServer:
    """WebSocket gateway server with optional channel support."""

    def __init__(self, config: OmniAgentConfig, agent: Optional["ReflexionAgent"] = None):
        """
        Initialize gateway server.

        Args:
            config: OmniAgent configuration
            agent: Optional ReflexionAgent instance for API endpoints
        """
        self.config = config
        self.agent = agent
        self.host = config.gateway.host
        self.port = config.gateway.port
        self._start_time = time.time()

        # Session management
        sessions_dir = Path.home() / ".omniagent" / "sessions"
        self.session_manager = SessionManager(
            storage_dir=sessions_dir,
            session_timeout=config.gateway.session_timeout,
        )

        # Message routing
        self.router = MessageRouter()

        # Channel system (lazy init)
        self.channel_manager = None
        self.message_bus = None

        # WebSocket connections
        self.connections: Dict[str, web.WebSocketResponse] = {}

        # Application
        self.app = web.Application()
        self.app.router.add_get("/ws", self.websocket_handler)
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_post("/message", self.http_message_handler)
        self.app.router.add_get("/", self.web_ui_handler)

        # API context and routes
        from .api import APIContext, create_api_router
        self.ctx = APIContext(
            agent=agent,
            session_manager=self.session_manager,
            config=self.config,
        )
        self.app["api_ctx"] = self.ctx
        self.app.router.add_routes(create_api_router(self.ctx))

        logger.info(
            "gateway_server_initialized",
            host=self.host,
            port=self.port,
        )

        # Setup approval event forwarding if agent is available
        if agent is not None:
            self._setup_approval_events(agent)

    def setup_channels(self) -> None:
        """Initialize channel system if channels are configured."""
        from omniagent.channels import ChannelManager, MessageBus

        channels_config = getattr(self.config, "channels", None)
        if channels_config is None:
            return

        # Check if any channel is actually configured.
        # Pydantic v2 with extra="allow" stores extra fields in __pydantic_extra__,
        # so we use model_dump() to include them.
        has_enabled = False
        if isinstance(channels_config, dict):
            sections = channels_config.values()
        else:
            sections = channels_config.model_dump().values()

        for section in sections:
            if section is None:
                continue
            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if enabled:
                has_enabled = True
                break

        if not has_enabled:
            return

        self.message_bus = MessageBus()
        self.router.set_message_bus(self.message_bus)
        self.channel_manager = ChannelManager(channels_config, self.message_bus)

        if self.channel_manager.enabled_channels:
            logger.info("channels_configured", channels=self.channel_manager.enabled_channels)

        self.ctx.channel_manager = self.channel_manager

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connections.

        Args:
            request: HTTP request

        Returns:
            WebSocket response
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Generate connection ID
        connection_id = id(ws)
        self.connections[connection_id] = ws

        logger.info("websocket_connected", connection_id=connection_id)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data, connection_id)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(
                        "websocket_error",
                        connection_id=connection_id,
                        error=ws.exception(),
                    )
        finally:
            self.connections.pop(connection_id, None)
            logger.info("websocket_disconnected", connection_id=connection_id)

        return ws

    async def _handle_message(
        self,
        ws: web.WebSocketResponse,
        data: str,
        connection_id: int,
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            ws: WebSocket response
            data: Message data (JSON)
            connection_id: Connection identifier
        """
        try:
            # Parse message
            payload = json.loads(data)
            logger.debug("message_received", payload=payload)

            # Handle approval response from client
            if payload.get("type") == "approval_response":
                await self._handle_approval_response(payload, connection_id)
                return

            # Extract fields
            session_id = payload.get("session_id")
            user_id = payload.get("user_id", "anonymous")
            channel_id = payload.get("channel_id", "web")
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})

            # Create incoming message
            incoming = IncomingMessage(
                session_id=session_id,
                user_id=user_id,
                channel_id=channel_id,
                content=content,
                metadata=metadata,
            )

            # Get or create session
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel_id=channel_id,
                session_id=session_id,
            )

            # Add user message to history
            session.add_message("user", content, metadata)

            # Update incoming message with session ID
            incoming.session_id = session.id

            # Route to agent
            outgoing = await self.router.route_to_agent(incoming)

            # Add assistant message to history
            session.add_message("assistant", outgoing.content, outgoing.metadata)

            # Send response
            response = {
                "session_id": outgoing.session_id,
                "content": outgoing.content,
                "metadata": outgoing.metadata,
            }
            await ws.send_json(response)

            logger.info(
                "message_handled",
                session_id=session.id,
                connection_id=connection_id,
            )

        except json.JSONDecodeError as e:
            logger.error("invalid_json", error=str(e))
            await ws.send_json({"error": "Invalid JSON"})
        except Exception as e:
            logger.error("message_handling_error", error=str(e))
            await ws.send_json({"error": str(e)})

    async def web_ui_handler(self, request: web.Request) -> web.Response:
        """Serve the web chat UI."""
        html_path = Path(__file__).parent / "web_ui.html"
        try:
            html = html_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            html = "<h1>OmniAgent Web UI not found</h1>"
        return web.Response(text=html, content_type="text/html", charset="utf-8")

    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Health check endpoint.

        Args:
            request: HTTP request

        Returns:
            Health status
        """
        status = {
            "status": "healthy",
            "sessions": len(self.session_manager.sessions),
            "connections": len(self.connections),
        }
        if self.channel_manager:
            status["channels"] = self.channel_manager.get_status()
        return web.json_response(status)

    async def http_message_handler(self, request: web.Request) -> web.Response:
        """
        HTTP message endpoint (alternative to WebSocket).

        Args:
            request: HTTP request

        Returns:
            Message response
        """
        try:
            # Parse request
            payload = await request.json()

            # Extract fields
            session_id = payload.get("session_id")
            user_id = payload.get("user_id", "anonymous")
            channel_id = payload.get("channel_id", "web")
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})

            # Create incoming message
            incoming = IncomingMessage(
                session_id=session_id,
                user_id=user_id,
                channel_id=channel_id,
                content=content,
                metadata=metadata,
            )

            # Get or create session
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel_id=channel_id,
                session_id=session_id,
            )

            # Add user message to history
            session.add_message("user", content, metadata)

            # Update incoming message with session ID
            incoming.session_id = session.id

            # Route to agent
            outgoing = await self.router.route_to_agent(incoming)

            # Add assistant message to history
            session.add_message("assistant", outgoing.content, outgoing.metadata)

            # Return response
            return web.json_response({
                "session_id": outgoing.session_id,
                "content": outgoing.content,
                "metadata": outgoing.metadata,
            })

        except Exception as e:
            logger.error("http_message_error", error=str(e))
            return web.json_response({"error": str(e)}, status=500)

    def _setup_approval_events(self, agent: "ReflexionAgent") -> None:
        """Subscribe to approval events and push to WebSocket clients."""

        from omniagent.agents.events import AgentEvent, EventType

        def on_approval_requested(event: AgentEvent):
            """Push approval request to all connected WebSocket clients."""
            for ws in list(self.connections.values()):
                try:
                    asyncio.create_task(ws.send_json({
                        "type": "approval_required",
                        "request_id": event.data["request_id"],
                        "tool": event.data["tool"],
                        "params": event.data.get("params", {}),
                        "description": event.data.get("description", ""),
                        "risk_level": event.data.get("risk_level", "medium"),
                    }))
                except Exception:
                    pass

        agent.event_bus.subscribe(
            EventType.APPROVAL_REQUESTED, on_approval_requested
        )
        logger.info("approval_events_subscribed")

    async def _handle_approval_response(
        self, payload: dict, connection_id: int
    ) -> None:
        """Handle approval/deny response from a WebSocket client."""
        if not self.agent:
            return

        request_id = payload.get("request_id", "")
        decision = payload.get("decision", "")

        if decision == "approve":
            self.agent.approval_manager.approve(request_id)
            logger.info("approval_approved_via_ws",
                        request_id=request_id, connection_id=connection_id)
        elif decision == "deny":
            self.agent.approval_manager.deny(request_id)
            logger.info("approval_denied_via_ws",
                        request_id=request_id, connection_id=connection_id)

        # Notify all clients that approval was resolved
        for ws in list(self.connections.values()):
            try:
                asyncio.create_task(ws.send_json({
                    "type": "approval_resolved",
                    "request_id": request_id,
                    "decision": decision,
                }))
            except Exception:
                pass

    async def start(self) -> None:
        """Start the gateway server."""
        logger.info("starting_gateway_server", host=self.host, port=self.port)

        # Setup and start channels
        self.setup_channels()
        if self.message_bus:
            await self.router.start_bridge()
        if self.channel_manager and self.channel_manager.enabled_channels:
            asyncio.create_task(self._run_channel_manager())

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

        # Run server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info("gateway_server_started", host=self.host, port=self.port)

    async def _run_channel_manager(self) -> None:
        """Run the channel manager in a background task."""
        try:
            await self.channel_manager.start_all()
        except Exception as e:
            logger.error("channel_manager_error", error=str(e))

    async def stop(self) -> None:
        """Stop the gateway server and channels."""
        if self.channel_manager:
            await self.channel_manager.stop_all()
        await self.router.stop_bridge()

    async def _cleanup_task(self) -> None:
        """Periodic cleanup of expired sessions."""
        while True:
            await asyncio.sleep(60)  # Run every minute
            try:
                cleaned = self.session_manager.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info("sessions_cleaned", count=cleaned)
            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    def run(self) -> None:
        """Run the gateway server (blocking)."""

        async def _run():
            await self.start()
            # Block forever
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
