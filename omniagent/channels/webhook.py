"""Generic Webhook channel for OmniAgent.

Receives messages via HTTP POST and sends responses via outbound HTTP POST.
"""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional
from aiohttp import web

from .base import BaseChannel
from .bus import OutboundMessage

logger = logging.getLogger(__name__)


class WebhookConfig:
    """Webhook channel configuration."""

    def __init__(
        self,
        enabled: bool = False,
        allow_from: list = None,
        inbound_path: str = "/webhook/in",
        inbound_secret: str = "",
        outbound_url: str = "",
    ):
        self.enabled = enabled
        self.allow_from = allow_from or []
        self.inbound_path = inbound_path
        self.inbound_secret = inbound_secret
        self.outbound_url = outbound_url


class WebhookChannel(BaseChannel):
    """Generic Webhook channel that receives and sends messages via HTTP."""

    name = "webhook"
    display_name = "Webhook"

    def __init__(self, config: WebhookConfig, bus):
        super().__init__(config, bus)
        self.config = config
        self._app: Optional[web.Application] = None

    async def _verify_signature(self, request) -> bool:
        """Verify HMAC-SHA256 signature if secret is configured."""
        if not self.config.inbound_secret:
            return True
        signature = request.headers.get("X-Signature-256", "")
        if not signature:
            return False
        timestamp = request.headers.get("X-Timestamp", "")
        body = await request.read()
        expected = hmac.new(
            self.config.inbound_secret.encode(),
            f"{timestamp}.{body}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    async def _inbound_handler(self, request) -> web.Response:
        """Handle inbound webhook POST request."""
        if request.method != "POST":
            return web.Response(status=405)

        if not await self._verify_signature(request):
            return web.Response(status=401, text="Invalid signature")

        try:
            body = await request.json()
            content = body.get("content", "")
            sender_id = body.get("sender_id", "")
            chat_id = body.get("chat_id", "")
            metadata = body.get("metadata", {})
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("webhook_invalid_payload", error=str(e))
            return web.Response(status=400, text=f"Invalid payload: {e}")

        await self._handle_message(sender_id, chat_id, content, metadata=metadata)
        return web.Response(text="OK")

    async def start(self) -> None:
        """Start the webhook HTTP server."""
        self._app = web.Application()
        self._app.router.add_post(self.config.inbound_path, self._inbound_handler)
        runner = web.AppRunner(self._app)
        site = web.TCPSite(runner)
        await site.start()
        self._running = True
        port = 8080
        logger.info("webhook_server_started", path=self.config.inbound_path, port=port)

    async def stop(self) -> None:
        """Stop the webhook HTTP server."""
        self._running = False
        if self._app:
            await self._app.shutdown()
        logger.info("webhook_server_stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send message via outbound HTTP POST."""
        if not self.config.outbound_url:
            logger.warning("webhook_no_outbound_url", chat_id=msg.chat_id)
            return

        try:
            import aiohttp
            payload = {
                "chat_id": msg.chat_id,
                "content": msg.content,
                "metadata": msg.metadata or {},
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.outbound_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(
                            "webhook_send_failed",
                            status=resp.status,
                            url=self.config.outbound_url,
                        )
        except Exception as e:
            logger.error("webhook_send_error", error=str(e))
