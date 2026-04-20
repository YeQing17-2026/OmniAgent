"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection.

No public IP or webhook required. The SDK's WebSocket client runs in a daemon
thread and bridges to the main asyncio event loop.

Reference: nanobot-main/nanobot/channels/feishu.py (simplified)
"""

import asyncio
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

import importlib.util
from pydantic import BaseModel, Field

from omniagent.infra import get_logger
from .base import BaseChannel
from .bus import OutboundMessage, MessageBus

logger = get_logger(__name__)

FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None

MSG_TYPE_PLACEHOLDER = {
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
    "video": "[video]",
}


class FeishuConfig(BaseModel):
    """Feishu/Lark channel configuration."""

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    allow_from: List[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "mention"


class FeishuChannel(BaseChannel):
    """Feishu/Lark channel using WebSocket long connection.

    No public IP required — the lark_oapi SDK maintains a persistent
    WebSocket connection to Feishu's servers.
    """

    name = "feishu"
    display_name = "Feishu"

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = FeishuConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None  # lark.Client for REST API
        self._ws_client: Any = None  # lark.ws.Client
        self._ws_thread: Optional[threading.Thread] = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error(
                "feishu_sdk_not_installed",
                hint="lark-oapi is missing from the current OmniAgent environment. Reinstall OmniAgent to restore bundled runtime dependencies.",
            )
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("feishu_credentials_missing")
            return

        import lark_oapi as lark

        self._running = True
        self._loop = asyncio.get_running_loop()

        # Lark REST API client (for sending messages)
        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

        # Event dispatcher for incoming messages
        event_handler = (
            lark.EventDispatcherHandler.builder(
                self.config.encrypt_key or "",
                self.config.verification_token or "",
            )
            .register_p2_im_message_receive_v1(self._on_message_sync)
            .build()
        )

        # WebSocket client
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.WARNING,
        )

        # Run WebSocket in a daemon thread with its own event loop.
        # lark_oapi's WS client is synchronous and calls asyncio.get_event_loop()
        # internally, so we give it a fresh loop.
        def run_ws():
            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            try:
                import lark_oapi.ws.client as _lark_ws_client
                _lark_ws_client.loop = ws_loop
            except Exception:
                pass
            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as e:
                        logger.warning("feishu_ws_error", error=str(e))
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        logger.info("feishu_started", app_id=self.config.app_id[:8] + "...")

        # Block until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        logger.info("feishu_stopped")

    # ── Message sending ──────────────────────────────────────────────

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu.

        Sends text messages and optional image attachments.
        """
        if not self._client:
            logger.warning("feishu_client_not_initialized")
            return

        try:
            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
            loop = asyncio.get_running_loop()

            # Send images first
            for file_path in msg.media:
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                if ext in {"png", "jpg", "jpeg", "gif", "bmp", "webp"}:
                    image_key = await loop.run_in_executor(
                        None, self._upload_image_sync, file_path
                    )
                    if image_key:
                        content = json.dumps({"image_key": image_key}, ensure_ascii=False)
                        await loop.run_in_executor(
                            None,
                            self._send_message_sync,
                            receive_id_type,
                            msg.chat_id,
                            "image",
                            content,
                        )

            # Send text
            if msg.content and msg.content.strip():
                text_body = json.dumps({"text": msg.content.strip()}, ensure_ascii=False)
                await loop.run_in_executor(
                    None,
                    self._send_message_sync,
                    receive_id_type,
                    msg.chat_id,
                    "text",
                    text_body,
                )

        except Exception as e:
            logger.error("feishu_send_error", error=str(e))
            raise

    def _send_message_sync(
        self, receive_id_type: str, receive_id: str, msg_type: str, content: str
    ) -> bool:
        """Send a message synchronously via Lark SDK."""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error("feishu_send_failed", code=response.code, msg=response.msg)
                return False
            return True
        except Exception as e:
            logger.error("feishu_send_exception", error=str(e))
            return False

    def _upload_image_sync(self, file_path: str) -> Optional[str]:
        """Upload image and return image_key."""
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        try:
            with open(file_path, "rb") as f:
                request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.image.create(request)
                if response.success():
                    return response.data.image_key
                return None
        except Exception as e:
            logger.error("feishu_upload_image_error", error=str(e))
            return None

    def _download_image_sync(self, message_id: str, image_key: str) -> Optional[bytes]:
        """Download image bytes from Feishu message."""
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                data = response.file
                if hasattr(data, "read"):
                    data = data.read()
                return data
            return None
        except Exception as e:
            logger.error("feishu_download_image_error", error=str(e))
            return None

    # ── Message receiving ────────────────────────────────────────────

    def _on_message_sync(self, data: Any) -> None:
        """Sync handler called from WebSocket thread. Bridges to async."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Deduplication
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            # Skip bot messages
            if sender.sender_type == "bot":
                return

            # Extract identifiers
            sender_id = (
                sender.sender_id.open_id if sender.sender_id else "unknown"
            )
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type

            # Group policy check
            if chat_type == "group" and not self._is_group_message_for_bot(message):
                logger.debug("feishu_skip_group_not_mentioned")
                return

            # Parse content
            content_parts: List[str] = []
            media_paths: List[str] = []

            try:
                content_json = json.loads(message.content) if message.content else {}
            except (json.JSONDecodeError, TypeError):
                content_json = {}

            if msg_type == "text":
                text = content_json.get("text", "")
                if text:
                    content_parts.append(text)

            elif msg_type == "post":
                text = self._extract_post_text(content_json)
                if text:
                    content_parts.append(text)

            elif msg_type == "image":
                image_key = content_json.get("image_key")
                if image_key:
                    loop = asyncio.get_running_loop()
                    data_bytes = await loop.run_in_executor(
                        None, self._download_image_sync, message_id, image_key
                    )
                    if data_bytes:
                        media_dir = Path.home() / ".omniagent" / "media"
                        media_dir.mkdir(parents=True, exist_ok=True)
                        import os
                        file_path = os.path.join(str(media_dir), f"{image_key[:16]}.jpg")
                        with open(file_path, "wb") as f:
                            f.write(data_bytes)
                        media_paths.append(file_path)
                    content_parts.append("[image]")

            elif msg_type in MSG_TYPE_PLACEHOLDER:
                content_parts.append(MSG_TYPE_PLACEHOLDER[msg_type])

            else:
                content_parts.append(f"[{msg_type}]")

            content = "\n".join(content_parts) if content_parts else ""

            if not content and not media_paths:
                return

            # Forward to bus
            # For group chats, reply to the chat; for p2p, reply to sender
            reply_chat_id = chat_id if chat_type == "group" else chat_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_chat_id,
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                },
            )

        except Exception as e:
            logger.error("feishu_message_processing_error", error=str(e))

    # ── Group policy ─────────────────────────────────────────────────

    def _is_bot_mentioned(self, message: Any) -> bool:
        """Check if the bot is @mentioned in the message."""
        mentions = getattr(message, "mentions", None) or []
        for mention in mentions:
            mid = getattr(mention, "id", None)
            if not mid:
                continue
            open_id = getattr(mid, "open_id", "")
            if open_id and open_id.startswith("ou_"):
                return True
        return False

    def _is_group_message_for_bot(self, message: Any) -> bool:
        """Allow group messages when policy is open or bot is @mentioned."""
        if self.config.group_policy == "open":
            return True
        return self._is_bot_mentioned(message)

    # ── Content parsing helpers ──────────────────────────────────────

    @staticmethod
    def _extract_post_text(content_json: dict) -> str:
        """Extract plain text from a Feishu post (rich text) message."""
        root = content_json
        if isinstance(root, dict) and isinstance(root.get("post"), dict):
            root = root["post"]

        for key in ("zh_cn", "en_us", "content"):
            block = root.get(key) if isinstance(root, dict) else None
            if not block or not isinstance(block, dict):
                continue
            texts = []
            if title := block.get("title"):
                texts.append(title)
            for row in block.get("content", []):
                if not isinstance(row, list):
                    continue
                for el in row:
                    if not isinstance(el, dict):
                        continue
                    tag = el.get("tag")
                    if tag in ("text", "a"):
                        texts.append(el.get("text", ""))
                    elif tag == "at":
                        texts.append(f"@{el.get('user_name', 'user')}")
            result = " ".join(texts).strip()
            if result:
                return result
        return ""


# Import at bottom to avoid circular
from pathlib import Path  # noqa: E402
