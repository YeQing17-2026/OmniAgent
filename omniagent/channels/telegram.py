"""Telegram channel for OmniAgent."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseChannel
from .bus import OutboundMessage

logger = logging.getLogger(__name__)


class TelegramConfig:
    """Telegram channel configuration."""

    def __init__(
        self,
        enabled: bool = False,
        bot_token: str = "",
        allow_from: list = None,
        group_policy: str = "mention",
    ):
        self.enabled = enabled
        self.bot_token = bot_token
        self.allow_from = allow_from or []
        self.group_policy = group_policy  # "mention" or "open"


class TelegramChannel(BaseChannel):
    """Telegram channel using python-telegram-bot library."""

    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config: TelegramConfig, bus):
        super().__init__(config, bus)
        self.config = config
        self._app = None
        self._updater = None

    def _ensure_client(self) -> bool:
        """Initialize telegram bot if not already done. Returns True if successful."""
        if self._app is not None:
            return True
        try:
            from telegram import Update, Application
            self._app = Application.builder().token(self.config.bot_token).build()
            logger.info("telegram_client_initialized")
            return True
        except ImportError:
            logger.warning("telegram_not_installed", msg="python-telegram-bot is not installed. Install with: pip install python-telegram-bot")
            return False

    async def start(self) -> None:
        """Start Telegram bot polling."""
        if not self._ensure_client():
            logger.error("telegram_start_failed", reason="python-telegram-bot not installed")
            return

        from telegram.ext import filters

        async def error_handler(update, context):
            logger.error("telegram_update_error", error=str(context.error))

        self._app.add_error_handler(error_handler)

        # Start polling
        await self._app.initialize()
        await self._app.start_polling()
        self._running = True
        logger.info("telegram_bot_started")

    async def stop(self) -> None:
        """Stop Telegram bot polling."""
        self._running = False
        if self._app:
            await self._app.stop()
            await self._app.shutdown()
        logger.info("telegram_bot_stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to Telegram chat."""
        if not self._ensure_client():
            logger.warning("telegram_send_no_client")
            return
        try:
            chat_id = msg.chat_id
            if chat_id:
                await self._app.bot.send_message(chat_id=chat_id, text=msg.content)
            logger.debug("telegram_message_sent", chat_id=chat_id)
        except Exception as e:
            logger.error("telegram_send_error", error=str(e))

    async def _on_message(self, update) -> None:
        """Handler for Telegram update events."""
        if not update.message:
            return

        message = update.message
        if message.from_user and message.from_user.is_bot:
            return

        content = message.text or ""
        sender_id = str(message.from_user.id)
        chat_id = str(message.chat.id)

        is_group = message.chat.type != "private"
        if is_group and self.config.group_policy == "mention":
            # Check for bot mention
            entities = message.entities or []
            is_mentioned = any(
                e.type == "mention" for e in entities
            )
            if not is_mentioned:
                return

        await self._handle_message(sender_id, chat_id, content)
