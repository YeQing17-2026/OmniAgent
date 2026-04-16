"""Discord channel for OmniAgent."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseChannel
from .bus import OutboundMessage

logger = logging.getLogger(__name__)


class DiscordConfig:
    """Discord channel configuration."""

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


class DiscordChannel(BaseChannel):
    """Discord channel using discord.py library."""

    name = "discord"
    display_name = "Discord"

    def __init__(self, config: DiscordConfig, bus):
        super().__init__(config, bus)
        self.config = config
        self._client = None
        self._bot = None

    def _ensure_client(self) -> bool:
        """Initialize discord client if not already done. Returns True if successful."""
        if self._client is not None:
            return True
        try:
            from discord import Client
            from discord.ext import commands
            self._client = Client(intents=Client.intents.default())
            self._bot = commands.Bot(command_prefix="!", intents=self._client.intents)
            self._commands = commands
            logger.info("discord_client_initialized")
            return True
        except ImportError:
            logger.warning("discord_not_installed", msg="discord.py is not installed. Install with: pip install discord.py")
            return False

    async def start(self) -> None:
        """Start Discord bot listener."""
        if not self._ensure_client():
            logger.error("discord_start_failed", reason="discord.py not installed")
            return
        await self._client.start(self.config.bot_token)
        self._running = True
        logger.info("discord_bot_started")

    async def stop(self) -> None:
        """Stop Discord bot."""
        self._running = False
        if self._client:
            await self._client.close()
        logger.info("discord_bot_stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to Discord channel."""
        if not self._ensure_client():
            logger.warning("discord_send_no_client")
            return
        try:
            channel = self._client.get_channel(int(msg.chat_id))
            if channel:
                await channel.send(msg.content)
            logger.debug("discord_message_sent", channel_id=msg.chat_id)
        except Exception as e:
            logger.error("discord_send_error", error=str(e))

    async def _on_message(self, message) -> None:
        """Handler for Discord message events."""
        if message.author.bot:
            return

        content = message.content or ""
        sender_id = str(message.author.id)
        chat_id = str(message.channel.id)
        is_group = message.channel.type != 0  # DM type

        if is_group and self.config.group_policy == "mention":
            # Check if bot is mentioned
            if not message.mentions:
                return

        await self._handle_message(sender_id, chat_id, content)
