"""Channel manager for coordinating chat channels."""

import asyncio
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger
from .base import BaseChannel
from .bus import MessageBus, OutboundMessage

logger = get_logger(__name__)

_RETRY_DELAYS = (1, 2, 4)
_MAX_RETRIES = 3


class ChannelManager:
    """Manages chat channels and coordinates message routing.

    Discovers enabled channels from config, starts/stops them,
    and dispatches outbound messages to the correct channel.
    """

    def __init__(self, channels_config: Any, bus: MessageBus):
        """
        Args:
            channels_config: ChannelsConfig instance with extra="allow".
                             Each enabled channel has a sub-field (e.g. .feishu).
            bus: The shared message bus.
        """
        self.config = channels_config
        self.bus = bus
        self.channels: Dict[str, BaseChannel] = {}
        self._dispatch_task: Optional[asyncio.Task] = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Discover and initialize enabled channels."""
        from .registry import discover_all

        # Pydantic v2 with extra="allow" stores extra fields in __pydantic_extra__,
        # so we use model_dump() to include them.
        if isinstance(self.config, dict):
            all_sections = self.config
        else:
            all_sections = self.config.model_dump()

        for name, cls in discover_all().items():
            section = all_sections.get(name) if isinstance(all_sections, dict) else None
            if section is None:
                continue

            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if not enabled:
                continue

            try:
                channel = cls(section, self.bus)
                self.channels[name] = channel
                logger.info("channel_enabled", name=name, display_name=cls.display_name)
            except Exception as e:
                logger.warning("channel_init_failed", name=name, error=str(e))

        # Validate allow_from not empty (would deny all)
        for name, ch in self.channels.items():
            allow_list = getattr(ch.config, "allow_from", None)
            if allow_list == []:
                logger.warning(
                    "channel_empty_allow_from",
                    name=name,
                    hint='Set ["*"] to allow everyone, or add specific user IDs.',
                )

    async def start_all(self) -> None:
        """Start all enabled channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("no_channels_enabled")
            return

        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        tasks = []
        for name, channel in self.channels.items():
            logger.info("starting_channel", name=name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("channel_stopped", name=name)
            except Exception as e:
                logger.error("channel_stop_error", name=name, error=str(e))

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        try:
            await channel.start()
        except Exception as e:
            logger.error("channel_start_error", name=name, error=str(e))

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the correct channel with retry."""
        logger.info("outbound_dispatcher_started")
        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)
                channel = self.channels.get(msg.channel)
                if channel:
                    await self._send_with_retry(channel, msg)
                else:
                    logger.warning("unknown_channel_for_outbound", channel=msg.channel)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("dispatch_outbound_error", error=str(e))

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        await channel.send(msg)

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        for attempt in range(_MAX_RETRIES):
            try:
                await self._send_once(channel, msg)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(
                        "send_failed_max_retries",
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        attempts=_MAX_RETRIES,
                        error=str(e),
                    )
                    return
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    "send_retry",
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    attempt=attempt + 1,
                    delay=delay,
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise

    def get_channel(self, name: str) -> Optional[BaseChannel]:
        return self.channels.get(name)

    def get_status(self) -> Dict[str, Any]:
        return {
            name: {"enabled": True, "running": ch.is_running}
            for name, ch in self.channels.items()
        }

    @property
    def enabled_channels(self) -> List[str]:
        return list(self.channels.keys())
