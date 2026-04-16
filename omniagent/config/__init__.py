"""Configuration system for OmniAgent."""

from .models import OmniAgentConfig, ToolsConfig, AgentConfig, GatewayConfig, MemorySearchConfig, ChannelsConfig
from .loader import load_config, save_config, get_default_config

__all__ = [
    "OmniAgentConfig",
    "ToolsConfig",
    "AgentConfig",
    "GatewayConfig",
    "MemorySearchConfig",
    "ChannelsConfig",
    "load_config",
    "save_config",
    "get_default_config",
]
