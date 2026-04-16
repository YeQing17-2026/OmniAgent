"""Configuration loading and saving."""

import json
import os
import re
import yaml
from pathlib import Path
from typing import Optional

from .models import OmniAgentConfig


DEFAULT_CONFIG_PATH = Path.home() / ".omniagent" / "config.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    For each key in override:
      - If both values are dicts, recurse.
      - Otherwise, override wins.

    Returns a new dict (does not mutate inputs).
    """
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _substitute_env_vars(data) -> any:
    """Recursively replace ${VAR} and ${VAR:-default} patterns in all string values.

    Supports the full config dict, lists, and nested structures.
    """
    if isinstance(data, str):
        def _replace_match(m):
            var_name = m.group(1)
            default_val = m.group(3)  # group(3) is the default after :-
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default_val is not None:
                return default_val
            return m.group(0)  # Leave unresolved if no env var and no default

        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}', _replace_match, data)
    elif isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    return data


def load_config(config_path: Optional[Path] = None) -> OmniAgentConfig:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        OmniAgentConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_config()

    # Load based on file extension
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")

    # Load API keys from environment if not in config
    if "api_key" not in data or data["api_key"] is None:
        data["api_key"] = os.getenv("OMNIAGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if "openai_api_key" not in data or data["openai_api_key"] is None:
        data["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if "anthropic_api_key" not in data or data["anthropic_api_key"] is None:
        data["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

    # Process includes: deep merge each include file
    includes = data.pop("includes", [])
    if includes:
        for include_path_str in includes:
            include_path = Path(include_path_str)
            if not include_path.is_absolute():
                # Resolve relative to the main config file's directory
                include_path = (config_path.parent / include_path_str).resolve()
            if include_path.exists():
                include_data = _load_raw_config(include_path)
                data = deep_merge(data, include_data)

    # Substitute environment variables
    data = _substitute_env_vars(data)

    # Validate and create config
    return OmniAgentConfig(**data)


def reload_config(config_path: Path, current_config: OmniAgentConfig) -> OmniAgentConfig:
    """
    Reload configuration from file, preserving values not present in the file.

    If the config file does not exist, returns the current_config unchanged.

    Args:
        config_path: Path to config file to reload from.
        current_config: The current in-memory config to use as fallback.

    Returns:
        New OmniAgentConfig instance with reloaded values.
    """
    if not config_path.exists():
        return current_config

    loaded = load_config(config_path)
    return loaded


def _load_raw_config(config_path: Path) -> dict:
    """Load a config file as a raw dict (used for includes merging)."""
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Include file must contain a mapping, got {type(data).__name__}")

    return data


def save_config(config: OmniAgentConfig, config_path: Optional[Path] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: OmniAgentConfig instance
        config_path: Path to save config. If None, uses default path.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Create parent directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = config.model_dump(mode="json")

    # Save based on file extension
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    elif config_path.suffix == ".json":
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def get_default_config() -> OmniAgentConfig:
    """
    Get default configuration with environment variable fallbacks.

    Returns:
        OmniAgentConfig with default values and env vars
    """
    # Load API keys from environment variables
    api_key = os.getenv("OMNIAGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    return OmniAgentConfig(
        api_key=api_key,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )
