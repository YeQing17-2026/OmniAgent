"""Plugin manifest parsing for OmniAgent extensions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PluginManifest:
    """Parsed plugin.yaml manifest."""

    name: str
    version: str = "0.1.0"
    module: str = ""
    extension_class: str = ""
    description: str = ""
    extra: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        """Create manifest from parsed YAML dict."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            module=data.get("module", ""),
            extension_class=data.get("class", ""),
            description=data.get("description", ""),
            extra=data.get("extra"),
        )

    @classmethod
    def load(cls, manifest_path: Path) -> "PluginManifest":
        """Load and parse a plugin.yaml file."""
        import yaml

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            data = yaml.safe_load(f) or {}

        manifest = cls.from_dict(data)
        if not manifest.name:
            manifest.name = manifest_path.parent.name

        if not manifest.module:
            manifest.module = manifest.name

        if not manifest.extension_class:
            manifest.extension_class = "Extension"

        return manifest
