"""Extension loader and registry for OmniAgent plugins."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Extension, ExtensionAPI

logger = logging.getLogger(__name__)


class ExtensionLoader:
    """Discovers and loads extensions from configured directories."""

    def __init__(self, extensions_dir: Path = None):
        self._extensions_dir = extensions_dir
        self.extensions: list = []

    async def discover_and_load(self, api: "ExtensionAPI") -> list:
        """Discover and load all extensions from search directories.

        Args:
            api: ExtensionAPI providing access to agent subsystems.

        Returns:
            List of loaded Extension instances.
        """
        if self._extensions_dir is None:
            return []

        ext_dir = self._extensions_dir
        if not ext_dir.exists():
            ext_dir.mkdir(parents=True, exist_ok=True)
            logger.info("extensions_dir_created", path=str(ext_dir))
            return []

        self.extensions = []
        loaded = []

        # Look for plugin.yaml manifests
        for manifest_path in sorted(ext_dir.glob("*/plugin.yaml")):
            ext_name = manifest_path.parent.name
            try:
                extension = self._load_extension(ext_name, manifest_path)
                if extension:
                    await extension.on_load(api)
                    self.extensions.append(extension)
                    loaded.append(ext_name)
                    logger.info("extension_loaded", name=ext_name)
            except Exception as e:
                logger.warning("extension_load_failed", name=ext_name, error=str(e))

        # Also look for directories with extension.py
        for ext_path in sorted(ext_dir.iterdir()):
            if not ext_path.is_dir() or ext_path.name.startswith("_"):
                continue
            if (ext_path / "plugin.yaml").exists():
                continue  # Already loaded above
            if (ext_path / "extension.py").exists():
                ext_name = ext_path.name
                if ext_name not in loaded:
                    try:
                        extension = self._load_direct_extension(ext_name, ext_path)
                        if extension:
                            await extension.on_load(api)
                            self.extensions.append(extension)
                            loaded.append(ext_name)
                            logger.info("extension_loaded", name=ext_name)
                    except Exception as e:
                        logger.warning("extension_load_failed", name=ext_name, error=str(e))

        logger.info("extensions_loaded", count=len(loaded), names=loaded)
        return self.extensions

    def _load_extension(self, ext_name: str, manifest_path: Path) -> "Extension":
        """Load extension from plugin.yaml manifest."""
        import yaml

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        module_name = manifest.get("module", ext_name)
        class_name = manifest.get("class", "Extension")
        version = manifest.get("version", "0.1.0")

        # Import the module from the extension directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(manifest_path.parent / f"{module_name}.py"),
            submodule_search_locations=[str(manifest_path.parent)],
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ext_class = getattr(module, class_name, None)
        if ext_class is None:
            raise ValueError(f"Class '{class_name}' not found in {module_name}")

        instance = ext_class()
        instance.name = ext_name
        if hasattr(instance, "version") and not instance.version:
            instance.version = version
        return instance

    def _load_direct_extension(self, ext_name: str, ext_path: Path) -> "Extension":
        """Load extension from extension.py without manifest."""
        import importlib.util

        module_name = ext_name
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(ext_path / "extension.py"),
            submodule_search_locations=[str(ext_path)],
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for a class that extends Extension
        from omniagent.extensions.base import Extension

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Extension) and attr is not Extension:
                instance = attr()
                instance.name = ext_name
                return instance

        raise ValueError(f"No Extension subclass found in {ext_name}/extension.py")

    async def unload_all(self) -> None:
        """Unload all extensions."""
        for ext in self.extensions:
            try:
                await ext.on_unload()
            except Exception as e:
                logger.warning("extension_unload_failed", name=ext.name, error=str(e))
        self.extensions = []
        logger.info("extensions_unloaded")


def discover_extension_names() -> list:
    """Discover available extension names from default directories."""
    from pathlib import Path

    ext_dir = Path.home() / ".omniagent" / "extensions"
    names = []
    if ext_dir.exists():
        for child in ext_dir.iterdir():
            if child.is_dir() and not child.name.startswith("_"):
                names.append(child.name)
    return sorted(names)
