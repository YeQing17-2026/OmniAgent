"""Auto-discovery for built-in channel modules."""

import importlib
import pkgutil
from typing import TYPE_CHECKING

from omniagent.infra import get_logger

if TYPE_CHECKING:
    from .base import BaseChannel

logger = get_logger(__name__)

# Internal modules that are not channel implementations
_INTERNAL = frozenset({"base", "manager", "registry", "bus", "__init__"})


def discover_channel_names() -> list[str]:
    """Return all built-in channel module names by scanning the package."""
    import omniagent.channels as pkg
    names = []
    for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if name not in _INTERNAL and not ispkg:
            names.append(name)
    return names


def load_channel_class(module_name: str) -> type["BaseChannel"]:
    """Import module and return the first BaseChannel subclass found."""
    from .base import BaseChannel as _Base
    mod = importlib.import_module(f"omniagent.channels.{module_name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base:
            return obj
    raise ImportError(f"No BaseChannel subclass in omniagent.channels.{module_name}")


def discover_all() -> dict[str, type["BaseChannel"]]:
    """Discover all built-in channels.

    Returns:
        Dict mapping channel name to its BaseChannel subclass.
    """
    result: dict[str, type["BaseChannel"]] = {}
    for modname in discover_channel_names():
        try:
            result[modname] = load_channel_class(modname)
        except ImportError as e:
            logger.debug("skipping_channel_module", name=modname, error=str(e))
    return result
