"""Logging setup using structlog."""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import structlog


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Setup structured logging with console and optional file output.

    Args:
        verbose: Enable verbose/debug logging to console. When False, only
                 WARNING and above are shown on console; INFO goes to file only.
        log_file: Path to log file (always DEBUG level). If None, no file logging.
    """
    # Console: WARNING by default, DEBUG with --verbose
    console_level = logging.DEBUG if verbose else logging.WARNING

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)

    # File handler (always DEBUG level for full traceability)
    file_handler = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers in debug mode
    for noisy in ("markdown_it", "markdown_it.main", "prompt_toolkit",
                  "httpx", "httpcore", "asyncio", "urllib3",
                  "lark_oapi", "lark_oapi.ws"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress common third-party UserWarnings (e.g. pkg_resources deprecation from lark_oapi)
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lark_oapi")
    if file_handler:
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> Any:
    """
    Get a structured logger.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
