"""Codex app-server subprocess client for OmniAgent LLM integration."""

import asyncio
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


class CodexBinaryResolver:
    """Finds the codex executable via env var or PATH."""

    @staticmethod
    def resolve() -> str:
        env_path = os.environ.get("OMNIAGENT_CODEX_BIN", "").strip()
        if env_path:
            return env_path
        which_path = shutil.which("codex")
        if which_path:
            return which_path
        raise RuntimeError(
            "codex binary not found. Install it with: npm install -g @openai/codex\n"
            "Or set OMNIAGENT_CODEX_BIN=/path/to/codex"
        )
