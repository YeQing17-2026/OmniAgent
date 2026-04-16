"""Ls tool for listing directory contents."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from omniagent.infra import safe_path, PathTraversalError
from .base import Tool, ToolResult


class LsTool(Tool):
    """List directory contents."""

    def __init__(self, work_dir: Optional[Path] = None):
        super().__init__(
            name="ls",
            description="List directory contents.",
        )
        self.work_dir = work_dir or Path.cwd()

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show hidden files (default: false)",
                },
            },
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", ".")
        show_hidden = params.get("show_hidden", False)

        try:
            target = safe_path(self.work_dir, Path(path), allow_home=True)
        except PathTraversalError as e:
            return ToolResult(success=False, output="", error=str(e))

        if not target.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        if not target.is_dir():
            return ToolResult(success=True, output=f"  {target.name}  (file)")

        try:
            entries = list(target.iterdir())
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {path}")

        # Sort: directories first, then files
        def sort_key(entry):
            return (0 if entry.is_dir() else 1, entry.name.lower())

        entries.sort(key=sort_key)

        lines = []
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue

            if entry.is_dir():
                suffix = "/"
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                suffix = f"  ({size_str})"

            lines.append(f"  {entry.name}{suffix}")

        if not lines:
            return ToolResult(
                success=True,
                output=f"Empty directory: {path}",
                metadata={"count": 0},
            )

        return ToolResult(
            success=True,
            output=f"Contents of {path}:\n" + "\n".join(lines),
            metadata={"count": len(lines)},
        )
