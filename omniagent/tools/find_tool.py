"""Find tool for locating files by glob pattern."""

import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.infra import safe_path, PathTraversalError
from .base import Tool, ToolResult


class FindTool(Tool):
    """Find files matching a glob pattern."""

    def __init__(self, work_dir: Optional[Path] = None):
        super().__init__(
            name="find",
            description="Find files matching a glob pattern (e.g., '**/*.py').",
        )
        self.work_dir = work_dir or Path.cwd()

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.py', '*.txt')",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in (default: current directory)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern", "")
        path = params.get("path", ".")

        if not pattern:
            return ToolResult(success=False, output="", error="Missing required parameter: pattern")

        try:
            target = safe_path(self.work_dir, Path(path), allow_home=True)
        except PathTraversalError as e:
            return ToolResult(success=False, output="", error=str(e))

        if not target.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        if not target.is_dir():
            return ToolResult(success=False, output="", error=f"Path is not a directory: {path}")

        try:
            matched = sorted(target.glob(pattern))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Invalid glob pattern: {e}")

        # Filter out directories, only return files
        files = [f for f in matched if f.is_file()]

        if not files:
            return ToolResult(
                success=True,
                output=f"No files found matching '{pattern}' in {path}",
                metadata={"count": 0},
            )

        lines = []
        for f in files:
            try:
                rel = f.relative_to(self.work_dir)
            except ValueError:
                rel = f
            lines.append(str(rel))

        output = "\n".join(lines)
        if len(files) > 100:
            output += f"\n... and {len(files) - 100} more files"

        return ToolResult(
            success=True,
            output=f"Found {len(files)} files matching '{pattern}' in {path}\n{output}",
            metadata={"count": len(files)},
        )
