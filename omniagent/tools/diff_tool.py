"""Diff tool for comparing files."""

import difflib
from pathlib import Path
from typing import Any, Dict, Optional

from omniagent.infra import safe_path, PathTraversalError
from .base import Tool, ToolResult


class DiffTool(Tool):
    """Compare two files and show differences."""

    def __init__(self, work_dir: Optional[Path] = None):
        super().__init__(
            name="diff",
            description="Compare two files and show unified diff.",
        )
        self.work_dir = work_dir or Path.cwd()

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path_a": {
                    "type": "string",
                    "description": "Path to the first (original) file",
                },
                "path_b": {
                    "type": "string",
                    "description": "Path to the second (modified) file",
                },
            },
            "required": ["path_a", "path_b"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path_a = params.get("path_a", "")
        path_b = params.get("path_b", "")

        if not path_a or not path_b:
            return ToolResult(success=False, output="", error="Missing required parameters: path_a and path_b")

        try:
            file_a = safe_path(self.work_dir, Path(path_a), allow_home=True)
            file_b = safe_path(self.work_dir, Path(path_b), allow_home=True)
        except PathTraversalError as e:
            return ToolResult(success=False, output="", error=str(e))

        if not file_a.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path_a}")
        if not file_b.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path_b}")

        try:
            lines_a = file_a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            lines_b = file_b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except (OSError, PermissionError) as e:
            return ToolResult(success=False, output="", error=f"Failed to read files: {e}")

        diff = list(difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=str(file_a),
            tofile=str(file_b),
            lineterm="",
        ))

        if not diff:
            return ToolResult(
                success=True,
                output=f"Files {path_a} and {path_b} are identical.",
                metadata={"changes": 0},
            )

        diff_text = "".join(diff)
        # Limit output
        if len(diff_text) > 10000:
            diff_text = diff_text[:10000] + "\n... (truncated)"

        changes = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        return ToolResult(
            success=True,
            output=diff_text,
            metadata={"changes": changes, "deletions": deletions},
        )
