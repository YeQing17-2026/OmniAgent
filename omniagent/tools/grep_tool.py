"""Grep tool for searching file contents using regular expressions."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.infra import safe_path, PathTraversalError
from .base import Tool, ToolResult


class GrepTool(Tool):
    """Search file contents using regular expressions."""

    def __init__(self, work_dir: Optional[Path] = None):
        super().__init__(
            name="grep",
            description="Search file contents for lines matching a regex pattern.",
        )
        self.work_dir = work_dir or Path.cwd()

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in (default: current directory)",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.py')",
                },
                "exclude": {
                    "type": "string",
                    "description": "Glob pattern for files to exclude (e.g., '*.pyc')",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern", "")
        path = params.get("path", ".")
        include = params.get("include")
        exclude = params.get("exclude")
        case_insensitive = params.get("case_insensitive", False)

        if not pattern:
            return ToolResult(success=False, output="", error="Missing required parameter: pattern")

        try:
            target = safe_path(self.work_dir, Path(path), allow_home=True)
        except PathTraversalError as e:
            return ToolResult(success=False, output="", error=str(e))

        if not target.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex pattern: {e}")

        matches = []
        files_searched = 0

        if target.is_file():
            files_searched = 1
            matches.extend(self._search_file(target, regex))
        else:
            for root, dirs, files in os.walk(target):
                # Skip common ignored directories
                dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
                for fname in sorted(files):
                    fpath = Path(root) / fname
                    if include and not fpath.match(include):
                        continue
                    if exclude and fpath.match(exclude):
                        continue
                    files_searched += 1
                    matches.extend(self._search_file(fpath, regex))

        if not matches:
            try:
                rel = target.relative_to(self.work_dir) if target != self.work_dir else target
            except ValueError:
                rel = target
            return ToolResult(
                success=True,
                output=f"No matches found for '{pattern}' in {rel} ({files_searched} files searched)",
                metadata={"files_searched": files_searched, "matches": 0},
            )

        # Format output
        lines = []
        for file_path, line_num, line_content in matches[:200]:
            try:
                rel = file_path.relative_to(self.work_dir)
            except ValueError:
                rel = file_path
            lines.append(f"{rel}:{line_num}: {line_content.strip()}")

        truncated = ""
        if len(matches) > 200:
            truncated = f"\n... and {len(matches) - 200} more matches"

        try:
            rel = target.relative_to(self.work_dir) if target != self.work_dir else target
        except ValueError:
            rel = target
        header = f"Found {len(matches)} matches for '{pattern}' in {rel} ({files_searched} files)\n"
        return ToolResult(
            success=True,
            output=header + "\n".join(lines) + truncated,
            metadata={"files_searched": files_searched, "matches": len(matches)},
        )

    def _search_file(self, file_path: Path, regex) -> List[tuple]:
        """Search a single file and return (path, line_num, line_content) tuples."""
        results = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append((file_path, i, line))
        except (OSError, PermissionError):
            pass
        return results
