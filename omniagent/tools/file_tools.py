"""File operation tools."""

from pathlib import Path
from typing import Any, Dict

from omniagent.infra import get_logger, read_file, write_file, safe_path
from .base import Tool, ToolResult

logger = get_logger(__name__)


class ReadTool(Tool):
    """Read file tool."""

    def __init__(self, work_dir: Path):
        """
        Initialize read tool.

        Args:
            work_dir: Working directory (for path safety)
        """
        super().__init__(
            name="read_file",
            description="Read contents of a file. Returns the file content with line numbers.",
        )
        self.work_dir = work_dir

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute read operation."""
        try:
            file_path = Path(params.get("path", ""))
            offset = params.get("offset", 0)
            limit = params.get("limit", 100)

            if not file_path:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: path",
                )

            logger.info("reading_file", path=str(file_path))

            # Read file with safety check
            content = read_file(file_path, base_dir=self.work_dir, allow_home=True)

            # Split into lines
            lines = content.split("\n")

            # Apply offset and limit
            start = offset
            end = offset + limit
            selected_lines = lines[start:end]

            # Add line numbers
            numbered_lines = [
                f"{start + i + 1:4d}  {line}"
                for i, line in enumerate(selected_lines)
            ]

            output = "\n".join(numbered_lines)

            # Add metadata
            total_lines = len(lines)
            showing = len(selected_lines)

            metadata = {
                "total_lines": total_lines,
                "showing_lines": showing,
                "offset": start,
            }

            if end < total_lines:
                metadata["has_more"] = True
                output += f"\n\n... ({total_lines - end} more lines)"

            return ToolResult(
                success=True,
                output=output,
                metadata=metadata,
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path}",
            )
        except Exception as e:
            logger.error("read_file_failed", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start reading from (default: 0)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100)",
                    "default": 100,
                },
            },
            "required": ["path"],
        }


class WriteTool(Tool):
    """Write file tool."""

    def __init__(self, work_dir: Path):
        """
        Initialize write tool.

        Args:
            work_dir: Working directory (for path safety)
        """
        super().__init__(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        )
        self.work_dir = work_dir

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute write operation."""
        try:
            file_path = Path(params.get("path", ""))
            content = params.get("content", "")

            if not file_path:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: path",
                )

            logger.info("writing_file", path=str(file_path))

            # Write file with safety check
            write_file(
                file_path,
                content,
                base_dir=self.work_dir,
                allow_home=True,
                create_dirs=True,
            )

            lines_written = len(content.split("\n"))

            return ToolResult(
                success=True,
                output=f"Successfully wrote {lines_written} lines to {file_path}",
                metadata={
                    "path": str(file_path),
                    "lines": lines_written,
                    "bytes": len(content),
                },
            )

        except Exception as e:
            logger.error("write_file_failed", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }


class EditTool(Tool):
    """Edit file tool (string replacement)."""

    def __init__(self, work_dir: Path):
        """
        Initialize edit tool.

        Args:
            work_dir: Working directory (for path safety)
        """
        super().__init__(
            name="edit_file",
            description="Edit a file by replacing old_text with new_text. The old_text must match exactly.",
        )
        self.work_dir = work_dir

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute edit operation."""
        try:
            file_path = Path(params.get("path", ""))
            old_text = params.get("old_text", "")
            new_text = params.get("new_text", "")

            if not file_path or not old_text:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameters: path, old_text",
                )

            logger.info("editing_file", path=str(file_path))

            # Read current content
            content = read_file(file_path, base_dir=self.work_dir, allow_home=True)

            # Check if old_text exists
            if old_text not in content:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Text not found in file: {old_text[:50]}...",
                )

            # Count occurrences
            count = content.count(old_text)

            if count > 1:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Text appears {count} times in file. Please be more specific.",
                )

            # Replace text
            new_content = content.replace(old_text, new_text, 1)

            # Write back
            write_file(
                file_path,
                new_content,
                base_dir=self.work_dir,
                allow_home=True,
            )

            return ToolResult(
                success=True,
                output=f"Successfully edited {file_path}",
                metadata={
                    "path": str(file_path),
                    "old_length": len(old_text),
                    "new_length": len(new_text),
                },
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path}",
            )
        except Exception as e:
            logger.error("edit_file_failed", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to replace (must match exactly)",
                },
                "new_text": {
                    "type": "string",
                    "description": "New text to insert",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }
