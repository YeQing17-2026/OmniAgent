"""JSON file tools."""

import json
from pathlib import Path
from typing import Any, Dict

from omniagent.infra import get_logger, read_file, write_file
from .base import Tool, ToolResult

logger = get_logger(__name__)


class LoadJSONTool(Tool):
    """Load JSON file tool."""

    def __init__(self, work_dir: Path):
        """
        Initialize load JSON tool.

        Args:
            work_dir: Working directory
        """
        super().__init__(
            name="load_json",
            description="Load and parse a JSON file. Returns the parsed JSON data.",
        )
        self.work_dir = work_dir

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute load JSON."""
        try:
            path = params.get("path", "")
            if not path:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: path",
                )

            file_path = Path(path)

            logger.info("loading_json", path=str(file_path))

            # Read file
            content = read_file(file_path, base_dir=self.work_dir)

            # Parse JSON
            data = json.loads(content)

            # Format output
            output = json.dumps(data, indent=2, ensure_ascii=False)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                },
            )

        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid JSON: {str(e)}",
            )
        except Exception as e:
            logger.error("load_json_failed", error=str(e))
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
                    "description": "Path to JSON file",
                },
            },
            "required": ["path"],
        }


class SaveJSONTool(Tool):
    """Save JSON file tool."""

    def __init__(self, work_dir: Path):
        """
        Initialize save JSON tool.

        Args:
            work_dir: Working directory
        """
        super().__init__(
            name="save_json",
            description="Save data to a JSON file. Accepts a JSON string or object.",
        )
        self.work_dir = work_dir

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute save JSON."""
        try:
            path = params.get("path", "")
            data = params.get("data", "")

            if not path:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: path",
                )

            if not data:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: data",
                )

            file_path = Path(path)

            logger.info("saving_json", path=str(file_path))

            # Parse data if string
            if isinstance(data, str):
                data = json.loads(data)

            # Format JSON
            content = json.dumps(data, indent=2, ensure_ascii=False)

            # Write file
            write_file(file_path, content, base_dir=self.work_dir)

            return ToolResult(
                success=True,
                output=f"JSON saved to {file_path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                },
            )

        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid JSON: {str(e)}",
            )
        except Exception as e:
            logger.error("save_json_failed", error=str(e))
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
                    "description": "Path to JSON file",
                },
                "data": {
                    "type": ["object", "string"],
                    "description": "JSON data to save (object or JSON string)",
                },
            },
            "required": ["path", "data"],
        }
