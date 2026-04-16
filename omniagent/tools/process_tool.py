"""Process management tools."""

import asyncio
import psutil
from typing import Any, Dict

from omniagent.infra import get_logger
from .base import Tool, ToolResult

logger = get_logger(__name__)


class ProcessListTool(Tool):
    """List running processes tool."""

    def __init__(self):
        """Initialize process list tool."""
        super().__init__(
            name="process_list",
            description="List running processes. Returns process information (PID, name, CPU%, memory%).",
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute process list."""
        try:
            name_filter = params.get("name_filter", "")
            limit = params.get("limit", 20)

            logger.info("listing_processes", name_filter=name_filter, limit=limit)

            # Get all processes
            processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    # Apply name filter if provided
                    if name_filter and name_filter.lower() not in info["name"].lower():
                        continue

                    processes.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu_percent": info["cpu_percent"] or 0.0,
                        "memory_percent": info["memory_percent"] or 0.0,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda p: p["cpu_percent"], reverse=True)

            # Limit results
            processes = processes[:limit]

            # Format output
            output = "Running Processes:\n"
            output += f"{'PID':<10} {'Name':<30} {'CPU%':<10} {'Memory%':<10}\n"
            output += "-" * 60 + "\n"

            for proc in processes:
                output += f"{proc['pid']:<10} {proc['name']:<30} {proc['cpu_percent']:<10.1f} {proc['memory_percent']:<10.1f}\n"

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "count": len(processes),
                    "name_filter": name_filter,
                },
            )

        except Exception as e:
            logger.error("process_list_failed", error=str(e))
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
                "name_filter": {
                    "type": "string",
                    "description": "Filter processes by name (optional)",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of processes to return (default: 20)",
                    "default": 20,
                },
            },
        }


class ProcessKillTool(Tool):
    """Kill process tool."""

    def __init__(self):
        """Initialize process kill tool."""
        super().__init__(
            name="process_kill",
            description="Kill a process by PID. This is a dangerous operation and requires approval.",
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute process kill."""
        try:
            pid = params.get("pid")

            if pid is None:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: pid",
                )

            # Convert to int
            try:
                pid = int(pid)
            except ValueError:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid PID: {pid}",
                )

            logger.info("killing_process", pid=pid)

            # Check if process exists
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
            except psutil.NoSuchProcess:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Process {pid} not found",
                )

            # Kill process
            try:
                proc.terminate()
                # Wait for process to terminate
                proc.wait(timeout=5)
                success = True
                message = f"Process {pid} ({proc_name}) terminated successfully"
            except psutil.TimeoutExpired:
                # Force kill if terminate doesn't work
                proc.kill()
                success = True
                message = f"Process {pid} ({proc_name}) force killed"
            except psutil.AccessDenied:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access denied: Cannot kill process {pid} ({proc_name})",
                )

            return ToolResult(
                success=success,
                output=message,
                metadata={
                    "pid": pid,
                    "name": proc_name,
                },
            )

        except Exception as e:
            logger.error("process_kill_failed", error=str(e))
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
                "pid": {
                    "type": "integer",
                    "description": "Process ID to kill",
                },
            },
            "required": ["pid"],
        }
