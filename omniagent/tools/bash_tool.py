"""Bash execution tool."""

import asyncio
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from omniagent.infra import get_logger
from .base import Tool, ToolResult

logger = get_logger(__name__)


# Dangerous commands that should be blocked
DANGEROUS_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    "> /dev/sda",
    "mv /* /dev/null",
    "wget * | sh",
    "curl * | sh",
    "fork bomb",
    ":(){ :|:& };:",
}


class BashTool(Tool):
    """Bash command execution tool."""

    def __init__(
        self,
        work_dir: Path,
        timeout: int = 30,
        allow_dangerous: bool = False,
    ):
        """
        Initialize bash tool.

        Args:
            work_dir: Working directory for command execution
            timeout: Command timeout in seconds
            allow_dangerous: Allow dangerous commands (not recommended)
        """
        super().__init__(
            name="bash",
            description=(
                "Execute a bash command in the working directory. "
                "Returns the command output (stdout and stderr). "
                "Use this for running shell commands, scripts, and system operations."
            ),
        )
        self.work_dir = work_dir
        self.timeout = timeout
        self.allow_dangerous = allow_dangerous

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute bash command."""
        try:
            command = params.get("command", "")
            timeout = params.get("timeout", self.timeout)
            background = params.get("background", False)

            if not command:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required parameter: command",
                )

            logger.info("executing_bash", command=command[:100])

            # Security check
            if not self.allow_dangerous:
                if self._is_dangerous(command):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Dangerous command blocked: {command[:50]}",
                    )

            # Execute command
            if background:
                # Background execution
                return await self._execute_background(command)
            else:
                # Foreground execution
                return await self._execute_foreground(command, timeout)

        except asyncio.TimeoutError:
            logger.error("bash_timeout", command=command[:100])
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            logger.error("bash_execution_failed", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    async def _execute_foreground(
        self,
        command: str,
        timeout: int,
    ) -> ToolResult:
        """Execute command in foreground."""
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.work_dir),
        )

        try:
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            # Combine output
            output = stdout_text
            if stderr_text:
                output += f"\n[stderr]\n{stderr_text}"

            # Check exit code
            success = process.returncode == 0

            if not success:
                logger.warning(
                    "bash_nonzero_exit",
                    command=command[:100],
                    exit_code=process.returncode,
                )

            return ToolResult(
                success=success,
                output=output.strip(),
                metadata={
                    "exit_code": process.returncode,
                    "command": command,
                },
            )

        except asyncio.TimeoutError:
            # Kill process on timeout
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            raise

    async def _execute_background(self, command: str) -> ToolResult:
        """Execute command in background."""
        # Create subprocess (don't wait)
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.work_dir),
        )

        logger.info(
            "bash_background_started",
            command=command[:100],
            pid=process.pid,
        )

        return ToolResult(
            success=True,
            output=f"Command started in background (PID: {process.pid})",
            metadata={
                "pid": process.pid,
                "command": command,
                "background": True,
            },
        )

    def _is_dangerous(self, command: str) -> bool:
        """Check if command is dangerous."""
        command_lower = command.lower().strip()

        # Check against dangerous patterns
        for dangerous in DANGEROUS_COMMANDS:
            if dangerous in command_lower:
                return True

        # Check for suspicious patterns
        suspicious_patterns = [
            "rm -rf",
            "rm -fr",
            "dd if=",
            "mkfs",
            "> /dev/",
            "format ",
            "fdisk",
            "parted",
        ]

        for pattern in suspicious_patterns:
            if pattern in command_lower:
                # Additional checks for rm -rf
                if pattern == "rm -rf" or pattern == "rm -fr":
                    # Only block if targeting root or system dirs directly
                    import re
                    if re.search(r'\s+/(?:\s|;|&|\||$)', command_lower) or any(
                        danger in command_lower
                        for danger in [" /*", " /bin", " /usr", " /etc"]
                    ):
                        return True
                else:
                    return True

        return False

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Command timeout in seconds (default: {self.timeout})",
                    "default": self.timeout,
                },
                "background": {
                    "type": "boolean",
                    "description": "Run command in background (default: false)",
                    "default": False,
                },
            },
            "required": ["command"],
        }
