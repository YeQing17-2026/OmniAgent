"""Base tool interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Tool execution result."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        if self.success:
            return self.output
        else:
            return f"Error: {self.error}"


class Tool(ABC):
    """Abstract tool interface."""

    def __init__(self, name: str, description: str):
        """
        Initialize tool.

        Args:
            name: Tool name
            description: Tool description (for LLM)
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute tool.

        Args:
            params: Tool parameters

        Returns:
            Tool result
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for LLM.

        Returns:
            Tool schema (OpenAI function calling format)
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        pass


class ToolRegistry:
    """Tool registry."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
        logger.info("tool_registry_initialized")

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None if not found
        """
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """
        List all registered tools.

        Returns:
            List of tools
        """
        return list(self.tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all tools.

        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools.values()]
