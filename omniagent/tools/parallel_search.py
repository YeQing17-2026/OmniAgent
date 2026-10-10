"""Opt-in anonymous web search through Parallel's Streamable HTTP MCP server."""

import asyncio
import json
import uuid
from typing import Any, Dict

from .base import Tool, ToolResult
from .web_tools import WebSearchTool


class ParallelSearchTool(WebSearchTool):
    """Keep the native web_search interface while using Parallel Search MCP."""

    MCP_URL = "https://search.parallel.ai/mcp"
    USER_AGENT = "OmniAgent/0.1 (Parallel Search MCP)"

    def __init__(self, timeout: int = 15, model_name: str = ""):
        Tool.__init__(
            self,
            name="web_search",
            description=(
                "Search the web using Parallel Search MCP. Returns titles, URLs, and relevant "
                "excerpts. No API key required; anonymous usage is rate limited."
            ),
        )
        self.timeout = timeout
        self.model_name = model_name
        self.session_id = str(uuid.uuid4())

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="", error="Missing required parameter: query")
        max_results = params.get("max_results", 5)
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            return ToolResult(success=False, output="", error="max_results must be an integer")
        max_results = min(max(max_results, 1), 10)

        try:
            # Lazy imports keep the default Brave installation dependency-free from MCP.
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error='Parallel search requires the optional dependencies: pip install "omniagent[parallel]"',
            )

        arguments = {
            "objective": query,
            "search_queries": [query],
            "session_id": self.session_id,
        }
        if self.model_name:
            arguments["model_name"] = self.model_name

        try:
            # Bound initialization, tool dispatch and cleanup together. Cancellation propagates.
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=self.timeout,
                    follow_redirects=True,
                ) as client:
                    async with streamable_http_client(self.MCP_URL, http_client=client) as (
                        read_stream,
                        write_stream,
                        _,
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool("web_search", arguments)

            text = "\n".join(item.text for item in result.content if item.type == "text")
            if result.isError:
                return ToolResult(success=False, output="", error=text or "Parallel search failed")
            data = result.structuredContent
            if data is None:
                data = json.loads(text)
            results = data["results"][:max_results]
            entries = [
                f"{i}. {item.get('title') or item['url']}\n   URL: {item['url']}\n   "
                + "\n   ".join(item.get("excerpts", []))
                for i, item in enumerate(results, 1)
            ]
            output = (
                f"Search results for: {query}\n\n" + "\n\n".join(entries)
                if entries
                else f"No results found for: {query}"
            )
            return ToolResult(
                success=True,
                output=output,
                metadata={"query": query, "results_count": len(results)},
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, output="", error=f"Search timed out after {self.timeout}s"
            )
        except Exception as exc:

            def messages(error: Exception) -> list[str]:
                if isinstance(error, ExceptionGroup):
                    return [text for child in error.exceptions for text in messages(child)]
                return [str(error)]

            return ToolResult(
                success=False,
                output="",
                error="Parallel search failed: " + "; ".join(messages(exc)),
            )
