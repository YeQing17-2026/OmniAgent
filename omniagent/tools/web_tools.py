"""Web tools for search and fetch."""

import asyncio
import json
import os
from html.parser import HTMLParser
from typing import Any, Dict
from urllib.parse import urlparse

import aiohttp

from omniagent.infra import get_logger
from .base import Tool, ToolResult

logger = get_logger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping tags/scripts/styles."""

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True
        elif tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._text_parts.append(text)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        # Collapse multiple blank lines
        lines = [line for line in raw.splitlines() if line.strip()]
        return "\n".join(lines)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


class WebFetchTool(Tool):
    """Fetch content from a URL."""

    MAX_CONTENT_LENGTH = 100_000

    def __init__(self, timeout: int = 30):
        super().__init__(
            name="web_fetch",
            description=(
                "Fetch content from a URL and return it as text. "
                "Supports HTML pages (converted to plain text) and JSON. "
                "Content is truncated at 100,000 characters."
            ),
        )
        self.timeout = timeout

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            url = params.get("url", "")

            if not url:
                return ToolResult(success=False, output="", error="Missing required parameter: url")

            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return ToolResult(success=False, output="", error=f"Invalid URL: {url}")

            logger.info("web_fetch", url=url)

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {
                    "User-Agent": "OmniAgent/0.1 (AI Agent)",
                    "Accept": "text/html,application/json,text/plain,*/*",
                }
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"HTTP {resp.status}: {error_text[:200]}",
                        )

                    content_type = resp.headers.get("Content-Type", "")

                    if "application/json" in content_type:
                        raw = await resp.text()
                        # Try to pretty-print JSON
                        try:
                            parsed_json = json.loads(raw)
                            text = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                        except json.JSONDecodeError:
                            text = raw
                    else:
                        # HTML or text
                        raw = await resp.text()
                        if "text/html" in content_type:
                            text = html_to_text(raw)
                        else:
                            text = raw

                    if len(text) > self.MAX_CONTENT_LENGTH:
                        text = text[:self.MAX_CONTENT_LENGTH] + f"\n\n[Content truncated at {self.MAX_CONTENT_LENGTH} characters]"

                    logger.info("web_fetch_success", url=url, content_length=len(text))

                    return ToolResult(
                        success=True,
                        output=text,
                        metadata={"url": url, "content_type": content_type, "length": len(text)},
                    )

        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error=f"Request timed out after {self.timeout}s")
        except Exception as e:
            logger.error("web_fetch_failed", error=str(e))
            return ToolResult(success=False, output="", error=str(e))

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from",
                },
            },
            "required": ["url"],
        }


class WebSearchTool(Tool):
    """Web search using Brave Search API."""

    def __init__(self, timeout: int = 15):
        super().__init__(
            name="web_search",
            description=(
                "Search the web using Brave Search API. "
                "Returns search results with titles, URLs, and descriptions. "
                "Requires BRAVE_API_KEY environment variable."
            ),
        )
        self.timeout = timeout
        self.api_key = os.getenv("BRAVE_API_KEY", "")

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            query = params.get("query", "")
            max_results = params.get("max_results", 5)

            if not query:
                return ToolResult(success=False, output="", error="Missing required parameter: query")

            if not self.api_key:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "Brave Search API key not configured. "
                        "Set BRAVE_API_KEY environment variable to enable web search."
                    ),
                )

            max_results = min(max(max_results, 1), 10)

            logger.info("web_search", query=query, max_results=max_results)

            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }

            params_api = {
                "q": query,
                "count": max_results,
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params=params_api,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("web_search_api_error", status=resp.status, error=error_text)
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"Brave Search API error: HTTP {resp.status}",
                        )

                    data = await resp.json()

            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])
            for i, r in enumerate(web_results[:max_results], 1):
                title = r.get("title", "")
                url = r.get("url", "")
                description = r.get("description", "")
                results.append(f"{i}. {title}\n   URL: {url}\n   {description}")

            if not results:
                output = f"No results found for: {query}"
            else:
                output = f"Search results for: {query}\n\n" + "\n\n".join(results)

            logger.info("web_search_success", query=query, results_count=len(results))

            return ToolResult(
                success=True,
                output=output,
                metadata={"query": query, "results_count": len(results)},
            )

        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error=f"Search timed out after {self.timeout}s")
        except Exception as e:
            logger.error("web_search_failed", error=str(e))
            return ToolResult(success=False, output="", error=str(e))

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10, default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }
