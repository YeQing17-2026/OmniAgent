"""Exa AI-powered search tool."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

from .base import Tool, ToolResult

logger = get_logger(__name__)


_VALID_SEARCH_TYPES = {"auto", "neural", "fast", "deep", "deep-lite", "deep-reasoning", "instant"}
_VALID_CATEGORIES = {
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "people",
}


@dataclass
class ExaSearchResult:
    """Typed Exa search result."""

    title: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    score: Optional[float] = None

    @classmethod
    def from_api(cls, item: Any) -> "ExaSearchResult":
        """Build from an exa-py result object (attribute-style or dict)."""

        def _attr(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        highlights = _attr(item, "highlights", []) or []
        if not isinstance(highlights, list):
            highlights = [str(highlights)]

        return cls(
            title=_attr(item, "title", "") or "",
            url=_attr(item, "url", "") or "",
            published_date=_attr(item, "published_date") or _attr(item, "publishedDate"),
            author=_attr(item, "author"),
            text=_attr(item, "text"),
            summary=_attr(item, "summary"),
            highlights=[str(h) for h in highlights],
            score=_attr(item, "score"),
        )

    def snippet(self, max_chars: int = 500) -> str:
        """Pick the best available content for a short snippet.

        Cascades through highlights → summary → text so partial results from
        the API still produce something readable.
        """
        if self.highlights:
            joined = " ... ".join(self.highlights)
            return joined[:max_chars]
        if self.summary:
            return self.summary[:max_chars]
        if self.text:
            return self.text[:max_chars]
        return ""


def _format_results(query: str, results: List[ExaSearchResult]) -> str:
    """Format results as human-readable text matching the WebSearch style."""
    if not results:
        return f"No results found for: {query}"

    lines: List[str] = [f"Search results for: {query}"]
    for i, r in enumerate(results, 1):
        block = [f"{i}. {r.title}", f"   URL: {r.url}"]
        if r.published_date:
            block.append(f"   Published: {r.published_date}")
        if r.author:
            block.append(f"   Author: {r.author}")
        snippet = r.snippet()
        if snippet:
            block.append(f"   {snippet}")
        lines.append("\n".join(block))

    return lines[0] + "\n\n" + "\n\n".join(lines[1:])


class ExaSearchTool(Tool):
    """Web search powered by the Exa AI search API.

    Exa returns embedding-based ("neural") and hybrid results plus optional
    page contents (highlights, summary, full text). Requires an EXA_API_KEY
    environment variable. See https://exa.ai for API access.
    """

    INTEGRATION_NAME = "omniagent"

    def __init__(self, timeout: int = 30):
        super().__init__(
            name="exa_search",
            description=(
                "Search the web using Exa's AI-powered search API. "
                "Returns titles, URLs, and content snippets (highlights or summary). "
                "Supports neural/keyword/auto search, category filters (company, "
                "research paper, news, personal site, financial report, people), "
                "domain include/exclude, and date ranges. "
                "Requires EXA_API_KEY environment variable."
            ),
        )
        self.timeout = timeout
        self.api_key = os.getenv("EXA_API_KEY", "")

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-25, default: 5)",
                    "default": 5,
                },
                "type": {
                    "type": "string",
                    "description": (
                        "Search type. 'auto' (default) lets Exa choose, 'neural' uses "
                        "embedding similarity, 'fast' is latency-optimized."
                    ),
                    "enum": sorted(_VALID_SEARCH_TYPES),
                    "default": "auto",
                },
                "category": {
                    "type": "string",
                    "description": "Optional content category filter.",
                    "enum": sorted(_VALID_CATEGORIES),
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict results to these domains.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains.",
                },
                "include_text": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Result text must contain these strings.",
                },
                "exclude_text": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Result text must not contain these strings.",
                },
                "start_published_date": {
                    "type": "string",
                    "description": "ISO 8601 lower bound on publish date.",
                },
                "end_published_date": {
                    "type": "string",
                    "description": "ISO 8601 upper bound on publish date.",
                },
                "content_mode": {
                    "type": "string",
                    "description": (
                        "Which page contents to fetch alongside results. "
                        "'highlights' (default) returns short matching passages; "
                        "'summary' returns an LLM-generated summary; "
                        "'text' returns full page text; 'all' returns everything."
                    ),
                    "enum": ["highlights", "summary", "text", "all", "none"],
                    "default": "highlights",
                },
                "summary_query": {
                    "type": "string",
                    "description": "Optional guidance for the summary content type.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = (params.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output="", error="Missing required parameter: query")

        if not self.api_key:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Exa API key not configured. "
                    "Set EXA_API_KEY environment variable to enable Exa search."
                ),
            )

        num_results = int(params.get("num_results", 5))
        num_results = max(1, min(num_results, 25))

        search_type = params.get("type", "auto")
        if search_type not in _VALID_SEARCH_TYPES:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid search type: {search_type}",
            )

        category = params.get("category")
        if category is not None and category not in _VALID_CATEGORIES:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid category: {category}",
            )

        request_kwargs: Dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "type": search_type,
        }
        if category:
            request_kwargs["category"] = category
        for key in (
            "include_domains",
            "exclude_domains",
            "include_text",
            "exclude_text",
            "start_published_date",
            "end_published_date",
        ):
            value = params.get(key)
            if value:
                request_kwargs[key] = value

        content_kwargs = self._build_content_kwargs(
            mode=params.get("content_mode", "highlights"),
            summary_query=params.get("summary_query"),
        )
        request_kwargs.update(content_kwargs)

        logger.info("exa_search", query=query, num_results=num_results, type=search_type)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, request_kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, output="", error=f"Exa search timed out after {self.timeout}s"
            )
        except ImportError as e:
            logger.error("exa_search_import_failed", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=(
                    "exa-py is not installed. Install it with: pip install exa-py>=2.0.0"
                ),
            )
        except Exception as e:
            logger.error("exa_search_failed", error=str(e))
            return ToolResult(success=False, output="", error=f"Exa search error: {e}")

        raw_results = getattr(response, "results", None)
        if raw_results is None and isinstance(response, dict):
            raw_results = response.get("results", [])
        raw_results = raw_results or []

        results = [ExaSearchResult.from_api(item) for item in raw_results]
        output = _format_results(query, results)

        logger.info("exa_search_success", query=query, results_count=len(results))

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "query": query,
                "results_count": len(results),
                "type": search_type,
            },
        )

    def _build_content_kwargs(
        self, mode: str, summary_query: Optional[str]
    ) -> Dict[str, Any]:
        """Map the user-facing content_mode to exa-py kwargs.

        Exa allows multiple content types per request; 'all' enables every
        type so the caller can pick whichever is most useful per result.
        """
        mode = (mode or "highlights").lower()
        kwargs: Dict[str, Any] = {}

        if mode == "none":
            return kwargs

        if mode in ("highlights", "all"):
            kwargs["highlights"] = True
        if mode in ("text", "all"):
            kwargs["text"] = True
        if mode in ("summary", "all"):
            kwargs["summary"] = (
                {"query": summary_query} if summary_query else True
            )

        return kwargs

    def _search_sync(self, request_kwargs: Dict[str, Any]) -> Any:
        """Run the blocking exa-py call. Executed inside a worker thread."""
        from exa_py import Exa  # imported lazily so the dep is optional at import time

        client = Exa(api_key=self.api_key)
        # Tag requests so Exa can attribute usage to this integration.
        try:
            client.headers["x-exa-integration"] = self.INTEGRATION_NAME
        except AttributeError:
            pass

        return client.search_and_contents(**request_kwargs)
