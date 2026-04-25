"""Tests for the Exa search tool."""

import sys
import types
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from omniagent.tools.exa_tool import (
    ExaSearchResult,
    ExaSearchTool,
    _format_results,
)


def _fake_search_response() -> Any:
    """Build a fake response object shaped like exa-py's SearchResponse."""
    results = [
        types.SimpleNamespace(
            title="Anthropic",
            url="https://anthropic.com",
            published_date="2025-01-15T00:00:00Z",
            author="Anthropic",
            text="Anthropic is an AI safety company.",
            highlights=["AI safety company", "Claude is a chatbot"],
            summary="Anthropic builds Claude.",
            score=0.91,
        ),
        types.SimpleNamespace(
            title="Exa",
            url="https://exa.ai",
            published_date=None,
            author=None,
            text=None,
            highlights=[],
            summary=None,
            score=0.7,
        ),
    ]
    return types.SimpleNamespace(results=results)


class _FakeExa:
    """Stand-in for exa_py.Exa used to capture calls and headers."""

    instances: List["_FakeExa"] = []

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers: Dict[str, str] = {}
        self.calls: List[Dict[str, Any]] = []
        _FakeExa.instances.append(self)

    def search_and_contents(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _fake_search_response()


@pytest.fixture(autouse=True)
def _install_fake_exa(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake exa_py module so the lazy import resolves to our stub."""
    _FakeExa.instances = []
    fake_module = types.ModuleType("exa_py")
    fake_module.Exa = _FakeExa  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)
    yield


def test_search_result_from_dict_shape():
    item = {
        "title": "Hello",
        "url": "https://example.com",
        "publishedDate": "2025-04-01",
        "highlights": ["a", "b"],
    }
    parsed = ExaSearchResult.from_api(item)
    assert parsed.title == "Hello"
    assert parsed.url == "https://example.com"
    assert parsed.published_date == "2025-04-01"
    assert parsed.highlights == ["a", "b"]


def test_snippet_prefers_highlights_then_summary_then_text():
    only_text = ExaSearchResult(title="t", url="u", text="full text body")
    only_summary = ExaSearchResult(title="t", url="u", summary="a summary")
    with_highlights = ExaSearchResult(
        title="t", url="u", highlights=["first hit", "second hit"], summary="x", text="y"
    )
    assert only_text.snippet() == "full text body"
    assert only_summary.snippet() == "a summary"
    assert with_highlights.snippet().startswith("first hit")


def test_format_results_handles_empty():
    out = _format_results("nothing", [])
    assert "No results" in out


def test_format_results_renders_url_and_snippet():
    results = [
        ExaSearchResult(
            title="Anthropic",
            url="https://anthropic.com",
            highlights=["AI safety"],
        )
    ]
    out = _format_results("anthropic", results)
    assert "Anthropic" in out
    assert "https://anthropic.com" in out
    assert "AI safety" in out


@pytest.mark.asyncio
async def test_execute_returns_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    tool = ExaSearchTool()
    result = await tool.execute({"query": "anthropic"})
    assert result.success is False
    assert "EXA_API_KEY" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_rejects_blank_query(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    tool = ExaSearchTool()
    result = await tool.execute({"query": "   "})
    assert result.success is False
    assert "query" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_calls_exa_with_expected_kwargs_and_header(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    tool = ExaSearchTool()

    result = await tool.execute(
        {
            "query": "anthropic",
            "num_results": 3,
            "type": "neural",
            "category": "company",
            "include_domains": ["anthropic.com"],
            "exclude_domains": ["example.com"],
            "start_published_date": "2024-01-01",
            "content_mode": "all",
            "summary_query": "What does the company do?",
        }
    )

    assert result.success is True
    assert "Anthropic" in result.output
    assert "https://anthropic.com" in result.output
    assert result.metadata["results_count"] == 2
    assert result.metadata["type"] == "neural"

    assert len(_FakeExa.instances) == 1
    fake = _FakeExa.instances[0]
    assert fake.api_key == "test-key"
    assert fake.headers.get("x-exa-integration") == "omniagent"

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["query"] == "anthropic"
    assert call["num_results"] == 3
    assert call["type"] == "neural"
    assert call["category"] == "company"
    assert call["include_domains"] == ["anthropic.com"]
    assert call["exclude_domains"] == ["example.com"]
    assert call["start_published_date"] == "2024-01-01"
    # content_mode='all' should turn on every content type
    assert call["highlights"] is True
    assert call["text"] is True
    assert call["summary"] == {"query": "What does the company do?"}


@pytest.mark.asyncio
async def test_execute_clamps_num_results(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    tool = ExaSearchTool()
    await tool.execute({"query": "x", "num_results": 999})
    call = _FakeExa.instances[0].calls[0]
    assert call["num_results"] == 25


@pytest.mark.asyncio
async def test_execute_rejects_invalid_search_type(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    tool = ExaSearchTool()
    result = await tool.execute({"query": "x", "type": "keyword"})
    assert result.success is False
    assert "Invalid search type" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_handles_partial_results_without_highlights(monkeypatch):
    """A result with only a summary should still produce a usable snippet."""
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    summary_only = types.SimpleNamespace(
        title="Summary Only",
        url="https://s.example/",
        published_date=None,
        author=None,
        text=None,
        highlights=[],
        summary="This is only a summary.",
        score=0.5,
    )

    class _SummaryOnly(_FakeExa):
        def search_and_contents(self, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace(results=[summary_only])

    fake_module = types.ModuleType("exa_py")
    fake_module.Exa = _SummaryOnly  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)

    tool = ExaSearchTool()
    result = await tool.execute({"query": "x", "content_mode": "summary"})
    assert result.success is True
    assert "This is only a summary." in result.output


@pytest.mark.asyncio
async def test_execute_surface_sdk_exceptions(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    class _Boom(_FakeExa):
        def search_and_contents(self, **kwargs):
            raise RuntimeError("upstream 500")

    fake_module = types.ModuleType("exa_py")
    fake_module.Exa = _Boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)

    tool = ExaSearchTool()
    result = await tool.execute({"query": "x"})
    assert result.success is False
    assert "upstream 500" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_reports_missing_sdk(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "exa_py", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "exa_py":
            raise ImportError("No module named 'exa_py'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        tool = ExaSearchTool()
        result = await tool.execute({"query": "x"})
    assert result.success is False
    assert "exa-py" in (result.error or "")


def test_schema_advertises_required_fields():
    schema = ExaSearchTool().get_schema()
    assert schema["name"] == "exa_search"
    assert "query" in schema["parameters"]["required"]
    props = schema["parameters"]["properties"]
    assert "num_results" in props
    assert "include_domains" in props
    assert "content_mode" in props
