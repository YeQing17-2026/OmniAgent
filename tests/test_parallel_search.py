"""Exercise native provider selection and real MCP protocol through a fake HTTP server."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from omniagent.agents.reflexion import ReflexionAgent
from omniagent.config import OmniAgentConfig
from omniagent.config.loader import load_config, save_config
from omniagent.tools.parallel_search import ParallelSearchTool
from omniagent.tools.web_tools import WebSearchTool


def make_agent(config, work_dir):
    config.memory.enabled = False
    config.enable_self_improving = False
    return ReflexionAgent(
        config,
        llm_provider=SimpleNamespace(supports_native_function_calling=True),
        work_dir=work_dir,
        enable_security=False,
    )


def test_provider_defaults_and_saved_selection(tmp_path):
    default = make_agent(OmniAgentConfig(), tmp_path)
    assert type(default.registry.get("web_search")) is WebSearchTool
    config = OmniAgentConfig(tools={"web_search_provider": "parallel"})
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    selected = make_agent(loaded, tmp_path)
    tool = selected.registry.get("web_search")
    assert isinstance(tool, ParallelSearchTool)
    assert tool.model_name == loaded.agent.model_id
    assert tool.get_schema()["parameters"] == default.tools["web_search"].get_schema()["parameters"]
    assert type(selected.tools["web_fetch"]) is type(default.tools["web_fetch"])


@pytest.fixture
def mcp_server(monkeypatch):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("mcp")
    requests = []
    response = {
        "results": [
            {
                "title": "Python docs",
                "url": "https://docs.python.org/3/",
                "excerpts": ["Python documentation"],
            },
            {"title": None, "url": "https://python.org/", "excerpts": ["Python home"]},
        ]
    }
    mode = {"value": "success"}

    async def handle(request):
        assert str(request.url) == ParallelSearchTool.MCP_URL
        assert request.headers["user-agent"] == ParallelSearchTool.USER_AGENT
        assert "authorization" not in request.headers
        requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            assert message["params"]["name"] == "web_search"
            args = message["params"]["arguments"]
            assert args["objective"] == args["search_queries"][0]
            assert len(args["session_id"]) == 36
            if mode["value"] == "hang":
                await asyncio.Event().wait()
            if mode["value"] == "http_error":
                return httpx.Response(503)
            if mode["value"] == "error":
                result = {
                    "content": [{"type": "text", "text": "Search quota exceeded"}],
                    "isError": True,
                }
            else:
                result = {"content": [{"type": "text", "text": json.dumps(response)}]}
                if mode["value"] != "text_only":
                    result["structuredContent"] = response
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    client_class = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["timeout"] > 0
        return client_class(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("BRAVE_API_KEY", "unrelated-brave-key")
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    return requests, response, mode


def test_native_dispatch_keyless_and_session_reuse(tmp_path, mcp_server):
    requests, _, _ = mcp_server
    agent = make_agent(OmniAgentConfig(tools={"web_search_provider": "parallel"}), tmp_path)

    async def run():
        first = await agent._execute_tool(
            {"tool": "web_search", "params": {"query": "Python documentation", "max_results": 1}}
        )
        second = await agent.tools["web_search"].execute(
            {"query": "Python documentation", "max_results": 10}
        )
        return first, second

    first, second = asyncio.run(run())
    assert "Python docs" in first and "https://docs.python.org/3/" in first
    assert "Python home" not in first
    assert second.success and second.metadata["results_count"] == 2
    calls = [
        json.loads(r.content)["params"]["arguments"]
        for r in requests
        if r.method == "POST" and json.loads(r.content)["method"] == "tools/call"
    ]
    assert len(calls) == 2
    assert calls[0]["session_id"] == calls[1]["session_id"]
    assert calls[0]["model_name"] == agent.config.agent.model_id


@pytest.mark.parametrize("mode", ["text_only", "error", "http_error", "empty"])
def test_responses(mcp_server, mode):
    _, response, state = mcp_server
    state["value"] = mode
    if mode == "empty":
        response["results"] = []
    result = asyncio.run(ParallelSearchTool().execute({"query": "Python docs"}))
    assert result.success == (mode in ("text_only", "empty"))
    if mode == "error":
        assert "quota exceeded" in result.error
    if mode == "empty":
        assert result.metadata["results_count"] == 0
        assert "No results found" in result.output


def test_timeout_and_cancellation(mcp_server):
    _, _, state = mcp_server
    state["value"] = "hang"
    result = asyncio.run(ParallelSearchTool(timeout=0.05).execute({"query": "test"}))
    assert not result.success and "timed out" in result.error

    async def cancel():
        task = asyncio.create_task(ParallelSearchTool().execute({"query": "test"}))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())


def test_missing_extra_and_invalid_parameters(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    tool = ParallelSearchTool()
    missing = asyncio.run(tool.execute({"query": "Python"}))
    assert not missing.success and "omniagent[parallel]" in missing.error
    for params in ({}, {"query": " "}, {"query": 123}, {"query": "ok", "max_results": "five"}):
        assert not asyncio.run(tool.execute(params)).success
