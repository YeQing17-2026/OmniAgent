import json
import os
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Task 1: CodexBinaryResolver
# ---------------------------------------------------------------------------

def test_resolver_uses_env_var(tmp_path):
    fake_bin = tmp_path / "codex"
    fake_bin.touch(mode=0o755)
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {"OMNIAGENT_CODEX_BIN": str(fake_bin)}):
        assert CodexBinaryResolver.resolve() == str(fake_bin)


def test_resolver_falls_back_to_which():
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {}, clear=True):
        with patch("shutil.which", return_value="/usr/local/bin/codex"):
            assert CodexBinaryResolver.resolve() == "/usr/local/bin/codex"


def test_resolver_raises_when_not_found():
    from omniagent.llm.codex_appserver import CodexBinaryResolver
    with patch.dict(os.environ, {}, clear=True):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="codex"):
                CodexBinaryResolver.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_process(stdout_lines: list):
    """Create a mock asyncio subprocess with scripted stdout responses."""
    proc = MagicMock()
    proc.returncode = None

    lines = [line.encode() for line in stdout_lines]
    lines_iter = iter(lines)

    async def mock_readline():
        try:
            return next(lines_iter)
        except StopIteration:
            return b""

    proc.stdout = MagicMock()
    proc.stdout.readline = mock_readline
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Task 2: CodexAppServerProcess
# ---------------------------------------------------------------------------

async def test_process_start_sends_initialize_and_returns_client():
    """Process sends initialize RPC on startup and returns a connected client."""
    from omniagent.llm.codex_appserver import CodexAppServerProcess

    fake_response = json.dumps({
        "id": 1,
        "result": {"userAgent": "codex/0.1.0"}
    }) + "\n"

    with patch("omniagent.llm.codex_appserver.CodexBinaryResolver.resolve", return_value="codex"):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = _make_mock_process(stdout_lines=[fake_response])
            mock_exec.return_value = mock_proc

            process = CodexAppServerProcess()
            client = await process.get_or_start()
            assert client is not None
            await process.close()


async def test_process_reuses_existing_client():
    """get_or_start() returns the same client on second call."""
    from omniagent.llm.codex_appserver import CodexAppServerProcess

    fake_response = json.dumps({"id": 1, "result": {"userAgent": "codex/0.1.0"}}) + "\n"

    with patch("omniagent.llm.codex_appserver.CodexBinaryResolver.resolve", return_value="codex"):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = _make_mock_process(stdout_lines=[fake_response])
            mock_exec.return_value = mock_proc

            process = CodexAppServerProcess()
            client1 = await process.get_or_start()
            client2 = await process.get_or_start()
            assert client1 is client2
            assert mock_exec.call_count == 1
            await process.close()


# ---------------------------------------------------------------------------
# Task 3: CodexAppServerLLMClient
# ---------------------------------------------------------------------------

async def test_llm_client_returns_assistant_text():
    """chat() collects item/message/delta notifications until turn/end and returns text."""
    from omniagent.llm.codex_appserver import CodexAppServerLLMClient, CodexAppServerProcess, _RpcClient
    from omniagent.agents.llm import LLMMessage

    notifications = [
        {"method": "item/agentMessage/delta", "params": {"delta": "Hello"}},
        {"method": "item/agentMessage/delta", "params": {"delta": " world"}},
        {"method": "turn/completed", "params": {"turnId": "turn-1"}},
    ]
    notif_iter = iter(notifications)

    async def mock_request(method: str, params: Any = None) -> Any:
        if method == "initialize":
            return {"userAgent": "codex/0.1.0"}
        if method == "thread/start":
            return {"thread": {"id": "thread-abc"}}
        if method == "turn/start":
            return {}
        return {}

    async def mock_get_notification(timeout: float = 60.0) -> Dict:
        return next(notif_iter)

    mock_client = MagicMock(spec=_RpcClient)
    mock_client.request = mock_request
    mock_client.get_notification = mock_get_notification

    process = MagicMock(spec=CodexAppServerProcess)
    process.get_or_start = AsyncMock(return_value=mock_client)

    llm_client = CodexAppServerLLMClient(process, model="gpt-5.4")
    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="Say hello"),
    ]
    result = await llm_client.chat(messages)
    assert result == "Hello world"


async def test_llm_client_maps_system_to_developer_instructions():
    """System message is passed as developerInstructions in thread/start params."""
    from omniagent.llm.codex_appserver import CodexAppServerLLMClient, CodexAppServerProcess, _RpcClient
    from omniagent.agents.llm import LLMMessage

    captured_thread_start: Dict = {}

    async def mock_request(method: str, params: Any = None) -> Any:
        if method == "initialize":
            return {"userAgent": "codex/0.1.0"}
        if method == "thread/start":
            captured_thread_start.update(params or {})
            return {"thread": {"id": "t1"}}
        if method == "turn/start":
            return {}
        return {}

    mock_client = MagicMock(spec=_RpcClient)
    mock_client.request = mock_request

    async def mock_get_notification(timeout: float = 60.0) -> Dict:
        return {"method": "turn/completed", "params": {"turnId": "turn-1"}}

    mock_client.get_notification = mock_get_notification

    process = MagicMock(spec=CodexAppServerProcess)
    process.get_or_start = AsyncMock(return_value=mock_client)

    llm_client = CodexAppServerLLMClient(process, model="gpt-5.4")
    messages = [
        LLMMessage(role="system", content="Be concise."),
        LLMMessage(role="user", content="Hi"),
    ]
    await llm_client.chat(messages)
    assert captured_thread_start.get("developerInstructions") == "Be concise."


# ---------------------------------------------------------------------------
# Task 4: OpenAICodexLLMProvider
# ---------------------------------------------------------------------------

async def test_openai_codex_provider_delegates_to_appserver():
    """OpenAICodexLLMProvider.chat() delegates to CodexAppServerLLMClient and wraps result."""
    from omniagent.agents.llm import OpenAICodexLLMProvider, LLMMessage

    with patch("omniagent.llm.codex_appserver.CodexAppServerLLMClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "mocked response"

        provider = OpenAICodexLLMProvider(model="gpt-5.4")
        messages = [LLMMessage(role="user", content="Hello")]
        result = await provider.chat(messages)

        assert result.content == "mocked response"
        assert result.finish_reason == "stop"
        assert result.metadata["provider"] == "openai-codex"
        mock_chat.assert_called_once_with(messages)
