import json
import os
import pytest
import pytest_asyncio
from typing import Dict
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
