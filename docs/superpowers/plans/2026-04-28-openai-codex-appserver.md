# OpenAI Codex App-Server LLM Provider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `OpenAICodexLLMProvider` (which directly calls `chatgpt.com/backend-api` and gets blocked by Cloudflare) with one that delegates to the local `codex app-server` subprocess via JSON-RPC over stdio.

**Architecture:** A new `omniagent/llm/codex_appserver.py` module provides three classes: `CodexBinaryResolver` (finds the `codex` binary via env var or PATH), `CodexAppServerProcess` (manages a single shared subprocess as a module-level singleton), and `CodexAppServerLLMClient` (stateless per-call: thread/start → turn/start → collect notifications → return text). `OpenAICodexLLMProvider` in `llm.py` is rewritten to delegate to `CodexAppServerLLMClient` instead of making HTTP requests.

**Tech Stack:** Python 3.10+, asyncio, `asyncio.create_subprocess_exec`, `shutil.which`, `json`, `pytest`, `pytest-asyncio`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `omniagent/llm/__init__.py` | Create | Empty package marker |
| `omniagent/llm/codex_appserver.py` | Create | `CodexBinaryResolver`, `CodexAppServerProcess`, `CodexAppServerLLMClient` |
| `omniagent/agents/llm.py` | Modify (lines 543-583) | Rewrite `OpenAICodexLLMProvider` to delegate to `CodexAppServerLLMClient` |
| `tests/llm/__init__.py` | Create | Empty package marker |
| `tests/llm/test_codex_appserver.py` | Create | Unit tests for all three new classes |

---

## Task 1: Create the `omniagent/llm` package and `CodexBinaryResolver`

**Files:**
- Create: `omniagent/llm/__init__.py`
- Create: `omniagent/llm/codex_appserver.py` (partial — `CodexBinaryResolver` only)
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_codex_appserver.py` (partial)

- [ ] **Step 1: Write failing tests for `CodexBinaryResolver`**

Create `tests/llm/__init__.py` (empty) and `tests/llm/test_codex_appserver.py`:

```python
import os
import pytest
from unittest.mock import patch


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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/wangxingjian/git/OmniAgent
python -m pytest tests/llm/test_codex_appserver.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'omniagent.llm'`

- [ ] **Step 3: Create package and implement `CodexBinaryResolver`**

Create `omniagent/llm/__init__.py` (empty file).

Create `omniagent/llm/codex_appserver.py`:

```python
"""Codex app-server subprocess client for OmniAgent LLM integration."""

import asyncio
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


class CodexBinaryResolver:
    """Finds the codex executable via env var or PATH."""

    @staticmethod
    def resolve() -> str:
        env_path = os.environ.get("OMNIAGENT_CODEX_BIN", "").strip()
        if env_path:
            return env_path
        which_path = shutil.which("codex")
        if which_path:
            return which_path
        raise RuntimeError(
            "codex binary not found. Install it with: npm install -g @openai/codex\n"
            "Or set OMNIAGENT_CODEX_BIN=/path/to/codex"
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/llm/test_codex_appserver.py -v 2>&1 | head -30
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omniagent/llm/__init__.py omniagent/llm/codex_appserver.py tests/llm/__init__.py tests/llm/test_codex_appserver.py
git commit -m "feat: add CodexBinaryResolver for dynamic codex path discovery"
```

---

## Task 2: Implement `CodexAppServerProcess` (subprocess lifecycle)

**Files:**
- Modify: `omniagent/llm/codex_appserver.py` — add `CodexAppServerProcess`
- Modify: `tests/llm/test_codex_appserver.py` — add process lifecycle tests

- [ ] **Step 1: Write failing tests for `CodexAppServerProcess`**

Append to `tests/llm/test_codex_appserver.py`:

```python
import asyncio
import pytest


@pytest.mark.asyncio
async def test_process_start_sends_initialize_and_returns_client(tmp_path):
    """Process sends initialize RPC on startup and returns a connected client."""
    from omniagent.llm.codex_appserver import CodexAppServerProcess

    # Simulate a codex app-server that responds to initialize with a valid response
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


def _make_mock_process(stdout_lines: list):
    """Create a mock asyncio subprocess with scripted stdout responses."""
    import io
    from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/llm/test_codex_appserver.py::test_process_start_sends_initialize_and_returns_client -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `CodexAppServerProcess` not defined yet

- [ ] **Step 3: Implement `CodexAppServerProcess`**

Append to `omniagent/llm/codex_appserver.py` (after `CodexBinaryResolver`):

```python

class _RpcClient:
    """Minimal JSON-RPC client over asyncio subprocess stdio."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._notifications: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None

    def start_reader(self) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                # EOF — process exited
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("codex app-server exited unexpectedly"))
                self._pending.clear()
                break
            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            if "id" in msg and "method" not in msg:
                # RPC response
                fut = self._pending.pop(msg["id"], None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"].get("message", "RPC error")))
                    else:
                        fut.set_result(msg.get("result"))
            else:
                # Notification or server request
                await self._notifications.put(msg)

    async def request(self, method: str, params: Any = None) -> Any:
        rpc_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rpc_id] = fut
        msg = {"id": rpc_id, "method": method}
        if params is not None:
            msg["params"] = params
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()
        return await fut

    async def get_notification(self, timeout: float = 60.0) -> Dict[str, Any]:
        return await asyncio.wait_for(self._notifications.get(), timeout=timeout)

    def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        try:
            self._proc.terminate()
        except Exception:
            pass


class CodexAppServerProcess:
    """Manages a single shared codex app-server subprocess (lazy singleton per instance)."""

    def __init__(self) -> None:
        self._client: Optional[_RpcClient] = None
        self._lock = asyncio.Lock()

    async def get_or_start(self) -> _RpcClient:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            binary = CodexBinaryResolver.resolve()
            logger.info("codex_appserver_starting", binary=binary)
            proc = await asyncio.create_subprocess_exec(
                binary, "app-server", "--listen", "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            client = _RpcClient(proc)
            client.start_reader()
            # Handshake
            await client.request("initialize", {
                "clientInfo": {"name": "omniagent", "version": "1.0.0"},
                "capabilities": {},
            })
            self._client = client
            logger.info("codex_appserver_ready")
            return client

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/llm/test_codex_appserver.py -v 2>&1 | head -40
```

Expected: All tests PASS (including the 3 from Task 1)

- [ ] **Step 5: Commit**

```bash
git add omniagent/llm/codex_appserver.py tests/llm/test_codex_appserver.py
git commit -m "feat: add _RpcClient and CodexAppServerProcess for codex subprocess management"
```

---

## Task 3: Implement `CodexAppServerLLMClient` (stateless per-call LLM)

**Files:**
- Modify: `omniagent/llm/codex_appserver.py` — add `CodexAppServerLLMClient`
- Modify: `tests/llm/test_codex_appserver.py` — add LLM client tests

- [ ] **Step 1: Write failing tests for `CodexAppServerLLMClient`**

Append to `tests/llm/test_codex_appserver.py`:

```python
@pytest.mark.asyncio
async def test_llm_client_returns_assistant_text():
    """chat() sends thread/start + turn/start, collects notifications, returns text."""
    from omniagent.llm.codex_appserver import CodexAppServerLLMClient, CodexAppServerProcess
    from omniagent.agents.llm import LLMMessage

    # Scripted responses: initialize, thread/start, turn/start, then notifications via queue
    responses = [
        json.dumps({"id": 1, "result": {"userAgent": "codex/0.1.0"}}) + "\n",
        json.dumps({"id": 2, "result": {"thread": {"id": "thread-abc"}}}) + "\n",
        json.dumps({"id": 3, "result": {}}) + "\n",
    ]
    notifications = [
        json.dumps({"method": "item/message/delta", "params": {"content": [{"type": "output_text", "text": "Hello"}]}}) + "\n",
        json.dumps({"method": "item/message/delta", "params": {"content": [{"type": "output_text", "text": " world"}]}}) + "\n",
        json.dumps({"method": "turn/end", "params": {"turnId": "turn-1"}}) + "\n",
    ]

    all_lines = responses[:1]  # initialize response first

    with patch("omniagent.llm.codex_appserver.CodexBinaryResolver.resolve", return_value="codex"):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            # Feed all lines (responses + notifications interleaved)
            all_output = responses + notifications
            mock_proc = _make_mock_process(stdout_lines=all_output)
            mock_exec.return_value = mock_proc

            process = CodexAppServerProcess()
            llm_client = CodexAppServerLLMClient(process, model="gpt-5.4")

            messages = [
                LLMMessage(role="system", content="You are helpful."),
                LLMMessage(role="user", content="Say hello"),
            ]
            result = await llm_client.chat(messages)
            assert result == "Hello world"
            await process.close()


@pytest.mark.asyncio
async def test_llm_client_maps_system_to_developer_instructions():
    """System message is passed as developerInstructions in thread/start params."""
    from omniagent.llm.codex_appserver import CodexAppServerLLMClient, CodexAppServerProcess, _RpcClient
    from omniagent.agents.llm import LLMMessage
    from unittest.mock import AsyncMock

    captured_thread_start: Dict = {}

    async def mock_request(method, params=None):
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

    async def mock_get_notification(timeout=60.0):
        return {"method": "turn/end", "params": {"turnId": "turn-1"}}

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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/llm/test_codex_appserver.py::test_llm_client_returns_assistant_text -v 2>&1 | head -20
```

Expected: `AttributeError` — `CodexAppServerLLMClient` not defined

- [ ] **Step 3: Implement `CodexAppServerLLMClient`**

Append to `omniagent/llm/codex_appserver.py`:

```python

_DEFAULT_TURN_TIMEOUT = float(os.environ.get("OMNIAGENT_CODEX_TURN_TIMEOUT", "60"))


class CodexAppServerLLMClient:
    """Stateless LLM client: each chat() starts a new codex thread, runs one turn, returns text."""

    def __init__(self, process: CodexAppServerProcess, model: str = "gpt-5.4") -> None:
        self._process = process
        self._model = model

    async def chat(self, messages: List[Any]) -> str:
        client = await self._process.get_or_start()

        # Extract system prompt and conversation history
        system_content: Optional[str] = None
        history_parts: List[str] = []
        last_user_text = ""

        for i, msg in enumerate(messages):
            role = msg.role
            content = msg.content or ""
            if role == "system" and i == 0:
                system_content = content
            elif i == len(messages) - 1 and role == "user":
                last_user_text = content
            else:
                history_parts.append(f"{role.upper()}: {content}")

        # Start a new thread
        thread_params: Dict[str, Any] = {
            "model": self._model,
            "cwd": os.getcwd(),
            "sandbox": {"type": "dangerFullAccess"},
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "serviceName": "OmniAgent",
            "experimentalRawEvents": True,
            "persistExtendedHistory": False,
        }
        if system_content:
            thread_params["developerInstructions"] = system_content

        thread_result = await client.request("thread/start", thread_params)
        thread_id = thread_result["thread"]["id"]
        logger.debug("codex_thread_started", thread_id=thread_id)

        # Build turn input
        user_input: Dict[str, Any] = {"type": "userText", "text": last_user_text}
        if history_parts:
            user_input["context"] = "\n".join(history_parts)

        turn_params = {
            "threadId": thread_id,
            "input": user_input,
        }
        await client.request("turn/start", turn_params)

        # Collect streamed text until turn/end
        chunks: List[str] = []
        timeout = _DEFAULT_TURN_TIMEOUT

        while True:
            notification = await client.get_notification(timeout=timeout)
            method = notification.get("method", "")
            params = notification.get("params") or {}

            if method == "turn/end":
                break
            if method == "item/message/delta":
                content_items = params.get("content") or []
                for item in content_items:
                    if item.get("type") == "output_text":
                        text = item.get("text", "")
                        if text:
                            chunks.append(text)

        result = "".join(chunks)
        logger.debug("codex_turn_complete", text_length=len(result))
        return result
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
python -m pytest tests/llm/test_codex_appserver.py -v 2>&1 | head -40
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add omniagent/llm/codex_appserver.py tests/llm/test_codex_appserver.py
git commit -m "feat: add CodexAppServerLLMClient stateless per-call LLM interface"
```

---

## Task 4: Rewrite `OpenAICodexLLMProvider` in `llm.py`

**Files:**
- Modify: `omniagent/agents/llm.py` — rewrite `OpenAICodexLLMProvider` (lines 543–583)

- [ ] **Step 1: Write a failing integration-style test**

Append to `tests/llm/test_codex_appserver.py`:

```python
@pytest.mark.asyncio
async def test_openai_codex_provider_delegates_to_appserver():
    """OpenAICodexLLMProvider.chat() delegates to CodexAppServerLLMClient."""
    from omniagent.agents.llm import OpenAICodexLLMProvider, LLMMessage
    from unittest.mock import AsyncMock, patch

    with patch("omniagent.llm.codex_appserver.CodexAppServerLLMClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "mocked response"

        provider = OpenAICodexLLMProvider(model="gpt-5.4")
        messages = [LLMMessage(role="user", content="Hello")]
        result = await provider.chat(messages)

        assert result.content == "mocked response"
        mock_chat.assert_called_once_with(messages)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/llm/test_codex_appserver.py::test_openai_codex_provider_delegates_to_appserver -v 2>&1 | head -20
```

Expected: `TypeError` or `AssertionError` — current provider tries to make HTTP requests

- [ ] **Step 3: Rewrite `OpenAICodexLLMProvider` in `llm.py`**

Replace lines 543–583 in `omniagent/agents/llm.py` (the entire `OpenAICodexLLMProvider` class):

```python
class OpenAICodexLLMProvider(LLMProvider):
    """OpenAI Codex provider via local codex app-server subprocess.

    Delegates all LLM calls to a local `codex app-server` process over JSON-RPC
    stdio. The codex process handles OAuth and Cloudflare internally.
    """

    def __init__(self, model: str = "gpt-5.4") -> None:
        from omniagent.llm.codex_appserver import CodexAppServerLLMClient, CodexAppServerProcess
        self._llm_client = CodexAppServerLLMClient(
            process=CodexAppServerProcess(),
            model=model,
        )
        self.model = model
        logger.info("openai_codex_llm_initialized", model=model)

    @property
    def supports_native_function_calling(self) -> bool:
        return False

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        text = await self._llm_client.chat(messages)
        return LLMResponse(
            content=text,
            finish_reason="stop",
            usage={},
            metadata={"model": self.model, "provider": "openai-codex"},
        )

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        text = await self._llm_client.chat(messages)
        yield text
```

Also update the factory function at line ~1168–1174 to remove `auth_profiles_path`:

```python
    elif provider == "openai-codex":
        return OpenAICodexLLMProvider(
            model=model or "gpt-5.4",
        )
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/llm/ tests/auth/ -v 2>&1 | tail -20
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add omniagent/agents/llm.py tests/llm/test_codex_appserver.py
git commit -m "feat: rewrite OpenAICodexLLMProvider to delegate to codex app-server"
```

---

## Task 5: Verify end-to-end with live codex binary (smoke test)

**Files:**
- No file changes — manual verification only

- [ ] **Step 1: Confirm codex is available**

```bash
which codex && codex --version
```

Expected: Prints path and version, e.g. `codex 0.x.x`

- [ ] **Step 2: Run a quick smoke test via Python REPL**

```bash
cd /Users/wangxingjian/git/OmniAgent
python3 - <<'EOF'
import asyncio
from omniagent.agents.llm import OpenAICodexLLMProvider, LLMMessage

async def main():
    provider = OpenAICodexLLMProvider(model="gpt-5.4")
    messages = [LLMMessage(role="user", content="Reply with just: OK")]
    result = await provider.chat(messages)
    print("Response:", result.content)
    # Clean up
    await provider._llm_client._process.close()

asyncio.run(main())
EOF
```

Expected: Prints `Response: OK` (or similar short confirmation)

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: All tests PASS

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete openai-codex app-server provider — replaces broken HTTP implementation"
```
