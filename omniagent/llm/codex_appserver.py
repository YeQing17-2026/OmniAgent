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


class _RpcClient:
    """Minimal JSON-RPC client over asyncio subprocess stdio."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
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
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rpc_id] = fut
        msg: Dict[str, Any] = {"id": rpc_id, "method": method}
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
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_or_start(self) -> _RpcClient:
        if self._client is not None:
            return self._client
        async with self._get_lock():
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


_DEFAULT_TURN_TIMEOUT = float(os.environ.get("OMNIAGENT_CODEX_TURN_TIMEOUT", "60"))


class CodexAppServerLLMClient:
    """Stateless LLM client: each chat() starts a new codex thread, runs one turn, returns text."""

    def __init__(self, process: CodexAppServerProcess, model: str = "gpt-5.4") -> None:
        self._process = process
        self._model = model

    async def chat(self, messages: List[Any]) -> str:
        client = await self._process.get_or_start()

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

        user_input: Dict[str, Any] = {"type": "userText", "text": last_user_text}
        if history_parts:
            user_input["context"] = "\n".join(history_parts)

        await client.request("turn/start", {
            "threadId": thread_id,
            "input": user_input,
        })

        chunks: List[str] = []
        timeout = _DEFAULT_TURN_TIMEOUT

        while True:
            notification = await client.get_notification(timeout=timeout)
            method = notification.get("method", "")
            params = notification.get("params") or {}

            if method == "turn/end":
                break
            if method == "item/message/delta":
                for item in (params.get("content") or []):
                    if item.get("type") == "output_text":
                        text = item.get("text", "")
                        if text:
                            chunks.append(text)

        result = "".join(chunks)
        logger.debug("codex_turn_complete", text_length=len(result))
        return result
