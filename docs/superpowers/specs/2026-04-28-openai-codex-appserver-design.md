# OpenAI Codex App-Server LLM Provider — Design

**Date:** 2026-04-28  
**Status:** Approved  
**Scope:** Replace the broken `OpenAICodexLLMProvider` HTTP implementation with one that delegates to a local `codex app-server` subprocess via JSON-RPC over stdio.

---

## Problem

The current `OpenAICodexLLMProvider` sends HTTP requests directly to `chatgpt.com/backend-api` using the `openai` Python SDK. Cloudflare Bot Management blocks these requests unconditionally, returning an HTML challenge page instead of a JSON response.

---

## Solution

Delegate all ChatGPT API calls to the official `codex` CLI's built-in `app-server` mode. The app-server handles OAuth token management and Cloudflare transparently. OmniAgent communicates with it over JSON-RPC on stdio.

---

## Architecture

```
OmniAgent ReflexionAgent
    │
    ▼
OpenAICodexLLMProvider.chat(messages)
    │
    ▼
CodexAppServerLLMClient        ← new, in omniagent/llm/codex_appserver.py
    │
    ▼
CodexAppServerProcess          ← module-level singleton, lazy init
    │  JSON-RPC over stdio (newline-delimited JSON)
    ▼
codex app-server subprocess    ← started once, reused across requests
    │  internal: OAuth + Cloudflare handled by codex
    ▼
chatgpt.com backend-api
```

---

## Components

### `omniagent/llm/codex_appserver.py` (new file)

**`CodexBinaryResolver`**

Discovers the `codex` executable path at runtime. Priority:
1. `OMNIAGENT_CODEX_BIN` environment variable
2. `shutil.which("codex")` — uses PATH
3. Raises `RuntimeError` with install instructions

Never hardcodes a path.

**`CodexAppServerProcess`**

Manages the `codex app-server --listen stdio://` subprocess lifecycle.

- Module-level singleton, lazy-started on first `chat()` call
- On first start: launches subprocess, sends `initialize` RPC, waits for response
- If subprocess exits unexpectedly: singleton is cleared; next call restarts it
- `close()`: terminates subprocess gracefully

**`CodexAppServerLLMClient`**

Stateless LLM calls. Each `chat()` call:

1. Calls `CodexAppServerProcess.get_or_start()` to get an initialized client
2. Sends `thread/start` RPC (new thread per call, `sandbox=dangerFullAccess`, passes `model` and `cwd`)
3. Sends `turn/start` RPC with user input constructed from `messages`
4. Collects streamed notifications until `turn/end` notification arrives
5. Returns concatenated assistant text
6. Thread is abandoned (not explicitly closed — codex GCs it)

### `omniagent/agents/llm.py` (modified)

`OpenAICodexLLMProvider`:
- No longer inherits `OpenAILLMProvider`
- Removes `CODEX_BASE_URL`, HTTP logic, and token injection
- `chat()` and `chat_stream()` delegate to `CodexAppServerLLMClient`

---

## Data Flow: One `chat()` Call

```
messages = [system, user1, assistant1, user2]   ← passed in by ReflexionAgent

→ thread/start  { model, cwd, sandbox, developerInstructions: system_prompt }
← { thread: { id } }

→ turn/start  { threadId, userText: last_user_message,
                context: prior_history_as_text }
← RPC ack

← notification: item/message/delta  { text: "..." }   (0 or more)
← notification: turn/end            { turnId }

→ return concatenated text
```

**Message mapping:**
- `messages[-1]` (last user message) → `userText` in `turn/start`
- `messages[0]` if role=system → `developerInstructions` in `thread/start`
- Remaining history → serialized as plain text in `turn/start` `context` field

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| `codex` binary not found | `RuntimeError` with install instructions and `OMNIAGENT_CODEX_BIN` hint |
| app-server startup failure | Exception propagated; singleton cleared so next call retries |
| Subprocess exits unexpectedly | Detected via EOF on stdout pipe; singleton cleared; error raised |
| `turn/start` RPC error response | `RuntimeError` with server error message |
| Turn timeout | `asyncio.wait_for` raises `TimeoutError` after configurable seconds |

---

## Configuration

```yaml
# ~/.omniagent/config.yaml
agent:
  model_provider: openai-codex
  model_id: gpt-5.4       # passed to thread/start as `model`
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIAGENT_CODEX_BIN` | (PATH lookup) | Override path to `codex` binary |
| `OMNIAGENT_CODEX_TURN_TIMEOUT` | `60` | Seconds to wait for a turn to complete |

---

## What Does NOT Change

- `omniagent/auth/openai_codex.py` — OAuth PKCE login flow unchanged
- `omniagent/auth/token_manager.py` / `token_store.py` — unchanged
- `omniagent/config/models.py` — `openai-codex` provider entry unchanged
- The `codex` CLI reads its own `~/.codex/auth.json` — OmniAgent does not pass tokens to the subprocess

---

## Files Changed

| File | Change |
|------|--------|
| `omniagent/llm/codex_appserver.py` | **New** — `CodexBinaryResolver`, `CodexAppServerProcess`, `CodexAppServerLLMClient` |
| `omniagent/agents/llm.py` | Modify `OpenAICodexLLMProvider` to delegate to `CodexAppServerLLMClient` |
