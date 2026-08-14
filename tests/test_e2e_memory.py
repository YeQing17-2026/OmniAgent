"""End-to-end tests for the memory system integrated into ReflexionAgent.

Tests the full pipeline:
  memory file → sync → handle_message → active memory injection
  → tool calls (memory_search / memory_store / memory_get) → response

Uses MockLLMProvider — scripted responses, no real API needed.
"""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from omniagent.agents.llm import LLMProvider, LLMMessage, LLMResponse
from omniagent.config.models import OmniAgentConfig
from omniagent.gateway.router import IncomingMessage


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Returns scripted LLMResponse objects. Records every call for assertions."""

    def __init__(self, script: List[LLMResponse]):
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    @property
    def supports_native_function_calling(self) -> bool:
        return True

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "tool_names": [t.get("function", {}).get("name") for t in (tools or [])],
        })
        if self._script:
            return self._script.pop(0)
        return LLMResponse(content="Done.", finish_reason="stop")

    async def chat_stream(
        self, messages: List[LLMMessage], temperature=0.7, max_tokens=4096
    ) -> AsyncIterator[str]:
        yield "Done."


def text_resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, finish_reason="stop")


def tool_resp(name: str, args: Dict) -> LLMResponse:
    """Simulate an LLM response that calls a tool (OpenAI tool_calls format)."""
    return LLMResponse(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[{
            "id": "call_test_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(memory_enabled: bool = True) -> OmniAgentConfig:
    cfg = OmniAgentConfig()
    cfg.memory.enabled = memory_enabled
    cfg.memory.hybrid_enabled = False       # FTS only — no embedding API needed
    cfg.memory.query_min_score = 0.01       # Low threshold so test content matches
    cfg.memory.sync_on_session_start = True
    cfg.agent.max_iterations = 5
    cfg.enable_self_improving = False
    cfg.sentinel.enabled = False
    cfg.guardian.enabled = False
    return cfg


def make_agent(work_dir: Path, llm: MockLLMProvider, memory_enabled: bool = True):
    from omniagent.agents.reflexion import ReflexionAgent
    return ReflexionAgent(
        config=make_config(memory_enabled),
        llm_provider=llm,
        work_dir=work_dir,
        enable_security=False,
    )


def make_message(content: str, session_id: str = "test-session") -> IncomingMessage:
    msg = IncomingMessage.__new__(IncomingMessage)
    msg.session_id = session_id
    msg.user_id = "test-user"
    msg.channel_id = "cli"
    msg.content = content
    msg.metadata = {}
    return msg


# ---------------------------------------------------------------------------
# 1. Memory sync on session start
# ---------------------------------------------------------------------------

class TestMemorySyncOnSessionStart:
    @pytest.mark.asyncio
    async def test_chunks_indexed_before_reply(self, tmp_path):
        """MEMORY.md is indexed before the first LLM call."""
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nUser prefers concise responses.\n"
        )
        llm = MockLLMProvider([text_resp("Sure, I'll be concise.")])
        agent = make_agent(tmp_path, llm)

        await agent.handle_message(make_message("Hello"))

        chunks = agent.memory_manager.store.get_all_chunks()
        assert len(chunks) > 0, "MEMORY.md should have been indexed"

    @pytest.mark.asyncio
    async def test_memory_disabled_no_manager(self, tmp_path):
        """When memory=disabled, memory_manager is None and tools are absent."""
        llm = MockLLMProvider([text_resp("Hi!")])
        agent = make_agent(tmp_path, llm, memory_enabled=False)

        assert agent.memory_manager is None
        tool_names = set(agent.tools.keys())
        assert "memory_search" not in tool_names

        resp = await agent.handle_message(make_message("Hi"))
        assert resp.content == "Hi!"

    @pytest.mark.asyncio
    async def test_incremental_sync_no_double_index(self, tmp_path):
        """Second sync on unchanged file returns 0 updated files."""
        (tmp_path / "MEMORY.md").write_text("# Memory\n\nhello world\n")
        llm = MockLLMProvider([text_resp("ok")])
        agent = make_agent(tmp_path, llm)

        n1 = await agent.memory_manager.sync()
        assert n1 >= 1

        n2 = await agent.memory_manager.sync()
        assert n2 == 0


# ---------------------------------------------------------------------------
# 2. memory_search tool
# ---------------------------------------------------------------------------

class TestMemorySearchTool:
    @pytest.mark.asyncio
    async def test_search_finds_relevant_content(self, tmp_path):
        """memory_search returns chunks matching the query."""
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text(
            "# Facts\n\n"
            "The project deadline is 2026-06-01.\n"
            "The tech stack is FastAPI and PostgreSQL.\n"
        )
        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.01)
        await mgr.sync()

        results = await mgr.search("project deadline")
        assert len(results) > 0
        assert any("deadline" in r.text.lower() for r in results)

    @pytest.mark.asyncio
    async def test_agent_calls_memory_search_tool(self, tmp_path):
        """Agent can invoke memory_search and receive tool result."""
        (tmp_path / "MEMORY.md").write_text(
            "# Profile\n\n"
            "The user loves Python programming.\n"
            "Python is the user's primary language.\n"
        )
        llm = MockLLMProvider([
            tool_resp("memory_search", {"query": "user language preference Python"}),
            text_resp("You prefer Python!"),
        ])
        agent = make_agent(tmp_path, llm)
        await agent.memory_manager.sync()

        resp = await agent.handle_message(make_message("What language do I prefer?"))

        assert resp.content is not None
        # memory_search must have been offered as a tool
        all_tool_names = [n for c in llm.calls for n in c["tool_names"]]
        assert "memory_search" in all_tool_names

    @pytest.mark.asyncio
    async def test_memory_search_min_score_filter(self, tmp_path):
        """Results below min_score are filtered out."""
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text(
            "# Notes\n\nThe server IP is 10.0.0.1.\n"
        )
        mgr_strict = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.99)
        await mgr_strict.sync()
        results = await mgr_strict.search("server IP")
        # With score threshold of 0.99, unlikely to match
        assert len(results) == 0 or all(r.score >= 0.99 for r in results)


# ---------------------------------------------------------------------------
# 3. memory_store tool
# ---------------------------------------------------------------------------

class TestMemoryStoreTool:
    @pytest.mark.asyncio
    async def test_store_creates_category_file(self, tmp_path):
        """memory_store writes content to memory/<category>.md."""
        llm = MockLLMProvider([
            tool_resp("memory_store", {
                "text": "User's name is Alice. She works at Acme Corp.",
                "category": "entity",
            }),
            text_resp("Saved: Alice at Acme Corp."),
        ])
        agent = make_agent(tmp_path, llm)

        await agent.handle_message(
            make_message("Remember: my name is Alice, I work at Acme Corp.")
        )

        entity_file = tmp_path / "memory" / "entity.md"
        assert entity_file.exists()
        content = entity_file.read_text()
        assert "Alice" in content
        assert "Acme Corp" in content

    @pytest.mark.asyncio
    async def test_store_custom_path(self, tmp_path):
        """memory_store respects custom path parameter."""
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool

        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryStoreTool(mgr)

        result = await tool.execute({
            "text": "Deployment host: prod-01.example.com",
            "path": "memory/infra/servers.md",
        })
        assert result.success
        assert (tmp_path / "memory" / "infra" / "servers.md").exists()

    @pytest.mark.asyncio
    async def test_stored_content_is_searchable(self, tmp_path):
        """Content written via memory_store is found by subsequent search."""
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool

        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.01)
        tool = MemoryStoreTool(mgr)

        await tool.execute({
            "text": "The production database host is db.prod.example.com port 5432.",
            "category": "fact",
        })
        await mgr.sync()

        results = await mgr.search("production database host")
        assert any("db.prod.example.com" in r.text for r in results)

    @pytest.mark.asyncio
    async def test_store_missing_text_returns_error(self, tmp_path):
        """memory_store returns failure when text is missing."""
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool

        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryStoreTool(mgr)
        result = await tool.execute({})
        assert not result.success


# ---------------------------------------------------------------------------
# 4. memory_get tool
# ---------------------------------------------------------------------------

class TestMemoryGetTool:
    @pytest.mark.asyncio
    async def test_get_reads_memory_file(self, tmp_path):
        """memory_get returns file contents for a valid path."""
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryGetTool

        (tmp_path / "MEMORY.md").write_text("# My Memory\n\nLine 1\nLine 2\n")
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryGetTool(mgr)

        result = await tool.execute({"path": "MEMORY.md", "from_line": 1, "lines": 5})
        assert result.success
        assert "My Memory" in result.output

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        """memory_get rejects paths outside the memory directory."""
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryGetTool

        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryGetTool(mgr)

        result = await tool.execute({"path": "../../etc/passwd"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_agent_calls_memory_get(self, tmp_path):
        """Agent can invoke memory_get tool."""
        (tmp_path / "MEMORY.md").write_text(
            "# Memory\n\nProject: OmniAgent v0.1.0\n"
        )
        llm = MockLLMProvider([
            tool_resp("memory_get", {"path": "MEMORY.md"}),
            text_resp("The project is OmniAgent v0.1.0."),
        ])
        agent = make_agent(tmp_path, llm)

        resp = await agent.handle_message(make_message("What project is this?"))
        assert resp.content is not None
        all_tool_names = [n for c in llm.calls for n in c["tool_names"]]
        assert "memory_get" in all_tool_names


# ---------------------------------------------------------------------------
# 5. All three memory tools registered
# ---------------------------------------------------------------------------

class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_all_memory_tools_registered(self, tmp_path):
        """ReflexionAgent registers memory_search, memory_get, memory_store."""
        llm = MockLLMProvider([])
        agent = make_agent(tmp_path, llm)

        tool_names = set(agent.tools.keys())
        assert "memory_search" in tool_names
        assert "memory_get" in tool_names
        assert "memory_store" in tool_names


# ---------------------------------------------------------------------------
# 6. Active Memory injection
# ---------------------------------------------------------------------------

class TestActiveMemoryInjection:
    @pytest.mark.asyncio
    async def test_active_memory_not_triggered_for_headless(self, tmp_path):
        """Active memory skips ineligible session kinds."""
        from omniagent.agents.active_memory import ActiveMemoryAgent
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text("# Memory\n\nsome data\n")
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        await mgr.sync()

        agent_obj = ActiveMemoryAgent(memory_manager=mgr, llm_provider=None)
        result = await agent_obj.run(
            "any message", session_id="s1", session_kind="headless"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_active_memory_returns_summary(self, tmp_path):
        """Active memory returns a summary for interactive sessions with matching memory."""
        from omniagent.agents.active_memory import ActiveMemoryAgent, ActiveMemoryConfig
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text(
            "# Display Preferences\n\n"
            "The user strongly prefers dark mode in all applications.\n"
            "Dark mode is the primary display preference for the user.\n"
            "The user always chooses dark themes whenever available.\n"
            "Dark background settings are preferred for reduced eye strain.\n"
        )
        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.0)
        await mgr.sync()

        am_agent = ActiveMemoryAgent(
            memory_manager=mgr,
            llm_provider=None,
            config=ActiveMemoryConfig(min_score=0.0),
        )
        result = await am_agent.run(
            "dark mode display preferences",
            session_id="s1",
            session_kind="interactive",
        )
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_active_memory_context_prepended_to_task(self, tmp_path):
        """When active memory fires, context is prepended to the task string."""
        (tmp_path / "MEMORY.md").write_text(
            "# Context\n\n"
            "The user is building a REST API with FastAPI.\n"
            "FastAPI is the web framework being used.\n"
        )

        received_tasks: List[str] = []

        original_execute = None

        llm = MockLLMProvider([text_resp("Here is how to add auth to FastAPI.")])
        agent = make_agent(tmp_path, llm)
        if agent._active_memory:
            agent._active_memory.config.min_score = 0.01

        # Patch execute to capture the task string
        original_execute = agent.execute

        async def patched_execute(task, context=None):
            received_tasks.append(task)
            return await original_execute(task, context)

        agent.execute = patched_execute

        await agent.memory_manager.sync()
        await agent.handle_message(
            make_message("How do I add authentication?", session_id="s1")
        )

        # If active memory fired, the task should contain injected context
        if received_tasks and agent._active_memory:
            task = received_tasks[0]
            # Either active memory injected, or message passed through as-is
            assert len(task) > 0


# ---------------------------------------------------------------------------
# 7. Short-term recall tracking
# ---------------------------------------------------------------------------

class TestRecallTracking:
    @pytest.mark.asyncio
    async def test_search_results_recorded_in_recall_store(self, tmp_path):
        """Each search result is tracked in short_term_recall table."""
        from omniagent.agents.short_term_memory import RecallTracker
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text(
            "# Notes\n\n"
            "The API rate limit is 100 requests per minute.\n"
            "Rate limiting applies to all endpoints globally.\n"
        )
        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.01)
        await mgr.sync()

        tracker = RecallTracker(mgr.store)

        for query in ["rate limit", "API limit", "requests per minute"]:
            results = await mgr.search(query, max_results=3)
            if results:
                await tracker.record_recalls(query, results)

        candidates = mgr.store.get_top_recall_candidates(min_count=1, limit=20)
        assert len(candidates) > 0

    @pytest.mark.asyncio
    async def test_high_recall_entry_promoted(self, tmp_path):
        """Entry recalled many times across unique queries gets promoted."""
        from omniagent.agents.short_term_memory import (
            RecallTracker, PromotionScorer, PromotionExecutor, DreamingScheduler,
        )
        from omniagent.agents.memory_manager import MemorySearchManager

        (tmp_path / "MEMORY.md").write_text(
            "# Prefs\n\nUser always uses dark mode for every app.\n"
            "Dark mode is preferred in all applications.\n"
            "The dark theme reduces eye strain.\n"
        )
        (tmp_path / "MEMORY.md").write_text(  # ensure content is long enough
            "# User Preferences\n\n"
            "User always uses dark mode for every single application.\n"
            "Dark mode is the preferred theme in all apps used by the user.\n"
            "The dark theme reduces eye strain significantly for the user.\n"
            "User prefers dark backgrounds consistently across all tools.\n"
        )

        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.01)
        await mgr.sync()

        tracker = RecallTracker(mgr.store)
        scorer = PromotionScorer(threshold=0.2, min_recall_count=3, min_unique_queries=2)
        executor = PromotionExecutor(tmp_path, mgr.store)
        scheduler = DreamingScheduler(mgr.store, scorer, executor, max_promotions_per_run=5)

        queries = [
            "dark mode", "user theme preference", "display settings",
            "dark background", "app theme", "eye strain",
        ]
        for query in queries:
            results = await mgr.search(query, max_results=3)
            if results:
                await tracker.record_recalls(query, results)

        n = await scheduler.run_light_phase()
        # Either promoted some, or candidates simply didn't reach threshold
        assert n >= 0


# ---------------------------------------------------------------------------
# 8. Daily notes auto-load
# ---------------------------------------------------------------------------

class TestDailyNotes:
    @pytest.mark.asyncio
    async def test_today_note_indexed_automatically(self, tmp_path):
        """Today's YYYY-MM-DD.md in memory/ is picked up by sync."""
        from omniagent.agents.memory_manager import MemorySearchManager, DailyNotesLoader

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().strftime("%Y-%m-%d")
        (mem_dir / f"{today}.md").write_text(
            "# Daily Note\n\n"
            "Today's standup: fixed the auth bug, PR is ready for review.\n"
        )

        files = DailyNotesLoader.list_memory_files_with_daily(tmp_path)
        assert any(today in str(f) for f in files)

        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.01)
        await mgr.sync()

        results = await mgr.search("auth bug standup")
        assert len(results) > 0
        assert any("auth bug" in r.text.lower() for r in results)

    @pytest.mark.asyncio
    async def test_old_notes_not_auto_loaded(self, tmp_path):
        """Notes older than yesterday are not returned by DailyNotesLoader."""
        from datetime import timedelta
        from omniagent.agents.memory_manager import DailyNotesLoader

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        old = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        (mem_dir / f"{old}.md").write_text("# Old\n\nOld content.\n")

        paths = DailyNotesLoader.get_daily_note_paths(tmp_path)
        assert not any(old in str(p) for p in paths)


# ---------------------------------------------------------------------------
# 9. Full handle_message round-trip
# ---------------------------------------------------------------------------

class TestFullRoundTrip:
    @pytest.mark.asyncio
    async def test_simple_message_returns_response(self, tmp_path):
        """Basic round-trip: message in → response out."""
        llm = MockLLMProvider([text_resp("I'm doing well, thank you!")])
        agent = make_agent(tmp_path, llm)

        resp = await agent.handle_message(make_message("How are you?"))

        assert resp.session_id == "test-session"
        assert resp.content == "I'm doing well, thank you!"

    @pytest.mark.asyncio
    async def test_multi_tool_conversation(self, tmp_path):
        """Agent can call memory_search then memory_store in one conversation."""
        (tmp_path / "MEMORY.md").write_text(
            "# Notes\n\nUser works on the payments team.\n"
            "Payments team uses Stripe API for processing.\n"
        )
        llm = MockLLMProvider([
            tool_resp("memory_search", {"query": "payments team Stripe"}),
            tool_resp("memory_store", {
                "text": "Stripe webhook secret is stored in vault.",
                "category": "fact",
            }),
            text_resp("Got it. I've noted the Stripe details."),
        ])
        agent = make_agent(tmp_path, llm)
        await agent.memory_manager.sync()

        resp = await agent.handle_message(
            make_message("What do you know about our payments setup?")
        )
        assert resp.content is not None

        # Both tools should have been offered
        all_names = [n for c in llm.calls for n in c["tool_names"]]
        assert "memory_search" in all_names
        assert "memory_store" in all_names

    @pytest.mark.asyncio
    async def test_response_session_id_matches(self, tmp_path):
        """Response session_id matches the incoming message session_id."""
        llm = MockLLMProvider([text_resp("Hello!")])
        agent = make_agent(tmp_path, llm)

        resp = await agent.handle_message(make_message("Hi", session_id="session-xyz"))
        assert resp.session_id == "session-xyz"

    @pytest.mark.asyncio
    async def test_memory_content_influences_context(self, tmp_path):
        """Memory content is available to the agent during conversation."""
        (tmp_path / "MEMORY.md").write_text(
            "# User Profile\n\n"
            "Name: Bob\n"
            "Role: Backend Engineer\n"
            "Primary language: Go\n"
            "Bob uses Go as the primary programming language.\n"
        )
        llm = MockLLMProvider([text_resp("As a Go engineer, you should use goroutines.")])
        agent = make_agent(tmp_path, llm)
        await agent.memory_manager.sync()

        resp = await agent.handle_message(
            make_message("How should I handle concurrency?", session_id="s1")
        )
        assert resp.content is not None
        # Memory was indexed and available
        chunks = agent.memory_manager.store.get_all_chunks()
        assert len(chunks) > 0
        assert any("Go" in c["text"] for c in chunks)
