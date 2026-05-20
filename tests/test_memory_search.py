"""Regression tests for memory file discovery, hybrid search, and provider fallbacks."""

import asyncio
import os

import pytest

from omniagent.agents.llm import DeepSeekLLMProvider, LLMMessage
from omniagent.agents.memory import MemoryStore, list_memory_files
from omniagent.agents.memory_manager import (
    MemorySearchManager,
    MemorySearchResult,
    MemorySearchTool,
)


def test_list_memory_files_deduplicates_same_file_identity(tmp_path):
    memory_file = tmp_path / "MEMORY.md"
    lower_case_alias = tmp_path / "memory.md"
    memory_file.write_text("same physical memory file", encoding="utf-8")
    try:
        os.link(memory_file, lower_case_alias)
    except FileExistsError:
        # Case-insensitive filesystems already treat both spellings as the same file.
        pass
    except OSError as exc:
        pytest.skip(f"hard links unavailable for this filesystem: {exc}")

    files = list_memory_files(tmp_path)

    assert len(files) == 1


def test_hybrid_search_embeds_query_and_merges_vector_results(tmp_path):
    async def run():
        manager = MemorySearchManager(
            workspace_dir=tmp_path,
            store_path=tmp_path / "memory.db",
            llm_provider=object(),
            hybrid_enabled=True,
            query_min_score=0.0,
            mmr_enabled=False,
        )
        manager.store.upsert_chunk(
            chunk_id="vector-best",
            path="memory/a.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-a",
            model="test",
            text="unmatched vector best",
        )
        manager.store.upsert_chunk(
            chunk_id="vector-other",
            path="memory/b.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-b",
            model="test",
            text="unmatched vector other",
        )
        manager.store.save_embedding("vector-best", "[1.0, 0.0]")
        manager.store.save_embedding("vector-other", "[0.0, 1.0]")

        async def embed_query(text):
            assert text == "semantic query"
            return [1.0, 0.0]

        manager._embed_text = embed_query

        results = await manager.search_async("semantic query", max_results=2, min_score=0.0)

        assert [r.chunk_id for r in results] == ["vector-best", "vector-other"]

    asyncio.run(run())


def test_hybrid_search_falls_back_to_fts_when_query_embedding_unavailable(tmp_path):
    async def run():
        manager = MemorySearchManager(
            workspace_dir=tmp_path,
            store_path=tmp_path / "memory.db",
            llm_provider=object(),
            hybrid_enabled=True,
        )
        manager.store.upsert_chunk(
            chunk_id="fallback",
            path="memory/fallback.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-fallback",
            model="test",
            text="semantic fallback query",
        )

        async def embed_query(_text):
            return None

        manager._embed_text = embed_query

        results = await manager.search_async("semantic fallback", max_results=2, min_score=0.0)

        assert [r.chunk_id for r in results] == ["fallback"]

    asyncio.run(run())


def test_hybrid_search_applies_mmr_after_merge(tmp_path):
    async def run():
        manager = MemorySearchManager(
            workspace_dir=tmp_path,
            store_path=tmp_path / "memory.db",
            llm_provider=object(),
            hybrid_enabled=True,
            vector_weight=1.0,
            text_weight=0.0,
            query_min_score=0.0,
            mmr_enabled=True,
            mmr_lambda=0.5,
        )
        manager.store.upsert_chunk(
            chunk_id="primary",
            path="memory/primary.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-primary",
            model="test",
            text="alpha beta semantic",
        )
        manager.store.upsert_chunk(
            chunk_id="duplicate",
            path="memory/duplicate.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-duplicate",
            model="test",
            text="alpha beta semantic duplicate",
        )
        manager.store.upsert_chunk(
            chunk_id="diverse",
            path="memory/diverse.md",
            source="memory",
            start_line=1,
            end_line=1,
            chunk_hash="hash-diverse",
            model="test",
            text="gamma delta semantic",
        )
        manager.store.save_embedding("primary", "[1.0, 0.0]")
        manager.store.save_embedding("duplicate", "[0.95, 0.0]")
        manager.store.save_embedding("diverse", "[0.8, 0.6]")

        async def embed_query(_text):
            return [1.0, 0.0]

        manager._embed_text = embed_query

        results = await manager.search_async("semantic", max_results=3, min_score=0.0)

        assert results[0].chunk_id == "primary"
        assert results[1].chunk_id == "diverse"

    asyncio.run(run())


def test_mmr_rerank_promotes_diverse_results(tmp_path):
    manager = MemorySearchManager(
        workspace_dir=tmp_path,
        store_path=tmp_path / "memory.db",
        mmr_enabled=True,
    )
    results = [
        MemorySearchResult("a", "a.md", "memory", 1, 1, "alpha beta", 1.0, "alpha beta"),
        MemorySearchResult("b", "b.md", "memory", 1, 1, "alpha beta copy", 0.95, "alpha beta"),
        MemorySearchResult("c", "c.md", "memory", 1, 1, "gamma delta", 0.8, "gamma delta"),
    ]

    reranked = manager._mmr_rerank(results, lambda_param=0.5)

    assert [r.chunk_id for r in reranked] == ["a", "c", "b"]


def test_fts5_tokenizer_is_configurable_and_migrates_existing_index(tmp_path):
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path, fts5_tokenizer="unicode61")
    store.upsert_chunk(
        chunk_id="task",
        path="MEMORY.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-task",
        model="test",
        text="running task",
    )

    assert store.search_fts("run", limit=5) == []

    migrated = MemoryStore(db_path, fts5_tokenizer="porter")
    rows = migrated.search_fts("run", limit=5)

    assert [row["id"] for row in rows] == ["task"]


def test_fts5_porter_tokenizer_stems_terms(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", fts5_tokenizer="porter")
    store.upsert_chunk(
        chunk_id="stemmed",
        path="MEMORY.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-stemmed",
        model="test",
        text="running tasks",
    )

    rows = store.search_fts("run", limit=5)

    assert [row["id"] for row in rows] == ["stemmed"]


def test_fts5_search_handles_hyphenated_query(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_chunk(
        chunk_id="hyphen",
        path="MEMORY.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-hyphen",
        model="test",
        text="Project codename: blue-lantern.",
    )

    rows = store.search_fts("blue-lantern", limit=5)

    assert [row["id"] for row in rows] == ["hyphen"]


def test_deepseek_tool_call_messages_preserve_reasoning_content():
    provider = DeepSeekLLMProvider(
        api_url="https://api.deepseek.com/v1/chat/completions",
        api_key="test",
        model="deepseek-v4-pro",
    )
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "memory_search", "arguments": "{\"query\":\"blue-lantern\"}"},
        }
    ]

    messages = provider._convert_messages([
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=tool_calls,
            reasoning_content="private reasoning trace",
        )
    ])

    assert messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "reasoning_content": "private reasoning trace",
        }
    ]


def test_session_isolation_filters_private_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_chunk(
        chunk_id="shared",
        path="memory/shared.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-shared",
        model="test",
        text="visibility token shared",
    )
    store.upsert_chunk(
        chunk_id="private-a",
        path="memory/private-a.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-private-a",
        model="test",
        text="visibility token private",
        session_id="session-a",
        is_shared=False,
    )

    rows = store.search_fts(
        "visibility",
        limit=5,
        session_id="session-b",
        session_isolation_enabled=True,
    )

    assert [row["id"] for row in rows] == ["shared"]


def test_session_isolation_keeps_legacy_memory_visible(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_chunk(
        chunk_id="legacy",
        path="memory/legacy.md",
        source="memory",
        start_line=1,
        end_line=1,
        chunk_hash="hash-legacy",
        model="test",
        text="legacy visibility token",
        session_id=None,
        is_shared=False,
    )

    rows = store.search_fts(
        "legacy",
        limit=5,
        session_id="session-a",
        session_isolation_enabled=True,
    )

    assert [row["id"] for row in rows] == ["legacy"]


def test_memory_search_tool_passes_hidden_session_id(tmp_path):
    class CapturingManager:
        captured_session_id = None

        async def search_async(self, query, max_results=None, min_score=None, session_id=None):
            self.captured_session_id = session_id
            return []

    async def run():
        manager = CapturingManager()
        tool = MemorySearchTool(manager)

        result = await tool.execute({
            "query": "visibility",
            "max_results": 3,
            "_session_id": "session-a",
        })

        assert result.success is True
        assert manager.captured_session_id == "session-a"

    asyncio.run(run())
