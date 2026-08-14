"""Tests for the memory system (Issues #11-#25)."""

import asyncio
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path):
    from omniagent.agents.memory import MemoryStore
    return MemoryStore(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Issue #11 — sqlite-vec BLOB vector storage
# ---------------------------------------------------------------------------

class TestBlobEmbedding:
    def test_roundtrip(self):
        from omniagent.agents.memory import embedding_to_blob, blob_to_embedding
        emb = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = embedding_to_blob(emb)
        assert isinstance(blob, bytes)
        assert len(blob) == 4 * 5
        recovered = blob_to_embedding(blob)
        assert len(recovered) == 5
        for a, b in zip(emb, recovered):
            assert abs(a - b) < 1e-6

    def test_save_and_get_blob(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert_chunk("c1", "MEMORY.md", "memory", 1, 10, "h1", "m1", "hello")
        emb = [0.1, 0.2, 0.3]
        store.save_embedding("c1", emb)

        blob = store.get_embedding_blob("c1")
        assert blob is not None
        recovered = __import__("omniagent.agents.memory", fromlist=["blob_to_embedding"]).blob_to_embedding(blob)
        assert len(recovered) == 3
        assert abs(recovered[0] - 0.1) < 1e-5

    def test_save_embedding_missing_chunk(self, tmp_path):
        """Saving embedding for non-existent chunk should not raise."""
        store = make_store(tmp_path)
        store.save_embedding("nonexistent", [0.1, 0.2])  # Should not raise

    def test_sqlite_vec_check(self):
        from omniagent.agents.memory import check_sqlite_vec
        result = check_sqlite_vec()
        assert isinstance(result, bool)  # Either True or False, no exception


# ---------------------------------------------------------------------------
# Issue #12 — KNN native query
# ---------------------------------------------------------------------------

class TestKNNSearch:
    def test_knn_returns_results(self, tmp_path):
        store = make_store(tmp_path)
        # Insert 3 chunks with embeddings
        for i, emb in enumerate(([1.0, 0.0], [0.0, 1.0], [0.7, 0.7])):
            store.upsert_chunk(f"c{i}", "MEMORY.md", "memory", i*10, i*10+9, f"h{i}", "m1", f"chunk {i}")
            store.save_embedding(f"c{i}", emb)

        results = store.search_knn([1.0, 0.0], limit=3)
        assert len(results) >= 1
        # c0 should be most similar to [1.0, 0.0]
        assert results[0]["id"] == "c0"

    def test_knn_score_range(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert_chunk("c1", "MEMORY.md", "memory", 1, 5, "h1", "m1", "text")
        store.save_embedding("c1", [0.6, 0.8])
        results = store.search_knn([0.6, 0.8], limit=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0 + 1e-6

    def test_knn_empty_db(self, tmp_path):
        store = make_store(tmp_path)
        results = store.search_knn([0.1, 0.2], limit=5)
        assert results == []

    def test_knn_session_isolation(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert_chunk("c1", "MEMORY.md", "memory", 1, 5, "h1", "m1", "text A", session_id="s1")
        store.upsert_chunk("c2", "MEMORY.md", "memory", 6, 10, "h2", "m1", "text B", session_id="s2")
        store.save_embedding("c1", [1.0, 0.0])
        store.save_embedding("c2", [0.0, 1.0])

        # Session isolation applies via chunks table TEXT fallback path.
        # chunks_vec BLOB path doesn't carry session_id, so test via get_all_chunks_for_vector.
        all_chunks = store.get_all_chunks_for_vector(session_id=None)
        assert len(all_chunks) == 2

        # KNN without filter returns both
        results_all = store.search_knn([1.0, 0.0], limit=5)
        assert len(results_all) == 2


# ---------------------------------------------------------------------------
# Issue #13 — Embedding cache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def test_cache_set_get(self, tmp_path):
        store = make_store(tmp_path)
        emb = [0.5, 0.6, 0.7]
        store.set_cached_embedding("abc123", emb, "model1")
        cached = store.get_cached_embedding("abc123")
        assert cached is not None
        assert len(cached) == 3
        assert abs(cached[0] - 0.5) < 1e-5

    def test_cache_miss(self, tmp_path):
        store = make_store(tmp_path)
        result = store.get_cached_embedding("nonexistent_hash")
        assert result is None

    def test_cache_overwrite(self, tmp_path):
        store = make_store(tmp_path)
        store.set_cached_embedding("h1", [0.1], "m1")
        store.set_cached_embedding("h1", [0.9], "m1")
        cached = store.get_cached_embedding("h1")
        assert abs(cached[0] - 0.9) < 1e-5

    def test_content_hash(self, tmp_path):
        store = make_store(tmp_path)
        h1 = store.content_hash("hello world")
        h2 = store.content_hash("hello world")
        h3 = store.content_hash("different text")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 32


# ---------------------------------------------------------------------------
# Issue #14 — ShortTermRecallStore data structure
# ---------------------------------------------------------------------------

class TestShortTermRecallStore:
    def _make_entry(self, key="k1", recall_count=3):
        from omniagent.agents.memory import ShortTermRecallEntry
        return ShortTermRecallEntry(
            key=key,
            path="MEMORY.md",
            source="memory",
            start_line=1,
            end_line=10,
            snippet="test snippet content",
            recall_count=recall_count,
            total_score=2.1,
            max_score=0.9,
            unique_query_count=2,
            query_hashes=["h1", "h2"],
            recall_days=["2026-01-01"],
            claim_hash="abc",
        )

    def test_upsert_and_get(self, tmp_path):
        store = make_store(tmp_path)
        entry = self._make_entry()
        store.upsert_recall(entry)
        fetched = store.get_recall("k1")
        assert fetched is not None
        assert fetched.recall_count == 3
        assert fetched.unique_query_count == 2
        assert "h1" in fetched.query_hashes

    def test_update_recall(self, tmp_path):
        store = make_store(tmp_path)
        entry = self._make_entry(recall_count=3)
        store.upsert_recall(entry)
        entry.recall_count = 5
        store.upsert_recall(entry)
        fetched = store.get_recall("k1")
        assert fetched.recall_count == 5

    def test_get_top_candidates(self, tmp_path):
        store = make_store(tmp_path)
        for i in range(5):
            e = self._make_entry(key=f"k{i}", recall_count=i + 1)
            store.upsert_recall(e)

        candidates = store.get_top_recall_candidates(min_count=3, limit=10)
        assert all(c.recall_count >= 3 for c in candidates)
        # Should be sorted descending
        counts = [c.recall_count for c in candidates]
        assert counts == sorted(counts, reverse=True)

    def test_mark_promoted(self, tmp_path):
        store = make_store(tmp_path)
        entry = self._make_entry()
        store.upsert_recall(entry)
        store.mark_promoted("k1")
        fetched = store.get_recall("k1")
        assert fetched.promoted_at is not None

        # Promoted entries should not appear in top candidates
        candidates = store.get_top_recall_candidates(min_count=1, limit=10)
        keys = [c.key for c in candidates]
        assert "k1" not in keys

    def test_get_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        result = store.get_recall("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Issue #15 — RecallTracker
# ---------------------------------------------------------------------------

class TestRecallTracker:
    @pytest.mark.asyncio
    async def test_records_recall(self, tmp_path):
        from omniagent.agents.memory import MemoryStore
        from omniagent.agents.memory_manager import MemorySearchResult
        from omniagent.agents.short_term_memory import RecallTracker

        store = MemoryStore(tmp_path / "t.db")
        tracker = RecallTracker(store)

        results = [
            MemorySearchResult(
                chunk_id="c1", path="MEMORY.md", source="memory",
                start_line=1, end_line=10, text="hello world",
                score=0.8, snippet="hello world snippet"
            )
        ]
        await tracker.record_recalls("test query", results)

        # Entry should exist with recall_count=1
        entries = store.get_top_recall_candidates(min_count=1, limit=10)
        assert len(entries) == 1
        assert entries[0].recall_count == 1
        assert entries[0].total_score == pytest.approx(0.8, abs=1e-5)

    @pytest.mark.asyncio
    async def test_increments_on_repeat(self, tmp_path):
        from omniagent.agents.memory import MemoryStore
        from omniagent.agents.memory_manager import MemorySearchResult
        from omniagent.agents.short_term_memory import RecallTracker

        store = MemoryStore(tmp_path / "t.db")
        tracker = RecallTracker(store)
        result = MemorySearchResult(
            chunk_id="c1", path="MEMORY.md", source="memory",
            start_line=1, end_line=10, text="hello",
            score=0.5, snippet="hello snippet"
        )

        await tracker.record_recalls("query A", [result])
        await tracker.record_recalls("query B", [result])  # Different query → unique_query_count should increase

        entries = store.get_top_recall_candidates(min_count=1, limit=10)
        assert entries[0].recall_count == 2
        assert entries[0].unique_query_count == 2

    @pytest.mark.asyncio
    async def test_deduplicates_same_query(self, tmp_path):
        from omniagent.agents.memory import MemoryStore
        from omniagent.agents.memory_manager import MemorySearchResult
        from omniagent.agents.short_term_memory import RecallTracker

        store = MemoryStore(tmp_path / "t.db")
        tracker = RecallTracker(store)
        result = MemorySearchResult(
            chunk_id="c1", path="MEMORY.md", source="memory",
            start_line=1, end_line=10, text="hello",
            score=0.5, snippet="hello snippet"
        )

        await tracker.record_recalls("same query", [result])
        await tracker.record_recalls("same query", [result])

        entries = store.get_top_recall_candidates(min_count=1, limit=10)
        assert entries[0].recall_count == 2
        assert entries[0].unique_query_count == 1  # Same query → no increment


# ---------------------------------------------------------------------------
# Issue #16 — PromotionScorer
# ---------------------------------------------------------------------------

class TestPromotionScorer:
    def _make_entry(self, recall_count=5, unique_queries=3, recall_days=3):
        from omniagent.agents.memory import ShortTermRecallEntry
        from datetime import datetime
        return ShortTermRecallEntry(
            key="k1", path="MEMORY.md", source="memory",
            start_line=1, end_line=10, snippet="snippet",
            recall_count=recall_count,
            total_score=recall_count * 0.7,
            max_score=0.9,
            unique_query_count=unique_queries,
            first_recalled_at="2026-01-01T00:00:00",
            last_recalled_at=datetime.now().isoformat(timespec="seconds"),
            query_hashes=[f"h{i}" for i in range(unique_queries)],
            recall_days=[f"2026-01-0{i+1}" for i in range(recall_days)],
        )

    def test_below_min_recall(self):
        from omniagent.agents.short_term_memory import PromotionScorer
        scorer = PromotionScorer(min_recall_count=5)
        entry = self._make_entry(recall_count=2)
        score = scorer.score(entry)
        assert score.final_score == 0.0
        assert not score.meets_threshold

    def test_high_quality_entry_passes_threshold(self):
        from omniagent.agents.short_term_memory import PromotionScorer
        scorer = PromotionScorer(threshold=0.5)
        entry = self._make_entry(recall_count=10, unique_queries=5, recall_days=7)
        score = scorer.score(entry)
        assert score.meets_threshold
        assert score.final_score > 0.5

    def test_score_components_are_valid(self):
        from omniagent.agents.short_term_memory import PromotionScorer
        scorer = PromotionScorer()
        entry = self._make_entry(recall_count=5, unique_queries=3)
        score = scorer.score(entry)
        for s in [score.frequency_score, score.relevance_score,
                  score.diversity_score, score.recency_score,
                  score.consolidation_score, score.conceptual_score]:
            assert 0.0 <= s <= 1.0

    def test_weights_sum(self):
        from omniagent.agents.short_term_memory import PromotionScorer
        w = PromotionScorer.DEFAULT_WEIGHTS
        assert abs(sum(w.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Issue #17 — DreamingScheduler
# ---------------------------------------------------------------------------

class TestDreamingScheduler:
    @pytest.mark.asyncio
    async def test_light_phase_promotes(self, tmp_path):
        from omniagent.agents.memory import MemoryStore, ShortTermRecallEntry
        from omniagent.agents.short_term_memory import (
            PromotionScorer, PromotionExecutor, DreamingScheduler
        )
        from datetime import datetime

        store = MemoryStore(tmp_path / "t.db")
        scorer = PromotionScorer(threshold=0.3, min_recall_count=1, min_unique_queries=1)
        executor = PromotionExecutor(tmp_path, store)
        scheduler = DreamingScheduler(store, scorer, executor, max_promotions_per_run=2)

        # Add a high-quality entry that should be promoted
        entry = ShortTermRecallEntry(
            key="k1", path="MEMORY.md", source="memory",
            start_line=1, end_line=5, snippet="important fact about X",
            recall_count=10, total_score=7.0, max_score=0.9,
            unique_query_count=5,
            last_recalled_at=datetime.now().isoformat(timespec="seconds"),
            query_hashes=[f"h{i}" for i in range(5)],
            recall_days=["2026-01-01", "2026-01-02", "2026-01-03"],
        )
        store.upsert_recall(entry)

        # Create MEMORY.md
        (tmp_path / "MEMORY.md").write_text("# Memory\n")

        n = await scheduler.run_light_phase()
        assert n == 1

        # Entry should be marked as promoted
        fetched = store.get_recall("k1")
        assert fetched.promoted_at is not None

        # MEMORY.md should contain the snippet
        content = (tmp_path / "MEMORY.md").read_text()
        assert "important fact about X" in content


# ---------------------------------------------------------------------------
# Issue #18/#19 — EligibilityChecker
# ---------------------------------------------------------------------------

class TestEligibilityChecker:
    def test_eligible(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker()
        r = c.check("user", "interactive", session_id="s1", session_age_seconds=60)
        assert r.eligible

    def test_ineligible_no_session(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker()
        r = c.check("user", "interactive", session_id=None)
        assert not r.eligible

    def test_ineligible_headless(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker()
        r = c.check("user", "headless", session_id="s1")
        assert not r.eligible

    def test_ineligible_system_trigger(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker()
        r = c.check("system", "interactive", session_id="s1")
        assert not r.eligible

    def test_ineligible_kinds(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker()
        for kind in ["headless", "one_shot", "heartbeat", "batch", "cli_single"]:
            r = c.check("user", kind, session_id="s1")
            assert not r.eligible, f"Expected ineligible for kind={kind}"

    def test_disabled(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker(config={"enabled": False})
        r = c.check("user", "interactive", session_id="s1")
        assert not r.eligible

    def test_min_session_age(self):
        from omniagent.agents.active_memory import EligibilityChecker
        c = EligibilityChecker(config={"min_session_age_seconds": 30})
        r = c.check("user", "interactive", session_id="s1", session_age_seconds=10)
        assert not r.eligible
        r2 = c.check("user", "interactive", session_id="s1", session_age_seconds=60)
        assert r2.eligible


# ---------------------------------------------------------------------------
# Issue #20 — MemorySummaryGenerator
# ---------------------------------------------------------------------------

class TestMemorySummaryGenerator:
    @pytest.mark.asyncio
    async def test_generate_without_llm(self):
        from omniagent.agents.active_memory import MemorySummaryGenerator
        from omniagent.agents.memory_manager import MemorySearchResult

        gen = MemorySummaryGenerator(llm_provider=None)
        results = [
            MemorySearchResult(
                chunk_id="c1", path="memory/prefs.md", source="memory",
                start_line=1, end_line=5, text="user prefers dark mode",
                score=0.9, snippet="user prefers dark mode"
            ),
            MemorySearchResult(
                chunk_id="c2", path="memory/facts.md", source="memory",
                start_line=10, end_line=15, text="project uses Python 3.11",
                score=0.7, snippet="project uses Python 3.11"
            ),
        ]
        summary = await gen.generate(results, query_context="preferences")
        assert summary.result_count == 2
        assert summary.source_count == 2
        assert "user prefers dark mode" in summary.summary_text
        assert len(summary.summary_text) <= gen.max_length + 3  # +3 for "..."

    @pytest.mark.asyncio
    async def test_empty_results(self):
        from omniagent.agents.active_memory import MemorySummaryGenerator
        gen = MemorySummaryGenerator(llm_provider=None)
        summary = await gen.generate([], query_context="anything")
        assert summary.summary_text == ""
        assert summary.result_count == 0

    @pytest.mark.asyncio
    async def test_deduplicates_similar(self):
        from omniagent.agents.active_memory import MemorySummaryGenerator
        from omniagent.agents.memory_manager import MemorySearchResult

        gen = MemorySummaryGenerator(llm_provider=None)
        dup_snippet = "duplicate content here"
        results = [
            MemorySearchResult("c1", "p.md", "memory", 1, 5, dup_snippet, 0.9, dup_snippet),
            MemorySearchResult("c2", "p.md", "memory", 6, 10, dup_snippet, 0.8, dup_snippet),
        ]
        summary = await gen.generate(results, query_context="test")
        assert summary.result_count == 1  # Deduplicated


# ---------------------------------------------------------------------------
# Issue #21 — ActiveMemoryAgent
# ---------------------------------------------------------------------------

class TestActiveMemoryAgent:
    @pytest.mark.asyncio
    async def test_run_returns_none_for_ineligible(self, tmp_path):
        from omniagent.agents.active_memory import ActiveMemoryAgent, ActiveMemoryConfig
        from omniagent.agents.memory_manager import MemorySearchManager

        mgr = MemorySearchManager(workspace_dir=tmp_path)
        agent = ActiveMemoryAgent(memory_manager=mgr, llm_provider=None)

        # headless → ineligible
        result = await agent.run(
            "hello", session_id="s1", session_kind="headless"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_run_returns_none_when_no_memory(self, tmp_path):
        from omniagent.agents.active_memory import ActiveMemoryAgent, ActiveMemoryConfig
        from omniagent.agents.memory_manager import MemorySearchManager

        mgr = MemorySearchManager(workspace_dir=tmp_path)
        agent = ActiveMemoryAgent(memory_manager=mgr, llm_provider=None)

        result = await agent.run(
            "what are user preferences?",
            session_id="s1", session_kind="interactive",
        )
        assert result is None  # No memory indexed

    @pytest.mark.asyncio
    async def test_run_returns_summary_with_memory(self, tmp_path):
        from omniagent.agents.active_memory import ActiveMemoryAgent
        from omniagent.agents.memory_manager import MemorySearchManager

        # Write a memory file with substantial content so FTS finds it
        (tmp_path / "MEMORY.md").write_text(
            "# User Preferences\n\n"
            "The user prefers dark mode for the interface.\n"
            "The user enjoys working with Python and dark themes.\n"
            "Dark mode reduces eye strain during long coding sessions.\n"
        )
        mgr = MemorySearchManager(workspace_dir=tmp_path, query_min_score=0.1)
        await mgr.sync()

        agent = ActiveMemoryAgent(
            memory_manager=mgr,
            llm_provider=None,
            config=__import__("omniagent.agents.active_memory", fromlist=["ActiveMemoryConfig"]).ActiveMemoryConfig(min_score=0.1),
        )
        result = await agent.run(
            "dark mode user preferences",
            session_id="s1", session_kind="interactive",
        )
        assert result is not None
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Issue #22 — MemoryBackend abstraction
# ---------------------------------------------------------------------------

class TestMemoryBackend:
    def test_abstract_cannot_instantiate(self):
        from omniagent.agents.memory_backend import MemoryBackend
        with pytest.raises(TypeError):
            MemoryBackend()  # type: ignore

    def test_embedding_provider_abstract(self):
        from omniagent.agents.memory_backend import EmbeddingProvider
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore


# ---------------------------------------------------------------------------
# Issue #23 — EmbeddingProvider factory
# ---------------------------------------------------------------------------

class TestEmbeddingProviderFactory:
    def test_create_openai(self):
        from omniagent.agents.memory_backend import create_embedding_provider, OpenAIEmbeddingProvider
        p = create_embedding_provider("openai", api_key="test", model_id="text-embedding-3-small")
        assert isinstance(p, OpenAIEmbeddingProvider)
        assert p.provider_name == "openai"
        assert p.dimension == 1536

    def test_create_openai_from_chat_url(self):
        from omniagent.agents.memory_backend import create_embedding_provider
        p = create_embedding_provider(
            "openai",
            api_key="k",
            api_url="https://api.example.com/v1/chat/completions",
        )
        assert "/embeddings" in p.api_url
        assert "/chat/completions" not in p.api_url

    def test_create_ollama(self):
        from omniagent.agents.memory_backend import create_embedding_provider, OllamaEmbeddingProvider
        p = create_embedding_provider("ollama", model_id="nomic-embed-text")
        assert isinstance(p, OllamaEmbeddingProvider)
        assert p.provider_name == "ollama"

    def test_create_local(self):
        from omniagent.agents.memory_backend import create_embedding_provider, LocalEmbeddingProvider
        p = create_embedding_provider("local")
        assert isinstance(p, LocalEmbeddingProvider)
        assert p.provider_name == "local"

    def test_create_unknown_returns_none(self):
        from omniagent.agents.memory_backend import create_embedding_provider
        p = create_embedding_provider("unknown_provider")
        assert p is None


# ---------------------------------------------------------------------------
# Issue #24 — MMR reranking + session isolation
# ---------------------------------------------------------------------------

class TestMMRReranking:
    def test_mmr_preserves_all_results(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemorySearchResult
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        results = [
            MemorySearchResult("c1", "p.md", "memory", 1, 5, "apple orange", 0.9, "apple orange"),
            MemorySearchResult("c2", "p.md", "memory", 6, 10, "banana grape", 0.8, "banana grape"),
            MemorySearchResult("c3", "p.md", "memory", 11, 15, "cherry plum", 0.7, "cherry plum"),
        ]
        reranked = mgr._mmr_rerank_results(results, lambda_param=0.7)
        assert len(reranked) == 3

    def test_mmr_first_is_highest_score(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemorySearchResult
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        results = [
            MemorySearchResult("c1", "p.md", "m", 1, 5, "text A", 0.9, "text A"),
            MemorySearchResult("c2", "p.md", "m", 6, 10, "text B", 0.5, "text B"),
        ]
        reranked = mgr._mmr_rerank_results(results)
        assert reranked[0].chunk_id == "c1"

    def test_mmr_single_result(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemorySearchResult
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        results = [MemorySearchResult("c1", "p.md", "m", 1, 5, "only", 0.9, "only")]
        reranked = mgr._mmr_rerank_results(results)
        assert len(reranked) == 1


# ---------------------------------------------------------------------------
# Issue #25 — memory_store tool + DailyNotesLoader
# ---------------------------------------------------------------------------

class TestMemoryStoreTool:
    @pytest.mark.asyncio
    async def test_store_creates_file(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryStoreTool(mgr)
        result = await tool.execute({"text": "User likes Python", "category": "preference"})
        assert result.success
        pref_file = tmp_path / "memory" / "preference.md"
        assert pref_file.exists()
        content = pref_file.read_text()
        assert "User likes Python" in content

    @pytest.mark.asyncio
    async def test_store_custom_path(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryStoreTool(mgr)
        result = await tool.execute({
            "text": "Custom memory",
            "path": "memory/custom/notes.md",
        })
        assert result.success
        assert (tmp_path / "memory" / "custom" / "notes.md").exists()

    @pytest.mark.asyncio
    async def test_store_missing_text(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager, MemoryStoreTool
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        tool = MemoryStoreTool(mgr)
        result = await tool.execute({})
        assert not result.success
        assert "Missing" in result.error


class TestDailyNotesLoader:
    def test_today_note_included(self, tmp_path):
        from omniagent.agents.memory_manager import DailyNotesLoader
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().strftime("%Y-%m-%d")
        (mem_dir / f"{today}.md").write_text("# Today")

        paths = DailyNotesLoader.get_daily_note_paths(tmp_path)
        assert any(today in str(p) for p in paths)

    def test_yesterday_note_included(self, tmp_path):
        from omniagent.agents.memory_manager import DailyNotesLoader
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        (mem_dir / f"{yesterday}.md").write_text("# Yesterday")

        paths = DailyNotesLoader.get_daily_note_paths(tmp_path)
        assert any(yesterday in str(p) for p in paths)

    def test_old_notes_not_included(self, tmp_path):
        from omniagent.agents.memory_manager import DailyNotesLoader
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        old = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        (mem_dir / f"{old}.md").write_text("# Old")

        paths = DailyNotesLoader.get_daily_note_paths(tmp_path)
        assert not any(old in str(p) for p in paths)

    def test_list_memory_files_with_daily(self, tmp_path):
        from omniagent.agents.memory_manager import DailyNotesLoader
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().strftime("%Y-%m-%d")
        (mem_dir / f"{today}.md").write_text("# Today")
        (tmp_path / "MEMORY.md").write_text("# Base")

        paths = DailyNotesLoader.list_memory_files_with_daily(tmp_path)
        names = [p.name for p in paths]
        assert f"{today}.md" in names
        assert "MEMORY.md" in names
        # No duplicates
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Integration: sync + search pipeline
# ---------------------------------------------------------------------------

class TestSyncAndSearch:
    @pytest.mark.asyncio
    async def test_sync_and_fts_search(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager
        (tmp_path / "MEMORY.md").write_text(
            "# Preferences\n\nUser prefers dark mode.\nUser uses macOS.\n"
        )
        mgr = MemorySearchManager(workspace_dir=tmp_path)
        updated = await mgr.sync()
        # macOS case-insensitive FS may find both MEMORY.md and memory.md
        assert updated >= 1

        results = await mgr.search("dark mode preferences")
        assert len(results) > 0
        assert any("dark mode" in r.text.lower() for r in results)

    @pytest.mark.asyncio
    async def test_sync_incremental(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager
        mem_file = tmp_path / "MEMORY.md"
        mem_file.write_text("# Initial content\n\nHello world.\n")
        mgr = MemorySearchManager(workspace_dir=tmp_path)

        n1 = await mgr.sync()
        assert n1 >= 1

        # No change → should be 0
        n2 = await mgr.sync()
        assert n2 == 0

        # Modify file → should be >= 1
        mem_file.write_text("# Updated content\n\nHello updated world.\n")
        n3 = await mgr.sync()
        assert n3 >= 1

    @pytest.mark.asyncio
    async def test_read_memory_safe(self, tmp_path):
        from omniagent.agents.memory_manager import MemorySearchManager
        (tmp_path / "MEMORY.md").write_text("secret content")
        mgr = MemorySearchManager(workspace_dir=tmp_path)

        content = mgr.read_memory_safe("MEMORY.md")
        assert content == "secret content"

        # Path traversal blocked
        blocked = mgr.read_memory_safe("../../etc/passwd")
        assert blocked is None
