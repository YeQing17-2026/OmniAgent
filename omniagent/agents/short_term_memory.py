"""Short-term memory promotion system (Issues #14, #15, #16, #17).

Implements:
- RecallTracker: records memory_search hits into short_term_recall table
- PromotionScorer: weighted scoring for promotion eligibility
- PromotionExecutor: writes promoted entries to MEMORY.md
- DreamingScheduler: background periodic promotion sweeps
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RecallTracker (#15)
# ---------------------------------------------------------------------------

def _compute_query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:12]


def _compute_claim_hash(snippet: str) -> str:
    return hashlib.sha256(snippet.encode()).hexdigest()[:16]


def _generate_recall_key(source: str, path: str, start_line: int, end_line: int, claim_hash: str) -> str:
    raw = f"{source}:{path}:{start_line}:{end_line}:{claim_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class RecallTracker:
    """Records memory_search hits into short_term_recall for later promotion."""

    def __init__(self, store):
        self.store = store

    async def record_recalls(
        self,
        query: str,
        results: List,
        session_id: Optional[str] = None,
    ) -> None:
        """Record recall events for search results (#15)."""
        from omniagent.agents.memory import ShortTermRecallEntry

        query_hash = _compute_query_hash(query)
        now = datetime.now().isoformat(timespec="seconds")
        today = datetime.now().strftime("%Y-%m-%d")

        for result in results:
            snippet = getattr(result, "snippet", "") or getattr(result, "text", "")[:200]
            claim_hash = _compute_claim_hash(snippet)
            key = _generate_recall_key(
                getattr(result, "source", "memory"),
                getattr(result, "path", ""),
                getattr(result, "start_line", 0),
                getattr(result, "end_line", 0),
                claim_hash,
            )

            existing = self.store.get_recall(key)
            if existing:
                entry = existing
                entry.recall_count += 1
                entry.total_score += getattr(result, "score", 0.0)
                entry.max_score = max(entry.max_score, getattr(result, "score", 0.0))
                entry.last_recalled_at = now
                # Unique query tracking
                if query_hash not in entry.query_hashes:
                    entry.query_hashes.append(query_hash)
                    entry.unique_query_count = len(entry.query_hashes)
                # Unique day tracking
                if today not in entry.recall_days:
                    entry.recall_days.append(today)
            else:
                entry = ShortTermRecallEntry(
                    key=key,
                    path=getattr(result, "path", ""),
                    source=getattr(result, "source", "memory"),
                    start_line=getattr(result, "start_line", 0),
                    end_line=getattr(result, "end_line", 0),
                    snippet=snippet,
                    recall_count=1,
                    total_score=getattr(result, "score", 0.0),
                    max_score=getattr(result, "score", 0.0),
                    unique_query_count=1,
                    first_recalled_at=now,
                    last_recalled_at=now,
                    query_hashes=[query_hash],
                    recall_days=[today],
                    claim_hash=claim_hash,
                )

            self.store.upsert_recall(entry)


# ---------------------------------------------------------------------------
# PromotionScorer (#16)
# ---------------------------------------------------------------------------

@dataclass
class PromotionScore:
    """Promotion scoring breakdown."""

    final_score: float
    frequency_score: float
    relevance_score: float
    diversity_score: float
    recency_score: float
    consolidation_score: float
    conceptual_score: float
    meets_threshold: bool


class PromotionScorer:
    """Weighted scoring for short-term → long-term memory promotion."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "frequency": 0.24,
        "relevance": 0.30,
        "diversity": 0.15,
        "recency": 0.15,
        "consolidation": 0.10,
        "conceptual": 0.06,
    }
    DEFAULT_THRESHOLD = 0.75
    DEFAULT_MIN_RECALL_COUNT = 3
    DEFAULT_MIN_UNIQUE_QUERIES = 2

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
        min_unique_queries: int = DEFAULT_MIN_UNIQUE_QUERIES,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self.min_recall_count = min_recall_count
        self.min_unique_queries = min_unique_queries

    def score(self, entry) -> PromotionScore:
        """Compute promotion score for a ShortTermRecallEntry."""
        import math

        # Pre-filter
        if entry.recall_count < self.min_recall_count:
            return self._zero_score()
        if entry.unique_query_count < self.min_unique_queries:
            return self._zero_score()

        # Frequency: log-normalized recall count
        freq = min(1.0, math.log(entry.recall_count + 1) / math.log(20))

        # Relevance: average relevance score
        avg_score = (entry.total_score / entry.recall_count) if entry.recall_count > 0 else 0.0
        relevance = min(1.0, avg_score)

        # Diversity: unique queries / total recalls
        diversity = min(1.0, entry.unique_query_count / max(entry.recall_count, 1))

        # Recency: exponential decay based on days since last recall
        recency = 0.0
        if entry.last_recalled_at:
            try:
                last = datetime.fromisoformat(entry.last_recalled_at)
                days_ago = (datetime.now() - last).days
                recency = math.exp(-days_ago / 7.0)  # half-life 7 days
            except Exception:
                recency = 0.5

        # Consolidation: number of distinct days
        consolidation = min(1.0, len(entry.recall_days) / 7.0)

        # Conceptual: bonus for having concept tags
        conceptual = min(1.0, len(entry.concept_tags) / 5.0)

        w = self.weights
        final = (
            w.get("frequency", 0.24) * freq
            + w.get("relevance", 0.30) * relevance
            + w.get("diversity", 0.15) * diversity
            + w.get("recency", 0.15) * recency
            + w.get("consolidation", 0.10) * consolidation
            + w.get("conceptual", 0.06) * conceptual
        )

        return PromotionScore(
            final_score=round(final, 4),
            frequency_score=round(freq, 4),
            relevance_score=round(relevance, 4),
            diversity_score=round(diversity, 4),
            recency_score=round(recency, 4),
            consolidation_score=round(consolidation, 4),
            conceptual_score=round(conceptual, 4),
            meets_threshold=final >= self.threshold,
        )

    def _zero_score(self) -> PromotionScore:
        return PromotionScore(
            final_score=0.0,
            frequency_score=0.0,
            relevance_score=0.0,
            diversity_score=0.0,
            recency_score=0.0,
            consolidation_score=0.0,
            conceptual_score=0.0,
            meets_threshold=False,
        )


# ---------------------------------------------------------------------------
# PromotionExecutor (#17)
# ---------------------------------------------------------------------------

class PromotionExecutor:
    """Writes high-scoring short-term memories to MEMORY.md."""

    def __init__(self, workspace_dir: Path, store):
        self.workspace_dir = workspace_dir
        self.store = store

    async def promote(self, entry) -> bool:
        """Append entry snippet to MEMORY.md and mark as promoted."""
        memory_path = self.workspace_dir / "MEMORY.md"
        timestamp = datetime.now().strftime("%Y-%m-%d")
        section = (
            f"\n## Auto-promoted: {entry.path}#L{entry.start_line}-L{entry.end_line} "
            f"({timestamp})\n\n{entry.snippet}\n"
        )
        try:
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(section)
            self.store.mark_promoted(entry.key)
            logger.info(
                "memory_promoted",
                key=entry.key[:12],
                path=entry.path,
                recall_count=entry.recall_count,
            )
            return True
        except Exception as e:
            logger.warning("promotion_failed", key=entry.key[:12], error=str(e))
            return False


# ---------------------------------------------------------------------------
# DreamingScheduler (#17)
# ---------------------------------------------------------------------------

class DreamingScheduler:
    """Periodically evaluates short-term recalls and promotes eligible entries."""

    def __init__(
        self,
        store,
        scorer: PromotionScorer,
        executor: PromotionExecutor,
        light_interval_minutes: int = 5,
        rem_interval_hours: float = 6.0,
        max_promotions_per_run: int = 3,
    ):
        self.store = store
        self.scorer = scorer
        self.executor = executor
        self.light_interval = light_interval_minutes * 60
        self.rem_interval = rem_interval_hours * 3600
        self.max_promotions_per_run = max_promotions_per_run
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("dreaming_scheduler_started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("dreaming_scheduler_stopped")

    async def run_light_phase(self) -> int:
        """Quick check on recent recalls. Returns number of promotions."""
        candidates = self.store.get_top_recall_candidates(
            min_count=self.scorer.min_recall_count,
            limit=self.max_promotions_per_run * 3,
        )
        promoted = 0
        for entry in candidates:
            if promoted >= self.max_promotions_per_run:
                break
            score = self.scorer.score(entry)
            if score.meets_threshold:
                success = await self.executor.promote(entry)
                if success:
                    promoted += 1
        return promoted

    async def run_rem_phase(self) -> int:
        """Deep sweep of all unpromotedcandidates. Returns number of promotions."""
        candidates = self.store.get_top_recall_candidates(
            min_count=self.scorer.min_recall_count,
            limit=50,
        )
        promoted = 0
        for entry in candidates:
            score = self.scorer.score(entry)
            if score.meets_threshold:
                success = await self.executor.promote(entry)
                if success:
                    promoted += 1
        return promoted

    async def _run_loop(self) -> None:
        light_elapsed = 0.0
        rem_elapsed = 0.0
        sleep_interval = 30

        while self._running:
            await asyncio.sleep(sleep_interval)
            light_elapsed += sleep_interval
            rem_elapsed += sleep_interval

            if light_elapsed >= self.light_interval:
                try:
                    n = await self.run_light_phase()
                    if n > 0:
                        logger.info("dreaming_light_phase", promoted=n)
                except Exception as e:
                    logger.warning("dreaming_light_phase_error", error=str(e))
                light_elapsed = 0.0

            if rem_elapsed >= self.rem_interval:
                try:
                    n = await self.run_rem_phase()
                    if n > 0:
                        logger.info("dreaming_rem_phase", promoted=n)
                except Exception as e:
                    logger.warning("dreaming_rem_phase_error", error=str(e))
                rem_elapsed = 0.0
