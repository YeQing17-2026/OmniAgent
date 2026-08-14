"""Active Memory system (Issues #18, #19, #20, #21).

Implements:
- EligibilityChecker: determines if active memory should activate
- MemorySummaryGenerator: generates coherent summaries from search results
- ActiveMemoryAgent: proactive sub-agent injecting memory before main reply
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# EligibilityChecker (#19)
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    """Result of an eligibility check."""

    eligible: bool
    reason: str
    session_kind: Optional[str] = None
    trigger_type: Optional[str] = None


class EligibilityChecker:
    """Determines whether active memory should activate for a session."""

    INELIGIBLE_KINDS = {
        "headless",
        "one_shot",
        "heartbeat",
        "batch",
        "cli_single",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.min_session_age_seconds = cfg.get("min_session_age_seconds", 0)

    def check(
        self,
        trigger_type: str,
        session_kind: str,
        session_id: Optional[str] = None,
        session_age_seconds: float = 0.0,
    ) -> EligibilityResult:
        """Check if active memory should activate."""
        if not self.enabled:
            return EligibilityResult(False, "active_memory_disabled")

        if trigger_type != "user":
            return EligibilityResult(
                False, f"non_user_trigger:{trigger_type}",
                session_kind=session_kind, trigger_type=trigger_type,
            )

        if not session_id:
            return EligibilityResult(
                False, "no_session_id",
                session_kind=session_kind, trigger_type=trigger_type,
            )

        if session_kind in self.INELIGIBLE_KINDS:
            return EligibilityResult(
                False, f"ineligible_session_kind:{session_kind}",
                session_kind=session_kind, trigger_type=trigger_type,
            )

        if session_age_seconds < self.min_session_age_seconds:
            return EligibilityResult(
                False, f"session_too_new:{session_age_seconds:.0f}s",
                session_kind=session_kind, trigger_type=trigger_type,
            )

        return EligibilityResult(
            True, "eligible",
            session_kind=session_kind, trigger_type=trigger_type,
        )


# ---------------------------------------------------------------------------
# MemorySummary / MemorySummaryGenerator (#20)
# ---------------------------------------------------------------------------

@dataclass
class MemorySummary:
    """Generated memory summary for context injection."""

    summary_text: str
    source_count: int
    result_count: int
    topics: List[str] = field(default_factory=list)


class MemorySummaryGenerator:
    """Generates coherent summaries from memory search results."""

    MAX_SUMMARY_LENGTH = 500
    MAX_RESULTS = 6

    def __init__(self, llm_provider=None, config: Optional[Dict[str, Any]] = None):
        self.llm = llm_provider
        cfg = config or {}
        self.max_length = cfg.get("max_summary_length", self.MAX_SUMMARY_LENGTH)
        self.max_results = cfg.get("max_results", self.MAX_RESULTS)

    async def generate(
        self,
        results: List,
        query_context: str,
    ) -> MemorySummary:
        """Generate a summary from search results.

        Uses LLM if available, otherwise falls back to structured formatting.
        """
        if not results:
            return MemorySummary("", 0, 0)

        deduplicated = self._dedup_results(results[: self.max_results])
        if not deduplicated:
            return MemorySummary("", 0, 0)

        sources = list({getattr(r, "path", "") for r in deduplicated})

        if self.llm:
            summary_text = await self._llm_summary(deduplicated, query_context)
        else:
            summary_text = self._format_summary(deduplicated)

        if len(summary_text) > self.max_length:
            summary_text = summary_text[: self.max_length] + "..."

        return MemorySummary(
            summary_text=summary_text,
            source_count=len(sources),
            result_count=len(deduplicated),
            topics=self._extract_topics(deduplicated),
        )

    def _dedup_results(self, results: List) -> List:
        """Remove near-duplicate results by snippet prefix."""
        seen = set()
        out = []
        for r in results:
            key = getattr(r, "snippet", "")[:80].strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def _format_summary(self, results: List) -> str:
        """Fallback: structured text formatting."""
        parts = []
        for r in results:
            path = getattr(r, "path", "")
            start = getattr(r, "start_line", 0)
            end = getattr(r, "end_line", 0)
            snippet = getattr(r, "snippet", getattr(r, "text", ""))[:200]
            parts.append(f"[{path}#L{start}-{end}]\n{snippet}")
        return "\n\n".join(parts)

    async def _llm_summary(self, results: List, query_context: str) -> str:
        """Generate summary via LLM."""
        from omniagent.agents.llm import LLMMessage

        context_parts = []
        for r in results:
            path = getattr(r, "path", "")
            snippet = getattr(r, "snippet", getattr(r, "text", ""))[:300]
            context_parts.append(f"Source: {path}\n{snippet}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"Based on the following memory snippets, write a brief, coherent summary "
            f"(max {self.max_length} chars) relevant to: {query_context[:200]}\n\n"
            f"Memory snippets:\n{context}\n\n"
            "Summary (be concise and factual, include source paths):"
        )

        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm.complete(messages, max_tokens=256, temperature=0.3)  # type: ignore[union-attr]
            return response.content.strip()
        except Exception as e:
            logger.warning("llm_summary_failed", error=str(e))
            return self._format_summary(results)

    def _extract_topics(self, results: List) -> List[str]:
        """Extract simple topic labels from paths."""
        topics = set()
        for r in results:
            path = getattr(r, "path", "")
            if path:
                stem = path.replace(".md", "").replace("/", " ").strip()
                if stem:
                    topics.add(stem.split()[-1])
        return list(topics)[:5]


# ---------------------------------------------------------------------------
# ActiveMemoryAgent (#18)
# ---------------------------------------------------------------------------

@dataclass
class ActiveMemoryConfig:
    """Configuration for ActiveMemoryAgent."""

    enabled: bool = True
    query_mode: str = "recent"       # "message" | "recent" | "full"
    prompt_style: str = "balanced"   # "balanced" | "strict" | "contextual"
    max_context_chars: int = 4000
    max_results: int = 6
    min_score: float = 0.3
    session_kind: str = "interactive"
    min_session_age_seconds: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActiveMemoryConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ActiveMemoryAgent:
    """Proactive sub-agent: recalls memory before main agent replies (#18).

    Called synchronously in handle_message() before the main LLM turn.
    Returns an injected memory summary string, or None if not eligible.
    """

    QUERY_MODE_MESSAGE = "message"
    QUERY_MODE_RECENT = "recent"
    QUERY_MODE_FULL = "full"

    def __init__(
        self,
        memory_manager,
        llm_provider=None,
        config: Optional[ActiveMemoryConfig] = None,
    ):
        self.memory_manager = memory_manager
        self.llm = llm_provider
        self.config = config or ActiveMemoryConfig()
        self.eligibility = EligibilityChecker(
            config={"enabled": self.config.enabled,
                    "min_session_age_seconds": self.config.min_session_age_seconds}
        )
        self.summary_gen = MemorySummaryGenerator(
            llm_provider=llm_provider,
            config={"max_summary_length": 500, "max_results": self.config.max_results},
        )

    async def run(
        self,
        message_text: str,
        session_id: Optional[str] = None,
        session_kind: str = "interactive",
        session_age_seconds: float = 60.0,
        recent_context: str = "",
    ) -> Optional[str]:
        """Run active memory recall. Returns summary text or None."""
        eligibility = self.eligibility.check(
            trigger_type="user",
            session_kind=session_kind,
            session_id=session_id,
            session_age_seconds=session_age_seconds,
        )
        if not eligibility.eligible:
            return None

        query = self._build_query(message_text, recent_context)
        if not query.strip():
            return None

        try:
            results = await self.memory_manager.search(
                query,
                max_results=self.config.max_results,
                min_score=self.config.min_score,
            )
        except Exception as e:
            logger.warning("active_memory_search_failed", error=str(e))
            return None

        if not results:
            return None

        summary = await self.summary_gen.generate(results, query_context=query)
        if not summary.summary_text:
            return None

        logger.info(
            "active_memory_injected",
            result_count=summary.result_count,
            source_count=summary.source_count,
        )
        return summary.summary_text

    def _build_query(self, message_text: str, recent_context: str) -> str:
        """Build search query based on query_mode."""
        mode = self.config.query_mode
        max_chars = self.config.max_context_chars

        if mode == self.QUERY_MODE_MESSAGE:
            return message_text[:max_chars]

        if mode == self.QUERY_MODE_FULL:
            return (recent_context + "\n" + message_text)[:max_chars]

        # Default: QUERY_MODE_RECENT — recent context + current message
        context_chars = max_chars - len(message_text) - 2
        context = recent_context[-context_chars:] if context_chars > 0 else ""
        return (context + "\n" + message_text).strip()[:max_chars]
