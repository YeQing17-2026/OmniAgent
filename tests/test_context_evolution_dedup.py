"""Context-evolution: lexical/semantic dedup, record-time merge, thresholds."""

import asyncio
import tempfile
from pathlib import Path

from omniagent.agents.context_evolution import LessonRecorder, LessonAnalyzer
from omniagent.agents.similarity import (
    SemanticEmbedder,
    is_duplicate,
    lexical_similarity,
)


def test_lexical_similarity_paraphrase_high_unrelated_low():
    a = "When analyzing code, use flowcharts instead of plain text"
    b = "When analyzing code, use flowcharts rather than plain text"
    c = "Always back up the database before running migrations"
    assert lexical_similarity(a, b) >= 0.72
    assert lexical_similarity(a, c) < 0.5


def test_is_duplicate_lexical_only():
    existing = ["When analyzing code, use flowcharts instead of plain text"]
    near = "When analyzing code, use flowcharts rather than plain text"
    far = "Always validate user input before saving to disk"
    assert asyncio.run(is_duplicate(near, existing, lex_threshold=0.72,
                                    sem_threshold=0.82, embedder=None))
    assert not asyncio.run(is_duplicate(far, existing, lex_threshold=0.72,
                                        sem_threshold=0.82, embedder=None))


class _FakeEmbedder:
    """Returns caller-controlled vectors; reports itself as available."""

    def __init__(self, mapping):
        self.mapping = mapping

    async def available(self):
        return True

    async def embed(self, texts):
        return [self.mapping.get(t, [0.0, 0.0, 1.0]) for t in texts]


def test_semantic_catches_paraphrase_that_lexical_misses():
    cand = "Prefer flowcharts over prose when explaining implementation"
    rule = "Use diagrams not paragraphs for module structure"
    assert lexical_similarity(cand, rule) < 0.72  # lexically distinct

    # No embedder -> lexical only -> not a duplicate
    assert not asyncio.run(is_duplicate(cand, [rule], lex_threshold=0.72,
                                        sem_threshold=0.82, embedder=None))
    # Embedder reports identical vectors -> semantic duplicate
    emb = _FakeEmbedder({cand: [1.0, 0.0, 0.0], rule: [1.0, 0.0, 0.0]})
    assert asyncio.run(is_duplicate(cand, [rule], lex_threshold=0.72,
                                    sem_threshold=0.82, embedder=emb))


def test_semantic_embedder_off_is_unavailable():
    emb = SemanticEmbedder(backend="off")
    assert asyncio.run(emb.available()) is False
    assert asyncio.run(emb.embed(["x"])) is None


def test_record_merges_paraphrases_and_accumulates_evidence():
    with tempfile.TemporaryDirectory() as d:
        rec = LessonRecorder(Path(d) / "learnings", dedup_record_threshold=0.7)
        base = "When analyzing code, use flowcharts instead of plain text"
        l1 = rec.record("failure", "approach", base, "")
        assert l1.evidence == 1 and l1.last_seen is not None

        l2 = rec.record("failure", "approach", base, "")  # exact repeat
        assert l2.lesson_hash == l1.lesson_hash and l2.evidence == 2

        para = "When analyzing code, use flowcharts rather than plain text"
        l3 = rec.record("failure", "approach", para, "")  # paraphrase -> merge
        assert l3.lesson_hash == l1.lesson_hash and l3.evidence == 3

        l4 = rec.record("failure", "approach",
                        "Always back up the database before running migrations", "")
        assert l4.lesson_hash != l1.lesson_hash and l4.evidence == 1


def _make_agents(work_dir: Path) -> Path:
    ad = work_dir / ".omniagent" / "AGENTS.md"
    ad.parent.mkdir(parents=True, exist_ok=True)
    ad.write_text("# AGENTS\n\nhuman rules\n\n## Learned Rules (auto-learned)\n",
                  encoding="utf-8")
    return ad


def test_single_user_feedback_not_promoted_until_threshold():
    """Reproduces #53: one feedback must not land in AGENTS.md."""
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ad = _make_agents(work)
        rec = LessonRecorder(work / ".omniagent" / "learnings")
        analyzer = LessonAnalyzer(rec, work, min_evidence=3,
                                  user_feedback_min_evidence=2, embedder=None)
        lesson = ("When the user asks for the previously received request JSON, "
                  "reconstruct it from logs instead of saying it is unavailable")

        rec.record("user_feedback", "preference", lesson, "")
        promoted = asyncio.run(analyzer.check_and_promote())
        assert promoted == []
        assert "reconstruct it from logs" not in ad.read_text()

        rec.record("user_feedback", "preference", lesson, "")  # evidence -> 2
        promoted = asyncio.run(analyzer.check_and_promote())
        assert len(promoted) == 1
        assert "reconstruct it from logs" in ad.read_text()
