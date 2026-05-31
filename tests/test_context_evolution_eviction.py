"""Context-evolution: full-section eviction, archive, pin/user protection."""

import asyncio
import json
import tempfile
from pathlib import Path

from omniagent.agents.context_evolution import LessonRecorder, LessonAnalyzer

PIN = "📌"


def _setup(work_dir: Path, rules):
    """rules: list of dicts {text, evidence, last_seen, pinned?, has_meta?, marker?}."""
    learn = work_dir / ".omniagent" / "learnings"
    learn.mkdir(parents=True, exist_ok=True)
    ad = work_dir / ".omniagent" / "AGENTS.md"
    lines = ["# AGENTS", "", "## Learned Rules (auto-learned)"]
    meta_lines = []
    for r in rules:
        marker = f"{PIN} " if r.get("marker") else ""
        lines.append(f"- [approach] {marker}{r['text']}")
        if r.get("has_meta", True):
            meta_lines.append(json.dumps({
                "rule_text": r["text"], "category": "approach", "source": "failure",
                "lesson_hash": "h_" + r["text"][:10], "evidence": r["evidence"],
                "promoted_at": r["last_seen"], "last_seen": r["last_seen"],
                "pinned": r.get("pinned", False),
            }, ensure_ascii=False))
    ad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (learn / "promoted_rules.jsonl").write_text(
        ("\n".join(meta_lines) + "\n") if meta_lines else "", encoding="utf-8")
    return ad


def _record(rec, text, evidence):
    for _ in range(evidence):
        rec.record("failure", "approach", text, "")


def _learned_lines(text):
    return [l for l in text.splitlines() if l.startswith("- ")]


def test_evicts_weakest_when_full_and_candidate_stronger():
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ad = _setup(work, [
            {"text": "deploy services via blue-green pipelines", "evidence": 5, "last_seen": "2026-01-01"},
            {"text": "emit application logs as structured json", "evidence": 4, "last_seen": "2026-01-02"},
            {"text": "invalidate the cache on every write", "evidence": 2, "last_seen": "2026-01-03"},
        ])
        rec = LessonRecorder(work / ".omniagent" / "learnings")
        _record(rec, "retry failed network calls with exponential backoff", 3)  # 3 > weakest 2
        analyzer = LessonAnalyzer(rec, work, min_evidence=1, max_agents_rules=3, embedder=None)

        promoted = asyncio.run(analyzer.check_and_promote())
        text = ad.read_text()
        assert len(promoted) == 1
        assert "invalidate the cache" not in text          # weakest evicted
        assert "exponential backoff" in text               # candidate added
        assert len(_learned_lines(text)) == 3              # cap respected
        archive = (work / ".omniagent" / "learnings" / "archived_rules.md").read_text()
        assert "invalidate the cache" in archive           # archived, not deleted


def test_no_evict_when_candidate_not_stronger():
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ad = _setup(work, [
            {"text": "deploy services via blue-green pipelines", "evidence": 5, "last_seen": "2026-01-01"},
            {"text": "emit application logs as structured json", "evidence": 4, "last_seen": "2026-01-02"},
            {"text": "invalidate the cache on every write", "evidence": 3, "last_seen": "2026-01-03"},
        ])
        rec = LessonRecorder(work / ".omniagent" / "learnings")
        _record(rec, "retry failed network calls with exponential backoff", 3)  # 3, not > weakest 3
        analyzer = LessonAnalyzer(rec, work, min_evidence=1, max_agents_rules=3, embedder=None)

        promoted = asyncio.run(analyzer.check_and_promote())
        text = ad.read_text()
        assert promoted == []
        assert "exponential backoff" not in text
        assert "invalidate the cache" in text


def test_pinned_weakest_is_never_evicted():
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ad = _setup(work, [
            {"text": "never log plaintext secrets or tokens", "evidence": 1,
             "last_seen": "2026-01-03", "marker": True},  # weakest, but pinned
            {"text": "deploy services via blue-green pipelines", "evidence": 5, "last_seen": "2026-01-01"},
            {"text": "emit application logs as structured json", "evidence": 4, "last_seen": "2026-01-02"},
        ])
        rec = LessonRecorder(work / ".omniagent" / "learnings")
        _record(rec, "retry failed network calls with exponential backoff", 3)
        analyzer = LessonAnalyzer(rec, work, min_evidence=1, max_agents_rules=3, embedder=None)

        promoted = asyncio.run(analyzer.check_and_promote())
        text = ad.read_text()
        # only evictable are evidence 5/4, both > candidate 3 -> nothing evicted
        assert promoted == []
        assert "never log plaintext secrets" in text  # pinned weakest survives


def test_user_placed_line_without_metadata_is_protected():
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ad = _setup(work, [
            {"text": "deploy services via blue-green pipelines", "evidence": 5, "last_seen": "2026-01-01"},
            {"text": "emit application logs as structured json", "evidence": 4, "last_seen": "2026-01-02"},
            {"text": "follow the SOC2 change-control checklist", "evidence": 1,
             "last_seen": "2026-01-03", "has_meta": False},  # user typed, no metadata
        ])
        rec = LessonRecorder(work / ".omniagent" / "learnings")
        _record(rec, "retry failed network calls with exponential backoff", 3)
        analyzer = LessonAnalyzer(rec, work, min_evidence=1, max_agents_rules=3, embedder=None)

        promoted = asyncio.run(analyzer.check_and_promote())
        text = ad.read_text()
        assert promoted == []                       # weakest evictable is 4 > candidate 3
        assert "SOC2 change-control" in text        # user line untouched


def test_user_section_untouched_and_stale_meta_dropped():
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        learn = work / ".omniagent" / "learnings"
        learn.mkdir(parents=True, exist_ok=True)
        ad = work / ".omniagent" / "AGENTS.md"
        ad.write_text(
            "# AGENTS\n\n## My Rules\n- my hand rule one\n- my hand rule two\n\n"
            "## Learned Rules (auto-learned)\n- [approach] auto rule one\n",
            encoding="utf-8")
        (learn / "promoted_rules.jsonl").write_text(
            json.dumps({"rule_text": "auto rule one", "category": "approach",
                        "source": "failure", "lesson_hash": "h1", "evidence": 2,
                        "promoted_at": "2026-01-01", "last_seen": "2026-01-01",
                        "pinned": False}) + "\n"
            + json.dumps({"rule_text": "GONE stale rule", "category": "approach",
                          "source": "failure", "lesson_hash": "h2", "evidence": 9,
                          "promoted_at": "2026-01-01", "last_seen": "2026-01-01",
                          "pinned": False}) + "\n",
            encoding="utf-8")
        rec = LessonRecorder(learn)
        _record(rec, "fresh guidance on debouncing duplicate webhooks", 2)
        analyzer = LessonAnalyzer(rec, work, min_evidence=1, max_agents_rules=30, embedder=None)

        promoted = asyncio.run(analyzer.check_and_promote())
        text = ad.read_text()
        assert "my hand rule one" in text and "my hand rule two" in text
        assert "## My Rules" in text                 # user section intact
        assert len(promoted) == 1                     # room available -> appended
        saved = (learn / "promoted_rules.jsonl").read_text()
        assert "GONE stale rule" not in saved         # stale metadata reconciled away
