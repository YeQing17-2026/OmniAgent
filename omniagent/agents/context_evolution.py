"""Context self-evolution system for OmniAgent.

Automatically extracts lessons from:
1. Failed executions and reflexion reflections
2. User feedback (corrections, preferences)
3. Error-recovery patterns

Validated lessons are promoted to AGENTS.md as learned rules.

Architecture:
  LessonRecorder → .omniagent/learnings/lessons.jsonl (append-only)
  LessonAnalyzer  → checks evidence count, promotes to AGENTS.md
  ContextEvolutionManager → orchestrator via EventBus
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger
from .similarity import SemanticEmbedder, is_duplicate, lexical_similarity

logger = get_logger(__name__)


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class Lesson:
    """A single recorded lesson (one JSONL line)."""

    timestamp: str
    lesson_hash: str
    source: str  # "failure", "reflection", "user_feedback", "error_recovery"
    category: str  # "approach", "preference", "constraint", "workflow"
    lesson: str
    context: str  # What situation triggered this lesson
    evidence: int = 1  # How many times confirmed
    promoted: bool = False
    promoted_at: Optional[str] = None
    last_seen: Optional[str] = None  # Last time evidence was reinforced

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class PromotedRule:
    """A lesson promoted to AGENTS.md."""

    rule: str
    category: str
    source: str
    promoted_at: str


# ── Lesson Recorder ────────────────────────────────────────────────


class LessonRecorder:
    """Records lessons from agent execution outcomes to append-only JSONL."""

    def __init__(self, learnings_dir: Path, max_learnings: int = 100,
                 dedup_record_threshold: float = 0.85):
        self.learnings_dir = learnings_dir
        self.max_learnings = max_learnings
        self.dedup_record_threshold = dedup_record_threshold
        self.learnings_dir.mkdir(parents=True, exist_ok=True)
        self._lessons_file = self.learnings_dir / "lessons.jsonl"
        # In-memory index: lesson_hash -> lesson
        self._index: Dict[str, Lesson] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load existing lessons into memory index."""
        if not self._lessons_file.exists():
            return
        with open(self._lessons_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    lesson = Lesson(**data)
                    self._index[lesson.lesson_hash] = lesson
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug("lesson_load_failed", error=str(e))

    def record(
        self,
        source: str,
        category: str,
        lesson: str,
        context: str,
    ) -> Optional[Lesson]:
        """Record a new lesson or increment evidence for an existing one.

        Args:
            source: Where the lesson came from ("failure", "reflection", etc.)
            category: Type of lesson ("approach", "preference", "constraint", "workflow")
            lesson: The lesson text
            context: Situation that triggered the lesson

        Returns:
            The recorded/updated lesson, or None if invalid.
        """
        if not lesson or len(lesson.strip()) < 10:
            return None

        lesson_text = lesson.strip()
        now = datetime.now(timezone.utc).isoformat()
        lesson_hash = self._compute_hash(category, lesson_text)

        # Merge into an existing lesson on an exact hash match OR a same-category
        # near-duplicate, so evidence accumulates across paraphrases (otherwise
        # the evidence thresholds would be effectively unreachable).
        existing = self._index.get(lesson_hash)
        if existing is None:
            existing = self._find_similar(category, lesson_text)
        if existing is not None:
            existing.evidence += 1
            existing.last_seen = now
            self._rewrite_file()
            logger.debug("lesson_evidence_incremented", hash=existing.lesson_hash, evidence=existing.evidence)
            return existing

        new_lesson = Lesson(
            timestamp=now,
            lesson_hash=lesson_hash,
            source=source,
            category=category,
            lesson=lesson_text,
            context=context[:500] if context else "",
            evidence=1,
            last_seen=now,
        )
        self._index[lesson_hash] = new_lesson
        self._prune_if_needed()
        with open(self._lessons_file, "a", encoding="utf-8") as f:
            f.write(new_lesson.to_jsonl() + "\n")
        logger.info("lesson_recorded", hash=lesson_hash, source=source, category=category)
        return new_lesson

    def _find_similar(self, category: str, lesson_text: str) -> Optional[Lesson]:
        """Find an existing same-category lesson that is a near-duplicate."""
        best: Optional[Lesson] = None
        best_score = 0.0
        for l in self._index.values():
            if l.category != category:
                continue
            score = lexical_similarity(lesson_text, l.lesson)
            if score >= self.dedup_record_threshold and score > best_score:
                best, best_score = l, score
        return best

    def get_lesson(self, lesson_hash: str) -> Optional[Lesson]:
        """Get a lesson by hash."""
        return self._index.get(lesson_hash)

    def get_candidates(
        self, min_evidence: int = 2, user_feedback_min_evidence: int = 1,
    ) -> List[Lesson]:
        """Get lessons that meet promotion threshold but are not yet promoted.

        User feedback lessons use a lower threshold (default 1) since they are
        explicit user corrections and should be promoted immediately.
        """
        candidates = []
        for l in self._index.values():
            if l.promoted:
                continue
            threshold = (
                user_feedback_min_evidence
                if l.source == "user_feedback"
                else min_evidence
            )
            if l.evidence >= threshold:
                candidates.append(l)
        return candidates

    def get_all_lessons(self) -> List[Lesson]:
        """Get all lessons."""
        return list(self._index.values())

    def mark_promoted(self, lesson_hash: str) -> None:
        """Mark a lesson as promoted."""
        if lesson_hash in self._index:
            self._index[lesson_hash].promoted = True
            self._index[lesson_hash].promoted_at = datetime.now(timezone.utc).isoformat()
            self._rewrite_file()

    def _prune_if_needed(self) -> None:
        """Rewrite file removing low-evidence lessons if count exceeds max."""
        if not self._lessons_file.exists():
            return
        if len(self._index) <= self.max_learnings:
            return

        # Prune promoted lessons first (they're in AGENTS.md already)
        all_lessons = sorted(self._index.values(), key=lambda l: (l.promoted, -l.evidence))
        keep = all_lessons[:self.max_learnings]
        self._index = {l.lesson_hash: l for l in keep}
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """Rewrite JSONL from in-memory index."""
        with open(self._lessons_file, "w", encoding="utf-8") as f:
            for lesson in self._index.values():
                f.write(lesson.to_jsonl() + "\n")

    @staticmethod
    def _compute_hash(category: str, lesson: str) -> str:
        """Compute deterministic hash for deduplication."""
        import hashlib
        text = f"{category}:{lesson[:200].lower()}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Lesson Analyzer ────────────────────────────────────────────────


class LessonAnalyzer:
    """Analyzes lessons and promotes validated ones to AGENTS.md."""

    LEARNED_RULES_SECTION = "## Learned Rules (auto-learned)"
    ARCHIVE_FILE = ".omniagent/learnings/archived_rules.md"

    def __init__(
        self,
        recorder: LessonRecorder,
        work_dir: Path,
        min_evidence: int = 2,
        max_agents_rules: int = 30,
        user_feedback_min_evidence: int = 1,
        lexical_threshold: float = 0.80,
        semantic_threshold: float = 0.78,
        embedder: Optional[SemanticEmbedder] = None,
        enable_eviction: bool = True,
        pin_marker: str = "📌",
    ):
        self.recorder = recorder
        self.work_dir = work_dir
        self.min_evidence = min_evidence
        self.max_agents_rules = max_agents_rules
        self.user_feedback_min_evidence = user_feedback_min_evidence
        self.lexical_threshold = lexical_threshold
        self.semantic_threshold = semantic_threshold
        self.embedder = embedder
        self.enable_eviction = enable_eviction
        self.pin_marker = pin_marker
        self._promoted_meta_file = (
            self.work_dir / ".omniagent" / "learnings" / "promoted_rules.jsonl"
        )

    async def check_and_promote(self) -> List[PromotedRule]:
        """Check for lessons ready for promotion and promote them.

        Dedup against existing AGENTS.md rules is lexical-or-semantic. When the
        auto-learned section is full, a strictly stronger candidate may displace
        the weakest evictable rule (archived, not deleted); user-authored or
        pinned rules are never touched.
        """
        candidates = self.recorder.get_candidates(
            min_evidence=self.min_evidence,
            user_feedback_min_evidence=self.user_feedback_min_evidence,
        )
        if not candidates:
            return []

        # Strongest first: they get scarce slots / get to displace the weakest.
        candidates.sort(key=lambda l: -l.evidence)

        agents_path = self._find_agents_md()
        if not agents_path:
            logger.debug("no_agents_md_found", work_dir=str(self.work_dir))
            return []

        current_lines = self._read_current_rules(agents_path)
        current_texts = [self._rule_text(line) for line in current_lines]
        meta = self._load_promoted_meta(current_texts)

        promoted: List[PromotedRule] = []
        for candidate in candidates:
            if await is_duplicate(
                candidate.lesson, current_texts,
                lex_threshold=self.lexical_threshold,
                sem_threshold=self.semantic_threshold,
                embedder=self.embedder,
            ):
                self.recorder.mark_promoted(candidate.lesson_hash)
                continue

            # Needs a slot.
            if len(current_texts) >= self.max_agents_rules:
                victim = (
                    self._pick_eviction_victim(current_lines, current_texts, meta, candidate)
                    if self.enable_eviction else None
                )
                if victim is None:
                    continue  # cannot make room for this candidate
                self._archive_rule(victim["text"])
                self._remove_rule_line(agents_path, victim["line"])
                meta.pop(victim["text"], None)
                idx = current_lines.index(victim["line"])
                current_lines.pop(idx)
                current_texts.pop(idx)

            rule = self._format_rule(candidate)
            if not rule:
                continue
            self._append_one_rule(agents_path, rule, meta)
            current_lines.append(f"- [{rule.category}] {rule.rule}")
            current_texts.append(rule.rule)
            promoted.append(rule)
            self.recorder.mark_promoted(rule._lesson_hash)

        if promoted:
            self._save_promoted_meta(meta)
            logger.info("rules_promoted_to_agents_md", count=len(promoted))
        return promoted

    def _find_agents_md(self) -> Optional[Path]:
        """Find AGENTS.md following BootstrapFiles search order."""
        search_paths = [
            self.work_dir / ".omniagent",
            Path.home() / ".omniagent",
        ]
        for search_dir in search_paths:
            path = search_dir / "AGENTS.md"
            if path.is_file():
                return path
        return None

    def _read_current_rules(self, agents_path: Path) -> List[str]:
        """Extract current learned rules from AGENTS.md."""
        content = agents_path.read_text(encoding="utf-8")
        rules = []

        in_section = False
        for line in content.split("\n"):
            if line.strip() == self.LEARNED_RULES_SECTION:
                in_section = True
                continue
            if in_section:
                if line.startswith("## ") and not line.strip() == self.LEARNED_RULES_SECTION:
                    break
                # Extract rule content (lines starting with -)
                if line.strip().startswith("- "):
                    rules.append(line.strip())

        return rules

    def _rule_text(self, line: str) -> str:
        """Extract the bare rule text from a '- [category] text' bullet line."""
        s = line.strip()
        if s.startswith("- "):
            s = s[2:].strip()
        s = re.sub(r"^\[[^\]]+\]\s*", "", s)  # drop leading [category]
        if self.pin_marker and s.startswith(self.pin_marker):
            s = s[len(self.pin_marker):].strip()
        return s

    def _is_pinned(self, line: str, text: str, meta: Dict[str, dict]) -> bool:
        """Protected if pin-marked, flagged pinned, or lacking metadata
        (a line the user placed by hand)."""
        if self.pin_marker and self.pin_marker in line:
            return True
        entry = meta.get(text)
        if entry is None:
            return True
        return bool(entry.get("pinned"))

    def _format_rule(self, lesson: Lesson) -> Optional[PromotedRule]:
        """Format a lesson as a promotable rule."""
        rule_text = lesson.lesson
        # Truncate if too long
        if len(rule_text) > 200:
            rule_text = rule_text[:200] + "..."

        rule = PromotedRule(
            rule=rule_text,
            category=lesson.category,
            source=lesson.source,
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )
        rule._lesson_hash = lesson.lesson_hash  # Attach for promotion tracking
        return rule

    def _append_one_rule(self, agents_path: Path, rule: PromotedRule, meta: Dict[str, dict]) -> None:
        """Insert one rule into the learned-rules section and record metadata."""
        content = agents_path.read_text(encoding="utf-8")
        line = f"- [{rule.category}] {rule.rule}"
        if self.LEARNED_RULES_SECTION in content:
            content = re.sub(
                rf"({re.escape(self.LEARNED_RULES_SECTION)}\n)",
                rf"\1{line}\n",
                content,
            )
        else:
            content = content.rstrip() + f"\n{self.LEARNED_RULES_SECTION}\n{line}\n"
        agents_path.write_text(content, encoding="utf-8")

        lesson = self.recorder.get_lesson(rule._lesson_hash)
        meta[rule.rule] = {
            "rule_text": rule.rule,
            "category": rule.category,
            "source": rule.source,
            "lesson_hash": rule._lesson_hash,
            "evidence": lesson.evidence if lesson else 1,
            "promoted_at": rule.promoted_at,
            "last_seen": (lesson.last_seen if lesson and lesson.last_seen else rule.promoted_at),
            "pinned": False,
        }

    def _load_promoted_meta(self, current_texts: List[str]) -> Dict[str, dict]:
        """Load promoted-rule metadata, dropping entries no longer in AGENTS.md."""
        meta: Dict[str, dict] = {}
        if self._promoted_meta_file.exists():
            for line in self._promoted_meta_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    meta[entry["rule_text"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
        present = set(current_texts)
        return {k: v for k, v in meta.items() if k in present}

    def _save_promoted_meta(self, meta: Dict[str, dict]) -> None:
        self._promoted_meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._promoted_meta_file, "w", encoding="utf-8") as f:
            for entry in meta.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _archive_rule(self, text: str) -> None:
        """Append an evicted rule to the archive (kept, not deleted)."""
        archive_path = self.work_dir / self.ARCHIVE_FILE
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(archive_path, "a", encoding="utf-8") as f:
            f.write(f"- [{ts}] {text}\n")

    def _remove_rule_line(self, agents_path: Path, line: str) -> None:
        content = agents_path.read_text(encoding="utf-8")
        kept = [ln for ln in content.split("\n") if ln.strip() != line.strip()]
        agents_path.write_text("\n".join(kept), encoding="utf-8")

    def _pick_eviction_victim(
        self, lines: List[str], texts: List[str], meta: Dict[str, dict], candidate: Lesson
    ) -> Optional[dict]:
        """Weakest evictable rule the candidate may displace, else None.

        Evictable = system-promoted (has metadata) and not pinned/protected.
        Ranked by (evidence asc, last_seen asc) so an old-but-reinforced rule
        survives. Returns a victim only if the candidate strictly out-evidences it.
        """
        evictable = []
        for line, text in zip(lines, texts):
            if self._is_pinned(line, text, meta):
                continue
            entry = meta.get(text)
            if entry is None:
                continue
            evictable.append((
                entry.get("evidence", 1),
                entry.get("last_seen") or entry.get("promoted_at") or "",
                line,
                text,
            ))
        if not evictable:
            return None
        evictable.sort(key=lambda x: (x[0], x[1]))
        weakest_evidence, _, line, text = evictable[0]
        if candidate.evidence > weakest_evidence:
            return {"line": line, "text": text, "evidence": weakest_evidence}
        return None


# ── Lesson Extractor ───────────────────────────────────────────────


class LessonExtractor:
    """Extracts lessons from conversation history using LLM."""

    def __init__(self, llm_provider, max_tokens: int = 1024):
        self.llm = llm_provider
        self.max_tokens = max_tokens

    async def extract_from_reflection(
        self, task: str, reflection: str, error: str = ""
    ) -> Optional[Dict[str, str]]:
        """Extract a lesson from a reflexion reflection.

        Returns dict with keys: category, lesson, context
        """
        from .llm import LLMMessage

        prompt = (
            "Analyze this failed task execution and extract ONE concise lesson.\n\n"
            f"Task: {task[:300]}\n"
            f"Error: {error[:300]}\n"
            f"Reflection: {reflection[:500]}\n\n"
            "Respond in JSON format only:\n"
            '{"category": "approach|preference|constraint|workflow", '
            '"lesson": "the lesson in one sentence", '
            '"context": "when this applies"}'
        )

        return await self._extract_with_llm(prompt)

    async def extract_from_failure(
        self, task: str, conversation_history: list, error: str = ""
    ) -> Optional[Dict[str, str]]:
        """Extract a lesson from a failed execution without reflexion."""
        from .llm import LLMMessage

        # Extract last few assistant messages for context
        context_snippets = []
        for msg in reversed(conversation_history[-10:]):
            if msg.role == "assistant" and msg.content:
                context_snippets.append(msg.content[:200])
            if len(context_snippets) >= 3:
                break
        context_text = "\n".join(reversed(context_snippets))

        prompt = (
            "Analyze this failed task and extract ONE concise lesson.\n\n"
            f"Task: {task[:300]}\n"
            f"Final context: {context_text[:500]}\n"
            f"Error: {error[:300]}\n\n"
            "Respond in JSON format only:\n"
            '{"category": "approach|preference|constraint|workflow", '
            '"lesson": "the lesson in one sentence", '
            '"context": "when this applies"}'
        )

        return await self._extract_with_llm(prompt)

    async def extract_from_user_feedback(
        self, task: str, user_feedback: str,
        prev_history: Optional[list] = None,
    ) -> Optional[Dict[str, str]]:
        """Extract a lesson from explicit user feedback/correction.

        Args:
            task: The original task description.
            user_feedback: The user's feedback/correction text.
            prev_history: Conversation history from the PREVIOUS execution
                (before the feedback), used to understand what the agent
                actually did and what the user is correcting.
        """
        from .llm import LLMMessage

        # Build a compact summary of the previous execution
        prev_summary = ""
        if prev_history:
            events = []
            for msg in prev_history[-20:]:
                if msg.role == "user" and msg.content:
                    events.append(f"User asked: {msg.content[:200]}")
                elif msg.role == "assistant":
                    if msg.content:
                        events.append(f"Agent did: {msg.content[:300]}")
                elif msg.role == "tool":
                    content = msg.content or ""
                    if content.startswith("Error:"):
                        events.append(f"Tool error: {content[:150]}")
            prev_summary = "\n".join(events[-10:])
            if len(prev_summary) > 2000:
                prev_summary = prev_summary[-2000:]

        prompt = (
            "Extract ONE specific, actionable lesson from this user feedback.\n\n"
            f"Task the agent was doing: {task[:300]}\n"
        )
        if prev_summary:
            prompt += (
                f"What the agent actually did (previous execution):\n"
                f"{prev_summary}\n\n"
            )
        prompt += (
            f"User feedback: {user_feedback[:500]}\n\n"
            "Rules:\n"
            '1. Category must be "preference" (user is expressing a preference)\n'
            "2. The lesson must be SPECIFIC — look at what the agent actually did "
            "above, then capture exactly what the user wants changed.\n"
            "3. Do NOT generalize — keep the concrete task type and concrete preference.\n"
            '4. Context must describe WHEN this applies (the specific task type).\n\n'
            "Respond in JSON format only:\n"
            '{"category": "preference", '
            '"lesson": "具体任务类型 + 具体偏好，一句话", '
            '"context": "适用场景描述"}\n\n'
            "Good example:\n"
            '{"category": "preference", '
            '"lesson": "When analyzing code implementation flow, use text-based flowcharts instead of plain text descriptions", '
            '"context": "When user asks to analyze code structure, implementation flow, or architecture"}'
        )

        return await self._extract_with_llm(prompt)

    async def _extract_with_llm(self, prompt: str) -> Optional[Dict[str, str]]:
        """Common LLM extraction logic."""
        from .llm import LLMMessage

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            content = (response.content or "").strip()
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            valid_categories = {"approach", "preference", "constraint", "workflow"}
            category = data.get("category", "").strip().lower()
            if category not in valid_categories:
                category = "approach"

            lesson = data.get("lesson", "").strip()
            if not lesson or len(lesson) < 10:
                return None

            return {
                "category": category,
                "lesson": lesson,
                "context": data.get("context", "").strip(),
            }
        except Exception as e:
            logger.warning("lesson_extraction_failed", error=str(e))
            return None


# ── Context Evolution Manager (Orchestrator) ───────────────────────


class ContextEvolutionManager:
    """Top-level orchestrator for context self-evolution.

    Subscribes to EventBus and coordinates LessonRecorder, LessonExtractor,
    and LessonAnalyzer.

    Lifecycle:
    1. AGENT_END with success + reflections → extract lesson from reflection
    2. AGENT_END with failure → extract lesson from failure
    3. After extraction → check_and_promote validated lessons to AGENTS.md
    """

    def __init__(
        self,
        event_bus,
        work_dir: Path,
        llm_provider,
        config,
    ):
        self.event_bus = event_bus
        self.work_dir = Path(work_dir)
        self.config = config
        self.llm = llm_provider

        self._learnings_dir = self.work_dir / ".omniagent" / "learnings"

        self.recorder = LessonRecorder(
            learnings_dir=self._learnings_dir,
            max_learnings=config.max_learnings,
            dedup_record_threshold=getattr(config, "dedup_record_threshold", 0.85),
        )
        self.extractor = LessonExtractor(
            llm_provider=llm_provider,
            max_tokens=config.compile_max_tokens,
        )
        embedder = SemanticEmbedder(
            backend=getattr(config, "embedding_backend", "auto"),
            ollama_url=getattr(config, "ollama_url", "http://localhost:11434"),
            ollama_model=getattr(config, "ollama_embed_model", "nomic-embed-text"),
            fastembed_model=getattr(config, "fastembed_model", "BAAI/bge-small-en-v1.5"),
        )
        self.analyzer = LessonAnalyzer(
            recorder=self.recorder,
            work_dir=self.work_dir,
            min_evidence=config.lesson_min_evidence,
            max_agents_rules=config.max_agents_rules,
            user_feedback_min_evidence=config.user_feedback_min_evidence,
            lexical_threshold=getattr(config, "dedup_lexical_threshold", 0.72),
            semantic_threshold=getattr(config, "dedup_semantic_threshold", 0.82),
            embedder=embedder,
            enable_eviction=getattr(config, "enable_rule_eviction", True),
            pin_marker=getattr(config, "rule_pin_marker", "📌"),
        )

        self._current_task: str = ""
        self._current_history: list = []
        self._current_reflections: List[str] = []
        self._current_success: bool = False
        self._current_error: str = ""
        self._current_user_feedback: List[str] = []
        self.last_session_results: Dict[str, Any] = {}

        from .events import EventType
        self.event_bus.subscribe(EventType.AGENT_START, self._on_agent_start)
        self.event_bus.subscribe(EventType.AGENT_END, self._on_agent_end)

        logger.info("context_evolution_initialized")

    async def _on_agent_start(self, event) -> None:
        """Capture execution context at start."""
        self._current_task = event.data.get("task", "")

    async def _on_agent_end(self, event) -> None:
        """Process completed execution."""
        if not self._current_task:
            self.last_session_results = {}
            return

        success = event.data.get("success", False)
        self._current_success = success
        lessons_extracted = 0
        lesson_details: List[str] = []

        if success and self._current_reflections:
            # Extract lessons from reflections (reflexion succeeded after retry)
            for reflection in self._current_reflections:
                try:
                    lesson_data = await self.extractor.extract_from_reflection(
                        task=self._current_task,
                        reflection=reflection,
                        error=self._current_error,
                    )
                    if lesson_data:
                        self.recorder.record(
                            source="reflection",
                            category=lesson_data["category"],
                            lesson=lesson_data["lesson"],
                            context=lesson_data["context"],
                        )
                        lessons_extracted += 1
                        lesson_details.append(lesson_data["lesson"])
                except Exception as e:
                    logger.warning("reflection_lesson_failed", error=str(e))

        elif not success:
            # Extract lesson from failure
            try:
                lesson_data = await self.extractor.extract_from_failure(
                    task=self._current_task,
                    conversation_history=self._current_history,
                    error=self._current_error,
                )
                if lesson_data:
                    self.recorder.record(
                        source="failure",
                        category=lesson_data["category"],
                        lesson=lesson_data["lesson"],
                        context=lesson_data["context"],
                    )
                    lessons_extracted += 1
                    lesson_details.append(lesson_data["lesson"])
            except Exception as e:
                logger.warning("failure_lesson_failed", error=str(e))

        elif success and self._has_tool_errors():
            # "False success" — agent reported success but encountered tool errors
            # (e.g., path traversal blocked, permission denied)
            try:
                tool_errors = self._extract_tool_errors()
                error_summary = "; ".join(tool_errors[:3])
                lesson_data = await self.extractor.extract_from_failure(
                    task=self._current_task,
                    conversation_history=self._current_history,
                    error=error_summary,
                )
                if lesson_data:
                    self.recorder.record(
                        source="failure",
                        category=lesson_data["category"],
                        lesson=lesson_data["lesson"],
                        context=lesson_data["context"],
                    )
                    lessons_extracted += 1
                    lesson_details.append(lesson_data["lesson"])
            except Exception as e:
                logger.warning("false_success_lesson_failed", error=str(e))

        # Process user feedback lessons
        for feedback in self._current_user_feedback:
            try:
                lesson_data = await self.extractor.extract_from_user_feedback(
                    task=self._current_task,
                    user_feedback=feedback,
                    prev_history=self._current_history,
                )
                if lesson_data:
                    self.recorder.record(
                        source="user_feedback",
                        category=lesson_data["category"],
                        lesson=lesson_data["lesson"],
                        context=lesson_data["context"],
                    )
                    lessons_extracted += 1
                    lesson_details.append(lesson_data["lesson"])
            except Exception as e:
                logger.warning("feedback_lesson_failed", error=str(e))

        # Check for promotion
        rules_promoted = 0
        rule_details: List[str] = []
        if self.config.promotion_enabled:
            try:
                promoted = await self.analyzer.check_and_promote()
                if promoted:
                    rules_promoted = len(promoted)
                    rule_details = [r.rule for r in promoted if r.rule]
                    logger.info("lessons_promoted", count=rules_promoted)
            except Exception as e:
                logger.warning("promotion_failed", error=str(e))

        from datetime import datetime
        self.last_session_results = {
            "lessons_extracted": lessons_extracted,
            "lesson_details": lesson_details,
            "rules_promoted": rules_promoted,
            "rule_details": rule_details,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def set_agent_refs(
        self,
        conversation_history: list,
        reflections: List[str],
        error: str = "",
        user_feedback: Optional[List[str]] = None,
    ) -> None:
        """Set references to agent state for event handlers."""
        self._current_history = conversation_history
        self._current_reflections = list(reflections)
        self._current_error = error
        self._current_user_feedback = list(user_feedback or [])

    def _has_tool_errors(self) -> bool:
        """Check if conversation history contains tool error messages."""
        if not self._current_history:
            return False
        for msg in self._current_history:
            if msg.role == "tool" and msg.content:
                content = msg.content if isinstance(msg.content, str) else ""
                if content.startswith("Error:") or "Path traversal" in content:
                    return True
        return False

    def _extract_tool_errors(self) -> List[str]:
        """Extract error messages from tool responses."""
        errors = []
        if not self._current_history:
            return errors
        for msg in self._current_history:
            if msg.role == "tool" and msg.content:
                content = msg.content if isinstance(msg.content, str) else ""
                if content.startswith("Error:") or "Path traversal" in content:
                    errors.append(content[:200])
        return errors
