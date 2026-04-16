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

    def __init__(self, learnings_dir: Path, max_learnings: int = 100):
        self.learnings_dir = learnings_dir
        self.max_learnings = max_learnings
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
        lesson_hash = self._compute_hash(category, lesson_text)

        if lesson_hash in self._index:
            # Increment evidence for existing lesson
            existing = self._index[lesson_hash]
            existing.evidence += 1
            self._rewrite_file()
            logger.debug("lesson_evidence_incremented", hash=lesson_hash, evidence=existing.evidence)
            return existing
        else:
            # New lesson
            new_lesson = Lesson(
                timestamp=datetime.now(timezone.utc).isoformat(),
                lesson_hash=lesson_hash,
                source=source,
                category=category,
                lesson=lesson_text,
                context=context[:500] if context else "",
                evidence=1,
            )
            self._index[lesson_hash] = new_lesson
            self._prune_if_needed()
            with open(self._lessons_file, "a", encoding="utf-8") as f:
                f.write(new_lesson.to_jsonl() + "\n")
            logger.info("lesson_recorded", hash=lesson_hash, source=source, category=category)
            return new_lesson

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
    ):
        self.recorder = recorder
        self.work_dir = work_dir
        self.min_evidence = min_evidence
        self.max_agents_rules = max_agents_rules
        self.user_feedback_min_evidence = user_feedback_min_evidence

    def check_and_promote(self) -> List[PromotedRule]:
        """Check for lessons ready for promotion and promote them."""
        candidates = self.recorder.get_candidates(
            min_evidence=self.min_evidence,
            user_feedback_min_evidence=self.user_feedback_min_evidence,
        )
        if not candidates:
            return []

        # Sort by evidence (most confirmed first)
        candidates.sort(key=lambda l: -l.evidence)

        promoted = []
        agents_path = self._find_agents_md()
        if not agents_path:
            logger.debug("no_agents_md_found", work_dir=str(self.work_dir))
            return []

        current_rules = self._read_current_rules(agents_path)
        remaining_slots = self.max_agents_rules - len(current_rules)

        for candidate in candidates[:remaining_slots]:
            if self._is_duplicate(candidate.lesson, current_rules):
                # Already a similar rule in AGENTS.md
                self.recorder.mark_promoted(candidate.lesson_hash)
                continue

            rule = self._format_rule(candidate)
            if not rule:
                continue

            promoted.append(rule)

        if promoted:
            self._append_rules_to_agents_md(agents_path, promoted)
            for rule in promoted:
                # Find the corresponding lesson hash
                self.recorder.mark_promoted(rule._lesson_hash)

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

    def _is_duplicate(self, lesson: str, existing_rules: List[str]) -> bool:
        """Check if a lesson is already captured by an existing rule."""
        lesson_lower = lesson.lower()
        for rule in existing_rules:
            rule_lower = rule.lower()
            # Check significant word overlap
            lesson_words = set(lesson_lower.split())
            rule_words = set(rule_lower.replace("-", "").split())
            if len(lesson_words) > 3 and len(rule_words) > 3:
                overlap = lesson_words & rule_words
                if len(overlap) / min(len(lesson_words), len(rule_words)) > 0.6:
                    return True
        return False

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

    def _append_rules_to_agents_md(self, agents_path: Path, rules: List[PromotedRule]) -> None:
        """Append learned rules section to AGENTS.md."""
        content = agents_path.read_text(encoding="utf-8")

        # Build rules text
        rules_lines = []
        for rule in rules:
            category_tag = f"[{rule.category}]"
            rules_lines.append(f"- {category_tag} {rule.rule}")

        rules_text = "\n".join(rules_lines)

        # Check if section already exists
        if self.LEARNED_RULES_SECTION in content:
            # Append to existing section
            content = re.sub(
                rf"({re.escape(self.LEARNED_RULES_SECTION)}\n)",
                rf"\1{rules_text}\n",
                content,
            )
        else:
            # Create new section at the end
            section = f"\n{self.LEARNED_RULES_SECTION}\n{rules_text}\n"
            content = content.rstrip() + section

        agents_path.write_text(content, encoding="utf-8")
        logger.info("rules_promoted_to_agents_md", count=len(rules))


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
        )
        self.extractor = LessonExtractor(
            llm_provider=llm_provider,
            max_tokens=config.compile_max_tokens,
        )
        self.analyzer = LessonAnalyzer(
            recorder=self.recorder,
            work_dir=self.work_dir,
            min_evidence=config.lesson_min_evidence,
            max_agents_rules=config.max_agents_rules,
            user_feedback_min_evidence=config.user_feedback_min_evidence,
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
                promoted = self.analyzer.check_and_promote()
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
