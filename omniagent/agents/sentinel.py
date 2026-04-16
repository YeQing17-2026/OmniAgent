"""Sentinel Agent — task decomposition and progress tracking.

A lightweight planning agent that activates for complex, multi-step tasks.
Does NOT execute any tools or make code changes. Pure planning + verification
role using lightweight LLM calls.

Activation conditions (any one):
  1. Task description contains ≥N multi-step keyword occurrences
     ("then"/"after that"/"next"/"然后"/"接着"/"之后"/"最后")
  2. Main agent has ≥N consecutive reflexion failures
  3. Bash operations span ≥N distinct directories
  4. LLM complexity estimation judges the task needs planning

Responsibilities:
  1. Task decomposition → structured milestone plan
  2. Progress tracking → verify each milestone completion
  3. Cross-session recovery → persist plans to .omniagent/sentinel/
"""

import json
import re
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omniagent.infra import get_logger
from omniagent.agents.llm import LLMMessage

logger = get_logger(__name__)

# Multi-step keywords (bilingual)
_MULTI_STEP_KEYWORDS_ZH = re.compile(r"(?:然后|接着|之后|最后|下一步|再|随后|继而)", re.UNICODE)
_MULTI_STEP_KEYWORDS_EN = re.compile(r"\b(?:then|after\s+that|next|afterwards|finally|afterward|subsequently)\b", re.IGNORECASE)

# Path-like patterns for bash directory extraction
_DIR_PATTERN = re.compile(r"(?:cd\s+|(?:mkdir|touch|rm|cp|mv|cat|sed|python|node|pip|npm)\s+['\"]?)([^'\"\s;|&]{2,200})")


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class Milestone:
    """A single milestone in a task plan."""

    index: int
    description: str
    dependencies: List[int] = field(default_factory=list)  # indices of prerequisites
    success_criteria: str = ""
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"
    result_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        return cls(**data)


@dataclass
class TaskPlan:
    """A complete task decomposition plan."""

    task_hash: str
    task_description: str
    milestones: List[Milestone]
    created_at: str
    updated_at: str
    current_milestone_idx: int = 0
    status: str = "active"  # "active" | "completed" | "abandoned"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        milestones = [Milestone.from_dict(m) for m in data["milestones"]]
        return cls(
            task_hash=data["task_hash"],
            task_description=data["task_description"],
            milestones=milestones,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            current_milestone_idx=data.get("current_milestone_idx", 0),
            status=data.get("status", "active"),
        )


@dataclass
class VerificationResult:
    """Result of verifying a milestone's completion."""

    passed: bool
    feedback: str
    suggested_adjustment: Optional[str] = None


# ── Activation Detection ───────────────────────────────────────────


def _count_multi_step_keywords(text: str) -> int:
    """Count multi-step keyword occurrences in task text."""
    zh_count = len(_MULTI_STEP_KEYWORDS_ZH.findall(text))
    en_count = len(_MULTI_STEP_KEYWORDS_EN.findall(text))
    return zh_count + en_count


def _extract_bash_directories(tool_calls: List[Dict[str, Any]]) -> set:
    """Extract distinct directory paths from bash tool calls."""
    dirs = set()
    for tc in tool_calls:
        if tc.get("tool_name") == "bash":
            cmd = tc.get("params", {}).get("command", "")
            for match in _DIR_PATTERN.findall(cmd):
                p = Path(match)
                if p.parts:
                    dirs.add(str(p.parent) if p.name else str(p))
    return dirs


# ── Sentinel Agent ─────────────────────────────────────────────────


class SentinelAgent:
    """Task decomposition and progress tracking agent.

    Activated for complex, multi-step tasks. Does NOT execute tools.
    Pure planning + verification role using lightweight LLM calls.
    """

    def __init__(
        self,
        config,
        main_agent_config,
        work_dir: str,
    ):
        """
        Args:
            config: SentinelConfig instance
            main_agent_config: AgentConfig for the main agent (inherit LLM settings)
            work_dir: Working directory for file operations
        """
        self.config = config
        self._main_agent_config = main_agent_config
        self._persist_path = Path(work_dir) / config.plan_persist_path
        self._persist_path.mkdir(parents=True, exist_ok=True)

        # Active plan for current session
        self._active_plan: Optional[TaskPlan] = None
        self._reflexion_failure_count: int = 0
        self._bash_dirs_seen: set = set()

    # ── Activation ────────────────────────────────────────────────

    def should_activate(
        self,
        task: str,
        reflexion_failure_count: int = 0,
        recent_bash_dirs: Optional[set] = None,
    ) -> Tuple[bool, str]:
        """Check activation conditions.

        Args:
            task: The user's task description
            reflexion_failure_count: Consecutive reflexion failures
            recent_bash_dirs: Set of directories seen in recent bash operations

        Returns:
            (should_activate, reason)
        """
        # 1. Multi-step keyword detection
        keyword_count = _count_multi_step_keywords(task)
        if keyword_count >= self.config.multi_step_keyword_threshold:
            reason = (
                f"multi_step_keywords({keyword_count})"
                f" >= threshold({self.config.multi_step_keyword_threshold})"
            )
            logger.info("sentinel_activated", trigger=reason)
            return True, reason

        # 2. Consecutive reflexion failures
        if reflexion_failure_count >= self.config.max_reflexion_failures_before_activate:
            reason = (
                f"reflexion_failures({reflexion_failure_count})"
                f" >= threshold({self.config.max_reflexion_failures_before_activate})"
            )
            logger.info("sentinel_activated", trigger=reason)
            return True, reason

        # 3. Bash directory spread
        if recent_bash_dirs and len(recent_bash_dirs) >= self.config.bash_dir_threshold:
            reason = (
                f"bash_dir_count({len(recent_bash_dirs)})"
                f" >= threshold({self.config.bash_dir_threshold})"
            )
            logger.info("sentinel_activated", trigger=reason)
            return True, reason

        return False, ""

    async def should_activate_with_llm(
        self,
        task: str,
        llm,
        reflexion_failure_count: int = 0,
        recent_bash_dirs: Optional[set] = None,
        skills_summary: str = "",
    ) -> Tuple[bool, str]:
        """Check activation conditions, falling back to LLM complexity estimation.

        First runs the synchronous rule-based checks (keywords, failures, dirs).
        If none trigger and LLM complexity estimation is enabled, makes a
        lightweight LLM call to judge whether the task needs planning.
        The LLM is informed about available skills so it can recognize when
        a skill already handles the task (no plan needed).

        Args:
            task: The user's task description
            llm: LLM provider instance (for complexity estimation)
            reflexion_failure_count: Consecutive reflexion failures
            recent_bash_dirs: Set of directories seen in recent bash operations
            skills_summary: Compact list of available skills for context

        Returns:
            (should_activate, reason)
        """
        # Try rule-based activation first (no LLM cost)
        activated, reason = self.should_activate(
            task,
            reflexion_failure_count=reflexion_failure_count,
            recent_bash_dirs=recent_bash_dirs,
        )
        if activated:
            return True, reason

        # Fall back to LLM complexity estimation
        if self.config.llm_complexity_enabled:
            try:
                needs_plan, complexity_reason = await self._estimate_complexity(
                    task, llm, skills_summary=skills_summary,
                )
                if needs_plan:
                    logger.info("sentinel_activated", trigger=f"llm_complexity: {complexity_reason}")
                    return True, f"llm_complexity: {complexity_reason}"
            except Exception as e:
                logger.warning("sentinel_complexity_estimation_failed", error=str(e))

        return False, ""

    async def _estimate_complexity(
        self, task: str, llm, skills_summary: str = ""
    ) -> Tuple[bool, str]:
        """Use a lightweight LLM call to estimate if a task needs planning.

        Args:
            task: The user's task description
            llm: LLM provider instance

        Returns:
            (needs_plan, reason)
        """
        skills_context = ""
        if skills_summary:
            skills_context = (
                "\nAvailable skills that can handle tasks directly (no plan needed if a skill matches):\n"
                f"{skills_summary}\n"
                "If the task matches one of these skills, answer needs_plan: false.\n\n"
            )

        prompt = (
            "Judge whether the following task requires a step-by-step plan to complete efficiently. "
            "Consider: does it involve multiple files, multiple steps, code exploration, "
            "analysis, refactoring, debugging, or any non-trivial workflow?\n\n"
            "Tasks that do NOT need a plan (answer needs_plan: false):\n"
            "- Running a single skill or command\n"
            "- Reading or searching files\n"
            "- Answering a question\n"
            "- Making a single edit or write\n"
            "- Any task that can be completed in 1-3 tool calls\n"
            "- Any task that matches an available skill listed below\n\n"
            f"{skills_context}"
            f"Task: {task}\n\n"
            'Answer with JSON only: {"needs_plan": true, "reason": "short explanation"} '
            'or {"needs_plan": false, "reason": "short explanation"}'
        )

        response = await llm.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=128,
        )

        return self._parse_complexity_response(response.content)

    def _parse_complexity_response(self, response_text: str) -> Tuple[bool, str]:
        """Parse LLM complexity estimation response.

        Args:
            response_text: Raw LLM response text

        Returns:
            (needs_plan, reason) — defaults to (False, "") on parse failure
        """
        try:
            decoder = json.JSONDecoder()
            idx = response_text.find("{")
            if idx == -1:
                raise ValueError("no JSON object found in response")
            data, _ = decoder.raw_decode(response_text, idx)
            needs_plan = bool(data.get("needs_plan", False))
            reason = str(data.get("reason", ""))
            return needs_plan, reason
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("sentinel_complexity_parse_failed", error=str(e))

        return False, ""

    # ── Plan Persistence ──────────────────────────────────────────

    def _plan_file(self, task_hash: str) -> Path:
        return self._persist_path / f"plan_{task_hash}.json"

    @staticmethod
    def _hash_task(task: str) -> str:
        return hashlib.sha256(task.encode()).hexdigest()[:12]

    def save_plan(self, plan: TaskPlan) -> None:
        """Persist plan to disk."""
        plan.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._plan_file(plan.task_hash)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug("sentinel_plan_saved", task_hash=plan.task_hash, path=str(path))

    def load_plan(self, task: str) -> Optional[TaskPlan]:
        """Load existing plan for a task (for cross-session recovery)."""
        task_hash = self._hash_task(task)
        path = self._plan_file(task_hash)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            plan = TaskPlan.from_dict(data)
            if plan.status == "active":
                logger.info("sentinel_plan_recovered", task_hash=task_hash,
                            milestone=f"{plan.current_milestone_idx}/{len(plan.milestones)}")
                return plan
        except Exception as e:
            logger.warning("sentinel_plan_load_failed", error=str(e))
        return None

    def archive_completed_plan(self, plan: TaskPlan) -> None:
        """Archive a completed/abandoned plan."""
        plan.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._plan_file(plan.task_hash)
        archive_path = self._persist_path / "archive"
        archive_path.mkdir(exist_ok=True)
        target = archive_path / path.name
        path.rename(target)
        logger.debug("sentinel_plan_archived", task_hash=plan.task_hash)

    # ── Decomposition ─────────────────────────────────────────────

    async def decompose(self, task: str, llm, skills_summary: str = "") -> TaskPlan:
        """Break task into ordered milestones via LLM.

        Args:
            task: The original user task
            llm: LLM provider instance (shared from main agent)
            skills_summary: Compact list of available skills for context

        Returns:
            TaskPlan with milestones
        """
        # Cap max milestones based on available iterations
        effective_max = min(
            self.config.max_milestones,
            max(2, int(self._main_agent_config.max_iterations / self.config.milestone_iteration_ratio))
        )
        max_ms = effective_max

        skills_context = ""
        if skills_summary:
            skills_context = (
                f"\nAvailable skills that can be used directly (reference by name in milestone descriptions):\n"
                f"{skills_summary}\n"
                f"If a milestone can be completed by using a skill, mention the skill name "
                f"in the description (e.g., 'Use skill-xxx to ...'). "
                f"Do NOT decompose skill-handled work into manual steps.\n"
            )

        prompt = (
            f"## Task Decomposition\n\n"
            f"Break the following task into sequential milestones (max {max_ms}).\n"
            f"Prefer FEWER, larger milestones over many small ones. Each milestone should "
            f"represent a meaningful unit of progress, not a single tool call.\n\n"
            f"{skills_context}\n"
            f"For each milestone, provide:\n"
            f"- description: what needs to be done\n"
            f"- success_criteria: how to verify it's done\n"
            f"- dependencies: indices of prerequisite milestones (empty list if none)\n\n"
            f"Task:\n{task}\n\n"
            f"Respond with a JSON array of milestones. Example:\n"
            f'[{{"description": "...", "success_criteria": "...", "dependencies": []}}]\n'
        )

        response = await llm.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=self.config.temperature,
            max_tokens=self.config.compile_max_tokens,
        )

        milestones = self._parse_milestones(response.content, task, max_ms)

        now = datetime.now(timezone.utc).isoformat()
        plan = TaskPlan(
            task_hash=self._hash_task(task),
            task_description=task,
            milestones=milestones,
            created_at=now,
            updated_at=now,
        )

        self._active_plan = plan
        self.save_plan(plan)
        logger.info("sentinel_plan_created",
                     milestone_count=len(milestones),
                     task_hash=plan.task_hash)

        # Mark first milestone as in_progress
        if milestones:
            milestones[0].status = "in_progress"
            self.save_plan(plan)

        return plan

    def _parse_milestones(
        self, response_text: str, task: str, max_milestones: int
    ) -> List[Milestone]:
        """Parse LLM response into Milestone objects."""
        milestones = []

        # Try JSON extraction
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                for i, item in enumerate(items[:max_milestones]):
                    if isinstance(item, dict):
                        ms = Milestone(
                            index=i,
                            description=item.get("description", f"Milestone {i+1}"),
                            success_criteria=item.get("success_criteria", ""),
                            dependencies=item.get("dependencies", []),
                        )
                        milestones.append(ms)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("sentinel_milestone_parse_failed", error=str(e))

        # Fallback: if parsing failed, create a single milestone
        if not milestones:
            milestones.append(Milestone(
                index=0,
                description=task,
                success_criteria="Task completed as described",
                dependencies=[],
            ))

        return milestones

    # ── Progress Verification ─────────────────────────────────────

    def get_current_milestone(self) -> Optional[Milestone]:
        """Get the current in-progress milestone."""
        if not self._active_plan:
            return None
        plan = self._active_plan
        for ms in plan.milestones:
            if ms.status == "in_progress":
                return ms
        return None

    async def verify_milestone(
        self, milestone: Milestone, execution_result: str, llm
    ) -> VerificationResult:
        """Verify if a milestone was completed successfully.

        Args:
            milestone: The milestone to verify
            execution_result: Summary of what was done
            llm: LLM provider instance

        Returns:
            VerificationResult with pass/fail and feedback
        """
        prompt = (
            f"## Milestone Verification\n\n"
            f"Milestone: {milestone.description}\n"
            f"Success criteria: {milestone.success_criteria}\n\n"
            f"Execution result:\n{execution_result}\n\n"
            f"Was this milestone completed successfully? "
            f"Answer with JSON:\n"
            f'{{"passed": true/false, "feedback": "...", "suggested_adjustment": "..." or null}}\n'
        )

        response = await llm.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=512,
        )

        return self._parse_verification(response.content)

    def _parse_verification(self, response_text: str) -> VerificationResult:
        """Parse LLM verification response."""
        try:
            decoder = json.JSONDecoder()
            idx = response_text.find("{")
            if idx == -1:
                raise ValueError("no JSON object found in response")
            data, _ = decoder.raw_decode(response_text, idx)
            return VerificationResult(
                    passed=bool(data.get("passed", True)),
                    feedback=data.get("feedback", ""),
                    suggested_adjustment=data.get("suggested_adjustment"),
                )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("sentinel_verification_parse_failed", error=str(e))

        return VerificationResult(passed=True, feedback="Parse failed, assuming passed")

    def mark_milestone_completed(self, milestone: Milestone, result_summary: str = "") -> None:
        """Mark a milestone as completed and advance to the next."""
        milestone.status = "completed"
        milestone.result_summary = result_summary

        if not self._active_plan:
            return

        plan = self._active_plan
        plan.current_milestone_idx = milestone.index + 1

        # Advance to next pending milestone
        for ms in plan.milestones[milestone.index + 1:]:
            if ms.status == "pending":
                ms.status = "in_progress"
                break

        # Check if all milestones are done
        if all(ms.status == "completed" for ms in plan.milestones):
            plan.status = "completed"
            self.archive_completed_plan(plan)
        else:
            self.save_plan(plan)

    def mark_milestone_failed(self, milestone: Milestone) -> None:
        """Mark a milestone as failed."""
        milestone.status = "failed"
        if self._active_plan:
            self.save_plan(self._active_plan)

    # ── Plan Summary ──────────────────────────────────────────────

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        if not self._active_plan:
            return ""

        plan = self._active_plan
        completed = sum(1 for ms in plan.milestones if ms.status == "completed")
        total = len(plan.milestones)

        lines = [
            f"[Sentinel] Progress: {completed}/{total} milestones",
        ]
        for ms in plan.milestones:
            icon = {"completed": "+", "in_progress": ">", "failed": "!", "pending": "-"}
            lines.append(
                f"  [{icon.get(ms.status, '?')}] {ms.description}"
                + (f" — {ms.result_summary}" if ms.result_summary else "")
            )

        return "\n".join(lines)

    @property
    def is_active(self) -> bool:
        """Whether Sentinel has an active plan."""
        return self._active_plan is not None and self._active_plan.status == "active"

    def reset(self) -> None:
        """Reset Sentinel state for a new task."""
        self._active_plan = None
        self._reflexion_failure_count = 0
        self._bash_dirs_seen = set()
