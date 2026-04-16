"""Guardian Agent — output quality review and safety gate.

A lightweight review agent that activates before high-impact operations.
Provides LLM-powered review on top of the static ToolPolicy system.

Activation conditions (any one):
  1. Bash command with high risk level in ToolPolicy
  2. Single response involves ≥N file write/edit operations
  3. Final response after a sequence of risky operations

Responsibilities:
  1. Pre-execution review — catch common mistakes before they happen
  2. Risk annotation — label operation risk level (low/medium/high/critical)
  3. Self-correction suggestions — provide specific fix advice, not just blocks
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omniagent.infra import get_logger
from .llm import LLMMessage

logger = get_logger(__name__)

# High-risk bash patterns (supplementary to ToolPolicy)
_HIGH_RISK_BASH_PATTERNS = [
    (r"\brm\s+(-[rfRF]+\s+)?/([\s;|&]|$)", "Removing root filesystem"),
    (r"\brm\s+(-[rfRF]+\s+)?~([\s;|&]|$)", "Removing home directory"),
    (r"\bsudo\s+", "Elevated privileges"),
    (r"\bchmod\s+[0-7]{3,4}\s+/([\s;|&]|$)", "Changing root permissions"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "System shutdown/reboot"),
    (r"\bdd\s+.*of=/dev/", "Direct device write"),
    (r"\bmkfs\b", "Filesystem formatting"),
    (r"\b(iptables|ufw)\b", "Firewall modification"),
    (r"\b(curl|wget)\b.*\|\s*(bash|sh)\b", "Piped remote script execution"),
    (r"\bpython\s+-c\s+.*(?:import\s+os|subprocess|shutil)", "Dynamic Python execution"),
    (r"\b(git\s+push\s+--force|git\s+reset\s+--hard)\b", "Destructive git operation"),
    (r"\bsed\s+-i\b.*\$\(", "sed -i with subshell (unpredictable scope)"),
    (r"\bfind\b.*-exec\s+rm\b", "find + rm combination"),
]

# Write-type tool names
_WRITE_TOOLS = {"write_file", "edit_file", "save_json", "bash"}


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class ReviewResult:
    """Result of reviewing a pending tool call."""

    risk_level: str = "low"  # "low" | "medium" | "high" | "critical"
    passed: bool = True
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "passed": self.passed,
            "findings": self.findings,
            "suggestions": self.suggestions,
        }


@dataclass
class FinalReviewResult:
    """Result of reviewing the agent's final response."""

    passed: bool = True
    summary: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class OperationRecord:
    """Record of a single operation for session tracking."""

    tool_name: str
    params_summary: str
    risk_level: str = "low"
    had_issues: bool = False


# ── Guardian Agent ─────────────────────────────────────────────────


class GuardianAgent:
    """Output quality review and safety gate agent.

    Activated before high-impact operations. LLM-powered review layer
    on top of static ToolPolicy.
    """

    _SKIP_REVIEW_COMMANDS = {
        'ls', 'find', 'grep', 'cat', 'head', 'tail', 'wc',
        'echo', 'pwd', 'which', 'type', 'file', 'stat',
        'date', 'whoami', 'uname', 'id', 'env', 'printenv',
        'sort', 'uniq', 'cut', 'tr', 'diff',
        'basename', 'dirname', 'realpath', 'readlink',
    }

    def __init__(self, config, main_agent_config):
        """
        Args:
            config: GuardianConfig instance
            main_agent_config: AgentConfig for the main agent (inherit LLM settings)
        """
        self.config = config
        self._main_agent_config = main_agent_config

        # Session state
        self._session_operations: List[OperationRecord] = []
        self._file_write_count: int = 0
        self._file_write_reset_threshold: int = 10  # reset counter after this many non-write ops
        self._non_write_ops: int = 0
        self._has_risky_operations: bool = False

        # Review cache to avoid re-reviewing identical operations
        self._review_cache: Dict[str, ReviewResult] = {}
        self._review_cache_max_size: int = 50

    # ── Activation Detection ────────────────────────────────────

    def should_activate_for_tool_call(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        policy_risk_level: str = "low",
    ) -> Tuple[bool, str]:
        """Check if a tool call should be reviewed by Guardian.

        Args:
            tool_name: Name of the tool being called
            tool_params: Parameters for the tool call
            policy_risk_level: Risk level from ToolPolicy (if available)

        Returns:
            (should_review, reason)
        """
        # 1. High-risk bash always activates
        if tool_name == "bash" and self.config.high_risk_bash_always_activate:
            cmd = tool_params.get("command", "")

            # Skip review for simple read-only commands
            if self._is_read_only_command(cmd):
                return False, ""

            static_risk = self._check_static_risk(cmd)
            if static_risk or policy_risk_level in ("high", "critical"):
                reason = f"high_risk_bash(policy={policy_risk_level}, static={static_risk})"
                return True, reason

            # Also activate when executing a script file — the command itself
            # may look benign but the script contents could be dangerous
            if self._extract_script_paths(cmd):
                reason = "bash_executes_script"
                return True, reason

        # 2. Accumulated file writes
        if tool_name in ("write_file", "edit_file", "save_json"):
            self._file_write_count += 1
            if self._file_write_count >= self.config.max_file_writes_before_activate:
                reason = f"file_write_count({self._file_write_count}) >= threshold({self.config.max_file_writes_before_activate})"
                return True, reason
        else:
            self._non_write_ops += 1
            if self._non_write_ops >= self._file_write_reset_threshold:
                self._file_write_count = 0
                self._non_write_ops = 0

        return False, ""

    def should_review_final_response(self) -> bool:
        """Check if the final response should be reviewed."""
        if not self.config.review_final_on_risky:
            return False
        return self._has_risky_operations

    # ── Static Risk Detection ───────────────────────────────────

    def _check_static_risk(self, command: str) -> Optional[str]:
        """Check a bash command against static high-risk patterns."""
        for pattern, description in _HIGH_RISK_BASH_PATTERNS:
            if re.search(pattern, command):
                return description
        return None

    def _is_read_only_command(self, command: str) -> bool:
        """Check if a bash command is a read-only operation that can skip review."""
        cmd = command.strip()
        if not cmd:
            return True
        first_word = cmd.split()[0] if cmd.split() else ''
        base_cmd = first_word.split('/')[-1]
        if base_cmd in self._SKIP_REVIEW_COMMANDS:
            # Even read-only commands can match high-risk patterns (e.g. pipe chains)
            for pattern, _ in _HIGH_RISK_BASH_PATTERNS:
                if re.search(pattern, command):
                    return False
            return True
        return False

    def _cache_key(self, tool_name: str, tool_params: Dict[str, Any]) -> str:
        """Generate a cache key for a tool call review."""
        import hashlib
        params_str = json.dumps(tool_params, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()[:16]

    # ── Script Content Extraction ────────────────────────────────

    _SCRIPT_EXTENSIONS = {".sh", ".bash", ".py", ".rb", ".js", ".ts"}
    _SCRIPT_PATTERNS = [
        r"\bbash\s+(\S+)",
        r"\bsh\s+(\S+)",
        r"\bpython\d*\s+(\S+)",
        r"\bruby\s+(\S+)",
        r"\bnode\s+(\S+)",
        r"\bsource\s+(\S+)",
        r"\.\s+(\S+)",  # dot-source
    ]

    def _extract_script_paths(self, command: str) -> List[str]:
        """Extract script file paths referenced in a bash command."""
        paths = []
        for pattern in self._SCRIPT_PATTERNS:
            for match in re.finditer(pattern, command):
                path = match.group(1)
                # Strip quotes and trailing args (e.g. --flag)
                path = path.strip("'\"")
                if not path.startswith("-") and any(
                    path.endswith(ext) or Path(path).suffix in self._SCRIPT_EXTENSIONS
                    for ext in self._SCRIPT_EXTENSIONS
                ):
                    paths.append(path)
        return paths

    def _read_script_contents(self, command: str, max_bytes: int = 4096) -> Dict[str, str]:
        """Read the contents of script files referenced in a command.

        Args:
            command: Bash command string
            max_bytes: Maximum bytes to read per file

        Returns:
            Dict mapping file path to its contents (empty if file not found)
        """
        contents: Dict[str, str] = {}
        for script_path in self._extract_script_paths(command):
            try:
                p = Path(script_path)
                if not p.is_file():
                    continue
                # Skip files larger than max_bytes
                if p.stat().st_size > max_bytes:
                    contents[script_path] = f"[File too large: {p.stat().st_size} bytes, skipped]"
                    continue
                contents[script_path] = p.read_text(errors="replace")
            except (OSError, PermissionError) as e:
                logger.debug("guardian_script_read_failed", path=script_path, error=str(e))
        return contents

    # ── LLM-Powered Review ──────────────────────────────────────

    async def review(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        context: str,
        llm,
    ) -> ReviewResult:
        """LLM-powered review of a pending tool call.

        Args:
            tool_name: Name of the tool
            tool_params: Tool call parameters
            context: Recent conversation context (truncated)
            llm: LLM provider instance

        Returns:
            ReviewResult with risk assessment and suggestions
        """
        # First, do static risk check
        static_risk = self._check_static_risk(
            tool_params.get("command", "")
        ) if tool_name == "bash" else None

        # Extract script file contents for deeper review
        script_contents = None
        if tool_name == "bash":
            command = tool_params.get("command", "")
            script_contents = self._read_script_contents(command)
            if script_contents:
                logger.debug(
                    "guardian_script_inspection",
                    scripts=list(script_contents.keys()),
                )

        prompt = self._build_review_prompt(tool_name, tool_params, context, static_risk, script_contents)

        # Check review cache
        cache_key = self._cache_key(tool_name, tool_params)
        if cache_key in self._review_cache:
            logger.debug("guardian_review_cache_hit", cache_key=cache_key)
            result = self._review_cache[cache_key]
            self._record_operation(tool_name, tool_params, result)
            return result

        try:
            response = await llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=self.config.temperature,
                max_tokens=self.config.max_review_tokens,
            )
            result = self._parse_review_response(response.content, static_risk)
        except Exception as e:
            logger.warning("guardian_review_failed", error=str(e))
            # On failure, block unless no static risk detected
            result = ReviewResult(
                risk_level="critical" if static_risk else "high",
                passed=False,
                findings=[static_risk] if static_risk else ["Review LLM call failed — blocked as precaution"],
                suggestions=[],
            )

        # Record operation
        self._record_operation(tool_name, tool_params, result)

        # Store in cache (evict oldest if full)
        if len(self._review_cache) >= self._review_cache_max_size:
            oldest_key = next(iter(self._review_cache))
            del self._review_cache[oldest_key]
        self._review_cache[cache_key] = result

        return result

    async def review_final_response(
        self,
        conversation_summary: str,
        llm,
    ) -> FinalReviewResult:
        """Review the agent's final response after risky operations.

        Args:
            conversation_summary: Summary of the conversation and operations
            llm: LLM provider instance

        Returns:
            FinalReviewResult with pass/fail and warnings
        """
        if not self._session_operations:
            return FinalReviewResult(passed=True, summary="No operations to review")

        ops_summary = "\n".join(
            f"- [{op.risk_level}] {op.tool_name}: {op.params_summary}"
            for op in self._session_operations[-10:]  # last 10 operations
        )

        prompt = (
            f"## Final Response Review\n\n"
            f"The agent completed the following operations:\n{ops_summary}\n\n"
            f"Summary of the agent's final response:\n{conversation_summary}\n\n"
            f"Review for any remaining issues. Answer with JSON:\n"
            f'{{"passed": true/false, "summary": "...", "warnings": ["..."]}}\n'
        )

        try:
            response = await llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=512,
            )
            return self._parse_final_review(response.content)
        except Exception as e:
            logger.warning("guardian_final_review_failed", error=str(e))
            return FinalReviewResult(
                passed=False,
                summary="Final review LLM call failed — blocked as precaution",
                warnings=[],
            )

    # ── Prompt Building ─────────────────────────────────────────

    def _build_review_prompt(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        context: str,
        static_risk: Optional[str],
        script_contents: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build the review prompt for the LLM."""
        # Truncate context to fit budget
        max_context = self.config.max_review_tokens - 300  # reserve for prompt
        if len(context) > max_context * 4:
            context = context[-(max_context * 4):]

        prompt_parts = [
            "## Operation Review\n",
            f"Review the following operation before it executes:\n",
            f"**Tool:** `{tool_name}`\n",
            f"**Parameters:**\n```json\n{json.dumps(tool_params, ensure_ascii=False, indent=2)}\n```\n",
        ]

        if static_risk:
            prompt_parts.append(f"**Static risk detected:** {static_risk}\n")

        if script_contents:
            prompt_parts.append("\n**Script file contents:**\n")
            for path, content in script_contents.items():
                prompt_parts.append(f"--- `{path}` ---\n```\n{content}\n```\n")
            prompt_parts.append(
                "\nIMPORTANT: Carefully review the script contents above. "
                "The command executes these scripts — check for hidden dangerous operations "
                "that are not obvious from the command alone.\n\n"
                "HIGH-RISK PATTERNS that MUST be rated 'critical' if found in scripts:\n"
                "- `find ... -exec rm -rf` or `find ... | xargs rm` (broad recursive deletion)\n"
                "- `rm -rf` with variable or glob expansion (e.g., rm -rf $DIR/*, rm -rf *)\n"
                "- Destructive loops that delete or overwrite files without confirmation\n"
                "- `sudo` combined with any destructive operation\n"
                "- Data exfiltration (curl/wget sending local files to remote servers)\n\n"
                "If any of these patterns are present, set risk_level to 'critical' and passed to false.\n"
            )

        prompt_parts.append(
            f"\n**Recent context:**\n{context}\n\n"
            "Check for:\n"
            "1. Common mistakes (typos, wrong paths, missing imports)\n"
            "2. Unintended side effects\n"
            "3. Security concerns\n\n"
            "Respond with JSON:\n"
            '{"risk_level": "low|medium|high|critical", '
            '"passed": true/false, '
            '"findings": ["..."], '
            '"suggestions": ["..."]}\n'
        )

        return "".join(prompt_parts)

    # ── Response Parsing ────────────────────────────────────────

    def _parse_review_response(
        self, response_text: str, static_risk: Optional[str]
    ) -> ReviewResult:
        """Parse LLM review response into ReviewResult."""
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                risk = data.get("risk_level", "medium")
                passed = bool(data.get("passed", True))

                # If static risk detected, never pass
                if static_risk:
                    passed = False
                    if risk in ("low", "medium"):
                        risk = "high"

                # Auto-block critical if configured
                if risk == "critical" and self.config.auto_block_on_critical:
                    passed = False

                return ReviewResult(
                    risk_level=risk,
                    passed=passed,
                    findings=data.get("findings", []),
                    suggestions=data.get("suggestions", []),
                )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("guardian_review_parse_failed", error=str(e))

        return ReviewResult(
            risk_level="critical" if static_risk else "medium",
            passed=not bool(static_risk),
            findings=[static_risk] if static_risk else ["Parse failed"],
            suggestions=[],
        )

    def _parse_final_review(self, response_text: str) -> FinalReviewResult:
        """Parse LLM final review response."""
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return FinalReviewResult(
                    passed=bool(data.get("passed", True)),
                    summary=data.get("summary", ""),
                    warnings=data.get("warnings", []),
                )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("guardian_final_parse_failed", error=str(e))

        return FinalReviewResult(passed=True, summary="Parse failed, allowing")

    # ── Session Tracking ────────────────────────────────────────

    def _record_operation(
        self, tool_name: str, tool_params: Dict[str, Any], result: ReviewResult
    ) -> None:
        """Record an operation for session-level tracking."""
        params_summary = str(tool_params)[:200]

        if result.risk_level in ("high", "critical"):
            self._has_risky_operations = True

        self._session_operations.append(OperationRecord(
            tool_name=tool_name,
            params_summary=params_summary,
            risk_level=result.risk_level,
            had_issues=not result.passed,
        ))

        # Keep session operations bounded
        if len(self._session_operations) > 100:
            self._session_operations = self._session_operations[-50:]

    def get_session_summary(self) -> str:
        """Get a summary of session operations."""
        if not self._session_operations:
            return ""

        risky = sum(1 for op in self._session_operations if op.risk_level in ("high", "critical"))
        blocked = sum(1 for op in self._session_operations if op.had_issues)

        return (
            f"[Guardian] Session: {len(self._session_operations)} ops, "
            f"{risky} risky, {blocked} blocked"
        )

    def reset(self) -> None:
        """Reset Guardian state for a new session."""
        self._session_operations.clear()
        self._file_write_count = 0
        self._non_write_ops = 0
        self._has_risky_operations = False
        self._review_cache.clear()
