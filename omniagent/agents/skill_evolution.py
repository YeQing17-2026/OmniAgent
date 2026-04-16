"""Skill self-evolution system for OmniAgent.

Two directions:
1. Skill Creation: Detect high-frequency successful patterns and compile into trial skills.
2. Skill Evolution: Detect error-recovery pairs in skill-guided execution and write patches.
"""

import hashlib
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
class ExecutionPattern:
    """A single recorded execution pattern (one JSONL line)."""

    timestamp: str
    task: str
    tool_sequence: List[str]
    tool_signatures: List[str]  # parameter-aware signatures for hash grouping
    tool_details: List[Dict[str, Any]]
    success: bool
    iterations: int
    duration_s: float
    active_skills: List[str]
    pattern_hash: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class SkillPatch:
    """A correction to be applied to an existing skill."""

    skill_name: str
    timestamp: str
    original_context: str
    error_description: str
    correction: str
    task_description: str

    def to_markdown(self) -> str:
        return (
            f"---\n"
            f"skill: {self.skill_name}\n"
            f"timestamp: {self.timestamp}\n"
            f"---\n\n"
            f"# Skill Patch: {self.skill_name}\n\n"
            f"## Task Context\n{self.task_description}\n\n"
            f"## Original Approach (from skill)\n{self.original_context}\n\n"
            f"## What Went Wrong\n{self.error_description}\n\n"
            f"## What Worked Instead\n{self.correction}\n"
        )


@dataclass
class CompiledSkill:
    """A candidate skill generated from pattern compilation."""

    name: str
    description: str
    content: str
    source_patterns: int
    confidence: float
    skill_type: str = "prompt"  # "prompt" or "script"
    script_content: Optional[str] = None  # Auto-generated main.sh for script-type skills


# ── Pattern Recorder ───────────────────────────────────────────────


class PatternRecorder:
    """Records successful execution patterns to append-only JSONL."""

    def __init__(self, patterns_dir: Path, max_file_size_mb: float = 50.0, work_dir: Optional[Path] = None):
        self.patterns_dir = patterns_dir
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = self.patterns_dir / "patterns.jsonl"
        self._work_dir = Path(work_dir) if work_dir else None

    async def record(
        self,
        task: str,
        conversation_history: list,
        tool_name_history: List[str],
        success: bool,
        iterations: int,
        duration_s: float,
        active_skills: Optional[List[str]] = None,
    ) -> Optional[ExecutionPattern]:
        """Record a pattern if the execution was successful."""
        if not success:
            return None

        tool_details = self._extract_tool_details(conversation_history)
        if not tool_details:
            return None

        tool_sequence = [d["name"] for d in tool_details]
        tool_signatures = [d["signature"] for d in tool_details]

        # Strip exploratory prefixes (ls, find, etc.) — they vary between runs
        # but don't represent the core pattern. Keep only the substantive operations.
        core_signatures = [
            s for s in tool_signatures
            if s not in self._EXPLORATORY_TOOLS
        ]
        if not core_signatures:
            return None

        pattern_hash = hashlib.sha256(
            ",".join(core_signatures).encode()
        ).hexdigest()[:16]

        pattern = ExecutionPattern(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=task[:500],
            tool_sequence=tool_sequence,
            tool_signatures=tool_signatures,
            tool_details=tool_details,
            success=True,
            iterations=iterations,
            duration_s=round(duration_s, 2),
            active_skills=active_skills or [],
            pattern_hash=pattern_hash,
        )

        self._append(pattern)
        return pattern

    def _extract_tool_details(self, history: list) -> List[Dict[str, Any]]:
        """Extract tool call details from conversation history.

        Each detail includes a parameter-aware 'signature' for hash grouping,
        so that different bash commands (e.g. cat vs rm) produce different
        pattern hashes even though the tool name is the same.
        """
        details = []
        for msg in history:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        params_keys = sorted(args.keys())
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                        params_keys = []
                    signature = self._compute_tool_signature(name, args, self._work_dir)
                    # Store truncated params for compilation context
                    params_summary = {}
                    for k, v in args.items():
                        v_str = str(v)
                        if len(v_str) > 150:
                            v_str = v_str[:150] + "..."
                        params_summary[k] = v_str
                    details.append({
                        "name": name,
                        "params_keys": params_keys,
                        "params_summary": params_summary,
                        "signature": signature,
                    })
        return details

    # Tools used for exploration/discovery — stripped from signature
    _EXPLORATORY_TOOLS = {"ls", "find", "bash:ls", "bash:find", "bash:tree", "bash:pwd"}

    @staticmethod
    def _compute_tool_signature(tool_name: str, args: Dict[str, Any], work_dir: Optional[Path] = None) -> str:
        """Compute a parameter-aware signature for a tool call.

        Rules:
        - bash: extract the command verb (first word after shell meta-chars)
        - read_file / write_file / edit_file: extract relative parent directory,
          with variable parts (dates, numbers) replaced by placeholders
        - other tools: use tool name as-is
        """
        if tool_name == "bash":
            command = args.get("command", "").strip()
            # Skip shell operators and find the actual command verb
            # Handles: "cat foo", "cd dir && ls", "sudo apt install"
            cmd = command
            for prefix in ("sudo ", "time ", "nice ", "nohup "):
                if cmd.startswith(prefix):
                    cmd = cmd[len(prefix):]

            # For pipes/chains, take the first command verb
            for separator in ("|", ";", "&&", "||"):
                if separator in cmd:
                    cmd = cmd.split(separator)[0].strip()

            # Extract the first word (the actual command)
            verb = cmd.split()[0] if cmd.split() else ""
            # Get basename (strip directory path like /usr/bin/cat → cat)
            verb = verb.split("/")[-1] if verb else ""
            return f"bash:{verb}" if verb else "bash"

        elif tool_name in ("read_file", "write_file", "edit_file"):
            path = args.get("path", "")
            if path:
                # Normalize path: make relative to work_dir
                p = Path(path)
                if work_dir and not p.is_absolute():
                    # Already relative
                    rel = p
                elif work_dir:
                    try:
                        rel = p.relative_to(work_dir)
                    except ValueError:
                        rel = p
                else:
                    rel = p

                # Get parent directory path as string
                parent_str = str(rel.parent)

                # Replace variable parts in path segments (dates, numbers)
                # e.g., "data/2026-04-01" → "data/$DATE", "logs/2026-04-01" → "logs/$DATE"
                import re
                parts = parent_str.split("/")
                normalized_parts = []
                for part in parts:
                    # Date pattern: YYYY-MM-DD or YYYYMMDD
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', part) or re.match(r'^\d{8}$', part):
                        normalized_parts.append("$DATE")
                    # Pure numeric segment (e.g., "12345" pid dirs)
                    elif re.match(r'^\d+$', part):
                        normalized_parts.append("$NUM")
                    else:
                        normalized_parts.append(part)
                normalized_parent = "/".join(normalized_parts)
                return f"{tool_name}:{normalized_parent}"
            return tool_name

        else:
            return tool_name

    def _append(self, pattern: ExecutionPattern) -> None:
        """Append pattern to JSONL with size guard."""
        self._prune_if_needed()
        with open(self._patterns_file, "a", encoding="utf-8") as f:
            f.write(pattern.to_jsonl() + "\n")

    def _prune_if_needed(self) -> None:
        """Remove oldest 20% of entries if file exceeds size limit."""
        if not self._patterns_file.exists():
            return
        if self._patterns_file.stat().st_size <= self.max_file_size_bytes:
            return

        logger.info("patterns_pruning", file=str(self._patterns_file))
        lines = self._patterns_file.read_text(encoding="utf-8").strip().split("\n")
        keep_count = int(len(lines) * 0.8)
        if keep_count < len(lines):
            self._patterns_file.write_text(
                "\n".join(lines[-keep_count:]) + "\n", encoding="utf-8"
            )

    def count_by_hash(self, pattern_hash: str) -> int:
        """Count occurrences of a pattern hash."""
        if not self._patterns_file.exists():
            return 0
        count = 0
        search = f'"pattern_hash": "{pattern_hash}"'
        with open(self._patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                if search in line:
                    count += 1
        return count

    def get_patterns_by_hash(self, pattern_hash: str, limit: int = 10) -> List[Dict]:
        """Retrieve full pattern records matching a hash."""
        if not self._patterns_file.exists():
            return []
        results = []
        search = f'"pattern_hash": "{pattern_hash}"'
        with open(self._patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                if search in line:
                    results.append(json.loads(line.strip()))
                    if len(results) >= limit:
                        break
        return results

    def remove_patterns_by_hash(self, pattern_hash: str) -> int:
        """Remove all pattern records matching a hash from the JSONL file.

        Called after a skill is successfully compiled from these patterns,
        to prevent re-compilation on future runs.

        Returns the number of removed records.
        """
        if not self._patterns_file.exists():
            return 0

        search = f'"pattern_hash": "{pattern_hash}"'
        kept_lines = []
        removed = 0
        with open(self._patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                if search in line:
                    removed += 1
                else:
                    kept_lines.append(line)

        if removed > 0:
            self._patterns_file.write_text(
                "".join(kept_lines), encoding="utf-8"
            )
            logger.info("patterns_removed", hash=pattern_hash, count=removed)

        return removed


# ── Pattern Analyzer ───────────────────────────────────────────────


class PatternAnalyzer:
    """Analyzes patterns for repetition and triggers skill compilation."""

    _TOOL_NAME_MAP = {
        "read_file": "read", "write_file": "write", "edit_file": "edit",
        "bash": "shell", "grep": "search", "find": "locate",
        "web_search": "websearch", "web_fetch": "fetch",
        "diff": "diff", "load_json": "json", "save_json": "json-save",
        "ls": "list", "memory_search": "recall",
    }

    def __init__(
        self,
        recorder: PatternRecorder,
        trial_skills_dir: Path,
        llm_provider,
        min_occurrences: int = 3,
        compile_max_tokens: int = 2048,
    ):
        self.recorder = recorder
        self.trial_skills_dir = trial_skills_dir
        self.llm = llm_provider
        self.min_occurrences = min_occurrences
        self.compile_max_tokens = compile_max_tokens
        self.trial_skills_dir.mkdir(parents=True, exist_ok=True)

    async def check_and_compile(self) -> Optional[CompiledSkill]:
        """Check for patterns that qualify for compilation.

        Flow:
        1. Build frequency table → find qualifying hashes
        2. Check if a skill already exists for the best hash (zero LLM cost)
        3. If not, compile via LLM → write skill → remove compiled patterns from JSONL
        """
        freq_table = self._build_frequency_table()
        qualifying = {
            h: count for h, count in freq_table.items()
            if count >= self.min_occurrences
        }
        if not qualifying:
            return None

        best_hash = max(qualifying, key=qualifying.get)
        existing_names = self._existing_trial_skill_names()

        # Check if a skill for this hash already exists (before any LLM call)
        patterns = self.recorder.get_patterns_by_hash(best_hash, limit=10)
        if not patterns:
            return None

        # Use tool-based name as a quick check — if a skill with this
        # fallback name already exists, the pattern has been compiled
        fallback_name = self._generate_tool_based_name(patterns)
        if fallback_name in existing_names:
            logger.info("pattern_compile_skipped", reason="trial_exists", skill=fallback_name)
            self.recorder.remove_patterns_by_hash(best_hash)
            return None

        # Compile via LLM
        compiled = await self._compile_skill(fallback_name, patterns)
        if compiled is None:
            return None

        # Check if LLM-generated name conflicts with existing skills
        if compiled.name in existing_names:
            logger.info("pattern_compile_skipped", reason="name_conflict", skill=compiled.name)
            # Pattern was already compiled (just under a different name), remove it
            self.recorder.remove_patterns_by_hash(best_hash)
            return None

        self._write_trial_skill(compiled)
        # Remove compiled patterns so they won't trigger re-compilation
        self.recorder.remove_patterns_by_hash(best_hash)
        logger.info(
            "trial_skill_created",
            skill=compiled.name,
            patterns_used=compiled.source_patterns,
        )
        return compiled

    def _build_frequency_table(self) -> Dict[str, int]:
        """Scan JSONL and count pattern hash frequencies."""
        freq: Dict[str, int] = {}
        patterns_file = self.recorder._patterns_file
        if not patterns_file.exists():
            return freq
        with open(patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    h = record.get("pattern_hash", "")
                    if h:
                        freq[h] = freq.get(h, 0) + 1
                except json.JSONDecodeError:
                    continue
        return freq

    def _generate_tool_based_name(self, patterns: List[Dict]) -> str:
        """Generate a fallback skill name from the most common tools.

        Used only when LLM compilation doesn't produce a name.
        """
        tool_counts: Dict[str, int] = {}
        for p in patterns:
            for name in p.get("tool_sequence", []):
                tool_counts[name] = tool_counts.get(name, 0) + 1

        sorted_tools = sorted(tool_counts, key=tool_counts.get, reverse=True)[:3]
        fragments = [self._TOOL_NAME_MAP.get(t, t.replace("_", "-")) for t in sorted_tools]
        return "-".join(fragments)

    def _existing_trial_skill_names(self) -> set:
        """List skill names already in the trial directory."""
        names: set = set()
        if self.trial_skills_dir.is_dir():
            for child in self.trial_skills_dir.iterdir():
                if child.is_dir() and (child / "SKILL.md").exists():
                    names.add(child.name)
        return names

    async def _compile_skill(
        self, skill_name: str, patterns: List[Dict]
    ) -> Optional[CompiledSkill]:
        """Use LLM to compile patterns into a SKILL.md.

        Dual-track: if the pattern is operation-heavy (>60% bash among core
        tools), generates a type: script skill with an auto-generated main.sh.
        Otherwise generates a type: prompt skill with step-by-step instructions.
        """
        from .llm import LLMMessage

        # Classify pattern type
        skill_type = self._classify_pattern_type(patterns)

        pattern_summaries = []
        for p in patterns:
            seq = " -> ".join(p.get("tool_sequence", []))
            task = p.get("task", "")[:200]
            # Include tool details with parameter values for richer compilation context
            details = p.get("tool_details", [])
            detail_lines = []
            for d in details:
                name = d.get("name", "?")
                params = d.get("params_summary", {})
                if params:
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    detail_lines.append(f"    {name}({param_str})")
                else:
                    detail_lines.append(f"    {name}()")
            details_str = "\n".join(detail_lines) if detail_lines else "    (no details)"
            pattern_summaries.append(
                f"- Task: {task}\n  Sequence: {seq}\n  Details:\n{details_str}"
            )

        patterns_text = "\n\n".join(pattern_summaries)

        # Load skill-creator spec as the canonical format reference
        skill_spec = self._load_skill_spec()

        if skill_type == "script":
            prompt = self._build_script_compile_prompt(patterns, patterns_text, skill_spec)
        else:
            prompt = self._build_prompt_compile_prompt(patterns, patterns_text, skill_spec)

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=self.compile_max_tokens,
            )
            content = (response.content or "").strip()
            if not content or len(content) < 100:
                logger.warning("skill_compile_too_short", length=len(content) if content else 0)
                return None

            # Post-process: strip thinking leaks and code block wrappers
            content = self._clean_llm_output(content)

            # For script-type, split SKILL.md and main.sh from combined output
            skill_md_content = content
            script_content = None
            if skill_type == "script":
                skill_md_content, script_content = self._split_script_output(content)

            # Fix missing closing frontmatter delimiter (---)
            skill_md_content = self._fix_frontmatter(skill_md_content)

            if not skill_md_content or len(skill_md_content) < 50:
                logger.warning("skill_compile_empty_after_clean", length=len(skill_md_content) if skill_md_content else 0)
                return None

            # Extract skill name from frontmatter (LLM-generated semantic name)
            extracted_name = self._extract_name_from_frontmatter(skill_md_content)
            final_name = extracted_name or skill_name  # fallback to tool-based name

            description = self._extract_description(skill_md_content) or f"Auto-compiled skill: {final_name}"

            return CompiledSkill(
                name=final_name,
                description=description,
                content=skill_md_content,
                source_patterns=len(patterns),
                confidence=min(1.0, len(patterns) / 10.0),
                skill_type=skill_type,
                script_content=script_content,
            )
        except Exception as e:
            logger.warning("skill_compile_failed", error=str(e))
            return None

    @staticmethod
    def _classify_pattern_type(patterns: List[Dict]) -> str:
        """Classify whether patterns are operation-type (script) or decision-type (prompt).

        An operation pattern is one where the agent executes a predictable sequence
        of tool calls without significant judgment or branching. This includes:
        - Bash-heavy patterns (>40% bash among core tools)
        - Repetitive file-read patterns (same tool ≥3 times per session)
        - Any pattern where core tools are all read-only operations in a fixed order

        Decision patterns require judgment, branching, or varied approaches.
        """
        total_core = 0
        bash_core = 0
        # Also count repetitive tool usage (e.g., read_file x4 = operation)
        tool_name_counts: Dict[str, int] = {}

        for p in patterns:
            for sig in p.get("tool_signatures", []):
                # Skip exploratory tools
                if sig in PatternRecorder._EXPLORATORY_TOOLS:
                    continue
                total_core += 1
                if sig.startswith("bash:"):
                    bash_core += 1
                # Track base tool name (e.g., "read_file" from "read_file:...")
                base = sig.split(":")[0]
                tool_name_counts[base] = tool_name_counts.get(base, 0) + 1

        if total_core == 0:
            return "prompt"

        # Rule 1: Bash-heavy → script
        if bash_core / total_core > 0.4:
            return "script"

        # Rule 2: Repetitive single tool (same tool ≥3 times on average per pattern)
        # This catches patterns like read_file x4 per session
        num_patterns = len(patterns)
        for tool_name, count in tool_name_counts.items():
            if count / num_patterns >= 3:
                return "script"

        # Rule 3: All core tools are deterministic/read-only operations
        # (bash, read_file, load_json, grep, find) → script
        deterministic_tools = {"bash", "read_file", "load_json", "grep", "find", "ls"}
        core_tool_names = set()
        for p in patterns:
            for sig in p.get("tool_signatures", []):
                if sig not in PatternRecorder._EXPLORATORY_TOOLS:
                    core_tool_names.add(sig.split(":")[0])

        if core_tool_names and core_tool_names.issubset(deterministic_tools):
            return "script"

        return "prompt"

    def _build_script_compile_prompt(
        self, patterns: List[Dict], patterns_text: str, skill_spec: str
    ) -> str:
        """Build the LLM prompt for script-type skill compilation."""
        return (
            "You are a skill compiler. Given the following repeated successful "
            "execution patterns that are repetitive operations, create TWO files:\n"
            "1. A SKILL.md file (frontmatter: name and description only)\n"
            "2. A scripts/main.sh that automates the operation\n\n"
            f"## Observed Patterns ({len(patterns)} occurrences)\n"
            f"{patterns_text}\n\n"
            f"{skill_spec}\n\n"
            "## Your Task\n"
            "1. Generate a semantic skill name from the task descriptions "
            "(e.g., 'service-status', not 'read-shell-write')\n"
            "2. Identify the variable parts across patterns (dates, project paths, etc.)\n"
            "3. Create the SKILL.md with:\n"
            "   - Frontmatter with `name` and `description` only (no `type` field)\n"
            "   - A 1-2 sentence introduction after the title\n"
            "   - 'When to Use' and 'When NOT to Use' sections\n"
            "   - A Usage section showing: `bash $SKILL_DIR/scripts/main.sh $ARG1 $ARG2`\n"
            "     Document what each argument means\n"
            "   - Output format description\n"
            "4. Create the main.sh script with:\n"
            "   - `set -euo pipefail` at the top\n"
            "   - Argument validation (check $# and print usage)\n"
            "   - The actual operations from the patterns, using $1, $2 etc. for variables\n"
            "   - Convert read_file operations to `cat` commands\n"
            "   - Use absolute paths from arguments, not hardcoded paths\n"
            "   - Proper error messages when files/dirs not found\n"
            "   - Formatted output with clear section headers\n\n"
            "## Output Format\n"
            "Output TWO files separated by a line containing exactly: ===SCRIPT===\n"
            "First the SKILL.md content (starting with ---), then ===SCRIPT===, "
            "then the main.sh content (starting with #!/bin/bash).\n\n"
            "IMPORTANT: Do NOT wrap in code blocks. Do NOT include thinking or analysis.\n"
            "Make sure the SKILL.md frontmatter has both opening and closing --- delimiters."
        )

    def _build_prompt_compile_prompt(
        self, patterns: List[Dict], patterns_text: str, skill_spec: str
    ) -> str:
        """Build the LLM prompt for prompt-type skill compilation."""
        return (
            "You are a skill compiler. Given the following repeated successful "
            "execution patterns, create a SKILL.md file.\n\n"
            f"## Observed Patterns ({len(patterns)} occurrences)\n"
            f"{patterns_text}\n\n"
            f"{skill_spec}\n\n"
            "## Your Task\n"
            "1. Generate a semantic skill name from the task descriptions "
            "(e.g., 'service-status', not 'read-shell-write')\n"
            "2. Frontmatter should contain `name` and `description` only (no `type` field)\n"
            "3. After the title heading, add a 1-2 sentence introduction explaining "
            "what this skill does and what problem it solves\n"
            "4. Include 'When to Use' and 'When NOT to Use' sections for precise activation\n"
            "5. Write concrete step-by-step instructions with $VARIABLE placeholders "
            "for variable parts (dates, paths, project names)\n"
            "6. Keep it concise (under 80 lines)\n\n"
            "IMPORTANT: Output the SKILL.md content directly, starting with ---. "
            "Make sure to include the closing --- delimiter after the frontmatter. "
            "Do NOT wrap in code blocks. Do NOT include thinking or analysis."
        )

    @staticmethod
    def _split_script_output(content: str) -> tuple:
        """Split LLM output into SKILL.md and main.sh parts.

        The LLM is instructed to output two files separated by '===SCRIPT==='.
        If no separator found, treat the whole content as SKILL.md only.
        """
        separator = "===SCRIPT==="
        if separator in content:
            parts = content.split(separator, 1)
            skill_md = parts[0].strip()
            script = parts[1].strip() if len(parts) > 1 else ""
            # Ensure script starts with shebang
            if script and not script.startswith("#!/bin/bash"):
                # Remove any leading code block markers
                script = re.sub(r'^```(?:bash|sh)?\s*\n?', '', script)
                script = script.rstrip('`')
                if not script.startswith("#!/bin/bash"):
                    script = "#!/bin/bash\n" + script
            return skill_md, script if script else None
        return content, None

    @staticmethod
    def _fix_frontmatter(content: str) -> str:
        """Ensure YAML frontmatter has both opening and closing --- delimiters.

        LLMs sometimes omit the closing ---. This method detects and fixes it.
        Also strips any `type:` field from frontmatter.
        """
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return content  # No frontmatter at all, leave as-is

        # Find the closing --- after the opening one
        found_closing = False
        closing_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                found_closing = True
                closing_idx = i
                break

        if not found_closing:
            # Insert closing --- before the first non-empty, non-YAML line
            insert_at = len(lines)
            for i in range(1, len(lines)):
                line = lines[i].strip()
                # A line that's not a YAML key:value and not empty = content start
                if line and not line.startswith("#") and ":" not in line:
                    insert_at = i
                    break
                # Also catch markdown headings (start of content)
                if line.startswith("#"):
                    insert_at = i
                    break

            lines.insert(insert_at, "---")
            closing_idx = insert_at

        # Strip `type:` lines from frontmatter
        new_lines = []
        for i, line in enumerate(lines):
            if 0 < i < closing_idx and line.strip().startswith("type:"):
                continue
            new_lines.append(line)

        return "\n".join(new_lines)

    @staticmethod
    def _clean_llm_output(content: str) -> str:
        """Clean LLM output by stripping thinking leaks and code block wrappers.

        Handles:
        1. <think...</think) or <thinking...</thinking) blocks (DeepSeek, Qwen)
        2. ```yaml...``` or ```...``` markdown code block wrappers
        3. Any preamble text before the first --- (YAML frontmatter start)
        """
        # Strip thinking tags
        content = re.sub(r'<think[^>]*>.*?</think\s*>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking\s*>', '', content, flags=re.DOTALL)

        # Strip markdown code block wrappers
        content = re.sub(r'^```(?:yaml|markdown)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)

        # Strip any preamble before the first YAML frontmatter start delimiter (---)
        # We look for --- at the start of a line (the frontmatter opening),
        # and remove everything before it — but KEEP the frontmatter itself.
        first_fm = re.search(r'^---\s*$', content, re.MULTILINE)
        if first_fm:
            content = content[first_fm.start():]

        return content.strip()

    @staticmethod
    def _load_skill_spec() -> str:
        """Load the skill-creator SKILL.md as the canonical format specification."""
        spec_paths = [
            Path(".omniagent/skills/skill-creator/SKILL.md"),
            Path.home() / ".omniagent" / "skills" / "skill-creator" / "SKILL.md",
        ]

        for p in spec_paths:
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8").strip()
                    # Extract only the specification parts (skip meta-thinking)
                    lines = content.split("\n")
                    spec_lines = []
                    in_spec = False
                    for line in lines:
                        # Skip thinking leaks in the skill-creator itself
                        if line.strip().startswith("让") or line.strip().startswith("Let me"):
                            continue
                        if "SKILL.md Specification" in line or "SKILL.md Format" in line:
                            in_spec = True
                        if in_spec:
                            spec_lines.append(line)
                    if spec_lines:
                        return "## SKILL.md Format Specification\n" + "\n".join(spec_lines)
                except Exception:
                    continue

        # Fallback spec if skill-creator not found
        return (
            "## SKILL.md Format Specification\n\n"
            "```yaml\n"
            "---\n"
            "name: skill-name\n"
            "description: What this skill does and when to use it.\n"
            "---\n\n"
            "# Skill Title\n\n"
            "## When to Use\n"
            "Conditions that trigger this skill.\n\n"
            "## When NOT to Use\n"
            "Conditions where this skill should NOT be activated.\n\n"
            "## Steps (prompt-type)\n"
            "Step-by-step instructions with $VARIABLE placeholders.\n\n"
            "## Usage (script-type, has scripts/main.sh)\n"
            "bash $SKILL_DIR/scripts/main.sh $ARGS\n\n"
            "## Output Format\n"
            "How to present results.\n"
            "```\n\n"
            "### Content Guidelines\n"
            "1. **Title**: Clear, descriptive heading\n"
            "2. **Frontmatter**: `name` and `description` only — no `type` field\n"
            "3. **When to Use / Not to Use**: Precise activation conditions\n"
            "4. **Usage** (script): Shows how to invoke scripts/main.sh with variable args\n"
            "5. **Steps** (prompt): Concrete tool calls with $VARIABLE placeholders\n"
            "6. **Output Format**: How to parse and present results\n"
        )

    @staticmethod
    def _extract_description(content: str) -> str:
        """Extract description from YAML frontmatter."""
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.strip().startswith("description:"):
                return line.split(":", 1)[1].strip().strip('"')
        return ""

    @staticmethod
    def _extract_name_from_frontmatter(content: str) -> Optional[str]:
        """Extract and validate skill name from YAML frontmatter.

        Returns None if no valid name found (caller should use fallback).
        """
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.strip().startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"').strip("'")
                # Validate: lowercase, hyphens, max 64 chars, no spaces
                if name and len(name) <= 64 and name == name.lower() and " " not in name:
                    return name
        return None

    def _write_trial_skill(self, skill: CompiledSkill) -> None:
        """Write a trial skill to disk.

        For script-type skills, also creates scripts/main.sh.
        """
        skill_dir = self.trial_skills_dir / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill.content, encoding="utf-8")

        if skill.skill_type == "script" and skill.script_content:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            script_path = scripts_dir / "main.sh"
            script_path.write_text(skill.script_content, encoding="utf-8")
            # Make executable
            script_path.chmod(script_path.stat().st_mode | 0o755)
            logger.info("skill_script_written", skill=skill.name, path=str(script_path))


# ── Skill Evolution Tracker ────────────────────────────────────────


class SkillEvolutionTracker:
    """Detects error-recovery pairs and writes patches to existing skills.

    Also handles direct user feedback that targets a specific skill.
    """

    def __init__(
        self,
        patches_dir: Path,
        llm_provider,
        max_patches_per_skill: int = 10,
    ):
        self.patches_dir = patches_dir
        self.llm = llm_provider
        self.max_patches_per_skill = max_patches_per_skill
        self.patches_dir.mkdir(parents=True, exist_ok=True)

    async def check_and_evolve(
        self,
        active_skills: List[str],
        conversation_history: list,
        task: str,
        user_feedback: Optional[List[str]] = None,
    ) -> Optional[SkillPatch]:
        """Check for evolution candidates and write patches.

        Two sources:
        1. Error-recovery pairs in tool execution history (always)
        2. User feedback that mentions a specific skill (when provided)
        """
        # Direction 1: error-recovery pairs (original logic)
        if active_skills:
            error_recovery_pairs = self._find_error_recovery_pairs(conversation_history)
            if error_recovery_pairs:
                for pair in error_recovery_pairs:
                    patch = await self._create_patch(pair, active_skills, task)
                    if patch:
                        self._write_patch(patch)
                        return patch

        # Direction 2: user feedback targeting a specific skill
        if user_feedback and active_skills:
            for feedback_text in user_feedback:
                patch = await self._create_feedback_patch(
                    feedback_text=feedback_text,
                    active_skills=active_skills,
                    task=task,
                )
                if patch:
                    self._write_patch(patch)
                    return patch

        return None

    def _find_error_recovery_pairs(self, history: list) -> List[Dict[str, Any]]:
        """Walk history to find tool errors followed by successful alternatives."""
        pairs = []
        i = 0
        while i < len(history):
            msg = history[i]
            if msg.role == "tool" and msg.content and msg.content.startswith("Error:"):
                error_tool = msg.name or "unknown"
                error_message = msg.content[:300]

                j = i + 1
                while j < min(i + 6, len(history)):
                    next_msg = history[j]
                    if next_msg.role == "tool" and next_msg.content:
                        if (
                            not next_msg.content.startswith("Error:")
                            and next_msg.name != error_tool
                        ):
                            context_before = ""
                            for k in range(max(0, j - 3), j):
                                if history[k].role == "assistant" and history[k].content:
                                    context_before = history[k].content[:300]
                                    break

                            pairs.append({
                                "error_tool": error_tool,
                                "error_message": error_message,
                                "recovery_tool": next_msg.name or "unknown",
                                "context_before": context_before,
                            })
                            i = j
                            break
                    j += 1
            i += 1

        return pairs

    async def _create_patch(
        self,
        pair: Dict[str, Any],
        active_skills: List[str],
        task: str,
    ) -> Optional[SkillPatch]:
        """Use LLM to create a patch from an error-recovery pair."""
        from .llm import LLMMessage

        skills_list = ", ".join(active_skills)

        prompt = (
            f"A OmniAgent agent was executing with skills: {skills_list}\n\n"
            f"Task: {task[:300]}\n\n"
            f'The agent tried tool "{pair["error_tool"]}" but got error:\n'
            f'{pair["error_message"]}\n\n'
            f'Context before error: {pair.get("context_before", "N/A")}\n\n'
            f'The agent recovered using tool "{pair["recovery_tool"]}".\n\n'
            f"Please identify which skill likely caused the wrong approach "
            f"and what correction is needed.\n\n"
            "Respond in this JSON format only:\n"
            '{"skill_name": "name", "original_context": "what the skill suggested", '
            '"error_description": "what went wrong", "correction": "what to do instead"}'
        )

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=500,
            )
            content = (response.content or "").strip()
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            skill_name = data.get("skill_name", "")
            if skill_name not in active_skills:
                return None

            return SkillPatch(
                skill_name=skill_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                original_context=data.get("original_context", ""),
                error_description=data.get("error_description", ""),
                correction=data.get("correction", ""),
                task_description=task[:300],
            )
        except Exception as e:
            logger.warning("evolution_patch_failed", error=str(e))
            return None

    async def _create_feedback_patch(
        self,
        feedback_text: str,
        active_skills: List[str],
        task: str,
    ) -> Optional[SkillPatch]:
        """Create a skill patch from direct user feedback.

        Only creates a patch if the feedback clearly targets a specific skill
        by name. If no skill is mentioned, returns None.
        """
        from .llm import LLMMessage

        skills_list = ", ".join(active_skills)

        prompt = (
            "A user gave direct feedback about an agent's execution:\n\n"
            f"Feedback: {feedback_text[:500]}\n"
            f"Task context: {task[:300]}\n\n"
            f"The agent was using these skills: {skills_list}\n\n"
            "Determine if this feedback targets a SPECIFIC skill by name.\n"
            "If it does not mention any skill, respond with:\n"
            '{"targets_skill": false}\n\n'
            "If it does target a skill, respond with:\n"
            '{"targets_skill": true, "skill_name": "the skill name", '
            '"correction": "what the skill should do instead"}'
        )

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=300,
            )
            content = (response.content or "").strip()
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            if data.get("targets_skill") is not True:
                return None

            skill_name = data.get("skill_name", "")
            if not skill_name or skill_name not in active_skills:
                return None

            correction = data.get("correction", "").strip()
            if not correction or len(correction) < 10:
                return None

            return SkillPatch(
                skill_name=skill_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                original_context="(user feedback, no tool error)",
                error_description=feedback_text[:300],
                correction=correction,
                task_description=task[:300],
            )
        except Exception as e:
            logger.warning("feedback_patch_failed", error=str(e))
            return None

    def _write_patch(self, patch: SkillPatch) -> None:
        """Write a patch file, respecting max_patches_per_skill."""
        patch_file = self.patches_dir / f"{patch.skill_name}.md"

        if patch_file.exists():
            self._prune_patches(patch_file)

        with open(patch_file, "a", encoding="utf-8") as f:
            f.write("\n---\n\n")
            f.write(patch.to_markdown())

        logger.info("skill_patch_written", skill=patch.skill_name)

    def _prune_patches(self, patch_file: Path) -> None:
        """Keep only the most recent patches for a skill."""
        content = patch_file.read_text(encoding="utf-8")
        sections = re.split(r"\n---\n\n", content.strip())
        if len(sections) > self.max_patches_per_skill:
            kept = sections[-(self.max_patches_per_skill - 1):]
            patch_file.write_text("\n---\n\n".join(kept) + "\n", encoding="utf-8")

    def get_patches_for_skill(self, skill_name: str) -> str:
        """Read all patches for a given skill."""
        patch_file = self.patches_dir / f"{skill_name}.md"
        if not patch_file.exists():
            return ""
        return patch_file.read_text(encoding="utf-8")


# ── Skill Evolution Manager (Orchestrator) ─────────────────────────


class SkillEvolutionManager:
    """Top-level orchestrator for skill self-evolution.

    Subscribes to EventBus and coordinates PatternRecorder, PatternAnalyzer,
    and SkillEvolutionTracker.
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

        self._patterns_dir = self.work_dir / ".omniagent" / "patterns"
        self._trial_skills_dir = self.work_dir / ".omniagent" / "skills"
        self._patches_dir = self.work_dir / ".omniagent" / "skill_patches"

        self.recorder = PatternRecorder(
            patterns_dir=self._patterns_dir,
            max_file_size_mb=config.pattern_max_file_size_mb,
            work_dir=self.work_dir,
        )
        self.analyzer = PatternAnalyzer(
            recorder=self.recorder,
            trial_skills_dir=self._trial_skills_dir,
            llm_provider=llm_provider,
            min_occurrences=config.pattern_min_occurrences,
            compile_max_tokens=config.compile_max_tokens,
        )
        self.evolution_tracker = SkillEvolutionTracker(
            patches_dir=self._patches_dir,
            llm_provider=llm_provider,
            max_patches_per_skill=config.patch_max_per_skill,
        )

        self._current_task: str = ""
        self._current_start_time: float = 0.0
        self._current_history: list = []
        self._current_tool_name_history: List[str] = []
        self._current_active_skills: List[str] = []
        self._current_user_feedback: List[str] = []
        self.last_session_results: Dict[str, Any] = {}
        self._skill_manager = None  # Set later via set_skill_manager()

        from .events import EventType
        self.event_bus.subscribe(EventType.AGENT_START, self._on_agent_start)
        self.event_bus.subscribe(EventType.AGENT_END, self._on_agent_end)

        logger.info("skill_evolution_initialized")

    async def _on_agent_start(self, event) -> None:
        """Capture execution context at start."""
        self._current_task = event.data.get("task", "")
        self._current_start_time = time.time()

    async def _on_agent_end(self, event) -> None:
        """Process completed execution.

        Pattern recording is synchronous (fast). Skill compilation and
        evolution checking are launched as background tasks so they
        don't block the agent response.
        """
        import asyncio

        success = event.data.get("success", False)
        if not self._current_task:
            self.last_session_results = {}
            return

        duration = time.time() - self._current_start_time
        history = self._current_history
        tool_names = self._current_tool_name_history
        iterations = event.data.get("iterations", 0)

        # Direction 1: Record pattern on success (fast, synchronous)
        pattern_recorded = False
        if success and history:
            try:
                pattern = await self.recorder.record(
                    task=self._current_task,
                    conversation_history=history,
                    tool_name_history=tool_names,
                    success=True,
                    iterations=iterations,
                    duration_s=duration,
                    active_skills=self._current_active_skills,
                )
                pattern_recorded = pattern is not None
            except Exception as e:
                logger.warning("pattern_recording_failed", error=str(e))

        # Launch compilation and evolution as background tasks
        # so they don't block the agent response
        async def _compile_and_evolve():
            skill_compiled = False
            patch_written = False
            compiled = None

            # Try skill compilation
            if pattern_recorded:
                try:
                    compiled = await self.analyzer.check_and_compile()
                    skill_compiled = compiled is not None
                    if skill_compiled:
                        # Invalidate skill cache so next request sees the new skill
                        if hasattr(self, '_skill_manager') and self._skill_manager:
                            self._skill_manager.invalidate_cache()
                except Exception as e:
                    logger.warning("skill_compilation_failed", error=str(e))

            # Check for evolution (error-recovery + user feedback)
            if self.config.evolution_enabled and self._current_active_skills:
                try:
                    user_fb = self._current_user_feedback if self._current_user_feedback else None
                    patch = await self.evolution_tracker.check_and_evolve(
                        active_skills=self._current_active_skills,
                        conversation_history=history or [],
                        task=self._current_task,
                        user_feedback=user_fb,
                    )
                    patch_written = patch is not None
                except Exception as e:
                    logger.warning("evolution_check_failed", error=str(e))

            from datetime import datetime
            self.last_session_results = {
                "patterns_recorded": pattern_recorded,
                "skill_compiled": skill_compiled,
                "skill_name": compiled.name if compiled else None,
                "skill_description": compiled.description if compiled else None,
                "skill_source_patterns": compiled.source_patterns if compiled else None,
                "patches_written": patch_written,
                "patch_skill_name": patch.skill_name if patch else None,
                "patch_description": patch.task_description if patch else None,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

        asyncio.create_task(_compile_and_evolve())

    def set_agent_refs(
        self,
        conversation_history: list,
        tool_name_history: List[str],
        active_skills: List[str],
        user_feedback: Optional[List[str]] = None,
    ) -> None:
        """Set references to agent state for event handlers."""
        self._current_history = conversation_history
        self._current_tool_name_history = tool_name_history
        self._current_active_skills = active_skills
        self._current_user_feedback = list(user_feedback or [])

    def set_skill_manager(self, skill_manager) -> None:
        """Set reference to SkillManager for cache invalidation after compilation."""
        self._skill_manager = skill_manager
