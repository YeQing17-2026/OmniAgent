"""Progressive context loading with token budget management.

Implements L0/L2 hierarchical context loading:
- L0: Brief overview — always in system prompt
- L2: Complete content — loaded on demand via tools

Tools use L0 (Available Tools list) + API tools array (JSON Schema).
Skills, Memory, Bootstrap use L0 summaries; full content via tools if needed.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)

# Characters-to-tokens heuristic (aligned with context_manager.py)
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Rough token count estimate based on character length."""
    return int(len(text) / CHARS_PER_TOKEN)


def truncate_with_ellipsis(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated, use tools to view full content]"


class ContextAssembler:
    """Assembles system prompt content within a token budget.

    Fills content by priority:
    1. Fixed blocks (base instructions, safety, workspace)
    2. Bootstrap L0 (section summaries from AGENTS.md/SOUL.md/CUSTOM.md)
    3. Tools L0 (name + one-line description)
    4. Skills L0 (name + description)
    5. Reflection hints / Memory hints
    """

    def __init__(self, token_budget: int):
        """Initialize with max tokens for the system prompt.

        Args:
            token_budget: Maximum tokens allowed for the assembled system prompt.
        """
        self.token_budget = token_budget

    def assemble(
        self,
        fixed_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        skills_l0: Optional[str] = "",
        memory_hints: Optional[str] = "",
        bootstrap_l0: Optional[str] = "",
        reflection_hints: Optional[str] = "",
    ) -> str:
        """Assemble the system prompt within token budget.

        Fills content by priority:
        1. Fixed blocks (base instructions, safety, workspace)
        2. Bootstrap L0 (section summaries from AGENTS.md/SOUL.md/CUSTOM.md)
        3. Tools L0 (name + one-line description)
        4. Skills L0 (name + description)
        4.5. Reflection hints (previous attempt context) — within budget
        5. Memory hints (recent search results)

        Args:
            fixed_prompt: Base instructions (safety, workspace, etc.)
            tools: List of tool dicts with keys: name, description, parameters
            skills_l0: Pre-formatted skills L0 string
            memory_hints: Pre-formatted memory hints string
            bootstrap_l0: Pre-formatted bootstrap L0 summaries string
            reflection_hints: Pre-formatted reflection/discovery hints from previous attempts

        Returns:
            Assembled system prompt string
        """
        parts: List[str] = []
        used_tokens = estimate_tokens(fixed_prompt)

        # 1. Fixed prompt (always included)
        parts.append(fixed_prompt)

        # 2. Bootstrap L0 (section summaries)
        if bootstrap_l0:
            bt_l0_tokens = estimate_tokens(bootstrap_l0)
            if used_tokens + bt_l0_tokens <= self.token_budget:
                parts.append(bootstrap_l0)
                used_tokens += bt_l0_tokens

        # 3. Tools L0
        if tools:
            tools_l0 = self._format_tools_l0(tools)
            tools_l0_tokens = estimate_tokens(tools_l0)
            if used_tokens + tools_l0_tokens <= self.token_budget:
                parts.append(tools_l0)
                used_tokens += tools_l0_tokens
            else:
                # Not enough budget even for L0 tools — skip L1 too
                tools = None

        # 4. Skills L0
        if skills_l0:
            skills_tokens = estimate_tokens(skills_l0)
            if used_tokens + skills_tokens <= self.token_budget:
                parts.append(skills_l0)
                used_tokens += skills_tokens

        # 4.5. Reflection hints (previous attempt context)
        if reflection_hints:
            refl_tokens = estimate_tokens(reflection_hints)
            if used_tokens + refl_tokens <= self.token_budget:
                parts.append(reflection_hints)
                used_tokens += refl_tokens

        # 5. Memory hints
        if memory_hints:
            hints_tokens = estimate_tokens(memory_hints)
            if used_tokens + hints_tokens <= self.token_budget:
                parts.append(memory_hints)
                used_tokens += hints_tokens

        prompt = "\n".join(parts)
        logger.debug(
            "context_assembled",
            used_tokens=used_tokens,
            budget=self.token_budget,
            utilization=f"{used_tokens / self.token_budget:.1%}",
        )
        return prompt

    def _format_tools_l0(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools as L0: one line per tool with name and description."""
        lines = ["## Available Tools"]
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(lines)

    @staticmethod
    def format_skills_l0(skills_xml: str) -> str:
        """Convert full skills XML to compact L0 format.

        Input: <available-skills><skill name="x" path="...">desc</skill>...</available-skills>
        Output: - x: desc
        """
        if not skills_xml:
            return ""

        entries = re.findall(
            r'<skill\s+name="([^"]+)"[^>]*>([^<]*)</skill>',
            skills_xml,
        )
        if not entries:
            return ""

        lines = ["## Available Skills"]
        lines.extend(f"- {name}: {desc.strip()}" for name, desc in entries)
        return "\n".join(lines)

    @staticmethod
    def format_memory_hints(search_results: List[Dict[str, Any]]) -> str:
        """Format memory search results as L0 hints for system prompt.

        Args:
            search_results: List of dicts with keys: path, snippet, score

        Returns:
            Formatted memory hints string, or empty string if no results.
        """
        if not search_results:
            return ""

        lines = ["## Memory Recall", "Recent memory search found relevant context:"]
        for result in search_results[:5]:
            path = result.get("path", "")
            snippet = result.get("snippet", "")
            # Truncate snippet to one line for L0
            first_line = snippet.split("\n")[0].strip()
            if len(first_line) > 120:
                first_line = first_line[:120] + "..."
            if first_line:
                lines.append(f"- {path}: {first_line}")

        return "\n".join(lines)
