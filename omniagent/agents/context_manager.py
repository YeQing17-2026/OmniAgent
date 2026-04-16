"""Context window management and compaction.

ts architecture:
- Adaptive chunk ratio (computeAdaptiveChunkRatio)
- Multi-stage summarization (summarizeInStages)
- History pruning (pruneHistoryForContextShare)
- Progressive fallback (summarizeWithFallback)
- Context overflow detection (aligned with run.ts overflow recovery)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omniagent.agents.llm import LLMMessage, LLMProvider
from omniagent.infra import get_logger

logger = get_logger(__name__)

#ts defaults
BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15
SAFETY_MARGIN = 1.2
DEFAULT_SUMMARY_FALLBACK = "No prior history."
MERGE_SUMMARIES_INSTRUCTIONS = (
    "Merge these partial summaries into a single cohesive summary. "
    "Preserve decisions, TODOs, open questions, and any constraints."
)
SUMMARIZATION_PROMPT = """Summarize the following conversation segment concisely. Use this structured format:

## Goal
[What the user was trying to accomplish]

## Progress
[What has been accomplished so far -- Done / In Progress / Blocked]

## Key Decisions
[Important decisions made]

## Next Steps
[What still needs to be done]

## Critical Context
[Essential details that must not be lost -- file paths, error messages, constraints]

Conversation to summarize:
{conversation}

Summary:"""

#ts
CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000
CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000

# Model to context window size mapping
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v3": 8192,
    "deepseek-v3.2": 8192,
    "deepseek-chat": 8192,
    "deepseek-r1": 8192,
    "gpt-3.5-turbo": 16384,
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 128000,
    "claude-3-opus": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-sonnet-4": 200000,
    "claude-opus-4": 200000,
}


def resolve_context_window_size(model_id: str, configured_size: int = 0) -> int:
    """Resolve context window size from model ID or config.

   ts resolveContextWindowInfo().
    Priority: configured_size > model_id lookup > default.
    """
    if configured_size > 0:
        return configured_size

    model_lower = model_id.lower()
    for key, size in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return size

    return 8192  # Default conservative value


def evaluate_context_guard(tokens: int, window_size: int) -> dict:
    """Evaluate if context window is too small.

   ts evaluateContextWindowGuard().
    """
    return {
        "should_warn": tokens < CONTEXT_WINDOW_WARN_BELOW_TOKENS,
        "should_block": tokens < CONTEXT_WINDOW_HARD_MIN_TOKENS,
    }


def estimate_tokens(messages: List[LLMMessage]) -> int:
    """Estimate token count using chars/4 heuristic with safety margin.

   ts estimateMessagesTokens().
    Uses SAFETY_MARGIN=1.2 for conservative estimate.
    """
    total_chars = sum(len(msg.content) for msg in messages)
    return int(total_chars / 4 * SAFETY_MARGIN)


def detect_context_overflow(error_msg: str) -> bool:
    """Detect if an error is a context overflow.

   ts overflow detection patterns.
    """
    patterns = [
        r"request_too_large",
        r"request exceeds the maximum size",
        r"context length exceeded",
        r"maximum context length",
        r"prompt is too long",
        r"413.*too large",
        r"context.*overflow",
        r"context window.*(too (?:large|long)|exceed|over|limit)",
        r"prompt.*(too (?:large|long)|exceed|over|limit)",
    ]
    for pattern in patterns:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return True
    return False


@dataclass
class ChunkResult:
    """Result of a compaction operation."""
    messages: List[LLMMessage]
    summary: str
    original_count: int
    summarized_count: int
    kept_count: int


class ContextManager:
    """Manages context window size and compaction.

   ts + compaction-safeguard.ts architecture.
    """

    def __init__(
        self,
        context_window_size: int = 8192,
        compaction_threshold: float = 0.80,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.context_window_size = context_window_size
        self.compaction_threshold = compaction_threshold
        self.llm = llm_provider

    def needs_compaction(self, messages: List[LLMMessage]) -> bool:
        """Check if compaction is needed."""
        estimated = estimate_tokens(messages)
        threshold = int(self.context_window_size * self.compaction_threshold)
        return estimated >= threshold

    def compute_adaptive_chunk_ratio(
        self, messages: List[LLMMessage], context_window: int
    ) -> float:
        """Compute adaptive chunk ratio based on average message size.

       ts computeAdaptiveChunkRatio().
        Reduces ratio when average message exceeds 10% of context window.
        """
        if not messages:
            return BASE_CHUNK_RATIO

        total_tokens = estimate_tokens(messages)
        avg_tokens = total_tokens / len(messages)
        safe_avg_tokens = avg_tokens * SAFETY_MARGIN
        avg_ratio = safe_avg_tokens / context_window

        if avg_ratio > 0.1:
            reduction = min(avg_ratio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO)
            return max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)
        return BASE_CHUNK_RATIO

    def _split_messages_by_tokens(
        self, messages: List[LLMMessage], parts: int = 2
    ) -> List[List[LLMMessage]]:
        """Split messages into roughly equal-token chunks.

       ts splitMessagesByTokenShare().
        Each message stays intact (no message splitting).
        """
        if not messages or parts <= 1:
            return [messages] if messages else []

        tokens_per_part = estimate_tokens(messages) / parts
        chunks = []
        current_chunk = []
        current_tokens = 0

        for msg in messages:
            msg_tokens = int(len(msg.content) / 4 * SAFETY_MARGIN)
            if current_tokens + msg_tokens > tokens_per_part * 1.5 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(msg)
            current_tokens += msg_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [messages]

    def _is_oversized(self, msg: LLMMessage, context_window: int) -> bool:
        """Check if a single message is too large for summarization.

       ts isOversizedForSummary().
        """
        msg_tokens = int(len(msg.content) / 4 * SAFETY_MARGIN)
        return msg_tokens > context_window * 0.5

    def _track_file_operations(self, messages: List[LLMMessage]) -> Dict[str, set]:
        """Extract file read/modified from tool calls in messages."""
        read_files = set()
        modified_files = set()

        for msg in messages:
            if msg.role == "tool" and msg.content:
                content = msg.content or ""
            elif msg.role == "assistant" and msg.tool_calls:
                for tc in (msg.tool_calls or []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    if name in ("read_file", "load_json"):
                        path = args.get("path", "")
                        if path:
                            read_files.add(path)
                    elif name in ("write_file", "edit_file", "save_json", "bash"):
                        path = args.get("path", "")
                        if path:
                            modified_files.add(path)

        return {"read": read_files, "modified": modified_files}

    def _build_conversation_text(self, messages: List[LLMMessage]) -> str:
        """Convert messages to text for summarization."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                continue
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    async def _summarize(
        self,
        messages: List[LLMMessage],
        previous_summary: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ) -> Optional[str]:
        """Summarize messages using LLM.

       ts generateSummary() behavior.
        """
        if not self.llm:
            return None

        conversation_text = self._build_conversation_text(messages)
        if not conversation_text.strip():
            return None

        prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)

        if previous_summary:
            prompt = (
                f"Previous summary (update it with new information):\n"
                f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
                + prompt
            )

        if custom_instructions:
            prompt += f"\n\n{custom_instructions}"

        try:
            response = await self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.content
        except Exception as e:
            logger.error("compaction_summary_failed", error=str(e))
            return None

    async def _summarize_with_fallback(
        self,
        messages: List[LLMMessage],
        previous_summary: Optional[str] = None,
    ) -> str:
        """Summarize with progressive fallback.

       ts summarizeWithFallback():
        1. Full summarization
        2. Partial (skip oversized messages)
        3. Static fallback
        """
        # Separate oversized messages
        normal_msgs = []
        oversized_msgs = []
        for msg in messages:
            if msg.role == "system":
                continue
            if self._is_oversized(msg, self.context_window_size):
                oversized_msgs.append(msg)
            else:
                normal_msgs.append(msg)

        # Level 1: Try full summarization
        summary = await self._summarize(normal_msgs, previous_summary)
        if summary:
            # Append notes for oversized messages
            if oversized_msgs:
                for msg in oversized_msgs:
                    msg_tokens = int(len(msg.content) / 4)
                    summary += f"\n\n[Large tool result (~{msg_tokens} tokens) omitted from summary]"
            return summary

        # Level 2: Partial summarization (already handled above if normal_msgs empty)
        if normal_msgs and not summary:
            # Retrying is handled by the caller (summarize_in_stages)
            pass

        # Level 3: Static fallback
        oversized_count = len(oversized_msgs)
        total_count = len(normal_msgs) + oversized_count
        return (
            f"Context contained {total_count} messages "
            f"({oversized_count} oversized). Summary unavailable due to size limits."
        )

    async def summarize_in_stages(
        self,
        messages: List[LLMMessage],
        parts: int = 2,
        previous_summary: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ) -> str:
        """Multi-stage summarization.

       ts summarizeInStages():
        1. If few messages, do single-stage
        2. Split into parts, summarize each
        3. Merge partial summaries into final summary
        """
        non_system = [m for m in messages if m.role != "system"]

        min_messages_for_split = 4
        if len(non_system) < min_messages_for_split:
            return await self._summarize_with_fallback(
                non_system, previous_summary
            )

        # Split and summarize each part
        chunks = self._split_messages_by_tokens(non_system, parts)
        partial_summaries = []

        for chunk in chunks:
            summary = await self._summarize(chunk)
            if summary:
                partial_summaries.append(summary)
            else:
                # Fallback: just truncate
                text = self._build_conversation_text(chunk)[:2000]
                partial_summaries.append(f"[Truncated]:\n{text}")

        if not partial_summaries:
            return DEFAULT_SUMMARY_FALLBACK

        if len(partial_summaries) == 1:
            return partial_summaries[0]

        # Merge partial summaries
        merged_text = "\n\n---\n\n".join(partial_summaries)
        merge_messages = [LLMMessage(role="user", content=merged_text)]
        merge_instructions = MERGE_SUMMARIES_INSTRUCTIONS
        if custom_instructions:
            merge_instructions += "\n\n" + custom_instructions
        merged_summary = await self._summarize(
            merge_messages,
            previous_summary=previous_summary,
            custom_instructions=merge_instructions,
        )

        return merged_summary if merged_summary else partial_summaries[0]

    def prune_history(
        self,
        messages: List[LLMMessage],
        max_context_tokens: Optional[int] = None,
        max_history_share: float = 0.5,
    ) -> ChunkResult:
        """Prune history to fit within budget.

       ts pruneHistoryForContextShare().
        Drops oldest chunks until messages fit within budget.
        """
        max_tokens = max_context_tokens or self.context_window_size
        budget_tokens = int(max_tokens * max_history_share)

        total_tokens = estimate_tokens(messages)

        if total_tokens <= budget_tokens:
            return ChunkResult(
                messages=messages,
                summary="",
                original_count=len(messages),
                summarized_count=0,
                kept_count=len(messages),
            )

        # Split into chunks and drop oldest first
        chunks = self._split_messages_by_tokens(messages, parts=2)
        kept = list(chunks)  # Start with all chunks

        while kept and estimate_tokens(
            [msg for chunk in kept for msg in chunk]
        ) > budget_tokens:
            dropped = kept.pop(0)  # Drop oldest chunk
            logger.info("history_chunk_dropped", chunk_size=len(dropped))

        # Flatten kept chunks
        result_messages = [msg for chunk in kept for msg in chunk]
        dropped_count = sum(len(c) for c in chunks) - len(result_messages)

        return ChunkResult(
            messages=result_messages,
            summary=f"[{dropped_count} older messages pruned to fit context budget]",
            original_count=len(messages),
            summarized_count=dropped_count,
            kept_count=len(result_messages),
        )

    async def compact(
        self,
        messages: List[LLMMessage],
    ) -> List[LLMMessage]:
        """Compact messages by summarizing older messages.

        Keeps the most recent messages intact, summarizes the rest.
        Integrated into ReflexionAgent.execute() before each LLM call.
        """
        if not self.llm:
            logger.warning("compaction_requested_but_no_llm")
            return messages

        # Separate system messages from history
        system_msgs = [m for m in messages if m.role == "system"]
        history_msgs = [m for m in messages if m.role != "system"]

        if len(history_msgs) < 4:
            return messages  # Not enough to compact

        # Find cut point: walk backwards, keep 50% of context window
        keep_budget = int(self.context_window_size * 0.50)
        keep_messages: List[LLMMessage] = []
        running_chars = 0

        for msg in reversed(history_msgs):
            msg_chars = len(msg.content)
            if running_chars + msg_chars > keep_budget * 4:
                break
            keep_messages.insert(0, msg)
            running_chars += msg_chars

        if len(keep_messages) >= len(history_msgs):
            return messages  # Nothing to compact

        to_summarize = history_msgs[: len(history_msgs) - len(keep_messages)]

        # Track file operations for incremental compaction context
        file_ops = self._track_file_operations(to_summarize)
        custom_instructions = ""
        if file_ops["read"] or file_ops["modified"]:
            ops_parts = []
            if file_ops["read"]:
                ops_parts.append(f"Files read: {', '.join(sorted(file_ops['read']))}")
            if file_ops["modified"]:
                ops_parts.append(f"Files modified: {', '.join(sorted(file_ops['modified']))}")
            custom_instructions = "File operations in this segment:\n" + "\n".join(ops_parts)

        # Compute adaptive chunk ratio
        ratio = self.compute_adaptive_chunk_ratio(
            to_summarize, self.context_window_size
        )
        parts = max(2, min(4, int(len(to_summarize) * ratio)))

        # Summarize with file tracking context
        summary = await self.summarize_in_stages(to_summarize, parts=parts, custom_instructions=custom_instructions or None)

        logger.info(
            "context_compacted",
            original_count=len(messages),
            summarized_count=len(to_summarize),
            kept_count=len(keep_messages),
        )

        # Return: system + summary + kept messages
        result = list(system_msgs)
        result.append(LLMMessage(role="compaction_summary", content=summary))
        result.extend(keep_messages)
        return result
