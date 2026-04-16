"""Tool hook system for intercepting and modifying tool execution."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional


@dataclass
class ToolCallContext:
    """Context passed to tool hooks."""

    tool_name: str
    params: dict
    call_id: Optional[str] = None


@dataclass
class ToolHookResult:
    """Result from a before-tool-call hook."""

    blocked: bool = False
    block_reason: str = ""
    override_content: Optional[str] = None
    override_is_error: Optional[bool] = None


# Type alias for async tool hook handlers
ToolHook = Callable[[ToolCallContext], Awaitable[ToolHookResult]]


class ToolHookManager:
    """Manages before/after tool execution hooks."""

    def __init__(self) -> None:
        self._before_hooks: List[ToolHook] = []
        self._after_hooks: List[ToolHook] = []

    def add_before_hook(self, hook: ToolHook) -> None:
        """Add a hook that runs before tool execution. Can block execution."""
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: ToolHook) -> None:
        """Add a hook that runs after tool execution. Can override result."""
        self._after_hooks.append(hook)

    async def run_before(self, ctx: ToolCallContext) -> ToolHookResult:
        """Run all before-hooks. Returns combined result (blocked if any hook blocks)."""
        result = ToolHookResult()
        for hook in self._before_hooks:
            hook_result = await hook(ctx)
            if hook_result.blocked:
                result.blocked = True
                result.block_reason = hook_result.block_reason or result.block_reason
                return result  # Short-circuit on first block
        return result

    async def run_after(
        self, ctx: ToolCallContext, result_content: str
    ) -> str:
        """Run all after-hooks. Returns possibly-modified result content."""
        content = result_content
        for hook in self._after_hooks:
            hook_result = await hook(ctx)
            if hook_result.override_content is not None:
                content = hook_result.override_content
        return content
