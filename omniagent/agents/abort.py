"""Abort controller for cancelling agent execution."""


class AbortError(Exception):
    """Raised when the agent is aborted."""


class AbortController:
    """Simple abort signal for cancelling agent execution."""

    def __init__(self) -> None:
        self._aborted = False

    @property
    def aborted(self) -> bool:
        """Check if abort has been requested."""
        return self._aborted

    def abort(self) -> None:
        """Request abort."""
        self._aborted = True

    def reset(self) -> None:
        """Reset abort state for reuse."""
        self._aborted = False

    def check(self) -> None:
        """Check abort state and raise AbortError if aborted."""
        if self._aborted:
            raise AbortError("Agent execution was aborted")
