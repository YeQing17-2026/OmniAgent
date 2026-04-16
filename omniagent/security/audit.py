"""Audit logging system."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


@dataclass
class AuditEvent:
    """Audit event."""

    timestamp: datetime
    event_type: str
    action: str
    user_id: str
    session_id: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "action": self.action,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "success": self.success,
            "details": self.details,
        }


class AuditLogger:
    """Audit logger."""

    def __init__(self, log_dir: Path):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = log_dir

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("audit_logger_initialized", log_dir=str(log_dir))

    def log_event(
        self,
        event_type: str,
        action: str,
        user_id: str,
        session_id: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Event type (tool_call, command_exec, etc.)
            action: Action performed
            user_id: User ID
            session_id: Session ID
            success: Whether action succeeded
            details: Additional details
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            action=action,
            user_id=user_id,
            session_id=session_id,
            success=success,
            details=details or {},
        )

        # Log to file
        self._write_event(event)

        # Log to structlog
        logger.info(
            "audit_event",
            event_type=event_type,
            action=action,
            user_id=user_id,
            session_id=session_id,
            success=success,
        )

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to log file."""
        # Use daily log files
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"

        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[AuditEvent]:
        """
        Query audit events.

        Args:
            start_date: Start date filter
            end_date: End date filter
            event_type: Event type filter
            user_id: User ID filter

        Returns:
            List of matching events
        """
        events = []

        # Read all log files in date range
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        event = AuditEvent(
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            event_type=data["event_type"],
                            action=data["action"],
                            user_id=data["user_id"],
                            session_id=data["session_id"],
                            success=data["success"],
                            details=data.get("details", {}),
                        )

                        # Apply filters
                        if start_date and event.timestamp < start_date:
                            continue
                        if end_date and event.timestamp > end_date:
                            continue
                        if event_type and event.event_type != event_type:
                            continue
                        if user_id and event.user_id != user_id:
                            continue

                        events.append(event)

            except Exception as e:
                logger.error("audit_query_error", file=str(log_file), error=str(e))

        return events
