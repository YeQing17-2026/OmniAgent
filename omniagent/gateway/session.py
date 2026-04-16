"""Session management for OmniAgent."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


class SessionState(str, Enum):
    """Session state."""

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class Message:
    """Message in session history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """Agent session."""

    id: str
    user_id: str
    channel_id: str
    state: SessionState = SessionState.ACTIVE
    history: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to history."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.history.append(message)
        self.last_active_at = datetime.now()

    def pause(self) -> None:
        """Pause session."""
        if self.state == SessionState.ACTIVE:
            self.state = SessionState.PAUSED
            logger.info("session_paused", session_id=self.id)

    def resume(self) -> None:
        """Resume session."""
        if self.state == SessionState.PAUSED:
            self.state = SessionState.ACTIVE
            self.last_active_at = datetime.now()
            logger.info("session_resumed", session_id=self.id)

    def close(self) -> None:
        """Close session."""
        self.state = SessionState.CLOSED
        logger.info("session_closed", session_id=self.id)

    def is_expired(self, timeout: int) -> bool:
        """Check if session is expired."""
        if self.state == SessionState.CLOSED:
            return True
        elapsed = datetime.now() - self.last_active_at
        return elapsed.total_seconds() > timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "state": self.state.value,
            "history": [msg.to_dict() for msg in self.history],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            channel_id=data["channel_id"],
            state=SessionState(data["state"]),
            history=[Message.from_dict(msg) for msg in data.get("history", [])],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active_at=datetime.fromisoformat(data["last_active_at"]),
        )


class SessionManager:
    """Manages agent sessions."""

    def __init__(self, storage_dir: Path, session_timeout: int = 3600):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory to store session files
            session_timeout: Session timeout in seconds
        """
        self.storage_dir = storage_dir
        self.session_timeout = session_timeout
        self.sessions: Dict[str, Session] = {}

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing sessions
        self._load_sessions()

        logger.info(
            "session_manager_initialized",
            storage_dir=str(storage_dir),
            timeout=session_timeout,
        )

    def create_session(
        self,
        user_id: str,
        channel_id: str,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Create new session.

        Args:
            user_id: User identifier
            channel_id: Channel identifier
            session_id: Optional session ID (generated if not provided)

        Returns:
            New session
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = Session(
            id=session_id,
            user_id=user_id,
            channel_id=channel_id,
        )

        self.sessions[session_id] = session
        self._save_session(session)

        logger.info(
            "session_created",
            session_id=session_id,
            user_id=user_id,
            channel_id=channel_id,
        )

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        return self.sessions.get(session_id)

    def get_or_create_session(
        self,
        user_id: str,
        channel_id: str,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Get existing session or create new one.

        Args:
            user_id: User identifier
            channel_id: Channel identifier
            session_id: Optional session ID

        Returns:
            Session
        """
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired(self.session_timeout):
                return session

        return self.create_session(user_id, channel_id, session_id)

    def pause_session(self, session_id: str) -> bool:
        """
        Pause session.

        Args:
            session_id: Session identifier

        Returns:
            True if paused, False if not found
        """
        session = self.get_session(session_id)
        if session:
            session.pause()
            self._save_session(session)
            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """
        Resume session.

        Args:
            session_id: Session identifier

        Returns:
            True if resumed, False if not found
        """
        session = self.get_session(session_id)
        if session:
            session.resume()
            self._save_session(session)
            return True
        return False

    def close_session(self, session_id: str) -> bool:
        """
        Close session.

        Args:
            session_id: Session identifier

        Returns:
            True if closed, False if not found
        """
        session = self.get_session(session_id)
        if session:
            session.close()
            self._save_session(session)
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired = [
            sid
            for sid, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]

        for session_id in expired:
            session = self.sessions.pop(session_id)
            self._delete_session(session_id)
            logger.info("session_expired", session_id=session_id)

        return len(expired)

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        state: Optional[SessionState] = None,
    ) -> List[Session]:
        """
        List sessions with optional filters.

        Args:
            user_id: Filter by user ID
            channel_id: Filter by channel ID
            state: Filter by state

        Returns:
            List of sessions
        """
        sessions = list(self.sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]

        if channel_id:
            sessions = [s for s in sessions if s.channel_id == channel_id]

        if state:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    def _save_session(self, session: Session) -> None:
        """Save session to disk."""
        file_path = self.storage_dir / f"{session.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _delete_session(self, session_id: str) -> None:
        """Delete session file."""
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()

    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = Session.from_dict(data)
                    self.sessions[session.id] = session
                    logger.debug("session_loaded", session_id=session.id)
            except Exception as e:
                logger.error("session_load_failed", file=str(file_path), error=str(e))
