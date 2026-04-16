"""Approval system for dangerous operations."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Approval status."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Approval request."""

    id: str
    action: str
    description: str
    risk_level: str  # "low", "medium", "high"
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ApprovalRequest":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            action=data["action"],
            description=data["description"],
            risk_level=data["risk_level"],
            status=ApprovalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            decided_at=(
                datetime.fromisoformat(data["decided_at"])
                if data.get("decided_at")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


class ApprovalManager:
    """Manages approval requests."""

    def __init__(self, storage_dir: Path, auto_approve_low_risk: bool = True):
        """
        Initialize approval manager.

        Args:
            storage_dir: Directory to store approvals
            auto_approve_low_risk: Auto-approve low risk operations
        """
        self.storage_dir = storage_dir
        self.auto_approve_low_risk = auto_approve_low_risk
        self.requests: Dict[str, ApprovalRequest] = {}
        self._events: Dict[str, asyncio.Event] = {}

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing approvals
        self._load_approvals()

        logger.info(
            "approval_manager_initialized",
            storage_dir=str(storage_dir),
            auto_approve_low_risk=auto_approve_low_risk,
        )

    def request_approval(
        self,
        action: str,
        description: str,
        risk_level: str = "medium",
        metadata: Optional[Dict] = None,
    ) -> ApprovalRequest:
        """
        Request approval for an action.

        Args:
            action: Action name
            description: Action description
            risk_level: Risk level (low, medium, high)
            metadata: Additional metadata

        Returns:
            Approval request
        """
        # Generate request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Create request
        request = ApprovalRequest(
            id=request_id,
            action=action,
            description=description,
            risk_level=risk_level,
            metadata=metadata or {},
        )

        # Auto-approve low risk if enabled
        if self.auto_approve_low_risk and risk_level == "low":
            request.status = ApprovalStatus.APPROVED
            request.decided_at = datetime.now()
            logger.info("auto_approved", request_id=request_id, action=action)

        # Store request
        self.requests[request_id] = request
        self._save_approval(request)

        # Create async event for wait_for_approval()
        self._events[request_id] = asyncio.Event()
        # If auto-approved, set the event immediately
        if request.status == ApprovalStatus.APPROVED:
            self._events[request_id].set()

        log_fn = logger.debug if request.status == ApprovalStatus.APPROVED else logger.info
        log_fn(
            "approval_requested",
            request_id=request_id,
            action=action,
            risk_level=risk_level,
            status=request.status.value,
        )

        return request

    def approve(self, request_id: str) -> bool:
        """
        Approve a request.

        Args:
            request_id: Request ID

        Returns:
            True if approved, False if not found
        """
        request = self.requests.get(request_id)
        if not request:
            return False

        request.status = ApprovalStatus.APPROVED
        request.decided_at = datetime.now()
        self._save_approval(request)

        # Wake up any coroutine waiting on this request
        event = self._events.get(request_id)
        if event:
            event.set()

        logger.debug("approval_granted", request_id=request_id)
        return True

    def deny(self, request_id: str) -> bool:
        """
        Deny a request.

        Args:
            request_id: Request ID

        Returns:
            True if denied, False if not found
        """
        request = self.requests.get(request_id)
        if not request:
            return False

        request.status = ApprovalStatus.DENIED
        request.decided_at = datetime.now()
        self._save_approval(request)

        # Wake up any coroutine waiting on this request
        event = self._events.get(request_id)
        if event:
            event.set()

        logger.info("approval_denied", request_id=request_id)
        return True

    def is_approved(self, request_id: str) -> bool:
        """
        Check if request is approved.

        Args:
            request_id: Request ID

        Returns:
            True if approved
        """
        request = self.requests.get(request_id)
        return request.status == ApprovalStatus.APPROVED if request else False

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """
        Get all pending requests.

        Returns:
            List of pending requests
        """
        return [
            req
            for req in self.requests.values()
            if req.status == ApprovalStatus.PENDING
        ]

    def _save_approval(self, request: ApprovalRequest) -> None:
        """Save approval to disk."""
        file_path = self.storage_dir / f"{request.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(request.to_dict(), f, indent=2)

    def _load_approvals(self) -> None:
        """Load approvals from disk."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    request = ApprovalRequest.from_dict(data)
                    self.requests[request.id] = request
            except Exception as e:
                logger.error("approval_load_failed", file=str(file_path), error=str(e))

    async def wait_for_approval(
        self, request_id: str, timeout: float = 120.0
    ) -> ApprovalStatus:
        """Wait for an approval decision asynchronously.

        Args:
            request_id: Request ID to wait for
            timeout: Maximum seconds to wait (default 120s)

        Returns:
            Final ApprovalStatus (APPROVED, DENIED, or EXPIRED)
        """
        event = self._events.get(request_id)
        if not event:
            return ApprovalStatus.EXPIRED

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.expire_request(request_id)
            return ApprovalStatus.EXPIRED

        # Clean up the event
        self._events.pop(request_id, None)
        return self.requests[request_id].status

    def expire_request(self, request_id: str) -> None:
        """Mark a request as expired."""
        request = self.requests.get(request_id)
        if request and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.EXPIRED
            request.decided_at = datetime.now()
            self._save_approval(request)
            self._events.pop(request_id, None)
            logger.info("approval_expired", request_id=request_id)
