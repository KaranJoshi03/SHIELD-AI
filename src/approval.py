"""
SHIELD AI — Human Approval Engine
Manages the human review/approval workflow for transactions
that require human oversight.

Approval thresholds (example):
- ≤ ₹2,000: ALLOW (no approval needed)
- ₹2,000 - ₹10,000: REVIEW (human approval required)
- > ₹10,000: BLOCK or REVIEW based on policy

Approval requests contain full context for informed decisions.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ApprovalRequest, ApprovalStatus, RiskLevel, SecurityDecision
)


class ApprovalEngine:

    def __init__(self):
        self._pending: Dict[str, ApprovalRequest] = {}
        self._history: List[ApprovalRequest] = []
        self._lock = threading.Lock()

    def request_approval(self, decision: SecurityDecision,
                         agent_name: str = "",
                         user_id: str = "",
                         tool_name: str = "",
                         amount: float = 0.0,
                         currency: str = "INR",
                         merchant_id: str = "",
                         purpose: str = "") -> ApprovalRequest:

        request = ApprovalRequest(
            request_id=decision.request_id,
            agent_id=decision.agent_id,
            agent_name=agent_name,
            user_id=user_id,
            tool_name=tool_name,
            amount=amount,
            currency=currency,
            merchant_id=merchant_id,
            purpose=purpose,
            risk_score=decision.overall_risk,
            risk_level=decision.risk_level,
            policy_violations=[r for r in decision.reasons if "VIOLATION" in r.upper() or "FAIL" in r.upper()],
            intent_alignment=decision.intent_alignment,
            behavior_anomaly=decision.behavior_score,
            reasons=decision.reasons,
            status=ApprovalStatus.PENDING,
        )

        with self._lock:
            self._pending[request.approval_id] = request

        return request

    def approve(self, approval_id: str, reviewer_id: str = "reviewer_001") -> Tuple[bool, str]:
        with self._lock:
            request = self._pending.get(approval_id)
            if request is None:
                return False, f"Approval request {approval_id} not found"

            if request.status != ApprovalStatus.PENDING:
                return False, f"Request already {request.status.value}"

            if datetime.utcnow() > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                self._history.append(request)
                del self._pending[approval_id]
                return False, "Approval request has expired"

            request.status = ApprovalStatus.APPROVED
            request.reviewer_id = reviewer_id
            request.reviewed_at = datetime.utcnow()
            self._history.append(request)
            del self._pending[approval_id]

        return True, "Approved"

    def reject(self, approval_id: str, reviewer_id: str = "reviewer_001",
               reason: str = "") -> Tuple[bool, str]:

        with self._lock:
            request = self._pending.get(approval_id)
            if request is None:
                return False, f"Approval request {approval_id} not found"

            if request.status != ApprovalStatus.PENDING:
                return False, f"Request already {request.status.value}"

            request.status = ApprovalStatus.REJECTED
            request.reviewer_id = reviewer_id
            request.reviewed_at = datetime.utcnow()
            if reason:
                request.reasons.append(f"REVIEWER_REJECTION: {reason}")
            self._history.append(request)
            del self._pending[approval_id]

        return True, "Rejected"

    def simulate_approval(self, approval_id: str,
                          auto_approve: bool = True) -> Tuple[bool, str]:

        if auto_approve:
            return self.approve(approval_id, "auto_reviewer")
        else:
            return self.reject(approval_id, "auto_reviewer",
                             "Automatically rejected by simulation")

    def get_pending(self) -> List[ApprovalRequest]:
        return list(self._pending.values())

    def get_history(self) -> List[ApprovalRequest]:
        return list(self._history)

    def get_stats(self) -> Dict[str, int]:
        approved = sum(1 for r in self._history if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self._history if r.status == ApprovalStatus.REJECTED)
        expired = sum(1 for r in self._history if r.status == ApprovalStatus.EXPIRED)
        return {
            "pending": len(self._pending),
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
            "total": len(self._history),
        }

    def reset(self) -> None:
        self._pending.clear()
        self._history.clear()
