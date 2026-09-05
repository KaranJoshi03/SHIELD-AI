"""
SHIELD AI — Audit Logging
Complete audit trail for every SHIELD evaluation.
Every request produces an audit event regardless of outcome.
Stored in SQLite for queryability.

Query functions:
- get_agent_history()
- get_blocked_transactions()
- get_high_risk_events()
- get_policy_violations()
- get_agent_trust_history()
- get_all_events()
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import AuditEvent, SecurityDecision, TransactionRequest


class AuditLogger:

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                agent_id TEXT,
                user_id TEXT,
                tool_name TEXT,
                amount REAL DEFAULT 0.0,
                currency TEXT DEFAULT 'INR',
                intent_text TEXT DEFAULT '',
                policy_result TEXT DEFAULT '',
                authorization_result TEXT DEFAULT '',
                risk_score REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT '',
                behavior_score REAL DEFAULT 0.0,
                workflow_result TEXT DEFAULT '',
                injection_result TEXT DEFAULT '',
                duplicate_result TEXT DEFAULT '',
                decision TEXT NOT NULL,
                reasons TEXT DEFAULT '[]',
                approval_required INTEGER DEFAULT 0,
                approval_status TEXT DEFAULT '',
                execution_result TEXT DEFAULT '',
                error TEXT DEFAULT '',
                trust_score REAL DEFAULT 0.0,
                intent_alignment REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_events(agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_events(decision)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)
        """)
        conn.commit()
        conn.close()

    def log(self, request: TransactionRequest,
            decision: SecurityDecision,
            intent_text: str = "",
            execution_result: str = "",
            error: str = "",
            approval_status: str = "") -> AuditEvent:

        event = AuditEvent(
            timestamp=datetime.utcnow(),
            request_id=request.request_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            tool_name=request.tool_name,
            amount=request.amount,
            currency=request.currency,
            intent_text=intent_text,
            policy_result=decision.policy_result.value if hasattr(decision.policy_result, 'value') else str(decision.policy_result),
            authorization_result=decision.authorization_result,
            risk_score=decision.overall_risk,
            risk_level=decision.risk_level.value if hasattr(decision.risk_level, 'value') else str(decision.risk_level),
            behavior_score=decision.behavior_score,
            workflow_result=decision.workflow_result,
            injection_result=decision.injection_result,
            duplicate_result=decision.duplicate_result,
            decision=decision.decision.value if hasattr(decision.decision, 'value') else str(decision.decision),
            reasons=decision.reasons,
            approval_required=decision.approval_required,
            approval_status=approval_status,
            execution_result=execution_result,
            error=error,
            trust_score=decision.trust_score,
            intent_alignment=decision.intent_alignment,
        )

        with self._lock:
            self._events.append(event)
            self._write_to_db(event)

        return event

    def _write_to_db(self, event: AuditEvent) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                event.event_id,
                event.timestamp.isoformat(),
                event.request_id,
                event.agent_id,
                event.user_id,
                event.tool_name,
                event.amount,
                event.currency,
                event.intent_text,
                event.policy_result,
                event.authorization_result,
                event.risk_score,
                event.risk_level,
                event.behavior_score,
                event.workflow_result,
                event.injection_result,
                event.duplicate_result,
                event.decision,
                json.dumps(event.reasons),
                int(event.approval_required),
                event.approval_status,
                event.execution_result,
                event.error,
                event.trust_score,
                event.intent_alignment,
                json.dumps(event.metadata),
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass 

    def get_all_events(self) -> List[AuditEvent]:
        return list(self._events)

    def get_agent_history(self, agent_id: str) -> List[AuditEvent]:
        return [e for e in self._events if e.agent_id == agent_id]

    def get_blocked_transactions(self) -> List[AuditEvent]:
        return [e for e in self._events if e.decision == "BLOCK"]

    def get_high_risk_events(self, threshold: float = 50.0) -> List[AuditEvent]:
        return [e for e in self._events if e.risk_score >= threshold]

    def get_policy_violations(self) -> List[AuditEvent]:
        return [e for e in self._events if e.policy_result == "FAIL"]

    def get_agent_trust_history(self, agent_id: str) -> List[Dict[str, Any]]:
        history = []
        for e in self._events:
            if e.agent_id == agent_id:
                history.append({
                    "timestamp": e.timestamp,
                    "trust_score": e.trust_score,
                    "decision": e.decision,
                    "risk_score": e.risk_score,
                })
        return history

    def get_decisions_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {
            "ALLOW": 0, "REVIEW": 0, "BLOCK": 0, "PAUSE_AGENT": 0,
        }
        for e in self._events:
            if e.decision in summary:
                summary[e.decision] += 1
        return summary

    def get_events_dataframe_data(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "request_id": e.request_id,
                "agent_id": e.agent_id,
                "user_id": e.user_id,
                "tool_name": e.tool_name,
                "amount": e.amount,
                "currency": e.currency,
                "policy_result": e.policy_result,
                "risk_score": e.risk_score,
                "risk_level": e.risk_level,
                "behavior_score": e.behavior_score,
                "decision": e.decision,
                "reasons": "; ".join(e.reasons),
                "trust_score": e.trust_score,
                "intent_alignment": e.intent_alignment,
                "workflow_result": e.workflow_result,
                "injection_result": e.injection_result,
                "duplicate_result": e.duplicate_result,
                "approval_required": e.approval_required,
                "execution_result": e.execution_result,
            }
            for e in self._events
        ]

    def get_stats(self) -> Dict[str, Any]:
        decisions = self.get_decisions_summary()
        total = len(self._events)
        return {
            "total_events": total,
            "decisions": decisions,
            "agents": len(set(e.agent_id for e in self._events)),
            "total_amount": sum(e.amount for e in self._events),
            "avg_risk_score": sum(e.risk_score for e in self._events) / max(total, 1),
            "policy_violations": len(self.get_policy_violations()),
            "high_risk_events": len(self.get_high_risk_events()),
        }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._init_db()
