"""
SHIELD AI — Workflow & Tool-Chain Security
Validates tool call sequences using DFA-style state machines.

Detects:
- Unexpected tool transitions
- Repeated tool flooding
- Privilege escalation
- Impossible transitions
- Retry loops

Valid workflow example:
    DISCOVER → SELECT → CREATE_ORDER → VERIFY_PAYMENT

Suspicious workflow:
    CREATE_ORDER → REFUND → CREATE_ORDER → PAYOUT
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from .models import ToolName, WorkflowState

VALID_TRANSITIONS: Dict[str, Set[str]] = {
    "create_order": {
        "fetch_payment", "verify_payment", "create_payment_link",
    },
    "fetch_payment": {
        "verify_payment", "refund_payment", "create_order",
        "fetch_payment", "create_payment_link",
    },
    "create_payment_link": {
        "fetch_payment", "verify_payment",
    },
    "refund_payment": {
        "fetch_payment", "verify_payment",
    },
    "create_payout": {
        "fetch_payment", "verify_payment",
    },
    "fetch_settlement": {
        "fetch_payment", "fetch_settlement",
    },
    "verify_payment": {
        "fetch_payment", "create_order", "refund_payment",
        "create_payment_link",
    },
}

SUSPICIOUS_SEQUENCES = [
    ["create_order", "create_payout"],
    ["create_order", "refund_payment", "create_order"],
    ["refund_payment", "create_payout"],
    ["create_payout", "create_payout", "create_payout"],
    ["refund_payment", "refund_payment", "refund_payment"],
]

ESCALATION_PAIRS = {
    ("fetch_payment", "create_payout"),
    ("fetch_payment", "refund_payment"),
    ("create_order", "create_payout"),
    ("verify_payment", "create_payout"),
    ("create_payment_link", "create_payout"),
}


class WorkflowValidator:
    """
    DFA-style workflow validator for tool-call sequences.
    Tracks the sequence of tool calls made by each agent within a session and validates each transition.
    """

    def __init__(self, transitions: Optional[Dict[str, Set[str]]] = None):
        self.transitions = transitions or VALID_TRANSITIONS
        self._sessions: Dict[str, WorkflowState] = {}
        self._max_tool_repeat = 3

    def get_or_create_session(self, session_id: str, agent_id: str) -> WorkflowState:
        if session_id not in self._sessions:
            self._sessions[session_id] = WorkflowState(
                session_id=session_id,
                agent_id=agent_id,
            )
        return self._sessions[session_id]

    def validate(self, session_id: str, agent_id: str,
                 tool_name: str) -> Tuple[str, List[str]]:
        """
        Validate a tool call within a workflow session.
        Returns:
            (result, reasons)
            result: "VALID" | "INVALID_TRANSITION" | "SUSPICIOUS_SEQUENCE" |
                    "TOOL_FLOODING" | "PRIVILEGE_ESCALATION"
        """
        session = self.get_or_create_session(session_id, agent_id)
        reasons: List[str] = []

        if not session.tool_sequence:
            session.add_tool(tool_name)
            return "VALID", ["First tool in session"]

        last_tool = session.last_tool()

        allowed_next = self.transitions.get(last_tool, set())
        if tool_name not in allowed_next and allowed_next:
            reasons.append(
                f"INVALID_TRANSITION: '{last_tool}' → '{tool_name}' "
                f"(allowed: {sorted(allowed_next)})"
            )
            session.add_tool(tool_name)
            return "INVALID_TRANSITION", reasons

        recent = session.tool_sequence[-self._max_tool_repeat:]
        if len(recent) >= self._max_tool_repeat and all(t == tool_name for t in recent):
            reasons.append(
                f"TOOL_FLOODING: '{tool_name}' called {self._max_tool_repeat}+ "
                f"times consecutively"
            )
            session.add_tool(tool_name)
            return "TOOL_FLOODING", reasons

        if self._check_suspicious(session.tool_sequence + [tool_name]):
            reasons.append(
                f"SUSPICIOUS_SEQUENCE: "
                f"{' → '.join(session.tool_sequence[-3:] + [tool_name])}"
            )
            session.add_tool(tool_name)
            return "SUSPICIOUS_SEQUENCE", reasons

        if last_tool and (last_tool, tool_name) in ESCALATION_PAIRS:
            reasons.append(
                f"PRIVILEGE_ESCALATION: '{last_tool}' → '{tool_name}'"
            )
            session.add_tool(tool_name)
            return "PRIVILEGE_ESCALATION", reasons

        session.add_tool(tool_name)
        return "VALID", ["Valid workflow transition"]

    def _check_suspicious(self, sequence: List[str]) -> bool:
        seq_str = "→".join(sequence)
        for pattern in SUSPICIOUS_SEQUENCES:
            pattern_str = "→".join(pattern)
            if pattern_str in seq_str:
                return True
        return False

    def get_session_history(self, session_id: str) -> List[str]:
        session = self._sessions.get(session_id)
        return session.tool_sequence if session else []

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reset(self) -> None:
        self._sessions.clear()
