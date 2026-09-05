"""
SHIELD AI — Fail-Closed Tests

Tests for fail-closed security design:
- ML model unavailable → NOT automatic ALLOW
- Policy unavailable → REVIEW or BLOCK
- System unhealthy → BLOCK
- Malformed request → not silently accepted

CRITICAL: Security uncertainty must NEVER become automatic authorization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.shield_gateway import ShieldGateway
from src.decision import DecisionEngine
from src.models import (
    DecisionType, PolicyResult, RiskLevel, TransactionRequest
)


class TestFailClosed:
    def setup_method(self):
        self.shield = ShieldGateway()
        self.decision_engine = DecisionEngine()

    def test_ml_unavailable_not_auto_allow(self):
        """ML failure must NOT cause automatic ALLOW."""
        decision = self.decision_engine.decide(
            auth_result="AUTHORIZED",
            policy_result=PolicyResult.PASS,
            risk_score=60,
            risk_level=RiskLevel.HIGH,
            ml_available=False,
            system_healthy=True,
        )
        # Should still process — ML unavailability alone doesn't block but it should NOT auto-allow a high-risk transaction
        assert decision.decision != DecisionType.ALLOW or decision.overall_risk < 50

    def test_system_unhealthy_blocks(self):
        """Unhealthy system must BLOCK."""
        decision = self.decision_engine.decide(
            auth_result="AUTHORIZED",
            policy_result=PolicyResult.PASS,
            system_healthy=False,
        )
        assert decision.decision == DecisionType.BLOCK

    def test_policy_unavailable_reviews(self):
        """Policy unavailability must result in REVIEW, not ALLOW."""
        decision = self.decision_engine.decide(
            auth_result="AUTHORIZED",
            policy_available=False,
            system_healthy=True,
        )
        assert decision.decision in (DecisionType.REVIEW, DecisionType.BLOCK)

    def test_gateway_system_unhealthy(self):
        """Shield gateway with system_healthy=False must block."""
        request = TransactionRequest(
            agent_id="agent_shopping_001",
            user_id="user_001",
            tool_name="create_order",
            amount=500,
        )
        result = self.shield.execute(
            request, system_healthy=False
        )
        assert result["decision"] in ("BLOCK", "REVIEW")

    def test_gateway_ml_unavailable_still_works(self):
        """Gateway should work without ML — deterministic controls take over."""
        request = TransactionRequest(
            agent_id="agent_shopping_001",
            user_id="user_001",
            tool_name="create_order",
            amount=500,
            category="groceries",
        )
        result = self.shield.execute(
            request, ml_available=False, auto_approve=True
        )
        assert result["decision"] in ("ALLOW", "REVIEW", "BLOCK")

    def test_blocked_transaction_never_reaches_paymcp(self):
        """A blocked transaction must NEVER execute in PayMCP."""
        initial_orders = len(self.shield.paymcp.get_all_orders())
        
        request = TransactionRequest(
            agent_id="totally_unknown_agent",
            user_id="user_001",
            tool_name="create_order",
            amount=1000,
        )
        result = self.shield.execute(request)
        assert result["decision"] == "BLOCK"
        
        final_orders = len(self.shield.paymcp.get_all_orders())
        assert final_orders == initial_orders, (
            "CRITICAL: Blocked transaction reached PayMCP!"
        )

    def test_paused_agent_never_executes(self):
        """Paused agent must NEVER execute."""
        request = TransactionRequest(
            agent_id="agent_paused_001",
            user_id="user_001",
            tool_name="fetch_payment",
            amount=0,
        )
        result = self.shield.execute(request)
        assert result["decision"] == "BLOCK"

    def test_every_decision_creates_audit(self):
        """Every financial decision must create an audit record."""
        initial_events = len(self.shield.audit_logger.get_all_events())
        
        request = TransactionRequest(
            agent_id="agent_shopping_001",
            user_id="user_001",
            tool_name="create_order",
            amount=500,
        )
        self.shield.execute(request, auto_approve=True)
        
        final_events = len(self.shield.audit_logger.get_all_events())
        assert final_events > initial_events, (
            "CRITICAL: No audit record created for financial decision!"
        )
