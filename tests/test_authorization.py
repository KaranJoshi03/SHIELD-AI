"""
SHIELD AI — Authorization Tests

Tests for capability authorization:
- Allowed capabilities pass
- Forbidden capabilities are blocked
- Unknown agents have no capabilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.agents import AgentRegistry
from src.authorization import AuthorizationEngine
from src.models import TransactionRequest


class TestCapabilityAuthorization:

    def setup_method(self):
        self.registry = AgentRegistry()
        self.auth = AuthorizationEngine(self.registry)

    def test_shopping_agent_can_create_order(self):
        """ShoppingAgent should be allowed to create orders."""
        ok, reason = self.auth.check_capability("agent_shopping_001", "create_order")
        assert ok is True

    def test_shopping_agent_cannot_create_payout(self):
        """ShoppingAgent must NOT be allowed to create payouts."""
        ok, reason = self.auth.check_capability("agent_shopping_001", "create_payout")
        assert ok is False
        assert "DENIED" in reason

    def test_shopping_agent_cannot_refund(self):
        """ShoppingAgent must NOT be allowed to refund."""
        ok, reason = self.auth.check_capability("agent_shopping_001", "refund_payment")
        assert ok is False

    def test_finance_agent_can_create_payout(self):
        """FinanceAgent should be allowed to create payouts."""
        ok, reason = self.auth.check_capability("agent_finance_001", "create_payout")
        assert ok is True

    def test_support_agent_can_refund(self):
        """SupportAgent should be allowed to refund."""
        ok, reason = self.auth.check_capability("agent_support_001", "refund_payment")
        assert ok is True

    def test_support_agent_cannot_create_order(self):
        """SupportAgent must NOT be allowed to create orders."""
        ok, reason = self.auth.check_capability("agent_support_001", "create_order")
        assert ok is False

    def test_unknown_agent_no_capabilities(self):
        """Unknown agent should have no capabilities."""
        ok, reason = self.auth.check_capability("unknown_agent_xyz", "create_order")
        assert ok is False
        assert "UNKNOWN" in reason

    def test_full_authorization_pass(self):
        """Full auth check should pass for valid agent + capability."""
        request = TransactionRequest(
            agent_id="agent_shopping_001",
            user_id="user_001",
            tool_name="create_order",
            amount=1000,
        )
        ok, reason, agent = self.auth.authorize(request)
        assert ok is True
        assert agent is not None

    def test_full_authorization_fail_unknown(self):
        """Full auth check must fail for unknown agent."""
        request = TransactionRequest(
            agent_id="totally_unknown",
            user_id="user_001",
            tool_name="create_order",
            amount=1000,
        )
        ok, reason, agent = self.auth.authorize(request)
        assert ok is False
        assert "UNKNOWN" in reason

    def test_full_authorization_fail_capability(self):
        """Full auth check must fail for unauthorized capability."""
        request = TransactionRequest(
            agent_id="agent_shopping_001",
            user_id="user_001",
            tool_name="create_payout",
            amount=5000,
        )
        ok, reason, agent = self.auth.authorize(request)
        assert ok is False
        assert "DENIED" in reason

    def test_unauthorized_tool_never_reaches_execution(self):
        """Unauthorized tool call must result in BLOCK decision."""
        decision = self.auth.create_block_decision(
            TransactionRequest(
                agent_id="agent_shopping_001",
                user_id="user_001",
                tool_name="create_payout",
                amount=5000,
            ),
            "CAPABILITY_DENIED"
        )
        assert decision.decision.value == "BLOCK"
