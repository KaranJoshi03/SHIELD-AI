"""
SHIELD AI — Policy Tests
Tests for deterministic policy evaluation:
- Max transaction limits
- Daily spending limits
- Blocked/allowed tools
- Category restrictions
- Currency restrictions
- Payout/refund limits
- Policy compilation from natural language
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.policies import PolicyEngine, PolicyCompiler
from src.models import Agent, AgentStatus, PolicyResult, TransactionRequest


def make_agent(role="shopping", trust=85.0) -> Agent:
    return Agent(
        agent_id="test_agent", agent_name="TestAgent",
        owner_id="test", status=AgentStatus.ACTIVE,
        capabilities=["create_order", "fetch_payment"],
        trust_score=trust, role=role,
    )


class TestPolicyEngine:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_shopping_under_limit_passes(self):
        """₹1,200 should pass for shopping agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_order", amount=1200),
            make_agent("shopping")
        )
        assert result.result == PolicyResult.PASS

    def test_shopping_over_limit_fails(self):
        """₹20,000 should fail for shopping agent (max 5000)."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_order", amount=20000),
            make_agent("shopping")
        )
        assert result.result == PolicyResult.FAIL

    def test_shopping_blocked_tool_fails(self):
        """Refund should fail for shopping agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="refund_payment", amount=500),
            make_agent("shopping")
        )
        assert result.result == PolicyResult.FAIL

    def test_shopping_payout_blocked(self):
        """Payout must be blocked for shopping agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_payout", amount=5000),
            make_agent("shopping")
        )
        assert result.result == PolicyResult.FAIL

    def test_support_refund_within_limit(self):
        """₹2,000 refund should pass for support agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="refund_payment", amount=2000),
            make_agent("support")
        )
        assert result.result in (PolicyResult.PASS, PolicyResult.REVIEW)

    def test_support_refund_over_limit(self):
        """₹5,000 refund should fail for support agent (max 3000)."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="refund_payment", amount=5000),
            make_agent("support")
        )
        assert result.result == PolicyResult.FAIL

    def test_finance_payout_within_limit(self):
        """₹20,000 payout should pass for finance agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_payout", amount=20000),
            make_agent("finance")
        )
        assert result.result in (PolicyResult.PASS, PolicyResult.REVIEW)

    def test_finance_payout_over_limit(self):
        """₹30,000 payout should fail for finance agent (max 25000)."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_payout", amount=30000),
            make_agent("finance")
        )
        assert result.result == PolicyResult.FAIL

    def test_blocked_category_fails(self):
        """Gambling category should be blocked for shopping agent."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_order", amount=500,
                             category="gambling"),
            make_agent("shopping")
        )
        assert result.result == PolicyResult.FAIL

    def test_policy_violations_have_reasons(self):
        """Policy failures should have human-readable reasons."""
        result = self.engine.evaluate(
            TransactionRequest(agent_id="t", user_id="u",
                             tool_name="create_order", amount=20000),
            make_agent("shopping")
        )
        assert len(result.reasons) > 0
        assert len(result.violations) > 0


class TestPolicyCompiler:
    def setup_method(self):
        self.compiler = PolicyCompiler()

    def test_compile_shopping_policy(self):
        """Compile natural language into structured shopping policy."""
        policy = self.compiler.compile(
            "Shopping agents can spend at most 5000 INR per transaction "
            "and cannot issue refunds or payouts.",
            "shopping"
        )
        assert policy.max_transaction == 5000
        assert "refund_payment" in policy.blocked_tools
        assert "create_payout" in policy.blocked_tools
        assert "INR" in policy.allowed_currencies

    def test_compile_extracts_amount(self):
        """Compiler should extract amount from natural language."""
        policy = self.compiler.compile(
            "Agents can spend up to 10000 INR",
            "general"
        )
        assert policy.max_transaction == 10000

    def test_compiled_policy_is_deterministic(self):
        """Same input should produce same compiled policy."""
        text = "Shopping agents maximum 3000 INR, cannot issue refunds"
        p1 = self.compiler.compile(text, "shopping")
        p2 = self.compiler.compile(text, "shopping")
        assert p1.max_transaction == p2.max_transaction
        assert p1.blocked_tools == p2.blocked_tools
