"""
SHIELD AI — Risk Tests
Tests for transaction risk evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.risk import TransactionRiskEngine
from src.models import Agent, AgentStatus, BehaviorProfile, RiskLevel, TransactionRequest


class TestRiskEngine:
    def setup_method(self):
        self.engine = TransactionRiskEngine()

    def test_low_amount_low_risk(self):
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order", amount=500
        )
        agent = Agent(agent_id="a", agent_name="A", owner_id="o",
                     trust_score=85, role="shopping")
        score, level, signals = self.engine.evaluate(request, agent)
        assert level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_high_amount_higher_risk(self):
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order", amount=50000
        )
        score, level, signals = self.engine.evaluate(request)
        assert score > 30  # Should be elevated

    def test_new_merchant_increases_risk(self):
        """New merchant should increase risk but NOT automatically block."""
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, merchant_id="brand_new_merchant"
        )
        score1, _, _ = self.engine.evaluate(request)
        
        # Same merchant again should have lower risk
        request2 = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, merchant_id="brand_new_merchant"
        )
        score2, _, _ = self.engine.evaluate(request2)
        assert score2 <= score1

    def test_low_trust_increases_risk(self):
        agent_low = Agent(agent_id="a", agent_name="A", owner_id="o",
                         trust_score=20, role="shopping")
        agent_high = Agent(agent_id="a", agent_name="A", owner_id="o",
                          trust_score=90, role="shopping")
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order", amount=1000
        )
        score_low, _, _ = self.engine.evaluate(request, agent_low)
        self.engine.reset()
        score_high, _, _ = self.engine.evaluate(request, agent_high)
        assert score_low > score_high

    def test_risk_score_bounded(self):
        """Risk score must be between 0 and 100."""
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order", amount=100000
        )
        score, _, _ = self.engine.evaluate(request)
        assert 0 <= score <= 100

    def test_injection_increases_risk(self):
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order", amount=1000
        )
        score_clean, _, _ = self.engine.evaluate(request, injection_detected=False)
        self.engine.reset()
        score_inject, _, _ = self.engine.evaluate(request, injection_detected=True)
        assert score_inject > score_clean
