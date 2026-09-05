"""
SHIELD AI — Behavior Tests
Tests for agent behavior analysis and anomaly detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.behavior import BehaviorEngine
from src.models import BehaviorProfile, TransactionRequest


class TestBehaviorEngine:
    def setup_method(self):
        self.engine = BehaviorEngine()
        self.profile = BehaviorProfile(
            agent_id="agent_001",
            avg_requests_per_hour=5.0,
            avg_transaction_amount=1500.0,
            std_transaction_amount=800.0,
            tool_distribution={"create_order": 0.7, "fetch_payment": 0.3},
        )

    def test_normal_behavior(self):
        request = TransactionRequest(
            agent_id="agent_001", user_id="u",
            tool_name="create_order", amount=1500
        )
        score, reasons = self.engine.analyze("agent_001", request, self.profile)
        assert score < 50, f"Normal behavior should have low score, got {score}"

    def test_amount_anomaly(self):
        """Very large amount should trigger anomaly."""
        request = TransactionRequest(
            agent_id="agent_001", user_id="u",
            tool_name="create_order", amount=50000
        )
        score, reasons = self.engine.analyze("agent_001", request, self.profile)
        assert score > 20, f"Amount anomaly expected, got {score}"

    def test_velocity_detection(self):
        """Many rapid requests should trigger velocity detection."""
        for i in range(15):
            req = TransactionRequest(
                agent_id="agent_fast", user_id="u",
                tool_name="create_order", amount=100
            )
            self.engine.record_action("agent_fast", req)

        req = TransactionRequest(
            agent_id="agent_fast", user_id="u",
            tool_name="create_order", amount=100
        )
        score, reasons = self.engine.analyze("agent_fast", req)
        assert score > 30

    def test_runaway_detection(self):
        """Runaway agent detection after burst of requests."""
        for i in range(15):
            req = TransactionRequest(
                agent_id="agent_runaway", user_id="u",
                tool_name="create_order", amount=100
            )
            self.engine.record_action("agent_runaway", req)

        is_runaway, reason = self.engine.is_runaway("agent_runaway")
        assert is_runaway is True

    def test_no_runaway_for_normal(self):
        """Normal activity should not trigger runaway."""
        req = TransactionRequest(
            agent_id="agent_normal", user_id="u",
            tool_name="create_order", amount=100
        )
        self.engine.record_action("agent_normal", req)
        is_runaway, _ = self.engine.is_runaway("agent_normal")
        assert is_runaway is False
