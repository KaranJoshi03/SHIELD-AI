"""
SHIELD AI — Identity Tests
Tests for agent identity validation:
- Known agent validation
- Unknown agent rejection
- Disabled agent rejection
- Expired agent rejection
- Paused agent rejection
- Invalid credential simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.agents import AgentRegistry
from src.models import AgentStatus, Agent


class TestAgentIdentity:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_known_active_agent_validates(self):
        """Active known agent should pass validation."""
        is_valid, reason = self.registry.validate_agent("agent_shopping_001")
        assert is_valid is True
        assert "ACTIVE" in reason

    def test_unknown_agent_rejected(self):
        """Unknown agent must be rejected."""
        is_valid, reason = self.registry.validate_agent("agent_unknown_xyz")
        assert is_valid is False
        assert "UNKNOWN" in reason

    def test_paused_agent_rejected(self):
        """Paused agent must be rejected."""
        is_valid, reason = self.registry.validate_agent("agent_paused_001")
        assert is_valid is False
        assert "PAUSED" in reason

    def test_expired_agent_rejected(self):
        """Expired agent must be rejected."""
        is_valid, reason = self.registry.validate_agent("agent_expired_001")
        assert is_valid is False
        assert "EXPIRED" in reason

    def test_disabled_agent_rejected(self):
        """Disabled agent must be rejected."""
        is_valid, reason = self.registry.validate_agent("agent_disabled_001")
        assert is_valid is False
        assert "DISABLED" in reason

    def test_all_default_agents_exist(self):
        """All default agents should be registered."""
        expected_ids = [
            "agent_shopping_001", "agent_finance_001", "agent_support_001",
            "agent_paused_001", "agent_expired_001", "agent_disabled_001",
        ]
        for agent_id in expected_ids:
            assert self.registry.is_known(agent_id), f"{agent_id} not found"

    def test_unknown_agent_never_known(self):
        """Unknown agent ID should never be found."""
        assert self.registry.is_known("completely_fake_agent") is False

    def test_pause_agent(self):
        """Pausing an active agent should change its status."""
        assert self.registry.pause_agent("agent_shopping_001")
        is_valid, reason = self.registry.validate_agent("agent_shopping_001")
        assert is_valid is False
        assert "PAUSED" in reason
        self.registry.resume_agent("agent_shopping_001")

    def test_resume_agent(self):
        """Resuming a paused agent should restore ACTIVE status."""
        self.registry.pause_agent("agent_shopping_001")
        self.registry.resume_agent("agent_shopping_001")
        is_valid, reason = self.registry.validate_agent("agent_shopping_001")
        assert is_valid is True

    def test_trust_score_update(self):
        """Trust score should update within bounds."""
        original = self.registry.get_agent("agent_shopping_001").trust_score
        self.registry.update_trust_score("agent_shopping_001", 5.0)
        updated = self.registry.get_agent("agent_shopping_001").trust_score
        assert updated == min(100.0, original + 5.0)

    def test_trust_score_cannot_exceed_100(self):
        """Trust score must not exceed 100."""
        for _ in range(50):
            self.registry.update_trust_score("agent_shopping_001", 5.0)
        agent = self.registry.get_agent("agent_shopping_001")
        assert agent.trust_score <= 100.0

    def test_trust_score_cannot_go_below_0(self):
        """Trust score must not go below 0."""
        for _ in range(50):
            self.registry.update_trust_score("agent_shopping_001", -10.0)
        agent = self.registry.get_agent("agent_shopping_001")
        assert agent.trust_score >= 0.0
