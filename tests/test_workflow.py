"""
SHIELD AI — Workflow Tests
Tests for workflow and tool-chain security.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.workflow import WorkflowValidator

class TestWorkflowValidator:
    def setup_method(self):
        self.validator = WorkflowValidator()

    def test_valid_sequence(self):
        """create_order → verify_payment should be valid."""
        result, _ = self.validator.validate("s1", "a1", "create_order")
        assert result == "VALID"
        result, _ = self.validator.validate("s1", "a1", "verify_payment")
        assert result == "VALID"

    def test_invalid_transition(self):
        """create_order → create_payout should be invalid."""
        self.validator.validate("s2", "a1", "create_order")
        result, reasons = self.validator.validate("s2", "a1", "create_payout")
        assert result != "VALID"

    def test_first_tool_always_valid(self):
        """First tool in a session should always be valid."""
        result, _ = self.validator.validate("s3", "a1", "create_payout")
        assert result == "VALID"

    def test_tool_flooding(self):
        """Same tool 3+ times should trigger flooding."""
        for i in range(4):
            result, _ = self.validator.validate("s4", "a1", "fetch_payment")
        assert result == "TOOL_FLOODING"

    def test_valid_read_sequence(self):
        """fetch_payment → verify_payment is valid."""
        self.validator.validate("s5", "a1", "fetch_payment")
        result, _ = self.validator.validate("s5", "a1", "verify_payment")
        assert result == "VALID"
