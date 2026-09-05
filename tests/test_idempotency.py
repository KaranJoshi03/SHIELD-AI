"""
SHIELD AI — Idempotency Tests
Tests for idempotency and replay defense:
- Same idempotency_key never creates multiple transactions
- Replay of expired requests is blocked
- Request ID uniqueness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from datetime import datetime, timedelta
from src.idempotency import IdempotencyEngine
from src.models import TransactionRequest


class TestIdempotencyEngine:
    def setup_method(self):
        self.engine = IdempotencyEngine()

    def test_unique_request_passes(self):
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, idempotency_key="key_001"
        )
        result, data = self.engine.check(request)
        assert result == "UNIQUE"

    def test_duplicate_idempotency_key_caught(self):
        """Same idempotency key should be caught as duplicate."""
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, idempotency_key="key_dup"
        )
        self.engine.reserve(request)
        self.engine.commit(request, {"status": "success"})
        
        request2 = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, idempotency_key="key_dup"
        )
        result, data = self.engine.check(request2)
        assert result == "DUPLICATE_IDEMPOTENCY"

    def test_duplicate_request_id_caught(self):
        """Same request_id should be caught."""
        request = TransactionRequest(
            request_id="req_same",
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000
        )
        self.engine.reserve(request)
        
        request2 = TransactionRequest(
            request_id="req_same",
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000
        )
        result, data = self.engine.check(request2)
        assert result == "DUPLICATE_REQUEST_ID"

    def test_expired_request_rejected(self):
        """Old request should be rejected as replay."""
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, timestamp=datetime.utcnow() - timedelta(hours=2)
        )
        result, data = self.engine.check(request)
        assert result == "REPLAY_EXPIRED"

    def test_reservation_prevents_double_execute(self):
        """After reservation, second reserve should fail."""
        request = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, idempotency_key="key_reserve"
        )
        assert self.engine.reserve(request) is True
        
        request2 = TransactionRequest(
            agent_id="a", user_id="u", tool_name="create_order",
            amount=1000, idempotency_key="key_reserve"
        )
        assert self.engine.reserve(request2) is False
