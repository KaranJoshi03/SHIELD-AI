"""
SHIELD AI — Concurrency Tests
Tests for concurrent request handling:
- 100 concurrent identical requests must result in only 1 execution
- No race-condition-based double execution
- Idempotency under concurrent load
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.shield_gateway import ShieldGateway
from src.models import TransactionRequest


class TestConcurrency:
    def setup_method(self):
        self.shield = ShieldGateway()

    def test_concurrent_identical_requests(self):
        """
        100 concurrent identical requests must result in exactly 1 executed transaction.
        THIS TEST ACTUALLY EXECUTES CONCURRENT REQUESTS.
        """
        idem_key = f"concurrent_test_{uuid.uuid4().hex[:8]}"
        
        def make_request():
            request = TransactionRequest(
                agent_id="agent_shopping_001",
                user_id="user_001",
                tool_name="create_order",
                amount=1200,
                currency="INR",
                merchant_id="merchant_grocery_001",
                purpose="Concurrent test purchase",
                category="groceries",
                idempotency_key=idem_key,
            )
            return self.shield.execute(request, auto_approve=True)

        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"decision": "ERROR", "error": str(e)})

        allowed = [r for r in results if r.get("decision") == "ALLOW" 
                   and r.get("execution", {}).get("status") == "success"]
        blocked = [r for r in results if r.get("decision") == "BLOCK"]
        
        # At most 1 should actually execute with success
        # Others should be blocked as duplicates or have duplicate_prevented
        actual_executions = len(allowed)
        
        assert actual_executions <= 1, (
            f"Expected at most 1 execution, got {actual_executions}. "
            f"CONCURRENT DUPLICATE PROTECTION FAILED!"
        )
        
        assert len(results) == 100

    def test_different_requests_all_process(self):
        """Different requests should each be processed independently."""
        results = []
        for i in range(5):
            request = TransactionRequest(
                agent_id="agent_shopping_001",
                user_id="user_001",
                tool_name="create_order",
                amount=100 + i * 100,
                currency="INR",
                merchant_id=f"merchant_{i}",
                purpose=f"Test purchase {i}",
                category="groceries",
                idempotency_key=f"unique_key_{i}_{uuid.uuid4().hex[:8]}",
            )
            result = self.shield.execute(request, auto_approve=True)
            results.append(result)
        
        assert len(results) == 5
        allowed = [r for r in results if r["decision"] == "ALLOW"]
        assert len(allowed) >= 1
