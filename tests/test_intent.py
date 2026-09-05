"""
SHIELD AI — Intent Tests
Tests for intent extraction, alignment, and drift detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from src.intent import IntentExtractor, IntentAlignmentScorer, IntentDriftDetector
from src.models import TransactionRequest


class TestIntentExtraction:
    def setup_method(self):
        self.extractor = IntentExtractor()

    def test_extract_hotel_intent(self):
        intent = self.extractor.extract("Book a hotel in Delhi under ₹5,000", "user_001")
        assert intent.max_amount == 5000
        assert intent.category == "travel"
        assert intent.currency == "INR"

    def test_extract_grocery_intent(self):
        intent = self.extractor.extract("Buy groceries under ₹2,000", "user_001")
        assert intent.max_amount == 2000
        assert intent.category == "groceries"

    def test_extract_no_amount(self):
        intent = self.extractor.extract("Check my payment status", "user_001")
        assert intent.max_amount == float('inf')

    def test_extract_usd_currency(self):
        intent = self.extractor.extract("Buy electronics for $500 USD", "user_001")
        assert intent.currency == "USD"


class TestIntentAlignment:
    def setup_method(self):
        self.scorer = IntentAlignmentScorer()
        self.extractor = IntentExtractor()

    def test_high_alignment(self):
        intent = self.extractor.extract("Book a hotel under ₹5,000", "user_001")
        request = TransactionRequest(
            agent_id="a", user_id="user_001",
            tool_name="create_order", amount=4500,
            category="travel", purpose="hotel booking"
        )
        score, reasons = self.scorer.score(intent, request)
        assert score >= 0.7, f"Expected high alignment, got {score}"

    def test_low_alignment_amount(self):
        intent = self.extractor.extract("Book a hotel under ₹5,000", "user_001")
        request = TransactionRequest(
            agent_id="a", user_id="user_001",
            tool_name="create_order", amount=18000,
            category="travel", purpose="hotel upgrade"
        )
        score, reasons = self.scorer.score(intent, request)
        assert score <= 0.6, f"Expected low alignment, got {score}"


class TestIntentDrift:
    def setup_method(self):
        self.drift = IntentDriftDetector()
        self.extractor = IntentExtractor()

    def test_no_drift_on_first_action(self):
        intent = self.extractor.extract("Buy groceries under ₹2,000", "user_001")
        request = TransactionRequest(
            agent_id="a", user_id="user_001",
            tool_name="create_order", amount=1800,
            category="groceries"
        )
        score, reasons = self.drift.detect_drift("s1", intent, request)
        assert score < 0.3

    def test_detects_tool_escalation(self):
        intent = self.extractor.extract("Buy groceries under ₹2,000", "user_001")
        req1 = TransactionRequest(
            agent_id="a", user_id="user_001",
            tool_name="create_order", amount=1800,
            category="groceries"
        )
        self.drift.record_action("s1", req1, 0.9)
        
        req2 = TransactionRequest(
            agent_id="a", user_id="user_001",
            tool_name="create_payout", amount=10000,
            category="general"
        )
        score, reasons = self.drift.detect_drift("s1", intent, req2)
        assert score >= 0.7, f"Expected drift detection, got {score}"
        assert any("ESCALATION" in r or "DRIFT" in r for r in reasons)
