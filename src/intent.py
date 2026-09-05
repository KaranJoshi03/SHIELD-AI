"""
SHIELD AI — Intent Engine
Extracts user financial intent from natural language and validates agent actions against the original intent.

Three core capabilities:
1. Intent Extraction — parse user instructions into structured constraints
2. Intent Alignment — score how well an agent action matches intent
3. Intent Drift — detect when agent deviates from original objective

NOTE: This uses rule-based extraction, not LLM-based. For a production system, LLM-assisted extraction could be added as a supporting signal,
but deterministic validation remains authoritative.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from .models import TransactionRequest, UserIntent

class IntentExtractor:
    """
    Extracts structured financial intent from natural language.
    
    Example:
        "Book a hotel in Delhi under ₹5,000"
        → max_amount=5000, category=travel/hotel, currency=INR,
          purpose=hotel booking, constraints={location: Delhi}
    """

    AMOUNT_PATTERNS = [
        re.compile(r'(?:under|below|less than|at most|up to|max|maximum|limit)\s*[₹$]?\s*([\d,]+(?:\.\d+)?)', re.IGNORECASE),
        re.compile(r'[₹$]\s*([\d,]+(?:\.\d+)?)', re.IGNORECASE),
        re.compile(r'([\d,]+(?:\.\d+)?)\s*(?:INR|USD|rupees?|dollars?)', re.IGNORECASE),
        re.compile(r'budget\s*(?:of|is)?\s*[₹$]?\s*([\d,]+(?:\.\d+)?)', re.IGNORECASE),
    ]

    CATEGORY_MAP = {
        r'\b(?:hotel|stay|accommodation|resort|booking)\b': "travel",
        r'\b(?:flight|airline|ticket|travel|trip)\b': "travel",
        r'\b(?:groceries|grocery|grocer|food|vegetable|fruit|supermarket|provision)\b': "groceries",
        r'\b(?:electronic|gadget|phone|laptop|computer|device)\b': "electronics",
        r'\b(?:cloth|wear|fashion|shirt|dress|shoe)\b': "clothing",
        r'\b(?:restaurant|dine|eat|cafe|lunch|dinner|breakfast)\b': "food",
        r'\b(?:medicine|pharma|health|doctor|hospital)\b': "health",
        r'\b(?:book|education|course|school|tuition|study)\b': "education",
        r'\b(?:bill|utility|electric|water|internet|recharge)\b': "utilities",
        r'\b(?:gift|present|surprise)\b': "gifts",
        r'\b(?:rent|lease|property)\b': "rent",
        r'\b(?:insurance|premium|policy)\b': "insurance",
        r'\b(?:invest|stock|mutual fund|SIP)\b': "investment",
        r'\b(?:refund|return|exchange)\b': "refund",
        r'\b(?:salary|payroll|wages)\b': "payroll",
        r'\b(?:vendor|supplier|procurement)\b': "vendor_payment",
    }

    PURPOSE_PATTERNS = [
        re.compile(r'(?:buy|purchase|order|get|book)\s+(?:a\s+)?(.+?)(?:\s+(?:under|below|for|in|at|from))', re.IGNORECASE),
        re.compile(r'(?:pay|send|transfer)\s+(?:for\s+)?(.+?)(?:\s+(?:under|below|of|to))', re.IGNORECASE),
        re.compile(r'(?:book|reserve)\s+(?:a\s+)?(.+?)(?:\s+(?:under|below|for|in|at))', re.IGNORECASE),
    ]

    LOCATION_PATTERN = re.compile(
        r'(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        re.IGNORECASE
    )

    MERCHANT_PATTERN = re.compile(
        r'(?:from|at|on)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
    )

    def extract(self, text: str, user_id: str = "") -> UserIntent:
        max_amount = float('inf')
        for pattern in self.AMOUNT_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    max_amount = float(match.group(1).replace(",", ""))
                    break
                except ValueError:
                    continue

        category = "general"
        for pattern, cat in self.CATEGORY_MAP.items():
            if re.search(pattern, text, re.IGNORECASE):
                category = cat
                break

        currency = "INR"
        if re.search(r'\$|USD|dollar', text, re.IGNORECASE):
            currency = "USD"
        elif re.search(r'€|EUR|euro', text, re.IGNORECASE):
            currency = "EUR"

        purpose = ""
        for pattern in self.PURPOSE_PATTERNS:
            match = pattern.search(text)
            if match:
                purpose = match.group(1).strip()
                break
        if not purpose:
            purpose = text[:100]

        location_match = self.LOCATION_PATTERN.search(text)
        location = location_match.group(1) if location_match else ""

        merchant_constraints = []
        merchant_matches = self.MERCHANT_PATTERN.findall(text)
        stop_words = {"INR", "USD", "EUR", "The", "This", "That", "Book", "Buy", "Pay", "Get"}
        merchant_constraints = [m for m in merchant_matches if m not in stop_words]

        extracted_constraints: Dict[str, Any] = {}
        if location:
            extracted_constraints["location"] = location
        if max_amount != float('inf'):
            extracted_constraints["max_amount"] = max_amount
        extracted_constraints["original_text"] = text

        return UserIntent(
            user_id=user_id,
            original_text=text,
            max_amount=max_amount,
            currency=currency,
            category=category,
            merchant_constraints=merchant_constraints,
            allowed_purpose=purpose,
            extracted_constraints=extracted_constraints,
        )

class IntentAlignmentScorer:
    """
    Scores how well an agent's proposed action aligns with the original user intent.
    Returns a score from 0.0 (no alignment) to 1.0 (perfect alignment).
    """

    def score(self, intent: UserIntent,
              request: TransactionRequest) -> Tuple[float, List[str]]:
        scores: List[float] = []
        reasons: List[str] = []

        # 1. Amount alignment (weight: 40%)
        amount_score, amount_reasons = self._score_amount(intent, request)
        scores.append(amount_score * 0.4)
        reasons.extend(amount_reasons)

        # 2. Category alignment (weight: 25%)
        category_score, category_reasons = self._score_category(intent, request)
        scores.append(category_score * 0.25)
        reasons.extend(category_reasons)

        # 3. Currency match (weight: 15%)
        currency_score = 1.0 if request.currency == intent.currency else 0.0
        scores.append(currency_score * 0.15)
        if currency_score < 1.0:
            reasons.append(f"Currency mismatch: expected {intent.currency}, got {request.currency}")

        # 4. Merchant constraints (weight: 10%)
        merchant_score = self._score_merchant(intent, request)
        scores.append(merchant_score * 0.10)

        # 5. Purpose alignment (weight: 10%)
        purpose_score = self._score_purpose(intent, request)
        scores.append(purpose_score * 0.10)

        total_score = min(1.0, max(0.0, sum(scores)))

        if total_score >= 0.8:
            reasons.insert(0, "HIGH ALIGNMENT: Action closely matches user intent")
        elif total_score >= 0.5:
            reasons.insert(0, "MEDIUM ALIGNMENT: Action partially matches user intent")
        else:
            reasons.insert(0, "LOW ALIGNMENT: Action significantly deviates from user intent")

        return total_score, reasons

    def _score_amount(self, intent: UserIntent,
                      request: TransactionRequest) -> Tuple[float, List[str]]:
        """Score amount alignment."""
        reasons = []
        
        if intent.max_amount == float('inf'):
            return 1.0, ["No amount constraint specified"]

        if request.amount <= intent.max_amount:
            ratio = request.amount / intent.max_amount
            score = 1.0
            if ratio > 0.9:
                reasons.append(f"Amount ₹{request.amount:,.0f} near limit ₹{intent.max_amount:,.0f}")
            else:
                reasons.append(f"Amount ₹{request.amount:,.0f} within limit ₹{intent.max_amount:,.0f}")
            return score, reasons
        else:
            overage_ratio = request.amount / intent.max_amount
            if overage_ratio <= 1.2:
                score = 0.5
                reasons.append(
                    f"Amount ₹{request.amount:,.0f} slightly exceeds limit "
                    f"₹{intent.max_amount:,.0f} ({(overage_ratio-1)*100:.0f}% over)"
                )
            elif overage_ratio <= 2.0:
                score = 0.2
                reasons.append(
                    f"Amount ₹{request.amount:,.0f} significantly exceeds limit "
                    f"₹{intent.max_amount:,.0f} ({(overage_ratio-1)*100:.0f}% over)"
                )
            else:
                score = 0.0
                reasons.append(
                    f"Amount ₹{request.amount:,.0f} drastically exceeds limit "
                    f"₹{intent.max_amount:,.0f} ({(overage_ratio-1)*100:.0f}% over)"
                )
            return score, reasons

    def _score_category(self, intent: UserIntent,
                        request: TransactionRequest) -> Tuple[float, List[str]]:
        """Score category alignment."""
        if intent.category == "general":
            return 1.0, []

        if request.category == intent.category:
            return 1.0, [f"Category '{request.category}' matches intent"]

        related = {
            "travel": ["hotel", "flight", "transport"],
            "groceries": ["food", "provisions"],
            "electronics": ["gadgets", "devices"],
            "food": ["restaurant", "dining", "groceries"],
        }
        if request.category in related.get(intent.category, []):
            return 0.7, [f"Category '{request.category}' is related to '{intent.category}'"]

        return 0.2, [f"Category mismatch: expected '{intent.category}', got '{request.category}'"]

    def _score_merchant(self, intent: UserIntent,
                        request: TransactionRequest) -> float:
        if not intent.merchant_constraints:
            return 1.0
        if request.merchant_id in intent.merchant_constraints:
            return 1.0
        # Check partial match
        for constraint in intent.merchant_constraints:
            if constraint.lower() in request.merchant_id.lower():
                return 0.8
        return 0.3

    def _score_purpose(self, intent: UserIntent,
                       request: TransactionRequest) -> float:
        if not intent.allowed_purpose or not request.purpose:
            return 0.7 

        intent_words = set(intent.allowed_purpose.lower().split())
        request_words = set(request.purpose.lower().split())
        
        if not intent_words:
            return 0.7

        overlap = len(intent_words & request_words) / len(intent_words)
        return max(0.3, overlap)


class IntentDriftDetector:
    """
    Detects when an agent's actions drift away from the original user objective over a session.
    
    Tracks:
    - Cumulative spending vs. intent
    - Category changes
    - Amount escalation
    - Tool escalation (e.g., order → payout)
    """

    def __init__(self):
        self._session_history: Dict[str, List[Dict[str, Any]]] = {}

    def record_action(self, session_id: str, request: TransactionRequest,
                      alignment_score: float) -> None:
        if session_id not in self._session_history:
            self._session_history[session_id] = []

        self._session_history[session_id].append({
            "request": request,
            "alignment_score": alignment_score,
            "timestamp": datetime.utcnow(),
        })

    def detect_drift(self, session_id: str, intent: UserIntent,
                     current_request: TransactionRequest) -> Tuple[float, List[str]]:
        """
        Detect intent drift in a session.
        
        Returns:
            (drift_score 0-1, drift_reasons)
            Higher score = more drift (worse)
        """
        history = self._session_history.get(session_id, [])
        drift_score = 0.0
        reasons: List[str] = []

        total_spent = sum(h["request"].amount for h in history) + current_request.amount
        if intent.max_amount != float('inf') and total_spent > intent.max_amount:
            overage = total_spent / intent.max_amount
            drift_score = max(drift_score, min(1.0, (overage - 1.0) * 2))
            reasons.append(
                f"AMOUNT DRIFT: Cumulative spend ₹{total_spent:,.0f} exceeds "
                f"intent limit ₹{intent.max_amount:,.0f}"
            )

        if history and intent.category != "general":
            prev_categories = set(h["request"].category for h in history)
            if current_request.category not in prev_categories and \
               current_request.category != intent.category:
                drift_score = max(drift_score, 0.6)
                reasons.append(
                    f"CATEGORY DRIFT: Shifted from '{intent.category}' "
                    f"to '{current_request.category}'"
                )

        if history:
            prev_amounts = [h["request"].amount for h in history]
            avg_prev = sum(prev_amounts) / len(prev_amounts)
            if current_request.amount > avg_prev * 3:
                drift_score = max(drift_score, 0.7)
                reasons.append(
                    f"AMOUNT ESCALATION: Current ₹{current_request.amount:,.0f} "
                    f"vs average ₹{avg_prev:,.0f}"
                )

        if history:
            prev_tools = set(h["request"].tool_name for h in history)
            escalation_tools = {"create_payout", "refund_payment"}
            if current_request.tool_name in escalation_tools and \
               current_request.tool_name not in prev_tools:
                drift_score = max(drift_score, 0.8)
                reasons.append(
                    f"TOOL ESCALATION: Escalated to '{current_request.tool_name}' "
                    f"from {prev_tools}"
                )

        if len(history) >= 2:
            recent_scores = [h["alignment_score"] for h in history[-3:]]
            if all(s < 0.5 for s in recent_scores):
                drift_score = max(drift_score, 0.7)
                reasons.append(
                    f"ALIGNMENT DECLINE: Recent scores {[f'{s:.2f}' for s in recent_scores]}"
                )

        if not reasons:
            reasons.append("No intent drift detected")

        return drift_score, reasons

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        history = self._session_history.get(session_id, [])
        if not history:
            return {"actions": 0, "total_spent": 0}

        return {
            "actions": len(history),
            "total_spent": sum(h["request"].amount for h in history),
            "tools_used": list(set(h["request"].tool_name for h in history)),
            "categories": list(set(h["request"].category for h in history)),
            "avg_alignment": sum(h["alignment_score"] for h in history) / len(history),
        }

    def reset_session(self, session_id: str) -> None:
        self._session_history.pop(session_id, None)

    def reset_all(self) -> None:
        self._session_history.clear()
