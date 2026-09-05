"""
SHIELD AI — Transaction Risk Engine
Deterministic risk scoring for financial transactions.

The risk engine evaluates multiple signals and produces
a risk_score (0-100) and risk_level (LOW/MEDIUM/HIGH/CRITICAL).

Signals evaluated:
- Amount deviation from historical behavior
- Transaction frequency
- New merchant / new device / unusual location
- Category risk
- Agent trust level
- Intent alignment
- Previous violations
- Recent failures
- Duplicate fingerprint
- Workflow anomaly
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Agent, BehaviorProfile, RiskLevel, TransactionRequest
)


class RiskSignal:
    def __init__(self, name: str, score: float, weight: float, reason: str = ""):
        self.name = name
        self.score = max(0.0, min(100.0, score))
        self.weight = weight
        self.reason = reason


class TransactionRiskEngine:
    """
    Deterministic risk scoring engine.
    Produces risk_score (0-100) from weighted combination of signals.
    
    Risk levels:
        0-25:   LOW
        25-50:  MEDIUM
        50-75:  HIGH
        75-100: CRITICAL
    
    IMPORTANT: "New device" or "new merchant" are risk SIGNALS, not automatic fraud indicators. They increase risk score
    but do not automatically result in BLOCK.
    """

    RISK_THRESHOLDS = {
        RiskLevel.LOW: (0, 25),
        RiskLevel.MEDIUM: (25, 50),
        RiskLevel.HIGH: (50, 75),
        RiskLevel.CRITICAL: (75, 100),
    }

    def __init__(self):
        self._recent_transactions: Dict[str, List[TransactionRequest]] = {}
        self._known_merchants: Dict[str, set] = {}
        self._known_devices: Dict[str, set] = {}
        self._historical_amounts: Dict[str, List[float]] = {}

    def evaluate(self, request: TransactionRequest,
                 agent: Optional[Agent] = None,
                 behavior_profile: Optional[BehaviorProfile] = None,
                 intent_alignment: float = 1.0,
                 duplicate_detected: bool = False,
                 workflow_violation: bool = False,
                 injection_detected: bool = False) -> Tuple[float, RiskLevel, List[RiskSignal]]:
        """
        Evaluate transaction risk.
        Returns:
            (risk_score, risk_level, signals)
        """
        signals: List[RiskSignal] = []

        # 1. Amount risk
        signals.append(self._evaluate_amount(request, agent, behavior_profile))

        # 2. Amount deviation
        signals.append(self._evaluate_amount_deviation(request, behavior_profile))

        # 3. Frequency/velocity
        signals.append(self._evaluate_velocity(request))

        # 4. New merchant
        signals.append(self._evaluate_merchant(request))

        # 5. New device
        signals.append(self._evaluate_device(request))

        # 6. Location
        signals.append(self._evaluate_location(request))

        # 7. Category risk
        signals.append(self._evaluate_category(request))

        # 8. Agent trust
        signals.append(self._evaluate_trust(agent))

        # 9. Intent alignment
        signals.append(self._evaluate_intent_alignment(intent_alignment))

        # 10. Previous violations
        signals.append(self._evaluate_violations(agent))

        # 11. Recent failures
        signals.append(self._evaluate_failures(agent))

        # 12. Duplicate
        signals.append(self._evaluate_duplicate(duplicate_detected))

        # 13. Workflow anomaly
        signals.append(self._evaluate_workflow(workflow_violation))

        # 14. Injection
        signals.append(self._evaluate_injection(injection_detected))

        total_weight = sum(s.weight for s in signals)
        if total_weight > 0:
            risk_score = sum(s.score * s.weight for s in signals) / total_weight
        else:
            risk_score = 50.0

        risk_score = max(0.0, min(100.0, risk_score))
        risk_level = self._classify_risk(risk_score)

        self._track_transaction(request)

        return risk_score, risk_level, signals

    def _evaluate_amount(self, request: TransactionRequest,
                         agent: Optional[Agent],
                         profile: Optional[BehaviorProfile]) -> RiskSignal:
        """Amount-based risk signal."""
        if request.amount <= 500:
            return RiskSignal("amount", 5.0, 0.15, "Low amount transaction")
        elif request.amount <= 2000:
            return RiskSignal("amount", 15.0, 0.15, "Moderate amount")
        elif request.amount <= 5000:
            return RiskSignal("amount", 30.0, 0.15, "Medium-high amount")
        elif request.amount <= 10000:
            return RiskSignal("amount", 50.0, 0.15, "High amount")
        elif request.amount <= 25000:
            return RiskSignal("amount", 70.0, 0.15, "Very high amount")
        else:
            return RiskSignal("amount", 90.0, 0.15, f"Extremely high amount: ₹{request.amount:,.0f}")

    def _evaluate_amount_deviation(self, request: TransactionRequest,
                                    profile: Optional[BehaviorProfile]) -> RiskSignal:
        """Amount deviation from historical average."""
        if profile is None:
            return RiskSignal("amount_deviation", 30.0, 0.10, "No behavior profile available")

        if profile.std_transaction_amount == 0:
            return RiskSignal("amount_deviation", 20.0, 0.10, "No standard deviation data")

        z_score = abs(request.amount - profile.avg_transaction_amount) / max(profile.std_transaction_amount, 1)
        
        if z_score <= 1.0:
            return RiskSignal("amount_deviation", 5.0, 0.10, f"Normal amount (z={z_score:.1f})")
        elif z_score <= 2.0:
            return RiskSignal("amount_deviation", 25.0, 0.10, f"Slightly unusual amount (z={z_score:.1f})")
        elif z_score <= 3.0:
            return RiskSignal("amount_deviation", 55.0, 0.10, f"Unusual amount (z={z_score:.1f})")
        else:
            return RiskSignal("amount_deviation", 85.0, 0.10, f"Highly unusual amount (z={z_score:.1f})")

    def _evaluate_velocity(self, request: TransactionRequest) -> RiskSignal:
        """Transaction frequency/velocity signal."""
        recent = self._recent_transactions.get(request.agent_id, [])
        count = len(recent)
        
        if count <= 5:
            return RiskSignal("velocity", 5.0, 0.10, f"Normal velocity ({count} recent)")
        elif count <= 15:
            return RiskSignal("velocity", 35.0, 0.10, f"Elevated velocity ({count} recent)")
        elif count <= 30:
            return RiskSignal("velocity", 65.0, 0.10, f"High velocity ({count} recent)")
        else:
            return RiskSignal("velocity", 95.0, 0.10, f"Extreme velocity ({count} recent)")

    def _evaluate_merchant(self, request: TransactionRequest) -> RiskSignal:
        known = self._known_merchants.get(request.agent_id, set())
        
        if not request.merchant_id:
            return RiskSignal("new_merchant", 10.0, 0.05, "No merchant specified")
        
        if request.merchant_id in known:
            return RiskSignal("new_merchant", 5.0, 0.05, "Known merchant")
        else:
            return RiskSignal("new_merchant", 35.0, 0.05,
                            f"First-time merchant: {request.merchant_id}")

    def _evaluate_device(self, request: TransactionRequest) -> RiskSignal:
        known = self._known_devices.get(request.agent_id, set())
        
        if not request.device_id:
            return RiskSignal("new_device", 10.0, 0.05, "No device specified")
        
        if request.device_id in known:
            return RiskSignal("new_device", 5.0, 0.05, "Known device")
        else:
            return RiskSignal("new_device", 35.0, 0.05,
                            f"New device: {request.device_id}")

    def _evaluate_location(self, request: TransactionRequest) -> RiskSignal:
        if not request.location:
            return RiskSignal("location", 10.0, 0.05, "No location data")

        # High-risk locations (simplified simulation)
        high_risk = {"unknown", "proxy", "tor", "vpn"}
        if request.location.lower() in high_risk:
            return RiskSignal("location", 70.0, 0.05, f"High-risk location: {request.location}")
        
        return RiskSignal("location", 5.0, 0.05, f"Normal location: {request.location}")

    def _evaluate_category(self, request: TransactionRequest) -> RiskSignal:
        high_risk_categories = {"gambling", "crypto", "adult", "cash_advance"}
        medium_risk_categories = {"luxury", "jewelry", "electronics"}
        
        if request.category in high_risk_categories:
            return RiskSignal("category", 80.0, 0.05, f"High-risk category: {request.category}")
        elif request.category in medium_risk_categories:
            return RiskSignal("category", 35.0, 0.05, f"Medium-risk category: {request.category}")
        else:
            return RiskSignal("category", 5.0, 0.05, f"Normal category: {request.category}")

    def _evaluate_trust(self, agent: Optional[Agent]) -> RiskSignal:
        if agent is None:
            return RiskSignal("agent_trust", 90.0, 0.10, "Unknown agent — no trust data")

        # Inverse: low trust = high risk
        risk = max(0, 100 - agent.trust_score)
        
        if agent.trust_score >= 80:
            return RiskSignal("agent_trust", risk, 0.10, f"High trust: {agent.trust_score:.0f}")
        elif agent.trust_score >= 50:
            return RiskSignal("agent_trust", risk, 0.10, f"Medium trust: {agent.trust_score:.0f}")
        else:
            return RiskSignal("agent_trust", risk, 0.10, f"Low trust: {agent.trust_score:.0f}")

    def _evaluate_intent_alignment(self, alignment: float) -> RiskSignal:
        risk = (1.0 - alignment) * 100
        if alignment >= 0.8:
            return RiskSignal("intent_alignment", risk, 0.10, f"Good alignment: {alignment:.2f}")
        elif alignment >= 0.5:
            return RiskSignal("intent_alignment", risk, 0.10, f"Moderate alignment: {alignment:.2f}")
        else:
            return RiskSignal("intent_alignment", risk, 0.10, f"Poor alignment: {alignment:.2f}")

    def _evaluate_violations(self, agent: Optional[Agent]) -> RiskSignal:
        if agent is None:
            return RiskSignal("violations", 50.0, 0.05, "Unknown agent")

        if agent.total_violations == 0:
            return RiskSignal("violations", 0.0, 0.05, "No previous violations")
        elif agent.total_violations <= 2:
            return RiskSignal("violations", 25.0, 0.05, f"{agent.total_violations} previous violations")
        elif agent.total_violations <= 5:
            return RiskSignal("violations", 50.0, 0.05, f"{agent.total_violations} previous violations")
        else:
            return RiskSignal("violations", 80.0, 0.05, f"{agent.total_violations} previous violations — high risk")

    def _evaluate_failures(self, agent: Optional[Agent]) -> RiskSignal:
        if agent is None:
            return RiskSignal("failures", 50.0, 0.05, "Unknown agent")

        if agent.total_requests == 0:
            return RiskSignal("failures", 10.0, 0.05, "No request history")

        failure_rate = agent.total_blocked / max(agent.total_requests, 1)
        if failure_rate <= 0.05:
            return RiskSignal("failures", 5.0, 0.05, f"Low failure rate: {failure_rate:.1%}")
        elif failure_rate <= 0.15:
            return RiskSignal("failures", 30.0, 0.05, f"Moderate failure rate: {failure_rate:.1%}")
        else:
            return RiskSignal("failures", 70.0, 0.05, f"High failure rate: {failure_rate:.1%}")

    def _evaluate_duplicate(self, duplicate_detected: bool) -> RiskSignal:
        if duplicate_detected:
            return RiskSignal("duplicate", 90.0, 0.05, "Duplicate transaction detected")
        return RiskSignal("duplicate", 0.0, 0.05, "Unique transaction")

    def _evaluate_workflow(self, workflow_violation: bool) -> RiskSignal:
        if workflow_violation:
            return RiskSignal("workflow", 80.0, 0.05, "Workflow violation detected")
        return RiskSignal("workflow", 0.0, 0.05, "Valid workflow")

    def _evaluate_injection(self, injection_detected: bool) -> RiskSignal:
        if injection_detected:
            return RiskSignal("injection", 95.0, 0.05, "Prompt injection indicators detected")
        return RiskSignal("injection", 0.0, 0.05, "No injection detected")

    def _classify_risk(self, score: float) -> RiskLevel:
        if score < 25:
            return RiskLevel.LOW
        elif score < 50:
            return RiskLevel.MEDIUM
        elif score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _track_transaction(self, request: TransactionRequest) -> None:
        if request.agent_id not in self._recent_transactions:
            self._recent_transactions[request.agent_id] = []
        self._recent_transactions[request.agent_id].append(request)
        
        if request.merchant_id:
            if request.agent_id not in self._known_merchants:
                self._known_merchants[request.agent_id] = set()
            self._known_merchants[request.agent_id].add(request.merchant_id)
        
        if request.device_id:
            if request.agent_id not in self._known_devices:
                self._known_devices[request.agent_id] = set()
            self._known_devices[request.agent_id].add(request.device_id)

        if request.agent_id not in self._historical_amounts:
            self._historical_amounts[request.agent_id] = []
        self._historical_amounts[request.agent_id].append(request.amount)

    def reset(self) -> None:
        self._recent_transactions.clear()
        self._known_merchants.clear()
        self._known_devices.clear()
        self._historical_amounts.clear()
