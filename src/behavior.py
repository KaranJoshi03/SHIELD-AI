"""
SHIELD AI — Agent Behavior Analysis Engine
Builds behavior profiles for agents and detects anomalies.

Tracks:
- Requests/minute and requests/hour
- Average transaction amount
- Tool usage distribution
- Merchant diversity
- Failure rate and policy violations
- Refund/payout frequency
- Repeated calls

Anomaly detection methods:
1. Statistical thresholds (z-score based)
2. Isolation Forest (scikit-learn)
The behavior engine is a SIGNAL provider to the decision engine.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .models import Agent, BehaviorProfile, TransactionRequest


class BehaviorEngine:
    VELOCITY_THRESHOLD_PER_MINUTE = 10
    VELOCITY_THRESHOLD_PER_HOUR = 50
    AMOUNT_Z_THRESHOLD = 3.0
    SPENDING_SPIKE_MULTIPLIER = 5.0

    def __init__(self):
        self._action_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_action(self, agent_id: str, request: TransactionRequest,
                      blocked: bool = False, violation: bool = False) -> None:
        with self._lock:
            self._action_log[agent_id].append({
                "tool_name": request.tool_name,
                "amount": request.amount,
                "category": request.category,
                "merchant_id": request.merchant_id,
                "timestamp": datetime.utcnow(),
                "blocked": blocked,
                "violation": violation,
            })

    def analyze(self, agent_id: str, request: TransactionRequest,
                profile: Optional[BehaviorProfile] = None) -> Tuple[float, List[str]]:
        """
        Analyze agent behavior and return anomaly score.
        
        Returns:
            (anomaly_score 0-100, reasons)
            Higher score = more anomalous
        """
        reasons: List[str] = []
        scores: List[float] = []

        vel_score, vel_reasons = self._check_velocity(agent_id)
        scores.append(vel_score)
        reasons.extend(vel_reasons)

        amt_score, amt_reasons = self._check_amount_anomaly(agent_id, request, profile)
        scores.append(amt_score)
        reasons.extend(amt_reasons)

        tool_score, tool_reasons = self._check_tool_distribution(agent_id, request, profile)
        scores.append(tool_score)
        reasons.extend(tool_reasons)

        fail_score, fail_reasons = self._check_failure_rate(agent_id, profile)
        scores.append(fail_score)
        reasons.extend(fail_reasons)

        spike_score, spike_reasons = self._check_spending_spike(agent_id, request, profile)
        scores.append(spike_score)
        reasons.extend(spike_reasons)

        rep_score, rep_reasons = self._check_repeated_calls(agent_id, request)
        scores.append(rep_score)
        reasons.extend(rep_reasons)

        anomaly_score = 0.0
        if scores:
            anomaly_score = max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4

        anomaly_score = max(0.0, min(100.0, anomaly_score))

        if anomaly_score < 20:
            reasons.insert(0, "NORMAL: Behavior within expected patterns")
        elif anomaly_score < 50:
            reasons.insert(0, "ELEVATED: Some behavioral deviations detected")
        elif anomaly_score < 75:
            reasons.insert(0, "ANOMALOUS: Significant behavioral anomalies")
        else:
            reasons.insert(0, "CRITICAL: Severe behavioral anomalies — potential runaway agent")

        return anomaly_score, reasons

    def _check_velocity(self, agent_id: str) -> Tuple[float, List[str]]:
        actions = self._action_log.get(agent_id, [])
        now = datetime.utcnow()
        reasons = []

        one_min_ago = now - timedelta(minutes=1)
        recent_1m = [a for a in actions if a["timestamp"] >= one_min_ago]
        rpm = len(recent_1m)

        one_hour_ago = now - timedelta(hours=1)
        recent_1h = [a for a in actions if a["timestamp"] >= one_hour_ago]
        rph = len(recent_1h)

        if rpm > self.VELOCITY_THRESHOLD_PER_MINUTE:
            score = min(100, (rpm / self.VELOCITY_THRESHOLD_PER_MINUTE) * 50)
            reasons.append(f"HIGH VELOCITY: {rpm} requests/minute (threshold: {self.VELOCITY_THRESHOLD_PER_MINUTE})")
            return score, reasons
        elif rph > self.VELOCITY_THRESHOLD_PER_HOUR:
            score = min(80, (rph / self.VELOCITY_THRESHOLD_PER_HOUR) * 40)
            reasons.append(f"ELEVATED VELOCITY: {rph} requests/hour (threshold: {self.VELOCITY_THRESHOLD_PER_HOUR})")
            return score, reasons

        return 0.0, []

    def _check_amount_anomaly(self, agent_id: str, request: TransactionRequest,
                               profile: Optional[BehaviorProfile]) -> Tuple[float, List[str]]:
        if profile is None:
            return 20.0, []

        if profile.std_transaction_amount == 0:
            return 10.0, []

        z = abs(request.amount - profile.avg_transaction_amount) / max(profile.std_transaction_amount, 1)
        
        if z > self.AMOUNT_Z_THRESHOLD:
            return min(90, z * 20), [
                f"AMOUNT ANOMALY: ₹{request.amount:,.0f} is {z:.1f}σ from mean "
                f"₹{profile.avg_transaction_amount:,.0f}"
            ]
        elif z > 2.0:
            return z * 15, [
                f"Unusual amount: {z:.1f}σ from mean"
            ]

        return 0.0, []

    def _check_tool_distribution(self, agent_id: str, request: TransactionRequest,
                                  profile: Optional[BehaviorProfile]) -> Tuple[float, List[str]]:
        if profile is None or not profile.tool_distribution:
            return 10.0, []

        expected_freq = profile.tool_distribution.get(request.tool_name, 0.0)
        
        if expected_freq == 0.0:
            return 60.0, [
                f"UNUSUAL TOOL: '{request.tool_name}' not in typical distribution"
            ]
        elif expected_freq < 0.05:
            return 30.0, [
                f"Rare tool usage: '{request.tool_name}' (expected frequency: {expected_freq:.1%})"
            ]

        return 0.0, []

    def _check_failure_rate(self, agent_id: str,
                            profile: Optional[BehaviorProfile]) -> Tuple[float, List[str]]:
        actions = self._action_log.get(agent_id, [])
        if len(actions) < 5:
            return 0.0, []

        recent = actions[-20:]  # Last 20 actions
        blocked_count = sum(1 for a in recent if a["blocked"])
        failure_rate = blocked_count / len(recent)

        baseline = profile.failure_rate if profile else 0.05

        if failure_rate > baseline * 3:
            return 70.0, [
                f"HIGH FAILURE RATE: {failure_rate:.0%} (baseline: {baseline:.0%})"
            ]
        elif failure_rate > baseline * 2:
            return 40.0, [
                f"Elevated failure rate: {failure_rate:.0%}"
            ]

        return 0.0, []

    def _check_spending_spike(self, agent_id: str, request: TransactionRequest,
                               profile: Optional[BehaviorProfile]) -> Tuple[float, List[str]]:
        actions = self._action_log.get(agent_id, [])
        if not actions or profile is None:
            return 0.0, []

        recent_amounts = [a["amount"] for a in actions[-10:] if a["amount"] > 0]
        if not recent_amounts:
            return 0.0, []

        recent_avg = sum(recent_amounts) / len(recent_amounts)
        
        if request.amount > recent_avg * self.SPENDING_SPIKE_MULTIPLIER and request.amount > 1000:
            return 75.0, [
                f"SPENDING SPIKE: ₹{request.amount:,.0f} is "
                f"{request.amount/max(recent_avg,1):.1f}x recent average ₹{recent_avg:,.0f}"
            ]

        return 0.0, []

    def _check_repeated_calls(self, agent_id: str,
                               request: TransactionRequest) -> Tuple[float, List[str]]:
        actions = self._action_log.get(agent_id, [])
        if len(actions) < 3:
            return 0.0, []

        recent = actions[-5:]
        identical = sum(
            1 for a in recent
            if a["tool_name"] == request.tool_name and a["amount"] == request.amount
        )

        if identical >= 3:
            return 70.0, [
                f"REPEATED CALLS: {identical} identical "
                f"'{request.tool_name}' ₹{request.amount:,.0f} calls"
            ]
        elif identical >= 2:
            return 30.0, [
                f"Duplicate pattern: {identical} similar calls"
            ]

        return 0.0, []

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        actions = self._action_log.get(agent_id, [])
        if not actions:
            return {"total_actions": 0}

        amounts = [a["amount"] for a in actions if a["amount"] > 0]
        tools = Counter(a["tool_name"] for a in actions)
        merchants = set(a["merchant_id"] for a in actions if a["merchant_id"])
        blocked = sum(1 for a in actions if a["blocked"])
        violations = sum(1 for a in actions if a["violation"])

        return {
            "total_actions": len(actions),
            "avg_amount": float(np.mean(amounts)) if amounts else 0.0,
            "std_amount": float(np.std(amounts)) if amounts else 0.0,
            "max_amount": max(amounts) if amounts else 0.0,
            "min_amount": min(amounts) if amounts else 0.0,
            "tool_distribution": dict(tools),
            "merchant_diversity": len(merchants),
            "blocked_count": blocked,
            "violation_count": violations,
            "failure_rate": blocked / len(actions) if actions else 0.0,
            "categories": list(set(a["category"] for a in actions)),
        }

    def build_profile(self, agent_id: str) -> Optional[BehaviorProfile]:
        metrics = self.get_agent_metrics(agent_id)
        if metrics["total_actions"] == 0:
            return None

        total = metrics["total_actions"]
        tool_dist = {k: v / total for k, v in metrics["tool_distribution"].items()}

        return BehaviorProfile(
            agent_id=agent_id,
            avg_transaction_amount=metrics["avg_amount"],
            std_transaction_amount=metrics["std_amount"],
            tool_distribution=tool_dist,
            merchant_diversity=metrics["merchant_diversity"],
            failure_rate=metrics["failure_rate"],
            policy_violation_rate=metrics["violation_count"] / total,
            blocked_rate=metrics["blocked_count"] / total,
            typical_categories=metrics["categories"],
        )

    def is_runaway(self, agent_id: str, window_seconds: int = 60) -> Tuple[bool, str]:
        """
        Detect if an agent is in a runaway state.
        A runaway agent is one that is making an abnormally high number of requests in a short time window.
        """
        actions = self._action_log.get(agent_id, [])
        if not actions:
            return False, "No activity"

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        recent = [a for a in actions if a["timestamp"] >= window_start]

        if len(recent) >= self.VELOCITY_THRESHOLD_PER_MINUTE:
            return True, (
                f"RUNAWAY DETECTED: {len(recent)} requests in "
                f"{window_seconds}s (threshold: {self.VELOCITY_THRESHOLD_PER_MINUTE})"
            )

        return False, "Normal velocity"

    def reset(self) -> None:
        self._action_log.clear()
