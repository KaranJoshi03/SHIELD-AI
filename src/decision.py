"""
SHIELD AI — Decision Engine
Fuses all security signals into a final authorization decision.

CRITICAL RULES:
1. Hard security rules ALWAYS override soft ML scores
2. Unknown agent = BLOCK (even if ML risk is low)
3. Unauthorized tool = BLOCK (even if transaction is small)
4. ML risk alone NEVER automatically authorizes
5. System failure = FAIL CLOSED (REVIEW or BLOCK)

Decision inputs:
- Authorization result
- Policy evaluation
- Intent alignment
- Risk score
- ML score
- Behavior anomaly
- Workflow validation
- Prompt injection scan
- Duplicate detection
- Velocity check
- Agent trust
- System health
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import (
    Agent, DecisionType, EvaluationContext, PolicyResult,
    RiskLevel, SecurityDecision, TransactionRequest
)


class DecisionEngine:
    """
    Fuses all security signals into a final deterministic decision.
    
    The decision hierarchy:
    1. HARD BLOCKS — identity, capability, critical policy (always BLOCK)
    2. SOFT BLOCKS — risk, behavior, intent (may BLOCK or REVIEW)
    3. REVIEW — approval thresholds, moderate signals
    4. ALLOW — all checks pass
    """

    def decide(self,
               auth_result: str = "AUTHORIZED",
               policy_result: PolicyResult = PolicyResult.PASS,
               policy_reasons: List[str] = None,
               
               risk_score: float = 0.0,
               risk_level: RiskLevel = RiskLevel.LOW,
               intent_alignment: float = 1.0,
               behavior_score: float = 0.0,
               ml_score: float = 0.0,
               trust_score: float = 50.0,
               
               duplicate_result: str = "UNIQUE",
               workflow_result: str = "VALID",
               injection_result: str = "CLEAN",
               velocity_ok: bool = True,
               
               agent: Optional[Agent] = None,
               request: Optional[TransactionRequest] = None,
               system_healthy: bool = True,
               ml_available: bool = True,
               policy_available: bool = True,
               approval_threshold_exceeded: bool = False,
               ) -> SecurityDecision:
        """
        Make final security decision.
        Returns:
            SecurityDecision with decision, reasons, and full context
        """
        reasons: List[str] = []
        decision = DecisionType.ALLOW

        if not system_healthy:
            reasons.append("SYSTEM_UNHEALTHY: Fail-closed — system health check failed")
            return self._build_decision(
                DecisionType.BLOCK, 100.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                policy_result=PolicyResult.FAIL,
                auth_result="SYSTEM_FAILURE",
            )

        if not policy_available:
            reasons.append("POLICY_UNAVAILABLE: Fail-closed — cannot evaluate policy")
            return self._build_decision(
                DecisionType.REVIEW, 80.0, RiskLevel.HIGH,
                reasons, request=request, agent=agent,
                policy_result=PolicyResult.FAIL,
                auth_result=auth_result,
            )

        if not ml_available:
            reasons.append("ML_UNAVAILABLE: Proceeding with deterministic controls only")


        # 1. Authorization failure
        if "UNAUTHORIZED" in auth_result.upper() or \
           "UNKNOWN" in auth_result.upper() or \
           "DENIED" in auth_result.upper() or \
           "PAUSED" in auth_result.upper() or \
           "DISABLED" in auth_result.upper() or \
           "EXPIRED" in auth_result.upper():
            reasons.append(f"AUTHORIZATION_FAILED: {auth_result}")
            return self._build_decision(
                DecisionType.BLOCK, 100.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
            )

        # 2. Critical policy violation
        if policy_result == PolicyResult.FAIL:
            reasons.extend(policy_reasons or ["Policy evaluation failed"])
            if agent and agent.total_violations > 5:
                reasons.append(f"REPEATED_VIOLATIONS: Agent has {agent.total_violations} violations")
                return self._build_decision(
                    DecisionType.PAUSE_AGENT, 95.0, RiskLevel.CRITICAL,
                    reasons, request=request, agent=agent,
                    auth_result=auth_result,
                    policy_result=policy_result,
                )
            return self._build_decision(
                DecisionType.BLOCK, 90.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
            )

        # 3. Prompt injection detected
        if injection_result == "INJECTION_DETECTED":
            reasons.append("PROMPT_INJECTION: Critical injection indicators detected")
            return self._build_decision(
                DecisionType.BLOCK, 95.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
                injection_result=injection_result,
            )

        # 4. Duplicate/replay
        if duplicate_result in ("DUPLICATE_IDEMPOTENCY", "DUPLICATE_REQUEST_ID",
                                "REPLAY_EXPIRED"):
            reasons.append(f"DUPLICATE_BLOCKED: {duplicate_result}")
            return self._build_decision(
                DecisionType.BLOCK, 70.0, RiskLevel.HIGH,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
                duplicate_result=duplicate_result,
            )

        # 5. Velocity violation (runaway agent)
        if not velocity_ok:
            reasons.append("VELOCITY_VIOLATION: Agent exceeding rate limits — pausing agent")
            return self._build_decision(
                DecisionType.PAUSE_AGENT, 90.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
            )

        # 6. Severe behavior anomaly
        if behavior_score >= 80:
            reasons.append(f"SEVERE_ANOMALY: Behavior score {behavior_score:.0f}/100")
            return self._build_decision(
                DecisionType.PAUSE_AGENT, 85.0, RiskLevel.CRITICAL,
                reasons, request=request, agent=agent,
                auth_result=auth_result,
                policy_result=policy_result,
            )

        # 7. Policy says REVIEW
        if policy_result == PolicyResult.REVIEW:
            reasons.extend(policy_reasons or ["Policy requires review"])
            decision = DecisionType.REVIEW

        # 8. Workflow violation
        if workflow_result not in ("VALID", ""):
            reasons.append(f"WORKFLOW_VIOLATION: {workflow_result}")
            decision = DecisionType.REVIEW

        # 9. Suspicious injection
        if injection_result == "SUSPICIOUS":
            reasons.append("INJECTION_SUSPICIOUS: Possible injection indicators")
            decision = DecisionType.REVIEW

        # 10. Duplicate fingerprint (soft duplicate)
        if duplicate_result == "DUPLICATE_FINGERPRINT":
            reasons.append("SOFT_DUPLICATE: Similar transaction fingerprint detected")
            decision = DecisionType.REVIEW

        # 11. High risk score
        if risk_level == RiskLevel.CRITICAL:
            reasons.append(f"CRITICAL_RISK: Risk score {risk_score:.0f}/100")
            decision = DecisionType.BLOCK
        elif risk_level == RiskLevel.HIGH:
            reasons.append(f"HIGH_RISK: Risk score {risk_score:.0f}/100")
            if decision != DecisionType.BLOCK:
                decision = DecisionType.REVIEW

        # 12. Low intent alignment
        if intent_alignment < 0.3:
            reasons.append(f"LOW_INTENT_ALIGNMENT: Score {intent_alignment:.2f}")
            decision = DecisionType.BLOCK
        elif intent_alignment < 0.6:
            reasons.append(f"MODERATE_INTENT_ALIGNMENT: Score {intent_alignment:.2f}")
            if decision == DecisionType.ALLOW:
                decision = DecisionType.REVIEW

        # 13. Moderate behavior anomaly
        if behavior_score >= 50 and decision == DecisionType.ALLOW:
            reasons.append(f"BEHAVIOR_ANOMALY: Score {behavior_score:.0f}/100")
            decision = DecisionType.REVIEW

        # 14. Low trust
        if trust_score < 30 and decision == DecisionType.ALLOW:
            reasons.append(f"LOW_TRUST: Agent trust score {trust_score:.0f}/100")
            decision = DecisionType.REVIEW

        # 15. Approval threshold
        if approval_threshold_exceeded and decision == DecisionType.ALLOW:
            reasons.append("APPROVAL_REQUIRED: Amount exceeds approval threshold")
            decision = DecisionType.REVIEW

        if decision == DecisionType.ALLOW:
            reasons.append("ALL_CHECKS_PASSED: Transaction authorized")

        overall_risk = self._calculate_overall_risk(
            risk_score, behavior_score, intent_alignment,
            trust_score, ml_score
        )

        return self._build_decision(
            decision, overall_risk, risk_level,
            reasons, request=request, agent=agent,
            auth_result=auth_result,
            policy_result=policy_result,
            intent_alignment=intent_alignment,
            behavior_score=behavior_score,
            duplicate_result=duplicate_result,
            workflow_result=workflow_result,
            injection_result=injection_result,
            trust_score=trust_score,
            velocity_ok=velocity_ok,
            approval_required=decision == DecisionType.REVIEW,
        )

    def _calculate_overall_risk(self, risk_score: float, behavior_score: float,
                                 intent_alignment: float, trust_score: float,
                                 ml_score: float) -> float:
        overall = (
            risk_score * 0.30 +
            behavior_score * 0.20 +
            (1 - intent_alignment) * 100 * 0.20 +
            (100 - trust_score) * 0.15 +
            ml_score * 0.15
        )
        return max(0.0, min(100.0, overall))

    def _build_decision(self, decision: DecisionType, risk: float,
                        risk_level: RiskLevel, reasons: List[str],
                        request=None, agent=None,
                        auth_result="", policy_result=PolicyResult.PASS,
                        intent_alignment=1.0, behavior_score=0.0,
                        duplicate_result="UNIQUE", workflow_result="VALID",
                        injection_result="CLEAN", trust_score=50.0,
                        velocity_ok=True, approval_required=False,
                        ) -> SecurityDecision:
        return SecurityDecision(
            decision=decision,
            overall_risk=risk,
            risk_level=risk_level,
            policy_result=policy_result if isinstance(policy_result, PolicyResult) else PolicyResult.PASS,
            authorization_result=auth_result,
            intent_alignment=max(0.0, min(1.0, intent_alignment)),
            behavior_score=behavior_score,
            anomaly_score=0.0,
            duplicate_result=duplicate_result,
            workflow_result=workflow_result,
            injection_result=injection_result,
            trust_score=trust_score,
            velocity_ok=velocity_ok,
            reasons=reasons,
            approval_required=approval_required,
            request_id=request.request_id if request else "",
            agent_id=request.agent_id if request else (agent.agent_id if agent else ""),
        )
