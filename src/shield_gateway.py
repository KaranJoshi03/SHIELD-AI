"""
SHIELD AI — Shield Gateway
The central security gateway that orchestrates all security checks.

CRITICAL INVARIANT:
    An AI agent can propose a financial action,
    but it cannot authorize its own financial action.
    SHIELD AI owns authorization.
    PayMCP owns execution.
    The agent owns proposal.

This is the ONLY supported route to PayMCP.
No tool call should bypass ShieldGateway.

Architecture:
    Agent Proposal → SHIELD Gateway → [Security Pipeline] → Decision → ALLOW → PayMCP → Result →
    REVIEW → Human Approval → BLOCK → Reject → PAUSE → Pause Agent
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .agents import AgentRegistry
from .approval import ApprovalEngine
from .audit import AuditLogger
from .authorization import AuthorizationEngine
from .behavior import BehaviorEngine
from .decision import DecisionEngine
from .idempotency import IdempotencyEngine
from .intent import IntentAlignmentScorer, IntentDriftDetector, IntentExtractor
from .models import (
    Agent, DecisionType, EvaluationContext, PolicyResult,
    RiskLevel, SecurityDecision, TransactionRequest, UserIntent,
    ToolCall,
)
from .paymcp import PayMCP
from .policies import PolicyEngine
from .prompt_security import PromptInjectionDetector
from .risk import TransactionRiskEngine
from .workflow import WorkflowValidator


class ShieldGateway:
    """
    Central SHIELD Security Gateway.
    
    Orchestrates the complete security pipeline:
    1. Identity & Authorization
    2. Idempotency Check
    3. Policy Evaluation
    4. Intent Alignment
    5. Prompt Injection Scan
    6. Workflow Validation
    7. Risk Assessment
    8. Behavior Analysis
    9. Decision Fusion
    10. Execution (if ALLOW) or Approval (if REVIEW)
    11. Audit Logging
    
    Every evaluation is audited regardless of outcome.
    """

    def __init__(self, paymcp: Optional[PayMCP] = None,
                 db_path: str = ":memory:"):
        self.registry = AgentRegistry()
        self.paymcp = paymcp or PayMCP(db_path=db_path)
        self.auth_engine = AuthorizationEngine(self.registry)
        self.policy_engine = PolicyEngine()
        self.risk_engine = TransactionRiskEngine()
        self.behavior_engine = BehaviorEngine()
        self.idempotency_engine = IdempotencyEngine()
        self.workflow_validator = WorkflowValidator()
        self.injection_detector = PromptInjectionDetector()
        self.decision_engine = DecisionEngine()
        self.approval_engine = ApprovalEngine()
        self.audit_logger = AuditLogger(db_path=db_path)
        self.intent_extractor = IntentExtractor()
        self.alignment_scorer = IntentAlignmentScorer()
        self.drift_detector = IntentDriftDetector()
        self._user_intents: Dict[str, UserIntent] = {}
        self._execution_count = 0
        self._block_count = 0

    def set_user_intent(self, user_id: str, intent_text: str) -> UserIntent:
        """
        Set the user's financial intent.
        This should be called before agent actions to establish the baseline for intent alignment checking.
        """
        intent = self.intent_extractor.extract(intent_text, user_id)
        self._user_intents[user_id] = intent
        return intent

    def get_user_intent(self, user_id: str) -> Optional[UserIntent]:
        return self._user_intents.get(user_id)

    def evaluate(self, request: TransactionRequest,
                 session_id: str = "",
                 system_healthy: bool = True,
                 ml_available: bool = True) -> SecurityDecision:
        """
        Evaluate a proposed financial action through the complete security pipeline.
        This is the core SHIELD function.
        
        Args:
            request: The transaction request to evaluate
            session_id: Session identifier for workflow tracking
            system_healthy: Whether the system is in healthy state
            ml_available: Whether the ML model is available
            
        Returns:
            SecurityDecision with complete evaluation results
        """
        start_time = time.perf_counter()
        
        if not session_id:
            session_id = f"session_{request.agent_id}"

        # STAGE 1: Identity & Authorization
        is_authorized, auth_reason, agent = self.auth_engine.authorize(request)
        
        if not is_authorized:
            decision = self.decision_engine.decide(
                auth_result=auth_reason,
                policy_result=PolicyResult.FAIL,
                system_healthy=system_healthy,
                request=request,
                agent=agent,
            )
            self._record_and_audit(request, decision, blocked=True, violation=True)
            return decision

        # STAGE 2: Idempotency Check
        dup_result, dup_data = self.idempotency_engine.check(request)
        
        if dup_result != "UNIQUE":
            decision = self.decision_engine.decide(
                auth_result="AUTHORIZED",
                duplicate_result=dup_result,
                system_healthy=system_healthy,
                request=request,
                agent=agent,
            )
            self._record_and_audit(request, decision, blocked=True)
            return decision

        # STAGE 3: Velocity / Runaway Check
        is_runaway, runaway_reason = self.behavior_engine.is_runaway(request.agent_id)
        velocity_ok = not is_runaway
        
        if is_runaway:
            self.registry.pause_agent(request.agent_id, runaway_reason)

        #STAGE 4: Policy Evaluation
        policy_eval = self.policy_engine.evaluate(request, agent)
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if agent.daily_spend_date != today:
            agent.daily_spend = 0.0
            agent.daily_spend_date = today

        #STAGE 5: Intent Alignment
        intent = self._user_intents.get(request.user_id)
        intent_alignment = 1.0
        intent_text = ""
        
        if intent:
            intent_alignment, alignment_reasons = self.alignment_scorer.score(
                intent, request
            )
            intent_text = intent.original_text
            
            drift_score, drift_reasons = self.drift_detector.detect_drift(
                session_id, intent, request
            )
            
            if drift_score > 0.5:
                intent_alignment = min(intent_alignment, 1.0 - drift_score)

        # STAGE 6: Prompt Injection Scan
        injection_result, injection_reasons = self.injection_detector.scan_request(
            purpose=request.purpose,
            metadata=request.metadata,
        )

        # STAGE 7: Workflow Validation 
        workflow_result, workflow_reasons = self.workflow_validator.validate(
            session_id, request.agent_id, request.tool_name
        )

        # STAGE 8: Risk Assessment 
        profile = self.registry.get_behavior_profile(request.agent_id)
        risk_score, risk_level, risk_signals = self.risk_engine.evaluate(
            request, agent, profile,
            intent_alignment=intent_alignment,
            duplicate_detected=(dup_result != "UNIQUE"),
            workflow_violation=(workflow_result != "VALID"),
            injection_detected=(injection_result != "CLEAN"),
        )

        # STAGE 9: Behavior Analysis
        behavior_score, behavior_reasons = self.behavior_engine.analyze(
            request.agent_id, request, profile
        )

        # STAGE 10: Decision Fusion
        approval_threshold = 2000.0
        policy = self.policy_engine.get_policy(agent.role)
        if policy:
            approval_threshold = policy.approval_threshold

        decision = self.decision_engine.decide(
            auth_result="AUTHORIZED",
            policy_result=policy_eval.result,
            policy_reasons=policy_eval.reasons,
            risk_score=risk_score,
            risk_level=risk_level,
            intent_alignment=intent_alignment,
            behavior_score=behavior_score,
            trust_score=agent.trust_score,
            duplicate_result=dup_result,
            workflow_result=workflow_result,
            injection_result=injection_result,
            velocity_ok=velocity_ok,
            agent=agent,
            request=request,
            system_healthy=system_healthy,
            ml_available=ml_available,
            policy_available=True,
            approval_threshold_exceeded=(request.amount > approval_threshold),
        )

        eval_time = (time.perf_counter() - start_time) * 1000  # ms
        decision.metadata["evaluation_time_ms"] = round(eval_time, 2)

        # STAGE 11: Record & Audit
        blocked = decision.decision in (DecisionType.BLOCK, DecisionType.PAUSE_AGENT)
        violation = policy_eval.result == PolicyResult.FAIL
        
        self._record_and_audit(
            request, decision,
            intent_text=intent_text,
            blocked=blocked,
            violation=violation,
        )

        if decision.decision == DecisionType.ALLOW:
            self.registry.update_trust_score(request.agent_id, 1.0)
        elif decision.decision == DecisionType.BLOCK:
            self.registry.update_trust_score(request.agent_id, -3.0)
        elif decision.decision == DecisionType.PAUSE_AGENT:
            self.registry.update_trust_score(request.agent_id, -10.0)

        if intent:
            self.drift_detector.record_action(
                session_id, request, intent_alignment
            )

        return decision

    def execute(self, request: TransactionRequest,
                session_id: str = "",
                auto_approve: bool = False,
                system_healthy: bool = True,
                ml_available: bool = True) -> Dict[str, Any]:
        """
        Protected execution: evaluate + conditionally execute.
        This is the ONLY supported route to PayMCP.
        
        Args:
            request: Transaction request
            session_id: Session for workflow tracking
            auto_approve: Auto-approve REVIEW decisions (for demo)
            system_healthy: System health status
            ml_available: ML availability status
            
        Returns:
            Complete result including decision and execution outcome
        """
        decision = self.evaluate(
            request, session_id, system_healthy, ml_available
        )

        result = {
            "decision": decision.decision.value,
            "risk_score": decision.overall_risk,
            "risk_level": decision.risk_level.value,
            "reasons": decision.reasons,
            "execution": None,
            "approval": None,
        }

        if decision.decision == DecisionType.ALLOW:
            reserved = self.idempotency_engine.reserve(request)
            if not reserved:
                result["execution"] = {"status": "duplicate_prevented"}
                return result

            exec_result = self.paymcp.execute_tool(
                request.tool_name,
                self._build_tool_args(request)
            )
            
            self.idempotency_engine.commit(request, exec_result)
            
            agent = self.registry.get_agent(request.agent_id)
            if agent:
                agent.daily_spend += request.amount
            
            self._execution_count += 1
            result["execution"] = exec_result

            self.audit_logger.log(
                request, decision,
                execution_result=exec_result.get("status", "unknown"),
            )

        elif decision.decision == DecisionType.REVIEW:
            agent = self.registry.get_agent(request.agent_id)
            approval_request = self.approval_engine.request_approval(
                decision,
                agent_name=agent.agent_name if agent else "",
                user_id=request.user_id,
                tool_name=request.tool_name,
                amount=request.amount,
                merchant_id=request.merchant_id,
                purpose=request.purpose,
            )
            result["approval"] = {
                "approval_id": approval_request.approval_id,
                "status": "PENDING",
            }

            if auto_approve:
                success, msg = self.approval_engine.approve(
                    approval_request.approval_id
                )
                if success:
                    reserved = self.idempotency_engine.reserve(request)
                    if reserved:
                        exec_result = self.paymcp.execute_tool(
                            request.tool_name,
                            self._build_tool_args(request)
                        )
                        self.idempotency_engine.commit(request, exec_result)
                        if agent:
                            agent.daily_spend += request.amount
                        self._execution_count += 1
                        result["execution"] = exec_result
                        result["approval"]["status"] = "APPROVED"
                else:
                    result["approval"]["status"] = "APPROVAL_FAILED"

        elif decision.decision == DecisionType.BLOCK:
            self._block_count += 1
            result["execution"] = {"status": "blocked", "reason": "; ".join(decision.reasons)}
            self.idempotency_engine.release(request)

        elif decision.decision == DecisionType.PAUSE_AGENT:
            self._block_count += 1
            self.registry.pause_agent(
                request.agent_id,
                reason="; ".join(decision.reasons)
            )
            result["execution"] = {
                "status": "agent_paused",
                "reason": "; ".join(decision.reasons)
            }

        return result

    def simulate_action(self, request: TransactionRequest,
                        session_id: str = "") -> Dict[str, Any]:
        """
        What-if security simulator.
        Evaluates what would happen if this action were proposed, WITHOUT executing anything.
        No transaction executes during simulation.
        """
        decision = self.evaluate(request, session_id or "simulation")
        
        return {
            "simulated": True,
            "decision": decision.decision.value,
            "risk_score": decision.overall_risk,
            "risk_level": decision.risk_level.value,
            "policy_result": decision.policy_result.value if hasattr(decision.policy_result, 'value') else str(decision.policy_result),
            "intent_alignment": decision.intent_alignment,
            "behavior_score": decision.behavior_score,
            "trust_score": decision.trust_score,
            "reasons": decision.reasons,
            "approval_required": decision.approval_required,
            "evaluation_time_ms": decision.metadata.get("evaluation_time_ms", 0),
        }

    def _build_tool_args(self, request: TransactionRequest) -> Dict[str, Any]:
        base_args = {
            "agent_id": request.agent_id,
            "user_id": request.user_id,
        }

        if request.tool_name == "create_order":
            base_args.update({
                "amount": request.amount,
                "currency": request.currency,
                "merchant_id": request.merchant_id,
                "purpose": request.purpose,
                "category": request.category,
                "idempotency_key": request.idempotency_key,
            })
        elif request.tool_name == "fetch_payment":
            base_args["payment_id"] = request.metadata.get("payment_id", "")
        elif request.tool_name == "create_payment_link":
            base_args.update({
                "amount": request.amount,
                "currency": request.currency,
                "purpose": request.purpose,
                "merchant_id": request.merchant_id,
                "idempotency_key": request.idempotency_key,
            })
        elif request.tool_name == "refund_payment":
            base_args.update({
                "payment_id": request.metadata.get("payment_id", ""),
                "amount": request.amount,
                "reason": request.purpose,
                "idempotency_key": request.idempotency_key,
            })
        elif request.tool_name == "create_payout":
            base_args.update({
                "amount": request.amount,
                "currency": request.currency,
                "recipient_id": request.merchant_id,
                "purpose": request.purpose,
                "idempotency_key": request.idempotency_key,
            })
        elif request.tool_name == "fetch_settlement":
            base_args["settlement_id"] = request.metadata.get("settlement_id", "")
            base_args["merchant_id"] = request.merchant_id
        elif request.tool_name == "verify_payment":
            base_args["payment_id"] = request.metadata.get("payment_id", "")

        return base_args

    def _record_and_audit(self, request: TransactionRequest,
                          decision: SecurityDecision,
                          intent_text: str = "",
                          blocked: bool = False,
                          violation: bool = False) -> None:
        self.behavior_engine.record_action(
            request.agent_id, request,
            blocked=blocked, violation=violation,
        )
        
        self.registry.record_activity(
            request.agent_id,
            blocked=blocked,
            violation=violation,
        )

        self.audit_logger.log(
            request, decision,
            intent_text=intent_text,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_evaluations": len(self.audit_logger.get_all_events()),
            "executions": self._execution_count,
            "blocks": self._block_count,
            "audit": self.audit_logger.get_stats(),
            "paymcp": self.paymcp.get_stats(),
            "idempotency": self.idempotency_engine.get_stats(),
            "approval": self.approval_engine.get_stats(),
        }

    def reset(self) -> None:
        self.registry.reset()
        self.paymcp.reset()
        self.risk_engine.reset()
        self.behavior_engine.reset()
        self.idempotency_engine.reset()
        self.workflow_validator.reset()
        self.approval_engine.reset()
        self.audit_logger.reset()
        self.drift_detector.reset_all()
        self._user_intents.clear()
        self._execution_count = 0
        self._block_count = 0
