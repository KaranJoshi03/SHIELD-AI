"""
SHIELD AI — Authorization Module
Implements identity validation and capability authorization.

Security principles:
- Unknown agents are ALWAYS rejected
- Disabled/Paused/Expired agents are ALWAYS rejected  
- Only explicitly granted capabilities are allowed (least privilege)
- Every authorization check produces a clear reason
"""

from __future__ import annotations
from typing import Optional, Tuple
from .models import (
    Agent, AgentStatus, TransactionRequest, SecurityDecision,
    DecisionType, RiskLevel, PolicyResult
)
from .agents import AgentRegistry


class AuthorizationEngine:
    """
    Handles identity validation and capability authorization.
    
    This is the first security gate in SHIELD's pipeline.
    If identity or capability checks fail, no further evaluation is needed — the request is BLOCKED.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def validate_identity(self, agent_id: str) -> Tuple[bool, str, Optional[Agent]]:
        is_valid, reason = self.registry.validate_agent(agent_id)
        agent = self.registry.get_agent(agent_id)
        return is_valid, reason, agent

    def check_capability(self, agent_id: str, tool_name: str) -> Tuple[bool, str]:
        return self.registry.check_capability(agent_id, tool_name)

    def authorize(self, request: TransactionRequest) -> Tuple[bool, str, Optional[Agent]]:
        is_valid, reason, agent = self.validate_identity(request.agent_id)
        if not is_valid:
            return False, reason, agent

        has_capability, cap_reason = self.check_capability(
            request.agent_id, request.tool_name
        )
        if not has_capability:
            return False, cap_reason, agent

        return True, "AUTHORIZED: Identity and capability verified", agent

    def create_block_decision(self, request: TransactionRequest,
                               reason: str) -> SecurityDecision:
        return SecurityDecision(
            decision=DecisionType.BLOCK,
            overall_risk=100.0,
            risk_level=RiskLevel.CRITICAL,
            authorization_result=reason,
            policy_result=PolicyResult.FAIL,
            reasons=[reason],
            request_id=request.request_id,
            agent_id=request.agent_id,
        )
