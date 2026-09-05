"""
SHIELD AI — Agent Registry & Identity Management
Manages AI agent identities, capabilities, and lifecycle.

Security principles:
- Known agent validation (unknown agents are always rejected)
- Least-privilege capability assignment
- Agent lifecycle management (ACTIVE/PAUSED/DISABLED/EXPIRED)
- Dynamic trust scoring
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .models import (
    Agent, AgentStatus, BehaviorProfile, RiskLevel, ToolName
)


class AgentRegistry:
 
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._behavior_profiles: Dict[str, BehaviorProfile] = {}
        self._lock = threading.Lock()
        self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        
        # ShoppingAgent — can create orders, fetch payments, create payment links
        self.register_agent(Agent(
            agent_id="agent_shopping_001",
            agent_name="ShoppingAgent",
            owner_id="owner_001",
            status=AgentStatus.ACTIVE,
            capabilities=[
                ToolName.CREATE_ORDER.value,
                ToolName.FETCH_PAYMENT.value,
                ToolName.CREATE_PAYMENT_LINK.value,
                ToolName.VERIFY_PAYMENT.value,
            ],
            trust_score=85.0,
            risk_level=RiskLevel.LOW,
            role="shopping",
        ))

        # FinanceAgent — can access settlements, fetch payments, create payouts
        self.register_agent(Agent(
            agent_id="agent_finance_001",
            agent_name="FinanceAgent",
            owner_id="owner_001",
            status=AgentStatus.ACTIVE,
            capabilities=[
                ToolName.FETCH_SETTLEMENT.value,
                ToolName.FETCH_PAYMENT.value,
                ToolName.CREATE_PAYOUT.value,
                ToolName.VERIFY_PAYMENT.value,
            ],
            trust_score=87.0,
            risk_level=RiskLevel.LOW,
            role="finance",
        ))

        # SupportAgent — can fetch payments and issue refunds
        self.register_agent(Agent(
            agent_id="agent_support_001",
            agent_name="SupportAgent",
            owner_id="owner_002",
            status=AgentStatus.ACTIVE,
            capabilities=[
                ToolName.FETCH_PAYMENT.value,
                ToolName.REFUND_PAYMENT.value,
                ToolName.VERIFY_PAYMENT.value,
            ],
            trust_score=61.0,
            risk_level=RiskLevel.MEDIUM,
            role="support",
        ))

        # PausedAgent — for testing paused agent rejection
        self.register_agent(Agent(
            agent_id="agent_paused_001",
            agent_name="PausedAgent",
            owner_id="owner_001",
            status=AgentStatus.PAUSED,
            capabilities=[ToolName.FETCH_PAYMENT.value],
            trust_score=30.0,
            risk_level=RiskLevel.MEDIUM,
            role="shopping",
        ))

        # ExpiredAgent — for testing expired agent rejection
        self.register_agent(Agent(
            agent_id="agent_expired_001",
            agent_name="ExpiredAgent",
            owner_id="owner_001",
            status=AgentStatus.EXPIRED,
            capabilities=[ToolName.CREATE_ORDER.value],
            trust_score=10.0,
            risk_level=RiskLevel.HIGH,
            role="shopping",
            created_at=datetime.utcnow() - timedelta(days=365),
        ))

        # DisabledAgent — for testing disabled agent rejection
        self.register_agent(Agent(
            agent_id="agent_disabled_001",
            agent_name="DisabledAgent",
            owner_id="owner_001",
            status=AgentStatus.DISABLED,
            capabilities=[],
            trust_score=0.0,
            risk_level=RiskLevel.CRITICAL,
            role="unknown",
        ))

        self._initialize_behavior_profiles()

    def _initialize_behavior_profiles(self) -> None:
        self._behavior_profiles["agent_shopping_001"] = BehaviorProfile(
            agent_id="agent_shopping_001",
            avg_requests_per_hour=5.0,
            avg_transaction_amount=1500.0,
            std_transaction_amount=800.0,
            tool_distribution={
                "create_order": 0.5,
                "fetch_payment": 0.3,
                "create_payment_link": 0.1,
                "verify_payment": 0.1,
            },
            merchant_diversity=8,
            failure_rate=0.03,
            policy_violation_rate=0.01,
            blocked_rate=0.02,
            typical_categories=["groceries", "electronics", "clothing", "food", "travel"],
            typical_hours=list(range(8, 22)),
        )
        self._behavior_profiles["agent_finance_001"] = BehaviorProfile(
            agent_id="agent_finance_001",
            avg_requests_per_hour=3.0,
            avg_transaction_amount=5000.0,
            std_transaction_amount=3000.0,
            tool_distribution={
                "fetch_settlement": 0.4,
                "fetch_payment": 0.3,
                "create_payout": 0.2,
                "verify_payment": 0.1,
            },
            merchant_diversity=5,
            failure_rate=0.02,
            policy_violation_rate=0.01,
            blocked_rate=0.01,
            typical_categories=["payroll", "vendor_payment", "settlement"],
            typical_hours=list(range(9, 18)),
        )
        self._behavior_profiles["agent_support_001"] = BehaviorProfile(
            agent_id="agent_support_001",
            avg_requests_per_hour=8.0,
            avg_transaction_amount=800.0,
            std_transaction_amount=500.0,
            tool_distribution={
                "fetch_payment": 0.6,
                "refund_payment": 0.3,
                "verify_payment": 0.1,
            },
            merchant_diversity=15,
            failure_rate=0.05,
            policy_violation_rate=0.03,
            blocked_rate=0.04,
            avg_refund_frequency=0.3,
            typical_categories=["refund", "support", "dispute"],
            typical_hours=list(range(8, 23)),
        )

    def register_agent(self, agent: Agent) -> None:
        with self._lock:
            self._agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def is_known(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def validate_agent(self, agent_id: str) -> tuple[bool, str]:

        agent = self.get_agent(agent_id)
        
        if agent is None:
            return False, "UNKNOWN_AGENT: Agent not found in registry"
        
        if agent.status == AgentStatus.PAUSED:
            return False, "PAUSED_AGENT: Agent is currently paused"
        
        if agent.status == AgentStatus.DISABLED:
            return False, "DISABLED_AGENT: Agent has been disabled"
        
        if agent.status == AgentStatus.EXPIRED:
            return False, "EXPIRED_AGENT: Agent has expired"
        
        if agent.status != AgentStatus.ACTIVE:
            return False, f"INVALID_STATUS: Agent status is {agent.status.value}"
        
        return True, "ACTIVE: Agent identity validated"

    def check_capability(self, agent_id: str, tool_name: str) -> tuple[bool, str]:

        agent = self.get_agent(agent_id)
        
        if agent is None:
            return False, "UNKNOWN_AGENT: Cannot check capabilities"
        
        if agent.has_capability(tool_name):
            return True, f"CAPABILITY_GRANTED: {agent.agent_name} has {tool_name}"
        
        return False, f"CAPABILITY_DENIED: {agent.agent_name} does not have {tool_name}"

    def get_behavior_profile(self, agent_id: str) -> Optional[BehaviorProfile]:
        return self._behavior_profiles.get(agent_id)

    def pause_agent(self, agent_id: str, reason: str = "") -> bool:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.PAUSED
                agent.metadata["pause_reason"] = reason
                agent.metadata["paused_at"] = datetime.utcnow().isoformat()
                return True
            return False

    def resume_agent(self, agent_id: str) -> bool:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent and agent.status == AgentStatus.PAUSED:
                agent.status = AgentStatus.ACTIVE
                agent.metadata.pop("pause_reason", None)
                agent.metadata["resumed_at"] = datetime.utcnow().isoformat()
                return True
            return False

    def update_trust_score(self, agent_id: str, delta: float) -> Optional[float]:

        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                delta = max(-10.0, min(5.0, delta))
                agent.trust_score = max(0.0, min(100.0, agent.trust_score + delta))
                return agent.trust_score
            return None

    def record_activity(self, agent_id: str, blocked: bool = False, 
                        violation: bool = False) -> None:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.update_activity()
                agent.total_requests += 1
                if blocked:
                    agent.total_blocked += 1
                if violation:
                    agent.total_violations += 1

    def get_all_agents(self) -> List[Agent]:
        return list(self._agents.values())

    def get_active_agents(self) -> List[Agent]:
        return [a for a in self._agents.values() if a.is_active()]

    def reset(self) -> None:
        self._agents.clear()
        self._behavior_profiles.clear()
        self._initialize_default_agents()
