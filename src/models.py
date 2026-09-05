"""
SHIELD AI : Data Models
All Pydantic models, enums, and state machines for the SHIELD AI system.

Key models:
- Agent: AI agent identity and capabilities
- TransactionRequest: Financial action proposed by an agent
- UserIntent: Original user's financial intent
- ToolCall: MCP tool invocation record
- SecurityDecision: SHIELD's authorization decision
- PaymentState: Payment lifecycle state machine
- AuditEvent: Complete audit trail record
"""

from __future__ import annotations
import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class AgentStatus(str, Enum):
    """Lifecycle status of an AI agent."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"
    EXPIRED = "EXPIRED"


class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    PAUSE_AGENT = "PAUSE_AGENT"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class PaymentState(str, Enum):
    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"
    CANCELLED = "CANCELLED"


VALID_PAYMENT_TRANSITIONS: Dict[PaymentState, List[PaymentState]] = {
    PaymentState.CREATED: [
        PaymentState.AUTHORIZED,
        PaymentState.FAILED,
        PaymentState.CANCELLED,
    ],
    PaymentState.AUTHORIZED: [
        PaymentState.CAPTURED,
        PaymentState.FAILED,
        PaymentState.CANCELLED,
    ],
    PaymentState.CAPTURED: [
        PaymentState.REFUNDED,
    ],
    PaymentState.FAILED: [],
    PaymentState.REFUNDED: [],
    PaymentState.CANCELLED: [],
}


def is_valid_transition(current: PaymentState, target: PaymentState) -> bool:
    return target in VALID_PAYMENT_TRANSITIONS.get(current, [])


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ToolName(str, Enum):
    CREATE_ORDER = "create_order"
    FETCH_PAYMENT = "fetch_payment"
    CREATE_PAYMENT_LINK = "create_payment_link"
    REFUND_PAYMENT = "refund_payment"
    CREATE_PAYOUT = "create_payout"
    FETCH_SETTLEMENT = "fetch_settlement"
    VERIFY_PAYMENT = "verify_payment"


class Agent(BaseModel):
    """
    An AI agent with identity, capabilities, and trust metadata.
    Agents operate under least-privilege: they only have access to the specific tools listed in their capabilities.
    """
    agent_id: str = Field(description="Unique agent identifier")
    agent_name: str = Field(description="Human-readable agent name")
    owner_id: str = Field(description="Owner/organization that created this agent")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of tool names this agent is allowed to use"
    )
    trust_score: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Dynamic trust score (0-100)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = Field(default=None)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    role: str = Field(default="unknown", description="Agent role (shopping, finance, support, unknown)")
    daily_spend: float = Field(default=0.0, description="Running daily spend total")
    daily_spend_date: Optional[str] = Field(default=None, description="Date for daily spend tracking")
    hourly_request_count: int = Field(default=0, description="Requests in current hour")
    hourly_request_hour: Optional[str] = Field(default=None, description="Hour for request counting")
    total_requests: int = Field(default=0)
    total_blocked: int = Field(default=0)
    total_violations: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_active(self) -> bool:
        return self.status == AgentStatus.ACTIVE

    def has_capability(self, tool_name: str) -> bool:
        return tool_name in self.capabilities

    def update_activity(self) -> None:
        self.last_activity = datetime.utcnow()

class TransactionRequest(BaseModel):
    """
    A financial action proposed by an AI agent.
    This represents the agent's PROPOSAL, not the execution.
    SHIELD evaluates this before any financial action occurs.
    """
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    agent_id: str = Field(description="Agent proposing the action")
    user_id: str = Field(description="End user on whose behalf agent acts")
    merchant_id: str = Field(default="", description="Target merchant")
    tool_name: str = Field(description="MCP tool being invoked")
    amount: float = Field(default=0.0, ge=0.0, description="Transaction amount")
    currency: str = Field(default="INR", description="Currency code")
    purpose: str = Field(default="", description="Purpose/description of transaction")
    category: str = Field(default="general", description="Transaction category")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: str = Field(default="", description="Device identifier")
    location: str = Field(default="", description="Geographic location")
    idempotency_key: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique key for idempotent execution"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """
        Generate a transaction fingerprint for duplicate detection.
        Based on: agent_id + user_id + tool_name + amount + merchant_id + purpose
        """
        data = f"{self.agent_id}:{self.user_id}:{self.tool_name}:{self.amount}:{self.merchant_id}:{self.purpose}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class UserIntent(BaseModel):

    intent_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique intent identifier"
    )
    user_id: str = Field(description="User who expressed the intent")
    original_text: str = Field(description="Original natural language instruction")
    max_amount: float = Field(default=float('inf'), description="Maximum allowed amount")
    currency: str = Field(default="INR")
    category: str = Field(default="general", description="Expected category")
    merchant_constraints: List[str] = Field(
        default_factory=list,
        description="Allowed/preferred merchants"
    )
    allowed_purpose: str = Field(default="", description="Expected purpose")
    expiration: Optional[datetime] = Field(
        default=None,
        description="When this intent expires"
    )
    extracted_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extracted constraints"
    )

    def is_expired(self) -> bool:
        if self.expiration is None:
            return False
        return datetime.utcnow() > self.expiration



class ToolCall(BaseModel):

    tool_name: str = Field(description="Name of the MCP tool")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments/parameters"
    )
    agent_id: str = Field(description="Agent making the call")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request ID"
    )

class SecurityDecision(BaseModel):
    """
    SHIELD's authorization decision.
    Combines all security signals into a final deterministic decision.
    Hard security rules always override soft ML scores.
    """
    decision: DecisionType = Field(description="Final decision: ALLOW/REVIEW/BLOCK/PAUSE_AGENT")
    overall_risk: float = Field(default=0.0, ge=0.0, le=100.0, description="Combined risk score 0-100")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    policy_result: PolicyResult = Field(default=PolicyResult.PASS)
    authorization_result: str = Field(default="AUTHORIZED")
    intent_alignment: float = Field(default=1.0, ge=0.0, le=1.0, description="Intent alignment score 0-1")
    behavior_score: float = Field(default=0.0, description="Behavior anomaly score")
    anomaly_score: float = Field(default=0.0, description="ML anomaly score")
    duplicate_result: str = Field(default="UNIQUE")
    workflow_result: str = Field(default="VALID")
    injection_result: str = Field(default="CLEAN")
    trust_score: float = Field(default=50.0)
    velocity_ok: bool = Field(default=True)
    reasons: List[str] = Field(default_factory=list, description="Human-readable decision reasons")
    approval_required: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default="")
    agent_id: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Policy(BaseModel):
    """
    A deterministic security policy for an agent role.
    Policies are compiled from natural language into structured, versioned, deterministic rules.
    """
    policy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: str = Field(description="Agent role this policy applies to")
    max_transaction: float = Field(default=float('inf'), description="Maximum single transaction amount")
    daily_limit: float = Field(default=float('inf'), description="Maximum daily spending")
    hourly_transaction_limit: int = Field(default=100, description="Max transactions per hour")
    blocked_tools: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)
    blocked_categories: List[str] = Field(default_factory=list)
    allowed_categories: List[str] = Field(default_factory=list)
    allowed_merchants: List[str] = Field(default_factory=list, description="Empty = all allowed")
    allowed_currencies: List[str] = Field(default_factory=lambda: ["INR"])
    approval_threshold: float = Field(default=2000.0, description="Amount above which approval is needed")
    max_payout: float = Field(default=0.0, description="Maximum payout amount")
    max_refund: float = Field(default=0.0, description="Maximum refund amount")
    time_restrictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Time-based restrictions (e.g., no transactions after hours)"
    )
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = Field(default="")


class PolicyViolation(BaseModel):
    rule: str = Field(description="Rule that was violated")
    expected: str = Field(description="What the policy requires")
    actual: str = Field(description="What was attempted")
    severity: str = Field(default="HIGH")


class PolicyEvaluation(BaseModel):
    result: PolicyResult
    violations: List[PolicyViolation] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    policy_id: str = Field(default="")
    policy_version: int = Field(default=1)

class ApprovalRequest(BaseModel):
    """
    A request for human review/approval.
    Contains full context for the reviewer to make an informed decision.
    """
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    agent_id: str
    agent_name: str = Field(default="")
    user_id: str
    tool_name: str
    amount: float
    currency: str = Field(default="INR")
    merchant_id: str = Field(default="")
    purpose: str = Field(default="")
    risk_score: float = Field(default=0.0)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    policy_violations: List[str] = Field(default_factory=list)
    intent_alignment: float = Field(default=1.0)
    behavior_anomaly: float = Field(default=0.0)
    reasons: List[str] = Field(default_factory=list)
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    reviewer_id: Optional[str] = Field(default=None)
    reviewed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )

class PaymentRecord(BaseModel):
    """
    A simulated payment record in PayMCP.
    Represents the result of executing a financial action through the simulated payment infrastructure.
    """
    payment_id: str = Field(default_factory=lambda: f"pay_{uuid.uuid4().hex[:12]}")
    order_id: str = Field(default="")
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    status: PaymentState = Field(default=PaymentState.CREATED)
    merchant_id: str = Field(default="")
    user_id: str = Field(default="")
    agent_id: str = Field(default="")
    method: str = Field(default="simulated")
    purpose: str = Field(default="")
    category: str = Field(default="general")
    idempotency_key: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrderRecord(BaseModel):
    order_id: str = Field(default_factory=lambda: f"order_{uuid.uuid4().hex[:12]}")
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    status: str = Field(default="created")
    merchant_id: str = Field(default="")
    user_id: str = Field(default="")
    agent_id: str = Field(default="")
    payment_id: Optional[str] = Field(default=None)
    purpose: str = Field(default="")
    category: str = Field(default="general")
    idempotency_key: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PayoutRecord(BaseModel):
    payout_id: str = Field(default_factory=lambda: f"pout_{uuid.uuid4().hex[:12]}")
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    status: str = Field(default="created")
    recipient_id: str = Field(default="")
    agent_id: str = Field(default="")
    purpose: str = Field(default="")
    idempotency_key: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RefundRecord(BaseModel):
    refund_id: str = Field(default_factory=lambda: f"rfnd_{uuid.uuid4().hex[:12]}")
    payment_id: str
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    status: str = Field(default="created")
    reason: str = Field(default="")
    agent_id: str = Field(default="")
    idempotency_key: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SettlementRecord(BaseModel):
    settlement_id: str = Field(default_factory=lambda: f"stl_{uuid.uuid4().hex[:12]}")
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    status: str = Field(default="processed")
    merchant_id: str = Field(default="")
    payment_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaymentLinkRecord(BaseModel):
    link_id: str = Field(default_factory=lambda: f"pl_{uuid.uuid4().hex[:12]}")
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    purpose: str = Field(default="")
    merchant_id: str = Field(default="")
    url: str = Field(default="")
    status: str = Field(default="active")
    expires_at: Optional[datetime] = Field(default=None)
    idempotency_key: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AuditEvent(BaseModel):
    """
    Complete audit trail record for every SHIELD evaluation.
    Every financial action proposal generates an audit event, regardless of the decision outcome.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default="")
    agent_id: str = Field(default="")
    user_id: str = Field(default="")
    tool_name: str = Field(default="")
    amount: float = Field(default=0.0)
    currency: str = Field(default="INR")
    intent_text: str = Field(default="")
    policy_result: str = Field(default="")
    authorization_result: str = Field(default="")
    risk_score: float = Field(default=0.0)
    risk_level: str = Field(default="")
    behavior_score: float = Field(default=0.0)
    workflow_result: str = Field(default="")
    injection_result: str = Field(default="")
    duplicate_result: str = Field(default="")
    decision: str = Field(default="")
    reasons: List[str] = Field(default_factory=list)
    approval_required: bool = Field(default=False)
    approval_status: str = Field(default="")
    execution_result: str = Field(default="")
    error: str = Field(default="")
    trust_score: float = Field(default=0.0)
    intent_alignment: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BehaviorProfile(BaseModel):
    """
    Behavioral baseline for an agent.
    Tracks historical patterns to detect anomalies.
    """
    agent_id: str
    avg_requests_per_hour: float = Field(default=5.0)
    avg_transaction_amount: float = Field(default=1000.0)
    std_transaction_amount: float = Field(default=500.0)
    tool_distribution: Dict[str, float] = Field(default_factory=dict)
    merchant_diversity: int = Field(default=5)
    failure_rate: float = Field(default=0.05)
    policy_violation_rate: float = Field(default=0.02)
    blocked_rate: float = Field(default=0.03)
    avg_refund_frequency: float = Field(default=0.1)
    avg_payout_frequency: float = Field(default=0.0)
    typical_categories: List[str] = Field(default_factory=list)
    typical_hours: List[int] = Field(default_factory=lambda: list(range(9, 21)))
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class WorkflowState(BaseModel):
    """
    Tracks an agent's tool-call sequence within a session.
    Used by the workflow validator to detect invalid tool sequences and potential escalation.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    tool_sequence: List[str] = Field(default_factory=list)
    timestamps: List[datetime] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)

    def add_tool(self, tool_name: str) -> None:
        self.tool_sequence.append(tool_name)
        self.timestamps.append(datetime.utcnow())

    def last_tool(self) -> Optional[str]:
        return self.tool_sequence[-1] if self.tool_sequence else None

class EvaluationContext(BaseModel):
    """
    Complete context for a SHIELD security evaluation.
    Bundles all the information SHIELD needs to make an authorization decision.
    """
    transaction: TransactionRequest
    agent: Optional[Agent] = None
    intent: Optional[UserIntent] = None
    workflow_state: Optional[WorkflowState] = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    previous_decisions: List[SecurityDecision] = Field(default_factory=list)
    system_healthy: bool = Field(default=True)
    ml_available: bool = Field(default=True)
    policy_available: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
