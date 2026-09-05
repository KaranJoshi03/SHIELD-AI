"""
SHIELD AI — Policy Engine & Policy Compiler
Deterministic policy enforcement for AI agent financial actions.

The policy engine:
1. Defines structured policies per agent role
2. Compiles natural language policies into deterministic rules
3. Evaluates transactions against applicable policies
4. Produces PASS/FAIL/REVIEW with human-readable reasons
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Agent, Policy, PolicyEvaluation, PolicyResult, PolicyViolation,
    TransactionRequest
)

DEFAULT_POLICIES: Dict[str, Policy] = {
    "shopping": Policy(
        policy_id="policy_shopping_v1",
        agent_role="shopping",
        max_transaction=5000.0,
        daily_limit=20000.0,
        hourly_transaction_limit=20,
        blocked_tools=["refund_payment", "create_payout", "fetch_settlement"],
        allowed_tools=["create_order", "fetch_payment", "create_payment_link", "verify_payment"],
        blocked_categories=["gambling", "adult", "crypto"],
        allowed_currencies=["INR"],
        approval_threshold=2000.0,
        max_payout=0.0,
        max_refund=0.0,
        description="Shopping agents can spend at most 5000 INR per transaction, "
                    "20000 INR daily, and cannot issue refunds or payouts.",
        version=1,
    ),
    "finance": Policy(
        policy_id="policy_finance_v1",
        agent_role="finance",
        max_transaction=50000.0,
        daily_limit=200000.0,
        hourly_transaction_limit=15,
        blocked_tools=["create_order", "create_payment_link"],
        allowed_tools=["fetch_settlement", "fetch_payment", "create_payout", "verify_payment"],
        allowed_currencies=["INR", "USD"],
        approval_threshold=10000.0,
        max_payout=25000.0,
        max_refund=0.0,
        description="Finance agents can create payouts up to 25000 INR, "
                    "with transactions reviewed above 10000 INR.",
        version=1,
    ),
    "support": Policy(
        policy_id="policy_support_v1",
        agent_role="support",
        max_transaction=10000.0,
        daily_limit=30000.0,
        hourly_transaction_limit=30,
        blocked_tools=["create_order", "create_payout", "create_payment_link", "fetch_settlement"],
        allowed_tools=["fetch_payment", "refund_payment", "verify_payment"],
        allowed_currencies=["INR"],
        approval_threshold=1500.0,
        max_payout=0.0,
        max_refund=3000.0,
        description="Support agents can issue refunds up to 3000 INR per transaction.",
        version=1,
    ),
    "unknown": Policy(
        policy_id="policy_unknown_v1",
        agent_role="unknown",
        max_transaction=0.0,
        daily_limit=0.0,
        hourly_transaction_limit=0,
        blocked_tools=[
            "create_order", "fetch_payment", "create_payment_link",
            "refund_payment", "create_payout", "fetch_settlement", "verify_payment",
        ],
        allowed_tools=[],
        allowed_currencies=[],
        approval_threshold=0.0,
        max_payout=0.0,
        max_refund=0.0,
        description="Unknown agents have no permissions.",
        version=1,
    ),
}

class PolicyCompiler:
    """
    Compiles natural language policy descriptions into structured, deterministic policy rules.
    
    The compilation pipeline:
    1. Parse natural language
    2. Extract constraints (amounts, tools, categories)
    3. Validate extracted constraints
    4. Generate structured Policy object
    5. Version the policy
    
    NOTE: This is NOT an LLM-based compiler. It uses rule-based extraction for deterministic, auditable compilation.
    """

    AMOUNT_PATTERN = re.compile(
        r'(?:at most|up to|maximum|max|limit|under|below|no more than)\s*'
        r'[₹$]?\s*([\d,]+(?:\.\d+)?)\s*(?:INR|USD|inr|usd)?',
        re.IGNORECASE
    )
    
    CURRENCY_PATTERN = re.compile(r'\b(INR|USD|EUR|GBP)\b', re.IGNORECASE)
    
    TOOL_BLOCK_PATTERN = re.compile(
        r'(?:cannot|can\'t|must not|forbidden|blocked|not allowed|prohibited)\s+'
        r'(?:to\s+)?(?:use\s+)?(?:issue\s+|create\s+|make\s+)?'
        r'((?:(?:refunds?|payouts?|orders?|payment\s*links?|settlements?)(?:\s*(?:or|and|,)\s*)?)+)',
        re.IGNORECASE
    )
    
    TOOL_ALLOW_PATTERN = re.compile(
        r'(?:can|allowed|permitted|may)\s+'
        r'(?:to\s+)?(?:use\s+)?(?:issue\s+|create\s+|make\s+|fetch\s+)?'
        r'(refund|payout|order|payment\s*link|settlement|payment)',
        re.IGNORECASE
    )

    ROLE_PATTERN = re.compile(
        r'(shopping|finance|support|admin)\s*agents?',
        re.IGNORECASE
    )

    DAILY_LIMIT_PATTERN = re.compile(
        r'(?:daily\s+(?:limit|spending|maximum)|per\s+day)\s*'
        r'(?:of\s+)?[₹$]?\s*([\d,]+(?:\.\d+)?)',
        re.IGNORECASE
    )

    TOOL_NAME_MAP = {
        "refund": "refund_payment",
        "refunds": "refund_payment",
        "payout": "create_payout",
        "payouts": "create_payout",
        "order": "create_order",
        "orders": "create_order",
        "payment link": "create_payment_link",
        "payment links": "create_payment_link",
        "paymentlink": "create_payment_link",
        "paymentlinks": "create_payment_link",
        "settlement": "fetch_settlement",
        "settlements": "fetch_settlement",
        "payment": "fetch_payment",
        "payments": "fetch_payment",
    }

    def compile(self, natural_language: str, base_role: str = "unknown") -> Policy:
        """
        Compile a natural language policy into a structured Policy.

        Args:
            natural_language: Policy description in plain English
            base_role: Base agent role to derive policy for
            
        Returns:
            Validated, structured Policy object
        """
        role_match = self.ROLE_PATTERN.search(natural_language)
        role = role_match.group(1).lower() if role_match else base_role

        amount_matches = self.AMOUNT_PATTERN.findall(natural_language)
        max_transaction = float(amount_matches[0].replace(",", "")) if amount_matches else 5000.0

        daily_match = self.DAILY_LIMIT_PATTERN.search(natural_language)
        daily_limit = float(daily_match.group(1).replace(",", "")) if daily_match else max_transaction * 4

        currency_matches = self.CURRENCY_PATTERN.findall(natural_language)
        currencies = list(set(c.upper() for c in currency_matches)) if currency_matches else ["INR"]

        blocked_matches = self.TOOL_BLOCK_PATTERN.findall(natural_language)
        blocked_tools = []
        for match in blocked_matches:
            for word, tool in self.TOOL_NAME_MAP.items():
                if re.search(r'\b' + word.replace(" ", r"\s*") + r'\b', match, re.IGNORECASE):
                    if tool not in blocked_tools:
                        blocked_tools.append(tool)

        allowed_matches = self.TOOL_ALLOW_PATTERN.findall(natural_language)
        allowed_tools = []
        for match in allowed_matches:
            tool_key = match.lower().replace(" ", "")
            if tool_key in self.TOOL_NAME_MAP:
                allowed_tools.append(self.TOOL_NAME_MAP[tool_key])

        policy = Policy(
            agent_role=role,
            max_transaction=max_transaction,
            daily_limit=daily_limit,
            blocked_tools=blocked_tools,
            allowed_tools=allowed_tools if allowed_tools else [],
            allowed_currencies=currencies,
            approval_threshold=max_transaction * 0.4,
            max_payout=0.0 if "create_payout" in blocked_tools else max_transaction,
            max_refund=0.0 if "refund_payment" in blocked_tools else max_transaction * 0.6,
            description=natural_language,
        )

        self._validate(policy)
        return policy

    def _validate(self, policy: Policy) -> None:
        if policy.max_transaction < 0:
            raise ValueError("max_transaction cannot be negative")
        if policy.daily_limit < policy.max_transaction:
            policy.daily_limit = policy.max_transaction
        if policy.approval_threshold > policy.max_transaction:
            policy.approval_threshold = policy.max_transaction * 0.4

    def to_json(self, policy: Policy) -> str:
        return policy.model_dump_json(indent=2)

    def from_json(self, json_str: str) -> Policy:
        return Policy.model_validate_json(json_str)


class PolicyEngine:
    """
    Deterministic policy evaluation engine.
    Evaluates transactions against structured policies and produces PASS/FAIL/REVIEW with explicit reasons.
    
    Policies are deterministic — the same input always produces the same output, regardless of ML state.
    """

    def __init__(self, policies: Optional[Dict[str, Policy]] = None):
        self.policies = policies or dict(DEFAULT_POLICIES)
        self.compiler = PolicyCompiler()

    def get_policy(self, agent_role: str) -> Optional[Policy]:
        return self.policies.get(agent_role, self.policies.get("unknown"))

    def add_policy(self, policy: Policy) -> None:
        self.policies[policy.agent_role] = policy

    def compile_and_add(self, natural_language: str, role: str = "unknown") -> Policy:
        """
        Pipeline: Natural language → Structured policy → Validation → Deterministic enforcement
        """
        policy = self.compiler.compile(natural_language, role)
        self.add_policy(policy)
        return policy

    def evaluate(self, request: TransactionRequest,
                 agent: Agent) -> PolicyEvaluation:
        """
        Evaluate a transaction against the agent's applicable policy.
        
        Checks (in order):
        1. Tool authorization (blocked/allowed tools)
        2. Transaction amount limit
        3. Daily spending limit
        4. Hourly transaction limit
        5. Category restrictions
        6. Currency restrictions
        7. Payout limits
        8. Refund limits
        9. Approval thresholds
        
        Returns:
            PolicyEvaluation with result, violations, and reasons
        """
        policy = self.get_policy(agent.role)
        if policy is None:
            return PolicyEvaluation(
                result=PolicyResult.FAIL,
                violations=[PolicyViolation(
                    rule="POLICY_MISSING",
                    expected="Valid policy for role",
                    actual=f"No policy found for role: {agent.role}",
                    severity="CRITICAL",
                )],
                reasons=["No policy found for agent role"],
            )

        violations: List[PolicyViolation] = []
        reasons: List[str] = []
        needs_review = False

        # 1. Tool authorization
        if request.tool_name in policy.blocked_tools:
            violations.append(PolicyViolation(
                rule="BLOCKED_TOOL",
                expected=f"Tool not in blocked list: {policy.blocked_tools}",
                actual=f"Attempted tool: {request.tool_name}",
                severity="CRITICAL",
            ))
            reasons.append(f"Tool '{request.tool_name}' is blocked for {agent.role} agents")

        if policy.allowed_tools and request.tool_name not in policy.allowed_tools:
            violations.append(PolicyViolation(
                rule="TOOL_NOT_ALLOWED",
                expected=f"Tool in allowed list: {policy.allowed_tools}",
                actual=f"Attempted tool: {request.tool_name}",
                severity="CRITICAL",
            ))
            reasons.append(f"Tool '{request.tool_name}' is not in allowed tools for {agent.role}")

        # 2. Transaction amount
        if request.amount > policy.max_transaction:
            violations.append(PolicyViolation(
                rule="MAX_TRANSACTION_EXCEEDED",
                expected=f"Amount ≤ {policy.max_transaction} {request.currency}",
                actual=f"Amount: {request.amount} {request.currency}",
                severity="CRITICAL",
            ))
            reasons.append(
                f"Amount ₹{request.amount:,.0f} exceeds maximum ₹{policy.max_transaction:,.0f}"
            )

        # 3. Daily spending limit
        if agent.daily_spend + request.amount > policy.daily_limit:
            violations.append(PolicyViolation(
                rule="DAILY_LIMIT_EXCEEDED",
                expected=f"Daily spend ≤ {policy.daily_limit} {request.currency}",
                actual=f"Projected daily: {agent.daily_spend + request.amount}",
                severity="HIGH",
            ))
            reasons.append(
                f"Daily spend would reach ₹{agent.daily_spend + request.amount:,.0f} "
                f"(limit: ₹{policy.daily_limit:,.0f})"
            )

        # 4. Hourly transaction limit
        if agent.hourly_request_count >= policy.hourly_transaction_limit:
            violations.append(PolicyViolation(
                rule="HOURLY_LIMIT_EXCEEDED",
                expected=f"Hourly transactions ≤ {policy.hourly_transaction_limit}",
                actual=f"Current hour: {agent.hourly_request_count}",
                severity="MEDIUM",
            ))
            reasons.append(f"Hourly transaction limit ({policy.hourly_transaction_limit}) exceeded")

        # 5. Category restrictions
        if request.category in policy.blocked_categories:
            violations.append(PolicyViolation(
                rule="BLOCKED_CATEGORY",
                expected=f"Category not in blocked list",
                actual=f"Category: {request.category}",
                severity="CRITICAL",
            ))
            reasons.append(f"Category '{request.category}' is blocked")

        if policy.allowed_categories and request.category not in policy.allowed_categories:
            violations.append(PolicyViolation(
                rule="CATEGORY_NOT_ALLOWED",
                expected=f"Category in allowed list: {policy.allowed_categories}",
                actual=f"Category: {request.category}",
                severity="MEDIUM",
            ))
            reasons.append(f"Category '{request.category}' not in allowed categories")

        # 6. Currency restrictions
        if policy.allowed_currencies and request.currency not in policy.allowed_currencies:
            violations.append(PolicyViolation(
                rule="CURRENCY_NOT_ALLOWED",
                expected=f"Currency in: {policy.allowed_currencies}",
                actual=f"Currency: {request.currency}",
                severity="HIGH",
            ))
            reasons.append(f"Currency '{request.currency}' is not allowed")

        # 7. Payout limits
        if request.tool_name == "create_payout" and request.amount > policy.max_payout:
            violations.append(PolicyViolation(
                rule="PAYOUT_LIMIT_EXCEEDED",
                expected=f"Payout ≤ {policy.max_payout}",
                actual=f"Payout: {request.amount}",
                severity="CRITICAL",
            ))
            reasons.append(
                f"Payout ₹{request.amount:,.0f} exceeds limit ₹{policy.max_payout:,.0f}"
            )

        # 8. Refund limits
        if request.tool_name == "refund_payment" and request.amount > policy.max_refund:
            violations.append(PolicyViolation(
                rule="REFUND_LIMIT_EXCEEDED",
                expected=f"Refund ≤ {policy.max_refund}",
                actual=f"Refund: {request.amount}",
                severity="CRITICAL",
            ))
            reasons.append(
                f"Refund ₹{request.amount:,.0f} exceeds limit ₹{policy.max_refund:,.0f}"
            )

        # 9. Approval threshold
        if request.amount > policy.approval_threshold and not violations:
            needs_review = True
            reasons.append(
                f"Amount ₹{request.amount:,.0f} exceeds approval threshold "
                f"₹{policy.approval_threshold:,.0f}"
            )

        if violations:
            has_critical = any(v.severity == "CRITICAL" for v in violations)
            if has_critical:
                result = PolicyResult.FAIL
            else:
                result = PolicyResult.REVIEW
        elif needs_review:
            result = PolicyResult.REVIEW
        else:
            result = PolicyResult.PASS

        if result == PolicyResult.PASS:
            reasons.append("All policy checks passed")

        return PolicyEvaluation(
            result=result,
            violations=violations,
            reasons=reasons,
            policy_id=policy.policy_id,
            policy_version=policy.version,
        )
