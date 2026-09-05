"""
SHIELD AI — Prompt Injection Defense
Detects prompt injection indicators in agent-generated content.

IMPORTANT: This module detects INDICATORS of prompt injection.
The actual defense is the combination of:
- Least privilege
- Deterministic authorization
- Policy enforcement
- Intent validation
- Tool isolation
- Human approval
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

INJECTION_PATTERNS = [
    (re.compile(r'ignore\s+(?:the\s+)?(?:previous|above|prior|user\'?s?)\s+(?:instructions?|limits?|rules?|constraints?)', re.I),
     "INSTRUCTION_OVERRIDE", "HIGH"),
    (re.compile(r'disregard\s+(?:the\s+)?(?:previous|above|prior|user)\s+', re.I),
     "INSTRUCTION_OVERRIDE", "HIGH"),
    (re.compile(r'forget\s+(?:the\s+)?(?:previous|above|prior|all)\s+(?:instructions?|rules?|limits?)', re.I),
     "INSTRUCTION_OVERRIDE", "HIGH"),
    (re.compile(r'override\s+(?:the\s+)?(?:security|policy|limit|restriction|authorization)', re.I),
     "POLICY_OVERRIDE", "CRITICAL"),

    (re.compile(r'(?:bypass|skip|circumvent|avoid)\s+(?:the\s+)?(?:security|authorization|verification|check|policy|limit)', re.I),
     "AUTHORIZATION_BYPASS", "CRITICAL"),
    (re.compile(r'(?:i\s+am|i\'m)\s+(?:authorized|admin|administrator|root|superuser)', re.I),
     "FALSE_AUTHORIZATION", "CRITICAL"),
    (re.compile(r'(?:admin|root|superuser|system)\s+(?:access|mode|privilege|override)', re.I),
     "PRIVILEGE_CLAIM", "CRITICAL"),

    (re.compile(r'(?:hidden|secret|internal)\s+(?:instruction|command|directive)', re.I),
     "HIDDEN_INSTRUCTION", "HIGH"),
    (re.compile(r'(?:system|internal)\s+(?:prompt|instruction|directive)\s*:', re.I),
     "SYSTEM_PROMPT_INJECTION", "CRITICAL"),
    (re.compile(r'<\s*(?:system|instruction|command)\s*>', re.I),
     "TAG_INJECTION", "HIGH"),

    (re.compile(r'(?:change|set|update|modify)\s+(?:the\s+)?(?:amount|limit|maximum|threshold)\s+to\s+', re.I),
     "AMOUNT_MANIPULATION", "HIGH"),
    (re.compile(r'(?:increase|raise|double|triple)\s+(?:the\s+)?(?:amount|limit|spending|transaction)', re.I),
     "AMOUNT_ESCALATION", "HIGH"),
    (re.compile(r'no\s+(?:limit|restriction|maximum|cap)', re.I),
     "LIMIT_REMOVAL", "HIGH"),

    (re.compile(r'(?:act|behave|operate)\s+as\s+(?:a\s+)?(?:different|new|admin|finance|privileged)', re.I),
     "ROLE_MANIPULATION", "CRITICAL"),
    (re.compile(r'(?:switch|change|upgrade)\s+(?:to\s+)?(?:role|agent|identity|privilege)', re.I),
     "ROLE_ESCALATION", "CRITICAL"),

    (re.compile(r'(?:urgent|emergency|immediately|right now)\s*[!.]*\s*(?:send|pay|transfer|process)', re.I),
     "URGENCY_PRESSURE", "MEDIUM"),

    (re.compile(r'(?:execute|run|process|send)\s+(?:this\s+)?(?:payment|transaction|transfer)\s+(?:now|immediately|without)', re.I),
     "DIRECT_EXECUTION", "HIGH"),

    (re.compile(r'(?:base64|encode|decrypt|decode)\s*:', re.I),
     "OBFUSCATION", "HIGH"),
    (re.compile(r'(?:eval|exec)\s*\(', re.I),
     "CODE_INJECTION", "CRITICAL"),
]


class PromptInjectionDetector:
    """
    Detects prompt injection indicators in content.
    This detector is ONE LAYER in SHIELD's defense-in-depth approach.
    
    The actual defenses against prompt injection are:
    1. Least privilege — agent can only use authorized tools
    2. Deterministic authorization — policies enforce limits
    3. Intent validation — actions must match user intent
    4. Amount limits — hard caps on transaction amounts
    5. Tool isolation — agent cannot bypass SHIELD
    6. Human approval — high-risk actions require human review
    
    Pattern matching provides an EARLY WARNING signal, not a complete defense.
    """

    def __init__(self, patterns=None):
        self.patterns = patterns or INJECTION_PATTERNS

    def scan(self, content: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Scan content for prompt injection indicators.
        
        Args:
            content: Text to scan (could be agent instructions, merchant metadata, tool arguments, etc.)
        Returns:
            (injection_detected, findings)
        """
        if not content:
            return False, []

        findings: List[Dict[str, Any]] = []

        for pattern, category, severity in self.patterns:
            matches = pattern.findall(content)
            if matches:
                findings.append({
                    "category": category,
                    "severity": severity,
                    "pattern_matched": pattern.pattern[:80],
                    "matches": matches[:3],
                })

        return len(findings) > 0, findings

    def scan_request(self, purpose: str = "", metadata: Dict = None,
                     merchant_data: str = "") -> Tuple[str, List[str]]:
        """
        Scan all content associated with a transaction request.
        Checks purpose, metadata values, and merchant data.
        Returns:
            (result, reasons)
            result: "CLEAN" | "SUSPICIOUS" | "INJECTION_DETECTED"
        """
        all_content = []
        reasons = []

        if purpose:
            all_content.append(purpose)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    all_content.append(value)

        if merchant_data:
            all_content.append(merchant_data)

        all_findings: List[Dict[str, Any]] = []
        for content in all_content:
            detected, findings = self.scan(content)
            all_findings.extend(findings)

        if not all_findings:
            return "CLEAN", ["No prompt injection indicators detected"]

        has_critical = any(f["severity"] == "CRITICAL" for f in all_findings)
        has_high = any(f["severity"] == "HIGH" for f in all_findings)

        for f in all_findings:
            reasons.append(
                f"[{f['severity']}] {f['category']}: "
                f"matched '{f['matches'][0] if f['matches'] else 'pattern'}'"
            )

        if has_critical:
            return "INJECTION_DETECTED", reasons
        elif has_high:
            return "SUSPICIOUS", reasons
        else:
            return "SUSPICIOUS", reasons

    def get_defense_explanation(self) -> str:
        """
        Return explanation of why pattern detection alone
        is insufficient for prompt injection defense.
        """
        return """
        Prompt Injection Defense Strategy:        
        Pattern detection (this module) is ONE signal. It cannot
        catch all prompt injection attacks, especially novel ones.
        
        The actual defense is DEFENSE IN DEPTH:
        
        1. LEAST PRIVILEGE: Agent can only use authorized tools.
           → Even if injected, agent cannot call tools it doesn't have.
        
        2. DETERMINISTIC AUTHORIZATION: Hard policy limits.
           → ₹5,000 limit means ₹50,000 is blocked regardless of instructions.
        
        3. INTENT VALIDATION: Actions must match original user intent.
           → User said "under ₹5,000" — SHIELD enforces this.
        
        4. CAPABILITY ISOLATION: Agent proposes, SHIELD authorizes.
           → Agent cannot authorize its own actions.
        
        5. HUMAN APPROVAL: High-risk actions require human review.
           → Suspicious requests go to human reviewers.
        
        6. AUDIT TRAIL: Every action is logged.
           → Injection attempts are recorded for analysis.
        
        The key insight: we don't need to perfectly detect every injection attempt. We need to ensure that injected
        instructions cannot bypass deterministic security controls.
        """
