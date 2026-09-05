# RAZORPAY BUILDATHON PITCH PROJECT
Hi, I’m Karan Joshi an M.Tech CSE student at NIT Delhi, and I’m particularly interested in how Razorpay is evolving from a traditional payment gateway toward AI-native and agentic payments. 
While exploring Razorpay’s MCP and agentic-payment ecosystem for the Buildathon, I started thinking about the problem: once AI agents are given the ability to interact with financial tools, how do we make sure they use those capabilities safely?

That led me to build **SHIELD AI** - an AI-agent security and governance layer that sits between an autonomous agent and financial MCP tools. It evaluates the agent’s identity, capabilities, user intent, transaction policy, risk, behavior, and tool usage before allowing an action to execute.

The idea is simple: The agent can propose a financial action, but it cannot authorize its own action. SHIELD AI makes that authorization decision before the payment layer executes it.




# SHIELD AI

**AI Agent Security & Governance Layer for MCP-Based Financial Actions**

*"An AI agent can propose a financial action, but it cannot authorize its own action."*

SHIELD AI is a robust, fail-closed security gateway designed to sit between autonomous AI agents and sensitive financial operations (like payments, payouts, and settlements). By leveraging strict deterministic policies, human intent alignment, and machine learning behavioral scoring, SHIELD ensures AI agents operate safely without running amok.

## Core Features

- **Strict Separation of Duties**: Agents propose actions, SHIELD authorizes, and PayMCP executes. Agents can never bypass SHIELD.
- **Fail-Closed Design**: If any security component (ML engine, policy engine) fails or errors out, the transaction defaults to BLOCK or REVIEW. Security uncertainty never becomes automatic ALLOW.
- **Deterministic Policy Engine**: Compile natural language constraints into hard limits (e.g., maximum transaction amounts, blocked tools like payouts, restricted merchant categories).
- **Intent Alignment & Drift Detection**: Scores whether an agent's proposed action mathematically aligns with the user's initial prompt, flagging "intent drift" over a session.
- **Runaway Agent Prevention**: Real-time velocity checks and anomaly detection instantly pause agents that flood the system with rapid requests.
- **Idempotency & Replay Defense**: Robust locking mechanism that prevents double-execution (even during extreme race conditions).
- **Tool-Chain & Workflow Validation**: Uses DFA-style transitions to catch impossible or escalated sequences (e.g., jumping directly to a payout without verifying a payment).
- **Immutable Audit Logging**: Every single financial decision (ALLOW, REVIEW, BLOCK, PAUSE) is recorded in an SQLite database with its exact rationale.

## Architecture

SHIELD AI processes every request through an **11-Stage Security Pipeline**:

1. **Identity Gating**: Verifies if the agent is known, active, and trusted.
2. **Idempotency Check**: Ensures the `request_id` or `idempotency_key` hasn't been used.
3. **Velocity / Runaway Check**: Prevents agents from flooding the API.
4. **Policy Evaluation**: Assesses hard deterministic limits (daily spend limits, blocked tools).
5. **Intent Alignment**: Scores action against the user's textual intent.
6. **Prompt Injection Scan**: Analyzes merchant notes/metadata for jailbreak instructions.
7. **Workflow Validation**: Ensures the sequence of tools called is legitimate.
8. **Risk Assessment**: Scores overall risk level based on amount, merchant, and context.
9. **Behavior Analysis**: Analyzes deviation from the agent's historical behavioral profile.
10. **Decision Fusion**: Combines all signals into a final strict decision (ALLOW, REVIEW, BLOCK, PAUSE_AGENT).
11. **Execution & Audit**: Routes allowed transactions to PayMCP and writes immutable audit logs.

## Running the Demo

The repository includes a complete interactive Jupyter Notebook (`demo.ipynb`) that demonstrates SHIELD's capabilities across six different scenarios:

1. Legitimate Shopping Transaction (ALLOW)
2. Policy Violation - Hard Limit Exceeded (BLOCK)
3. Capability & Privilege Escalation Denial (BLOCK)
4. Prompt Injection Detection (BLOCK/REVIEW)
5. Intent Drift & Runaway Prevention (PAUSE_AGENT)
6. Immutable Audit Logging (pandas DataFrame visualization)

### Prerequisites

- Python 3.11+
- The dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Launch the Demo

Start Jupyter Notebook and open `demo.ipynb`:

```bash
jupyter notebook demo.ipynb
```

## Testing

SHIELD AI includes a comprehensive, 83-test Pytest suite that validates every layer of the security architecture,including concurrent duplicate requests handling and fail-closed edge cases.

```bash
python -m pytest tests/ -v
```

## Project Structure

```
SHIELD_AI/
├── src/
│   ├── agents.py             # Agent registry and state tracking
│   ├── approval.py           # Human-in-the-loop review workflow
│   ├── audit.py              # SQLite-backed immutable logging
│   ├── authorization.py      # Least-privilege capability engine
│   ├── behavior.py           # Anomaly and runaway detection
│   ├── decision.py           # Final fusion decision engine
│   ├── idempotency.py        # Concurrent replay defense
│   ├── intent.py             # Intent alignment and drift tracking
│   ├── models.py             # Core Pydantic data schemas
│   ├── paymcp.py             # Payment execution simulator
│   ├── policies.py           # Deterministic policy engine
│   ├── prompt_security.py    # Injection/jailbreak detection
│   ├── risk.py               # Risk scoring engine
│   └── shield_gateway.py     # Central orchestration pipeline
├── tests/
│   ├── test_authorization.py
│   ├── test_behavior.py
│   ├── test_concurrency.py
│   ├── test_fail_closed.py
│   ├── test_idempotency.py
│   ├── test_identity.py
│   ├── test_intent.py
│   ├── test_policy.py
│   ├── test_prompt_injection.py
│   ├── test_risk.py
│   └── test_workflow.py
├── demo.ipynb                # Scenario based demo
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
