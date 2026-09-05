"""
SHIELD AI — PayMCP: Payment MCP Simulator

A simulated MCP-style payment server that represents a generic
payment infrastructure. PayMCP operates ONLY on synthetic
in-memory/SQLite data. No real money ever moves.

Exposed tools:
- create_order()
- fetch_payment()
- create_payment_link()
- refund_payment()
- create_payout()
- fetch_settlement()
- verify_payment()

IMPORTANT: PayMCP itself does NOT perform SHIELD authorization.
SHIELD must authorize BEFORE calling PayMCP. This demonstrates separation of PROPOSAL → AUTHORIZATION → EXECUTION.

Razorpay relationship:
"Razorpay's MCP/agentic infrastructure demonstrates how AI agents can interact with payment capabilities. SHIELD AI explores the
security and governance layer required when autonomous agents are given such capabilities."
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import (
    OrderRecord, PaymentRecord, PaymentLinkRecord, PayoutRecord,
    RefundRecord, SettlementRecord, PaymentState,
    is_valid_transition, AuditEvent
)


class PayMCPError(Exception):
    pass


class InvalidTransitionError(PayMCPError):
    pass


class IdempotencyError(PayMCPError):
    pass


class PayMCP:
    """
    Simulated MCP-Style Payment Server.
    All state is maintained in-memory with optional SQLite persistence.
    Every operation generates an audit event.
    
    This server has NO security logic - it simply executes financial operations. All security is SHIELD's responsibility.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        self._orders: Dict[str, OrderRecord] = {}
        self._payments: Dict[str, PaymentRecord] = {}
        self._payment_links: Dict[str, PaymentLinkRecord] = {}
        self._payouts: Dict[str, PayoutRecord] = {}
        self._refunds: Dict[str, RefundRecord] = {}
        self._settlements: Dict[str, SettlementRecord] = {}
        self._idempotency_store: Dict[str, Dict[str, Any]] = {}
        self._audit_events: List[AuditEvent] = []
        
        self._init_db()
        self._seed_data()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paymcp_audit (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                tool_name TEXT,
                agent_id TEXT,
                user_id TEXT,
                amount REAL,
                currency TEXT,
                status TEXT,
                result TEXT,
                idempotency_key TEXT,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paymcp_orders (
                order_id TEXT PRIMARY KEY,
                amount REAL,
                currency TEXT,
                status TEXT,
                merchant_id TEXT,
                user_id TEXT,
                agent_id TEXT,
                payment_id TEXT,
                purpose TEXT,
                category TEXT,
                idempotency_key TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paymcp_payments (
                payment_id TEXT PRIMARY KEY,
                order_id TEXT,
                amount REAL,
                currency TEXT,
                status TEXT,
                merchant_id TEXT,
                user_id TEXT,
                agent_id TEXT,
                purpose TEXT,
                category TEXT,
                idempotency_key TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _seed_data(self) -> None:
        """Seed synthetic payment data for testing."""
        seed_payments = [
            PaymentRecord(
                payment_id="pay_seed_001",
                order_id="order_seed_001",
                amount=1200.0,
                currency="INR",
                status=PaymentState.CAPTURED,
                merchant_id="merchant_grocery_001",
                user_id="user_001",
                agent_id="agent_shopping_001",
                purpose="Grocery purchase",
                category="groceries",
            ),
            PaymentRecord(
                payment_id="pay_seed_002",
                order_id="order_seed_002",
                amount=4500.0,
                currency="INR",
                status=PaymentState.CAPTURED,
                merchant_id="merchant_hotel_001",
                user_id="user_001",
                agent_id="agent_shopping_001",
                purpose="Hotel booking",
                category="travel",
            ),
            PaymentRecord(
                payment_id="pay_seed_003",
                order_id="order_seed_003",
                amount=800.0,
                currency="INR",
                status=PaymentState.AUTHORIZED,
                merchant_id="merchant_electronics_001",
                user_id="user_002",
                agent_id="agent_shopping_001",
                purpose="Phone case",
                category="electronics",
            ),
            PaymentRecord(
                payment_id="pay_seed_004",
                order_id="order_seed_004",
                amount=15000.0,
                currency="INR",
                status=PaymentState.CAPTURED,
                merchant_id="merchant_appliance_001",
                user_id="user_003",
                agent_id="agent_finance_001",
                purpose="Office supplies",
                category="business",
            ),
        ]
        for p in seed_payments:
            self._payments[p.payment_id] = p

        seed_orders = [
            OrderRecord(
                order_id="order_seed_001", amount=1200.0, merchant_id="merchant_grocery_001",
                user_id="user_001", agent_id="agent_shopping_001",
                status="paid", payment_id="pay_seed_001", category="groceries",
            ),
            OrderRecord(
                order_id="order_seed_002", amount=4500.0, merchant_id="merchant_hotel_001",
                user_id="user_001", agent_id="agent_shopping_001",
                status="paid", payment_id="pay_seed_002", category="travel",
            ),
        ]
        for o in seed_orders:
            self._orders[o.order_id] = o

        self._settlements["stl_seed_001"] = SettlementRecord(
            settlement_id="stl_seed_001",
            amount=5700.0,
            merchant_id="merchant_grocery_001",
            payment_ids=["pay_seed_001"],
            status="processed",
        )

    def _log_audit(self, tool_name: str, agent_id: str, user_id: str,
                   amount: float, currency: str, status: str, result: str,
                   idempotency_key: str = "", metadata: Dict = None) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            tool_name=tool_name,
            agent_id=agent_id,
            user_id=user_id,
            amount=amount,
            currency=currency,
            execution_result=status,
            decision=result,
            metadata=metadata or {},
        )
        self._audit_events.append(event)
        return event

    def _check_idempotency(self, key: str, tool_name: str) -> Optional[Dict[str, Any]]:
        idem_key = f"{tool_name}:{key}"
        return self._idempotency_store.get(idem_key)

    def _store_idempotency(self, key: str, tool_name: str, result: Dict[str, Any]) -> None:
        idem_key = f"{tool_name}:{key}"
        self._idempotency_store[idem_key] = result

    def create_order(self, amount: float, currency: str = "INR",
                     merchant_id: str = "", user_id: str = "",
                     agent_id: str = "", purpose: str = "",
                     category: str = "general",
                     idempotency_key: str = "",
                     metadata: Dict = None) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            amount: Order amount (must be > 0)
            currency: Currency code
            merchant_id: Target merchant
            user_id: End user
            agent_id: Agent creating the order
            purpose: Order purpose
            category: Order category
            idempotency_key: Unique key for idempotent execution
            metadata: Additional metadata
            
        Returns:
            Dict with order details and status
        """
        with self._lock:
            if amount <= 0:
                self._log_audit("create_order", agent_id, user_id, amount,
                              currency, "FAILED", "INVALID_AMOUNT")
                return {"status": "error", "error": "Amount must be positive"}

            if idempotency_key:
                prev = self._check_idempotency(idempotency_key, "create_order")
                if prev is not None:
                    self._log_audit("create_order", agent_id, user_id, amount,
                                  currency, "IDEMPOTENT_REPLAY", "REPLAYED")
                    return {**prev, "idempotent_replay": True}

            order = OrderRecord(
                amount=amount, currency=currency, merchant_id=merchant_id,
                user_id=user_id, agent_id=agent_id, purpose=purpose,
                category=category, idempotency_key=idempotency_key,
                metadata=metadata or {},
            )
            self._orders[order.order_id] = order

            payment = PaymentRecord(
                order_id=order.order_id, amount=amount, currency=currency,
                status=PaymentState.CREATED, merchant_id=merchant_id,
                user_id=user_id, agent_id=agent_id, purpose=purpose,
                category=category, idempotency_key=idempotency_key,
            )
            self._payments[payment.payment_id] = payment
            order.payment_id = payment.payment_id

            payment.status = PaymentState.AUTHORIZED
            payment.status = PaymentState.CAPTURED
            payment.updated_at = datetime.utcnow()
            order.status = "paid"

            result = {
                "status": "success",
                "order_id": order.order_id,
                "payment_id": payment.payment_id,
                "amount": amount,
                "currency": currency,
                "payment_status": payment.status.value,
                "merchant_id": merchant_id,
                "created_at": order.created_at.isoformat(),
                "simulated": True,
            }

            if idempotency_key:
                self._store_idempotency(idempotency_key, "create_order", result)

            self._log_audit("create_order", agent_id, user_id, amount,
                          currency, "SUCCESS", "EXECUTED", idempotency_key)
            return result

    def fetch_payment(self, payment_id: str, agent_id: str = "",
                      user_id: str = "") -> Dict[str, Any]:
        """
        Fetch payment details by payment ID.
        This is a read-only operation.
        """
        payment = self._payments.get(payment_id)
        if payment is None:
            self._log_audit("fetch_payment", agent_id, user_id, 0, "INR",
                          "NOT_FOUND", "FETCH_FAILED")
            return {"status": "error", "error": f"Payment {payment_id} not found"}

        self._log_audit("fetch_payment", agent_id, user_id, payment.amount,
                      payment.currency, "SUCCESS", "FETCHED")
        return {
            "status": "success",
            "payment_id": payment.payment_id,
            "order_id": payment.order_id,
            "amount": payment.amount,
            "currency": payment.currency,
            "payment_status": payment.status.value,
            "merchant_id": payment.merchant_id,
            "user_id": payment.user_id,
            "created_at": payment.created_at.isoformat(),
            "simulated": True,
        }

    def create_payment_link(self, amount: float, currency: str = "INR",
                            purpose: str = "", merchant_id: str = "",
                            agent_id: str = "", user_id: str = "",
                            idempotency_key: str = "",
                            expires_in_hours: int = 24) -> Dict[str, Any]:
        with self._lock:
            if amount <= 0:
                return {"status": "error", "error": "Amount must be positive"}

            if idempotency_key:
                prev = self._check_idempotency(idempotency_key, "create_payment_link")
                if prev is not None:
                    return {**prev, "idempotent_replay": True}

            link = PaymentLinkRecord(
                amount=amount, currency=currency, purpose=purpose,
                merchant_id=merchant_id,
                url=f"https://pay.shield-sim.local/pl/{uuid.uuid4().hex[:8]}",
                expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours),
                idempotency_key=idempotency_key,
            )
            self._payment_links[link.link_id] = link

            result = {
                "status": "success",
                "link_id": link.link_id,
                "url": link.url,
                "amount": amount,
                "currency": currency,
                "expires_at": link.expires_at.isoformat() if link.expires_at else None,
                "simulated": True,
            }

            if idempotency_key:
                self._store_idempotency(idempotency_key, "create_payment_link", result)

            self._log_audit("create_payment_link", agent_id, user_id, amount,
                          currency, "SUCCESS", "EXECUTED", idempotency_key)
            return result

    def refund_payment(self, payment_id: str, amount: Optional[float] = None,
                       reason: str = "", agent_id: str = "",
                       user_id: str = "",
                       idempotency_key: str = "") -> Dict[str, Any]:
        """
        Refund a captured payment (full or partial).
        Only CAPTURED payments can be refunded.
        Validates state transition: CAPTURED → REFUNDED.
        """
        with self._lock:
            if idempotency_key:
                prev = self._check_idempotency(idempotency_key, "refund_payment")
                if prev is not None:
                    return {**prev, "idempotent_replay": True}

            payment = self._payments.get(payment_id)
            if payment is None:
                return {"status": "error", "error": f"Payment {payment_id} not found"}

            if not is_valid_transition(payment.status, PaymentState.REFUNDED):
                raise InvalidTransitionError(
                    f"Cannot refund payment in state {payment.status.value}. "
                    f"Only CAPTURED payments can be refunded."
                )

            refund_amount = amount if amount is not None else payment.amount
            if refund_amount <= 0 or refund_amount > payment.amount:
                return {"status": "error", "error": "Invalid refund amount"}

            refund = RefundRecord(
                payment_id=payment_id, amount=refund_amount,
                currency=payment.currency, status="processed",
                reason=reason, agent_id=agent_id,
                idempotency_key=idempotency_key,
            )
            self._refunds[refund.refund_id] = refund

            payment.status = PaymentState.REFUNDED
            payment.updated_at = datetime.utcnow()

            result = {
                "status": "success",
                "refund_id": refund.refund_id,
                "payment_id": payment_id,
                "refund_amount": refund_amount,
                "currency": payment.currency,
                "simulated": True,
            }

            if idempotency_key:
                self._store_idempotency(idempotency_key, "refund_payment", result)

            self._log_audit("refund_payment", agent_id, user_id, refund_amount,
                          payment.currency, "SUCCESS", "REFUNDED", idempotency_key)
            return result

    def create_payout(self, amount: float, currency: str = "INR",
                      recipient_id: str = "", purpose: str = "",
                      agent_id: str = "", user_id: str = "",
                      idempotency_key: str = "") -> Dict[str, Any]:
        with self._lock:
            if amount <= 0:
                return {"status": "error", "error": "Amount must be positive"}

            if idempotency_key:
                prev = self._check_idempotency(idempotency_key, "create_payout")
                if prev is not None:
                    return {**prev, "idempotent_replay": True}

            payout = PayoutRecord(
                amount=amount, currency=currency, recipient_id=recipient_id,
                status="processed", agent_id=agent_id, purpose=purpose,
                idempotency_key=idempotency_key,
            )
            self._payouts[payout.payout_id] = payout

            result = {
                "status": "success",
                "payout_id": payout.payout_id,
                "amount": amount,
                "currency": currency,
                "recipient_id": recipient_id,
                "simulated": True,
            }

            if idempotency_key:
                self._store_idempotency(idempotency_key, "create_payout", result)

            self._log_audit("create_payout", agent_id, user_id, amount,
                          currency, "SUCCESS", "PAYOUT_CREATED", idempotency_key)
            return result


    def fetch_settlement(self, settlement_id: str = "", merchant_id: str = "",
                         agent_id: str = "", user_id: str = "") -> Dict[str, Any]:
        if settlement_id:
            settlement = self._settlements.get(settlement_id)
            if settlement is None:
                return {"status": "error", "error": f"Settlement {settlement_id} not found"}
            
            self._log_audit("fetch_settlement", agent_id, user_id,
                          settlement.amount, settlement.currency,
                          "SUCCESS", "FETCHED")
            return {
                "status": "success",
                "settlement_id": settlement.settlement_id,
                "amount": settlement.amount,
                "currency": settlement.currency,
                "merchant_id": settlement.merchant_id,
                "payment_count": len(settlement.payment_ids),
                "settlement_status": settlement.status,
                "simulated": True,
            }

        if merchant_id:
            results = [
                s for s in self._settlements.values()
                if s.merchant_id == merchant_id
            ]
            self._log_audit("fetch_settlement", agent_id, user_id, 0, "INR",
                          "SUCCESS", "FETCHED")
            return {
                "status": "success",
                "merchant_id": merchant_id,
                "settlements": [
                    {
                        "settlement_id": s.settlement_id,
                        "amount": s.amount,
                        "status": s.status,
                    }
                    for s in results
                ],
                "simulated": True,
            }

        return {"status": "error", "error": "Provide settlement_id or merchant_id"}

    def verify_payment(self, payment_id: str, agent_id: str = "",
                       user_id: str = "") -> Dict[str, Any]:
        """
        Verify a payment's current status.
        Read-only check used to confirm payment state.
        """
        payment = self._payments.get(payment_id)
        if payment is None:
            self._log_audit("verify_payment", agent_id, user_id, 0, "INR",
                          "NOT_FOUND", "VERIFY_FAILED")
            return {
                "status": "error",
                "error": f"Payment {payment_id} not found",
                "verified": False,
            }

        is_successful = payment.status in (PaymentState.CAPTURED, PaymentState.AUTHORIZED)

        self._log_audit("verify_payment", agent_id, user_id, payment.amount,
                      payment.currency, "SUCCESS", "VERIFIED")
        return {
            "status": "success",
            "payment_id": payment.payment_id,
            "payment_status": payment.status.value,
            "amount": payment.amount,
            "currency": payment.currency,
            "verified": is_successful,
            "simulated": True,
        }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a PayMCP tool by name.
        This is the main entry point used by SHIELD Gateway after authorization.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        tool_map = {
            "create_order": self.create_order,
            "fetch_payment": self.fetch_payment,
            "create_payment_link": self.create_payment_link,
            "refund_payment": self.refund_payment,
            "create_payout": self.create_payout,
            "fetch_settlement": self.fetch_settlement,
            "verify_payment": self.verify_payment,
        }

        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}

        try:
            return tool_fn(**arguments)
        except TypeError as e:
            return {"status": "error", "error": f"Invalid arguments: {str(e)}"}
        except PayMCPError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": f"Execution error: {str(e)}"}

    def get_all_orders(self) -> List[OrderRecord]:
        return list(self._orders.values())

    def get_all_payments(self) -> List[PaymentRecord]:
        return list(self._payments.values())

    def get_all_refunds(self) -> List[RefundRecord]:
        return list(self._refunds.values())

    def get_all_payouts(self) -> List[PayoutRecord]:
        return list(self._payouts.values())

    def get_audit_events(self) -> List[AuditEvent]:
        return list(self._audit_events)

    def get_execution_count(self) -> int:
        return len([e for e in self._audit_events if e.execution_result == "SUCCESS"])

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_orders": len(self._orders),
            "total_payments": len(self._payments),
            "total_refunds": len(self._refunds),
            "total_payouts": len(self._payouts),
            "total_settlements": len(self._settlements),
            "total_payment_links": len(self._payment_links),
            "total_audit_events": len(self._audit_events),
            "idempotency_entries": len(self._idempotency_store),
        }

    def reset(self) -> None:
        self._orders.clear()
        self._payments.clear()
        self._payment_links.clear()
        self._payouts.clear()
        self._refunds.clear()
        self._settlements.clear()
        self._idempotency_store.clear()
        self._audit_events.clear()
        self._seed_data()
