"""
SHIELD AI — Idempotency & Replay Defense
Prevents duplicate execution and replay attacks.

Two distinct protections:
1. Idempotency — same idempotency_key never creates multiple transactions
2. Replay defense — old/expired requests are rejected

Uses:
- idempotency_key tracking
- Transaction fingerprinting (hash of key fields)
- Timestamp validation
- Nonce/request ID uniqueness
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .models import TransactionRequest


class IdempotencyEngine:
    """
    Prevents duplicate transaction execution.
    Guarantees: same request sent N times results in exactly 1 execution and N-1 replayed responses.
    """

    MAX_REQUEST_AGE = timedelta(minutes=30)

    def __init__(self):
        self._idempotency_store: Dict[str, Dict[str, Any]] = {}
        self._request_ids: Dict[str, datetime] = {}
        self._fingerprints: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def check(self, request: TransactionRequest) -> Tuple[str, Optional[Dict[str, Any]]]:

        with self._lock:
            age = datetime.utcnow() - request.timestamp
            if age > self.MAX_REQUEST_AGE:
                return "REPLAY_EXPIRED", {
                    "reason": f"Request expired: age {age.total_seconds():.0f}s "
                              f"exceeds {self.MAX_REQUEST_AGE.total_seconds():.0f}s limit",
                    "expired": True,
                }

            if request.request_id in self._request_ids:
                return "DUPLICATE_REQUEST_ID", {
                    "reason": f"Request ID {request.request_id} already processed",
                    "original_time": self._request_ids[request.request_id].isoformat(),
                }

            if request.idempotency_key:
                idem_key = f"{request.tool_name}:{request.idempotency_key}"
                if idem_key in self._idempotency_store:
                    prev = self._idempotency_store[idem_key]
                    return "DUPLICATE_IDEMPOTENCY", prev

            fingerprint = request.fingerprint()
            if fingerprint in self._fingerprints:
                prev = self._fingerprints[fingerprint]
                prev_time = prev.get("timestamp", datetime.min)
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time)
                time_diff = (datetime.utcnow() - prev_time).total_seconds()
                if time_diff < 300:  # 5 minute window for fingerprint duplicates
                    return "DUPLICATE_FINGERPRINT", prev

            return "UNIQUE", None

    def reserve(self, request: TransactionRequest) -> bool:
        """
        Reserve an idempotency slot BEFORE execution.
        This prevents race conditions where concurrent requests both pass the check and both try to execute.
        Returns:
            True if reservation successful, False if already reserved.
        """
        with self._lock:
            if request.idempotency_key:
                idem_key = f"{request.tool_name}:{request.idempotency_key}"
                if idem_key in self._idempotency_store:
                    return False
                self._idempotency_store[idem_key] = {
                    "status": "RESERVED",
                    "reserved_at": datetime.utcnow().isoformat(),
                    "request_id": request.request_id,
                }

            if request.request_id in self._request_ids:
                return False
            self._request_ids[request.request_id] = datetime.utcnow()

            return True

    def commit(self, request: TransactionRequest,
               result: Dict[str, Any]) -> None:
        """
        Commit execution result for future idempotent replay.
        Called AFTER successful execution.
        """
        with self._lock:
            if request.idempotency_key:
                idem_key = f"{request.tool_name}:{request.idempotency_key}"
                self._idempotency_store[idem_key] = {
                    **result,
                    "committed_at": datetime.utcnow().isoformat(),
                    "request_id": request.request_id,
                }

            fingerprint = request.fingerprint()
            self._fingerprints[fingerprint] = {
                "request_id": request.request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result.get("status", "unknown"),
            }

    def release(self, request: TransactionRequest) -> None:
        """
        Release a reservation (e.g., if execution was blocked).
        This allows the same idempotency key to be retried if the request was never actually executed.
        """
        with self._lock:
            if request.idempotency_key:
                idem_key = f"{request.tool_name}:{request.idempotency_key}"
                entry = self._idempotency_store.get(idem_key)
                if entry and entry.get("status") == "RESERVED":
                    del self._idempotency_store[idem_key]

    def get_stats(self) -> Dict[str, int]:
        return {
            "idempotency_entries": len(self._idempotency_store),
            "request_ids_tracked": len(self._request_ids),
            "fingerprints_tracked": len(self._fingerprints),
        }

    def reset(self) -> None:
        with self._lock:
            self._idempotency_store.clear()
            self._request_ids.clear()
            self._fingerprints.clear()
