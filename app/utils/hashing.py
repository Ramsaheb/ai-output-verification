"""
Hashing & ID-generation utilities.

Used for:
  • Anonymising queries / answers in audit logs  (SHA-256)
  • Creating unique request & log identifiers     (UUID-based)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timezone
from typing import List


def generate_hash(text: str) -> str:
    """Return a hex SHA-256 digest of the given text."""
    salt = os.getenv("HASH_SALT")
    if salt:
        return hmac.new(salt.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_context_hash(chunks: List[str]) -> str:
    """Deterministic hash for a set of context chunks (order-independent)."""
    combined = "||".join(sorted(chunks))
    return generate_hash(combined)


def generate_request_id() -> str:
    """Unique, time-prefixed request identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"req_{ts}_{uuid.uuid4().hex[:8]}"


def generate_log_id() -> str:
    """Unique audit-log identifier."""
    return f"log_{uuid.uuid4().hex}"