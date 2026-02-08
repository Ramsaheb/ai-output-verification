"""
Audit Logger — immutable record of every verification decision.

Storage strategy (MVP):
  • SQLAlchemy + SQLite  → structured, queryable audit trail
  • Daily JSONL files    → human-readable backup / export

Both run in parallel so you get the best of both worlds.
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Boolean,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings
from app.utils.hashing import generate_log_id

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  SQLAlchemy ORM models
# ═══════════════════════════════════════════════════════════

class Base(DeclarativeBase):
    pass


class AuditRecord(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(64), unique=True, index=True)
    request_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_hash = Column(String(64))
    answer_hash = Column(String(64))
    context_hash = Column(String(64))
    decision = Column(String(10))
    confidence_score = Column(Float)
    hallucination_detected = Column(Boolean, default=False)
    policy_results = Column(Text)          # JSON string
    verification_summary = Column(Text)    # JSON string
    processing_time_ms = Column(Float)
    extra_metadata = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "query_hash": self.query_hash,
            "answer_hash": self.answer_hash,
            "context_hash": self.context_hash,
            "decision": self.decision,
            "confidence_score": self.confidence_score,
            "hallucination_detected": self.hallucination_detected,
            "policy_results": json.loads(self.policy_results) if self.policy_results else {},
            "verification_summary": json.loads(self.verification_summary) if self.verification_summary else {},
            "processing_time_ms": self.processing_time_ms,
            "metadata": json.loads(self.extra_metadata) if self.extra_metadata else None,
        }


# ═══════════════════════════════════════════════════════════
#  Audit Logger class
# ═══════════════════════════════════════════════════════════

class AuditLogger:
    """Dual-write logger: SQLite DB + daily JSONL flat files."""

    def __init__(self) -> None:
        # ── SQLite setup ──────────────────────────────────
        self._engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(bind=self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)

        # ── JSONL flat-file dir ───────────────────────────
        self._log_dir = settings.AUDIT_LOG_DIR
        os.makedirs(self._log_dir, exist_ok=True)

        logger.info("AuditLogger ready  (db=%s, files=%s)", settings.DATABASE_URL, self._log_dir)

    # ── helpers ────────────────────────────────────────────

    def _jsonl_path(self, dt: datetime) -> str:
        return os.path.join(self._log_dir, f"audit_{dt.strftime('%Y%m%d')}.jsonl")

    def _write_jsonl(self, entry: Dict[str, Any], dt: datetime) -> None:
        try:
            with open(self._jsonl_path(dt), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("JSONL write failed: %s", exc)

    def _read_last_entry(self, dt: datetime) -> Optional[Dict[str, Any]]:
        path = self._jsonl_path(dt)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                fh.seek(-read_size, os.SEEK_END)
                chunk = fh.read(read_size).splitlines()
                if not chunk:
                    return None
                last_line = chunk[-1] if chunk[-1] else (chunk[-2] if len(chunk) > 1 else b"")
                if not last_line:
                    return None
                return json.loads(last_line.decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to read last JSONL entry: %s", exc)
            return None

    def _build_integrity(self, entry: Dict[str, Any], dt: datetime) -> Dict[str, Any]:
        last_entry = self._read_last_entry(dt)
        prev_hash = None
        if last_entry:
            meta = last_entry.get("metadata") or {}
            prev_hash = (meta.get("integrity") or {}).get("entry_hash")

        payload = json.dumps(entry, sort_keys=True, default=str)
        chain_input = f"{prev_hash or ''}|{payload}".encode("utf-8")
        entry_hash = hashlib.sha256(chain_input).hexdigest()

        hmac_val = None
        if settings.AUDIT_HMAC_KEY:
            hmac_val = hmac.new(
                settings.AUDIT_HMAC_KEY.encode("utf-8"),
                entry_hash.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        return {
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
            "hmac": hmac_val,
        }

    # ── write ──────────────────────────────────────────────

    def log_transaction(
        self,
        *,
        request_id: str,
        query_hash: str,
        answer_hash: str,
        context_hash: str,
        decision: str,
        confidence_score: float,
        hallucination_detected: bool,
        policy_results: Dict[str, Any],
        verification_summary: Dict[str, Any],
        processing_time_ms: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an audit record.  Returns the generated log_id."""
        log_id = generate_log_id()
        now = datetime.now(timezone.utc)

        record = AuditRecord(
            log_id=log_id,
            request_id=request_id,
            timestamp=now,
            query_hash=query_hash,
            answer_hash=answer_hash,
            context_hash=context_hash,
            decision=decision,
            confidence_score=confidence_score,
            hallucination_detected=hallucination_detected,
            policy_results=json.dumps(policy_results, default=str),
            verification_summary=json.dumps(verification_summary, default=str),
            processing_time_ms=processing_time_ms,
            extra_metadata=json.dumps(extra_metadata, default=str) if extra_metadata else None,
        )

        entry = record.to_dict()
        metadata = entry.get("metadata") or {}
        integrity = self._build_integrity(entry, now)
        metadata["integrity"] = integrity
        entry["metadata"] = metadata
        record.extra_metadata = json.dumps(metadata, default=str)

        # Write to DB
        db: Session = self._SessionLocal()
        db_write_ok = True
        try:
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("DB write failed for log_id=%s", log_id)
            db_write_ok = False
        finally:
            db.close()

        if not db_write_ok:
            metadata["db_write_failed"] = True
            entry["metadata"] = metadata
            metadata["integrity"] = self._build_integrity(entry, now)
            entry["metadata"] = metadata

        # Write to JSONL backup
        self._write_jsonl(entry, now)

        return log_id

    # ── read ───────────────────────────────────────────────

    def get_by_id(self, lookup_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an audit record by log_id OR request_id."""
        db: Session = self._SessionLocal()
        try:
            row = (
                db.query(AuditRecord)
                .filter(
                    (AuditRecord.log_id == lookup_id)
                    | (AuditRecord.request_id == lookup_id)
                )
                .first()
            )
            return row.to_dict() if row else None
        finally:
            db.close()

    def get_recent(self, limit: int = 50, decision: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return most recent audit records."""
        db: Session = self._SessionLocal()
        try:
            query = db.query(AuditRecord)
            if decision:
                query = query.filter(AuditRecord.decision == decision)
            rows = query.order_by(AuditRecord.timestamp.desc()).limit(limit).all()
            return [r.to_dict() for r in rows]
        finally:
            db.close()

    def get_statistics(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Aggregate statistics for a given date (defaults to today)."""
        target = date or datetime.now(timezone.utc)
        start = target.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        db: Session = self._SessionLocal()
        try:
            rows = (
                db.query(AuditRecord)
                .filter(AuditRecord.timestamp >= start, AuditRecord.timestamp < end)
                .all()
            )
            if not rows:
                return {"date": start.strftime("%Y-%m-%d"), "total": 0}

            decisions = [r.decision for r in rows]
            scores = [r.confidence_score for r in rows]
            hall_count = sum(1 for r in rows if r.hallucination_detected)

            return {
                "date": start.strftime("%Y-%m-%d"),
                "total": len(rows),
                "decisions": {
                    "ALLOW": decisions.count("ALLOW"),
                    "FLAG": decisions.count("FLAG"),
                    "REFUSE": decisions.count("REFUSE"),
                },
                "avg_confidence": round(sum(scores) / len(scores), 4),
                "hallucination_count": hall_count,
                "hallucination_rate": round(hall_count / len(rows), 4),
            }
        finally:
            db.close()


# ── module-level singleton ─────────────────────────────────
audit_logger = AuditLogger()