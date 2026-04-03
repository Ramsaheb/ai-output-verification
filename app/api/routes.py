"""
API routes — the gateway into AOVP.

Endpoints:
    POST  /verify            → verify an LLM output
    POST  /policies          → create / update default policy rules
    GET   /policies          → view active policy rules
    GET   /audit/{lookup_id} → fetch an audit record
    GET   /audit/recent      → list recent audit records
    GET   /audit/stats/today → today's aggregate stats
    GET   /health            → liveness / readiness probe
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from models.schemas import (
    Decision,
    HealthResponse,
    PolicyConfig,
    VerificationRequest,
    VerificationResponse,
)
from app.verification.engine import VerificationEngine
from app.policies.rules import PolicyEngine
from app.audit.logger import audit_logger
from app.utils.hashing import (
    generate_context_hash,
    generate_hash,
    generate_request_id,
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ── singletons (loaded once on first import) ──────────────
verifier = VerificationEngine()
policy_engine = PolicyEngine()


# ═══════════════════════════════════════════════════════════
#  POST /verify
# ═══════════════════════════════════════════════════════════

@router.post("/verify", response_model=VerificationResponse)
async def verify_output(request: VerificationRequest):
    """Verify an LLM-generated answer against the provided context."""
    start = time.perf_counter()
    request_id = generate_request_id()

    try:
        policy_overrides = (
            request.policy_config.model_dump(exclude_unset=True)
            if request.policy_config
            else None
        )

        # 1 ─ Verification Engine
        vr = verifier.verify(
            query=request.query,
            context=request.context,
            answer=request.generated_answer,
            policy_config=policy_overrides,
        )

        # 2 ─ Policy Engine
        pr = policy_engine.evaluate(
            verification_result=vr,
            policy_config=policy_overrides,
            answer=request.generated_answer,
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        # 3 ─ Audit log
        audit_id = audit_logger.log_transaction(
            request_id=request_id,
            query_hash=generate_hash(request.query),
            answer_hash=generate_hash(request.generated_answer),
            context_hash=generate_context_hash(request.context),
            decision=pr["decision"],
            confidence_score=vr["score"],
            hallucination_detected=vr["hallucination_detected"],
            policy_results=pr,
            verification_summary={
                "score": vr["score"],
                "similarity_score": vr["similarity_score"],
                "avg_similarity": vr["avg_similarity"],
                "entailment_label": vr["entailment"]["label"],
                "entailment_scores": vr["entailment"]["scores"],
                "coverage": vr["context_coverage"],
                "coverage_percent": vr.get("coverage_percent", 0.0),
                "confidence_components": vr.get("confidence_components", {}),
                "strict_mode_applied": vr.get("strict_mode_applied", False),
                "strict_mode_source": vr.get("strict_mode_source", "unknown"),
                "inference_time_ms": vr.get("inference_time_ms"),
                "sentence_count": len(vr.get("sentence_level_analysis", [])),
                "sentence_contradictions": sum(
                    1
                    for s in vr.get("sentence_level_analysis", [])
                    if s.get("label") == "contradiction"
                ),
                "warnings": vr.get("warnings", []),
            },
            processing_time_ms=elapsed_ms,
        )

        # 4 ─ Build explanation
        decision = pr["decision"]
        reasons = pr.get("reasons", [])
        support_gaps = vr.get("support_gaps", [])
        if decision == "ALLOW":
            explanation = (
                f"Output verified — confidence {vr['score']:.2%}. "
                "Content is grounded in the provided context."
            )
            if vr.get("warnings"):
                explanation = f"{explanation} Warnings: {'; '.join(vr['warnings'])}"
        elif decision == "FLAG":
            if support_gaps:
                gap = support_gaps[0]
                explanation = (
                    "Flagged for review. "
                    f"Potentially unsupported phrase: '{gap.get('sentence', '')}'. "
                    f"Reason: {gap.get('reason', 'Insufficient support evidence')}."
                )
            else:
                explanation = f"Flagged for review. {'; '.join(reasons)}"
        else:
            if support_gaps:
                gap = support_gaps[0]
                explanation = (
                    "Output refused. "
                    f"Conflicting phrase: '{gap.get('sentence', '')}'. "
                    f"Reason: {gap.get('reason', 'Contradiction with context')}"
                )
            else:
                explanation = f"Output refused. {'; '.join(reasons)}"

        # 5 ─ Response
        return VerificationResponse(
            request_id=request_id,
            audit_id=audit_id,
            decision=Decision(decision),
            confidence_score=vr["score"],
            hallucination_detected=vr["hallucination_detected"],
            explanation=explanation,
            verification_details={
                "similarity_score": vr["similarity_score"],
                "avg_similarity": vr["avg_similarity"],
                "entailment": vr["entailment"],
                "context_coverage": vr["context_coverage"],
                "coverage_percent": vr.get("coverage_percent", 0.0),
                "hallucination_severity": vr.get("hallucination_severity", "none"),
                "hallucination_reason": vr.get("hallucination_reason", "grounded"),
                "sentence_analysis": vr["sentence_level_analysis"],
                "support_gaps": vr.get("support_gaps", []),
                "inference_time_ms": vr.get("inference_time_ms"),
                "confidence_components": vr.get("confidence_components", {}),
                "strict_mode_applied": vr.get("strict_mode_applied", False),
                "strict_mode_source": vr.get("strict_mode_source", "unknown"),
                "warnings": vr.get("warnings", []),
            },
            policy_results={
                "passed": pr["passed_checks"],
                "failed": pr["failed_checks"],
                "flagged": pr["flagged_checks"],
                "details": pr.get("all_check_details", {}),
            },
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as exc:
        logger.exception("Verification failed for request %s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ═══════════════════════════════════════════════════════════
#  POST /policies
# ═══════════════════════════════════════════════════════════

@router.post("/policies")
async def update_policies(config: PolicyConfig):
    """
    Create or update default policy rules.

    Overrides the global defaults used when a /verify request
    does not include its own policy_config.
    """
    updated = policy_engine.update_defaults(
        config.model_dump(exclude_none=True, exclude_unset=True)
    )
    return {
        "status": "updated",
        "active_policy": updated,
    }


@router.get("/policies")
async def get_policies():
    """Return the currently active default policy."""
    return policy_engine.DEFAULTS


# ═══════════════════════════════════════════════════════════
#  GET /audit
# ═══════════════════════════════════════════════════════════

@router.get("/audit/stats/today")
async def audit_stats_today():
    """Return aggregate verification stats for the current day."""
    return audit_logger.get_statistics()


@router.get("/audit/recent")
async def audit_recent(limit: int = 50, decision: Decision | None = None):
    """Return the latest audit records."""
    return audit_logger.get_recent(
        limit=limit,
        decision=decision.value if decision else None,
    )


@router.get("/audit/{lookup_id}")
async def get_audit_log(lookup_id: str):
    """Fetch a single audit record by log_id or request_id."""
    record = audit_logger.get_by_id(lookup_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Audit record not found: {lookup_id}")
    return record


# ═══════════════════════════════════════════════════════════
#  GET /health
# ═══════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — checks models and database."""
    # Check models
    models_ok = (
        hasattr(verifier, "_embedding")
        and verifier._embedding is not None
        and hasattr(verifier, "_nli")
        and verifier._nli is not None
    )

    # Check DB
    db_ok = False
    try:
        from sqlalchemy import text
        db = audit_logger._SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        pass

    status = "healthy" if (models_ok and db_ok) else "degraded"

    return HealthResponse(
        status=status,
        version=settings.VERSION,
        models_loaded=models_ok,
        db_connected=db_ok,
        timestamp=datetime.now(timezone.utc),
    )