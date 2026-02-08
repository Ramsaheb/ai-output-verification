"""
Policy Engine — rule-based gatekeeper.

Receives the verification result dict from VerificationEngine and evaluates
it against a set of configurable rules to produce ALLOW / FLAG / REFUSE.

Each check returns:
    (passed: bool | None, reason: str)
        True  → check passed
        None  → borderline, flag for review
        False → hard failure, refuse
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

# Map coverage labels to ordinal values for comparison
_COVERAGE_RANK = {"Low": 1, "Partial": 2, "Full": 3}


class PolicyEngine:
    """Evaluate verification results against configurable policy rules."""

    # ── default thresholds (overridable per-request) ──────
    DEFAULTS: Dict[str, Any] = {
        "min_confidence": settings.CONFIDENCE_ALLOW,
        "flag_threshold": settings.CONFIDENCE_FLAG,
        "allow_hallucination": False,
        "require_source_coverage": True,
        "require_sources": False,
        "min_coverage_level": "Partial",
        "blocked_keywords": [],
        "max_contradiction_sentences": 0,
        "max_inference_time_ms": settings.MAX_INFERENCE_TIME_MS,
        "similarity_weight": settings.SIMILARITY_WEIGHT,
        "entailment_weight": settings.ENTAILMENT_WEIGHT,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        logger.info("PolicyEngine initialised with defaults: %s", self.DEFAULTS)

    def update_defaults(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Thread-safe update of the global default policy. Returns the new defaults."""
        with self._lock:
            self.DEFAULTS.update(overrides)
            logger.info("PolicyEngine defaults updated: %s", self.DEFAULTS)
            return dict(self.DEFAULTS)

    # ── private checks ────────────────────────────────────

    @staticmethod
    def _check_confidence(
        score: float, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        allow = float(policy["min_confidence"])
        flag = float(policy["flag_threshold"])
        if score >= allow:
            return True, f"Confidence {score:.2%} meets threshold ({allow:.0%})"
        if score >= flag:
            return None, f"Confidence {score:.2%} is below {allow:.0%} — flagged"
        return False, f"Confidence {score:.2%} too low (minimum {flag:.0%})"

    @staticmethod
    def _check_hallucination(
        detected: bool, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        if not detected:
            return True, "No hallucination detected"
        if policy.get("allow_hallucination"):
            return None, "Hallucination detected but tolerated by policy — flagged"
        return False, "Hallucination detected — answer not grounded in context"

    @staticmethod
    def _check_coverage(
        coverage: str, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        if not policy.get("require_source_coverage", True):
            return True, "Coverage check disabled by policy"
        required = _COVERAGE_RANK.get(policy["min_coverage_level"], 2)
        actual = _COVERAGE_RANK.get(coverage, 0)
        if actual >= required:
            return True, f"Coverage '{coverage}' meets '{policy['min_coverage_level']}'"
        return False, f"Coverage '{coverage}' below required '{policy['min_coverage_level']}'"

    @staticmethod
    def _check_entailment(
        entailment: Dict,
    ) -> Tuple[Optional[bool], str]:
        label = entailment.get("label", "neutral")
        if label == "contradiction":
            return False, "Answer contradicts the provided context"
        if label == "entailment":
            return True, "Answer is entailed by context"
        return None, "Answer is neutral — not directly supported or contradicted"

    @staticmethod
    def _check_sentence_contradictions(
        sentence_analysis: List[Dict], policy: Dict
    ) -> Tuple[Optional[bool], str]:
        limit = policy.get("max_contradiction_sentences", 0)
        bad = [s for s in sentence_analysis if s.get("label") == "contradiction"]
        if len(bad) <= limit:
            return True, f"Contradicting sentences ({len(bad)}) within limit ({limit})"
        previews = [s["sentence"][:60] for s in bad[:3]]
        return False, f"{len(bad)} sentence(s) contradict context: {previews}"

    @staticmethod
    def _check_blocked_keywords(
        answer: str, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        blocked: List[str] = policy.get("blocked_keywords", [])
        if not blocked:
            return True, "No blocked keywords configured"
        lower = answer.lower()
        found = [kw for kw in blocked if kw.lower() in lower]
        if found:
            return False, f"Blocked keywords detected: {', '.join(found)}"
        return True, "No blocked keywords found in answer"

    @staticmethod
    def _has_citations(answer: str) -> bool:
        if not answer:
            return False
        patterns = [
            r"\[\d+\]",
            r"\(source:[^)]+\)",
            r"https?://",
        ]
        lower = answer.lower()
        return any(re.search(p, lower) for p in patterns)

    @staticmethod
    def _check_sources(
        answer: str, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        if not policy.get("require_sources"):
            return True, "Source citations not required by policy"
        if PolicyEngine._has_citations(answer):
            return True, "Source citations detected"
        return False, "Missing required source citations"

    @staticmethod
    def _check_inference_time(
        inference_time_ms: float, policy: Dict
    ) -> Tuple[Optional[bool], str]:
        limit = policy.get("max_inference_time_ms")
        if limit is None:
            return True, "Inference time limit not configured"
        if inference_time_ms <= float(limit):
            return True, f"Inference time {inference_time_ms:.0f} ms within limit"
        return None, f"Inference time {inference_time_ms:.0f} ms exceeded limit"

    # ── public API ─────────────────────────────────────────

    def evaluate(
        self,
        verification_result: Dict[str, Any],
        policy_config: Optional[Dict[str, Any]] = None,
        answer: str = "",
    ) -> Dict[str, Any]:
        """
        Run every policy check and aggregate into a single decision.

        Decision logic:
            • Any check → False  ⇒  REFUSE
            • Any check → None   ⇒  FLAG
            • All checks → True  ⇒  ALLOW
        """
        # Merge caller overrides on top of defaults (thread-safe read)
        with self._lock:
            policy = {**self.DEFAULTS}
        if policy_config:
            policy.update(policy_config)

        # Run all checks
        checks: Dict[str, Tuple[Optional[bool], str]] = {
            "confidence": self._check_confidence(
                verification_result.get("score", 0.0), policy
            ),
            "hallucination": self._check_hallucination(
                verification_result.get("hallucination_detected", False), policy
            ),
            "coverage": self._check_coverage(
                verification_result.get("context_coverage", "Low"), policy
            ),
            "entailment": self._check_entailment(
                verification_result.get("entailment", {})
            ),
            "sentence_contradictions": self._check_sentence_contradictions(
                verification_result.get("sentence_level_analysis", []), policy
            ),
            "inference_time": self._check_inference_time(
                verification_result.get("inference_time_ms", 0.0), policy
            ),
        }
        if answer:
            checks["blocked_keywords"] = self._check_blocked_keywords(answer, policy)
            checks["sources"] = self._check_sources(answer, policy)

        # Aggregate
        statuses = {name: result[0] for name, result in checks.items()}
        reasons_map = {name: result[1] for name, result in checks.items()}

        passed = [n for n, v in statuses.items() if v is True]
        failed = [n for n, v in statuses.items() if v is False]
        flagged = [n for n, v in statuses.items() if v is None]

        if failed:
            decision = "REFUSE"
        elif flagged:
            decision = "FLAG"
        else:
            decision = "ALLOW"

        # Collect human-readable reasons for the decision
        decision_reasons = (
            [reasons_map[n] for n in failed]
            if failed
            else [reasons_map[n] for n in flagged]
        )

        return {
            "decision": decision,
            "passed_checks": passed,
            "failed_checks": failed,
            "flagged_checks": flagged,
            "reasons": decision_reasons,
            "all_check_details": reasons_map,
            "policy_used": policy,
        }