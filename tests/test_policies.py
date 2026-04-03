"""
Tests for the Policy Engine.

Run with:  pytest tests/ -v
"""

import pytest

from app.policies.rules import PolicyEngine


@pytest.fixture
def policy():
    return PolicyEngine()


def _make_verification_result(
    score=0.85,
    hallucination=False,
    hallucination_severity="none",
    coverage="Full",
    entailment_label="entailment",
    sentence_analysis=None,
):
    """Helper to build a mock verification result dict."""
    return {
        "score": score,
        "hallucination_detected": hallucination,
        "hallucination_severity": hallucination_severity,
        "context_coverage": coverage,
        "entailment": {
            "label": entailment_label,
            "scores": {
                "contradiction": 0.05 if entailment_label != "contradiction" else 0.8,
                "entailment": 0.85 if entailment_label == "entailment" else 0.1,
                "neutral": 0.1,
            },
            "confidence": 0.85,
        },
        "sentence_level_analysis": sentence_analysis or [],
    }


class TestPolicyDecisions:
    def test_allow_on_high_confidence(self, policy):
        vr = _make_verification_result(score=0.9)
        result = policy.evaluate(vr)
        assert result["decision"] == "ALLOW"

    def test_flag_on_medium_confidence(self, policy):
        vr = _make_verification_result(score=0.55, entailment_label="neutral")
        result = policy.evaluate(vr)
        assert result["decision"] == "FLAG"

    def test_refuse_on_low_confidence(self, policy):
        vr = _make_verification_result(
            score=0.3,
            entailment_label="contradiction",
            hallucination=True,
            hallucination_severity="high",
        )
        result = policy.evaluate(vr)
        assert result["decision"] == "REFUSE"

    def test_refuse_on_hallucination(self, policy):
        vr = _make_verification_result(hallucination=True, hallucination_severity="high")
        result = policy.evaluate(vr)
        assert result["decision"] == "REFUSE"

    def test_flag_on_low_severity_hallucination(self, policy):
        vr = _make_verification_result(
            hallucination=True,
            hallucination_severity="low",
            entailment_label="neutral",
        )
        result = policy.evaluate(vr)
        assert result["decision"] in ("FLAG", "ALLOW")

    def test_refuse_on_blocked_keywords(self, policy):
        vr = _make_verification_result()
        result = policy.evaluate(
            vr,
            policy_config={"blocked_keywords": ["medical", "diagnosis"]},
            answer="This is a medical diagnosis for the patient.",
        )
        assert result["decision"] == "REFUSE"
        assert "blocked_keywords" in result["failed_checks"]

    def test_custom_threshold_override(self, policy):
        vr = _make_verification_result(score=0.6, entailment_label="neutral")
        result = policy.evaluate(vr, policy_config={"min_confidence": 0.5, "flag_threshold": 0.3})
        assert result["decision"] in ("ALLOW", "FLAG")

    def test_refuse_on_sentence_contradiction(self, policy):
        analysis = [
            {"label": "entailment", "sentence": "Paris is in France."},
            {"label": "contradiction", "sentence": "Berlin is the capital of France."},
        ]
        vr = _make_verification_result(sentence_analysis=analysis)
        result = policy.evaluate(vr)
        assert result["decision"] == "REFUSE"

    def test_result_structure(self, policy):
        vr = _make_verification_result()
        result = policy.evaluate(vr)
        assert "decision" in result
        assert "passed_checks" in result
        assert "failed_checks" in result
        assert "flagged_checks" in result
        assert "reasons" in result
        assert "policy_used" in result
