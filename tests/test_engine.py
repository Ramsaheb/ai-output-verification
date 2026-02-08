"""
Tests for the Verification Engine.

Run with:  pytest tests/ -v
"""

import pytest

from app.verification.engine import VerificationEngine


@pytest.fixture(scope="module")
def engine():
    """Load models once for the entire test module."""
    return VerificationEngine()


# ── Similarity tests ──────────────────────────────────────

class TestSimilarity:
    def test_high_similarity_with_matching_text(self, engine):
        result = engine.compute_similarity(
            answer="Paris is the capital of France.",
            context_chunks=["Paris is the capital city of France."],
        )
        assert result["max"] > 0.8

    def test_low_similarity_with_unrelated_text(self, engine):
        result = engine.compute_similarity(
            answer="Quantum physics explains particle behaviour.",
            context_chunks=["Paris is the capital of France."],
        )
        assert result["max"] < 0.5

    def test_empty_context_returns_zero(self, engine):
        result = engine.compute_similarity(answer="anything", context_chunks=[])
        assert result["max"] == 0.0


# ── Entailment tests ──────────────────────────────────────

class TestEntailment:
    def test_entailed_answer(self, engine):
        result = engine.check_entailment(
            context_chunks=["Paris is the capital of France."],
            answer="The capital of France is Paris.",
        )
        assert result["label"] == "entailment"

    def test_contradicted_answer(self, engine):
        result = engine.check_entailment(
            context_chunks=["Paris is the capital of France."],
            answer="Berlin is the capital of France.",
        )
        assert result["label"] == "contradiction"

    def test_neutral_answer(self, engine):
        result = engine.check_entailment(
            context_chunks=["France is in Europe."],
            answer="France has beautiful weather.",
        )
        # Neutral or at least not entailment
        assert result["label"] in ("neutral", "contradiction")


# ── Full verify pipeline ──────────────────────────────────

class TestVerify:
    def test_allow_case(self, engine):
        result = engine.verify(
            query="What is the capital of France?",
            context=["Paris is the capital of France.", "France is in Western Europe."],
            answer="The capital of France is Paris.",
        )
        assert result["score"] > 0.5
        assert result["hallucination_detected"] is False
        assert result["context_coverage"] in ("Full", "Partial")

    def test_hallucination_case(self, engine):
        result = engine.verify(
            query="What is the capital of France?",
            context=["Paris is the capital of France."],
            answer="Berlin is the capital of France.",
        )
        assert result["hallucination_detected"] is True

    def test_result_contains_required_keys(self, engine):
        result = engine.verify(
            query="test",
            context=["some context"],
            answer="some answer",
        )
        required = {
            "score", "similarity_score", "avg_similarity",
            "entailment", "hallucination_detected",
            "context_coverage", "sentence_level_analysis",
            "chunk_similarities",
        }
        assert required.issubset(result.keys())
