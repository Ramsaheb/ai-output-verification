"""
Verification Engine — the brain of AOVP.

Uses pre-trained HuggingFace models (no training required):
  • Sentence-Transformers  → semantic similarity between answer & context
  • CrossEncoder NLI       → entailment / contradiction / neutral detection

Pipeline:
  1. Compute similarity (answer ↔ each context chunk)
  2. Run NLI entailment   (combined context → answer)
  3. Sentence-level NLI   (combined context → each answer sentence)
  4. Weighted confidence   (similarity × w1  +  entailment × w2)
  5. Hallucination flag    (contradiction or low score)
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Set

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from app.core.config import settings

logger = logging.getLogger(__name__)


class VerificationEngine:
    """Stateless (after init) verification service — thread-safe for FastAPI."""

    _STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for",
        "from", "in", "is", "it", "may", "of", "on", "or", "that", "the",
        "to", "was", "were", "with", "will", "would", "should", "could",
        "include", "includes", "including", "also", "such", "like", "cases",
        "case", "common", "serious", "rare", "issue", "issues",
    }

    _SEMANTIC_NORMALIZATION_RULES = [
        (r"\bin rare cases?\b", "may include"),
        (r"\bin uncommon cases?\b", "may include"),
        (r"\bcan occur\b", "may include"),
        (r"\bcan cause\b", "may include"),
        (r"\bmay cause\b", "may include"),
        (r"\bcould cause\b", "may include"),
        (r"\bserious issues?\b", "serious side effects"),
        (r"\badverse effects?\b", "side effects"),
    ]

    @staticmethod
    def _support_thresholds(strict_mode: bool) -> Dict[str, float]:
        if strict_mode:
            return {
                "hard_contradiction": 0.50,
                "strong_entailment": 0.65,
                "semantic_support": 0.65,
                "lexical_support": 0.60,
                "contradiction_guard": 0.35,
                "min_confidence_for_hallucination": 0.50,
            }
        return {
            "hard_contradiction": 0.60,
            "strong_entailment": 0.55,
            "semantic_support": 0.58,
            "lexical_support": 0.45,
            "contradiction_guard": 0.45,
            "min_confidence_for_hallucination": 0.40,
        }

    @staticmethod
    def _contains_domain_keywords(text: str) -> bool:
        haystack = f" {text.lower()} "
        return any(f" {kw} " in haystack for kw in settings.strict_domain_keywords)

    def _resolve_strict_mode(
        self,
        query: str,
        answer: str,
        policy_config: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if policy_config and "strict_mode" in policy_config:
            return {
                "enabled": bool(policy_config["strict_mode"]),
                "source": "policy_override",
            }

        if settings.STRICT_MODE_DEFAULT:
            return {"enabled": True, "source": "settings_default"}

        if settings.AUTO_STRICT_BY_DOMAIN and self._contains_domain_keywords(f"{query} {answer}"):
            return {"enabled": True, "source": "auto_domain"}

        return {"enabled": False, "source": "balanced_default"}

    def __init__(self) -> None:
        self._configure_hf_auth()
        logger.info("Loading verification models …")
        self._embedding = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._nli = CrossEncoder(settings.NLI_MODEL)
        logger.info("Models loaded successfully.")

    @staticmethod
    def _configure_hf_auth() -> None:
        """Expose HF auth token to model loading libraries when configured."""
        token = (settings.HF_TOKEN or "").strip()
        if not token:
            return

        os.environ["HF_TOKEN"] = token
        # Compatibility with clients that still inspect the legacy variable.
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naive but reliable sentence splitter."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _truncate(text: str, max_chars: int = 2000) -> str:
        return text[:max_chars] if len(text) > max_chars else text

    @staticmethod
    def _normalize(text: str) -> str:
        """Unicode normalization + whitespace collapse before NLI."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _semantic_normalize(cls, text: str) -> str:
        """Normalize semantically equivalent risk qualifiers before inference."""
        normalized = cls._normalize(text.lower())
        for pattern, replacement in cls._SEMANTIC_NORMALIZATION_RULES:
            normalized = re.sub(pattern, replacement, normalized)
        return cls._normalize(normalized)

    @classmethod
    def _keyword_overlap(cls, sentence: str, context_chunks: List[str]) -> float:
        """Token-overlap coverage that is resilient to phrasing differences."""
        sent_tokens = {
            t for t in re.findall(r"[a-zA-Z]{3,}", sentence.lower())
            if t not in cls._STOPWORDS
        }
        if not sent_tokens:
            return 0.0

        context_text = " ".join(context_chunks).lower()
        context_tokens = {
            t for t in re.findall(r"[a-zA-Z]{3,}", context_text)
            if t not in cls._STOPWORDS
        }
        if not context_tokens:
            return 0.0

        overlap = sent_tokens.intersection(context_tokens)
        return round(len(overlap) / len(sent_tokens), 4)

    @staticmethod
    def _support_score(
        entailment_score: float,
        semantic_similarity: float,
        lexical_overlap: float,
        contradiction_score: float,
        hard_contradiction_threshold: float,
    ) -> float:
        if contradiction_score >= hard_contradiction_threshold:
            return 0.0
        # Use the stronger signal between direct entailment and grounded overlap,
        # which improves robustness on paraphrased risk qualifiers.
        overlap_grounding = 0.65 * semantic_similarity + 0.35 * lexical_overlap
        score = max(entailment_score, overlap_grounding)
        return round(float(min(max(score, 0.0), 1.0)), 4)

    @staticmethod
    def _resolve_weights(policy_config: Dict[str, Any] | None) -> Dict[str, float]:
        sim_w = settings.SIMILARITY_WEIGHT
        ent_w = settings.ENTAILMENT_WEIGHT
        if policy_config:
            if "similarity_weight" in policy_config:
                sim_w = float(policy_config["similarity_weight"])
            if "entailment_weight" in policy_config:
                ent_w = float(policy_config["entailment_weight"])
        total = sim_w + ent_w
        if total <= 0:
            sim_w, ent_w = settings.SIMILARITY_WEIGHT, settings.ENTAILMENT_WEIGHT
            total = sim_w + ent_w
        return {"similarity_weight": sim_w / total, "entailment_weight": ent_w / total}

    # ── similarity ─────────────────────────────────────────

    def compute_similarity(
        self, answer: str, context_chunks: List[str]
    ) -> Dict[str, Any]:
        """Cosine similarity between the answer embedding and each chunk."""
        if not context_chunks:
            return {"max": 0.0, "avg": 0.0, "per_chunk": []}

        try:
            answer_norm = self._normalize(answer)
            chunks_norm = [self._semantic_normalize(c) for c in context_chunks]
            answer_norm = self._semantic_normalize(answer_norm)

            ans_emb = self._embedding.encode(answer_norm, convert_to_tensor=True)
            ctx_emb = self._embedding.encode(chunks_norm, convert_to_tensor=True)
            sims = util.cos_sim(ans_emb, ctx_emb)[0].cpu().numpy()

            per_chunk = [
                {
                    "index": i,
                    "preview": chunk[:120],
                    "similarity": round(float(s), 4),
                }
                for i, (chunk, s) in enumerate(zip(context_chunks, sims))
            ]

            return {
                "max": round(float(np.max(sims)), 4),
                "avg": round(float(np.mean(sims)), 4),
                "per_chunk": per_chunk,
            }
        except Exception as exc:
            logger.error("Similarity computation failed: %s", exc)
            return {"max": 0.0, "avg": 0.0, "per_chunk": [], "error": str(exc)}

    # ── NLI: full answer ───────────────────────────────────

    def check_entailment(
        self, context_chunks: List[str], answer: str
    ) -> Dict[str, Any]:
        """Run NLI on (combined context → answer).  Labels: 0=contradiction 1=entailment 2=neutral."""
        if not context_chunks:
            return {
                "label": "neutral",
                "scores": {"contradiction": 0.33, "entailment": 0.33, "neutral": 0.34},
                "confidence": 0.34,
            }

        try:
            premise = self._truncate(self._semantic_normalize(" ".join(context_chunks)))
            hypothesis = self._semantic_normalize(answer)
            raw = self._nli.predict([(premise, hypothesis)], apply_softmax=True)

            labels = ["contradiction", "entailment", "neutral"]
            score_map = {l: round(float(raw[0][i]), 4) for i, l in enumerate(labels)}
            best = labels[int(np.argmax(raw[0]))]

            return {
                "label": best,
                "scores": score_map,
                "confidence": round(float(np.max(raw[0])), 4),
            }
        except Exception as exc:
            logger.error("Entailment check failed: %s", exc)
            return {
                "label": "neutral",
                "scores": {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0},
                "confidence": 0.0,
                "error": str(exc),
            }

    # ── NLI: per-sentence ──────────────────────────────────

    def sentence_level_analysis(
        self, context_chunks: List[str], answer: str, strict_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Entailment check for every sentence in the answer."""
        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        try:
            premise = self._truncate(self._semantic_normalize(" ".join(context_chunks)))
            norm_sents = [self._semantic_normalize(s) for s in sentences]
            pairs = [(premise, s) for s in norm_sents]
            raw = self._nli.predict(pairs, apply_softmax=True)

            chunks_norm = [self._semantic_normalize(c) for c in context_chunks] if context_chunks else []
            sentence_similarities = np.zeros(len(sentences), dtype=float)
            if chunks_norm:
                sent_emb = self._embedding.encode(norm_sents, convert_to_tensor=True)
                ctx_emb = self._embedding.encode(chunks_norm, convert_to_tensor=True)
                sim_matrix = util.cos_sim(sent_emb, ctx_emb).cpu().numpy()
                sentence_similarities = np.max(sim_matrix, axis=1)

            thresholds = self._support_thresholds(strict_mode)
            labels = ["contradiction", "entailment", "neutral"]
            results: List[Dict[str, Any]] = []
            for idx, (sent, scores) in enumerate(zip(sentences, raw)):
                best = labels[int(np.argmax(scores))]

                entailment_score = round(float(scores[1]), 4)
                contradiction_score = round(float(scores[0]), 4)
                semantic_similarity = round(float(sentence_similarities[idx]), 4)
                lexical_overlap = self._keyword_overlap(sent, context_chunks)
                support_score = self._support_score(
                    entailment_score,
                    semantic_similarity,
                    lexical_overlap,
                    contradiction_score,
                    thresholds["hard_contradiction"],
                )

                hard_contradiction = contradiction_score >= thresholds["hard_contradiction"]
                strong_entailment = entailment_score >= thresholds["strong_entailment"]
                semantic_support = (
                    semantic_similarity >= thresholds["semantic_support"]
                    and contradiction_score < thresholds["contradiction_guard"]
                )
                lexical_support = (
                    lexical_overlap >= thresholds["lexical_support"]
                    and contradiction_score < thresholds["contradiction_guard"]
                )

                is_supported = (
                    strong_entailment
                    or semantic_support
                    or lexical_support
                    or support_score >= thresholds["strong_entailment"]
                )

                support_reason = "Insufficient support evidence for this claim"
                if hard_contradiction:
                    support_reason = "Contradiction signal is high"
                elif strong_entailment:
                    support_reason = "Direct entailment from context"
                elif semantic_support and lexical_support:
                    support_reason = "Semantic and keyword overlap support this claim"
                elif semantic_support:
                    support_reason = "Semantic similarity suggests this claim is grounded"
                elif lexical_support:
                    support_reason = "Key context entities support this claim"

                adjusted_label = best
                if hard_contradiction:
                    adjusted_label = "contradiction"
                elif is_supported:
                    adjusted_label = "entailment"
                else:
                    adjusted_label = "neutral"

                results.append(
                    {
                        "index": idx,
                        "sentence": sent,
                        "label": adjusted_label,
                        "entailment_score": entailment_score,
                        "contradiction_score": contradiction_score,
                        "similarity_score": semantic_similarity,
                        "lexical_overlap": lexical_overlap,
                        "support_score": support_score,
                        "support_reason": support_reason,
                        "is_supported": is_supported,
                    }
                )
            return results
        except Exception as exc:
            logger.error("Sentence-level NLI failed: %s", exc)
            return [{"index": 0, "sentence": answer, "label": "neutral",
                     "entailment_score": 0.0, "contradiction_score": 0.0,
                     "is_supported": False, "error": str(exc)}]

    # ── coverage ───────────────────────────────────────────

    @staticmethod
    def _coverage_label(similarity: float) -> str:
        if similarity >= 0.75:
            return "Full"
        if similarity >= 0.50:
            return "Partial"
        return "Low"

    @staticmethod
    def _coverage_label_from_percent(coverage_percent: float) -> str:
        if coverage_percent >= 80.0:
            return "Full"
        if coverage_percent >= 40.0:
            return "Partial"
        return "Low"

    # ── hallucination ──────────────────────────────────────

    @staticmethod
    def _is_hallucination(
        entailment_label: str,
        combined_score: float,
        sentence_analysis: List[Dict],
        strict_mode: bool = False,
    ) -> Dict[str, Any]:
        thresholds = VerificationEngine._support_thresholds(strict_mode)
        if entailment_label == "contradiction":
            return {"detected": True, "severity": "high", "reason": "overall_contradiction"}
        if combined_score < thresholds["min_confidence_for_hallucination"]:
            return {"detected": True, "severity": "medium", "reason": "low_combined_confidence"}
        if any(s.get("label") == "contradiction" for s in sentence_analysis):
            return {"detected": True, "severity": "high", "reason": "sentence_level_contradiction"}

        unsupported_count = sum(1 for s in sentence_analysis if not s.get("is_supported"))
        if unsupported_count > 0:
            return {"detected": True, "severity": "low", "reason": "partial_support_gaps"}

        return {"detected": False, "severity": "none", "reason": "grounded"}

    # ── public API ─────────────────────────────────────────

    def verify(
        self,
        query: str,
        context: List[str],
        answer: str,
        policy_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run the full verification pipeline.

        Returns a dict consumed by PolicyEngine and the API response builder.
        """
        t0 = time.perf_counter()

        # 1. Semantic similarity
        sim = self.compute_similarity(answer, context)

        # 1.5 Strictness profile resolution
        strict_profile = self._resolve_strict_mode(query, answer, policy_config)
        strict_mode = strict_profile["enabled"]

        # 2. NLI (whole answer)
        ent = self.check_entailment(context, answer)

        # 3. NLI (per sentence)
        sent_analysis = self.sentence_level_analysis(context, answer, strict_mode=strict_mode)

        # 4. Weighted confidence
        weights = self._resolve_weights(policy_config)
        score = round(
            weights["similarity_weight"] * sim["max"]
            + weights["entailment_weight"] * ent["scores"]["entailment"],
            4,
        )

        # 5. Numeric coverage percentage (graded support score across answer sentences)
        total_sents = max(len(sent_analysis), 1)
        support_values = [float(s.get("support_score", 0.0)) for s in sent_analysis]
        coverage_pct = round(sum(support_values) / total_sents * 100, 1)

        # 6. Coverage & hallucination
        coverage = self._coverage_label_from_percent(coverage_pct)
        hallucination_eval = self._is_hallucination(
            ent["label"], score, sent_analysis, strict_mode=strict_mode
        )
        hallucinated = hallucination_eval["detected"]

        # Consistency rule: entailment implies non-hallucination unless there is strong contradiction.
        if ent["label"] == "entailment" and hallucination_eval["severity"] != "high":
            hallucinated = False
            hallucination_eval = {
                "detected": False,
                "severity": "none",
                "reason": "entailed_answer_consistency_guard",
            }

        # Build concise gap diagnostics for reviewer-facing explanations.
        support_gaps = [
            {
                "sentence": s.get("sentence", "")[:180],
                "label": s.get("label", "neutral"),
                "support_score": s.get("support_score", 0.0),
                "reason": s.get("support_reason", "Insufficient support evidence"),
            }
            for s in sent_analysis
            if not s.get("is_supported") or s.get("label") == "contradiction"
        ]

        # 7. Inference time check
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        warnings: List[str] = []
        if elapsed > settings.MAX_INFERENCE_TIME_MS:
            logger.warning(
                "Inference took %.0f ms (limit: %.0f ms)",
                elapsed, settings.MAX_INFERENCE_TIME_MS,
            )
            warnings.append(
                f"Inference time {elapsed:.0f} ms exceeded {settings.MAX_INFERENCE_TIME_MS:.0f} ms"
            )
        if "error" in sim:
            warnings.append("Similarity computation failed; using fallback scores")
        if "error" in ent:
            warnings.append("Entailment check failed; using fallback scores")
        if any("error" in s for s in sent_analysis):
            warnings.append("Sentence-level entailment failed; using fallback scores")

        return {
            "score": score,
            "similarity_score": sim["max"],
            "avg_similarity": sim["avg"],
            "entailment": ent,
            "hallucination_detected": hallucinated,
            "hallucination_severity": hallucination_eval["severity"],
            "hallucination_reason": hallucination_eval["reason"],
            "context_coverage": coverage,
            "coverage_percent": coverage_pct,
            "sentence_level_analysis": sent_analysis,
            "support_gaps": support_gaps,
            "chunk_similarities": sim["per_chunk"],
            "inference_time_ms": elapsed,
            "confidence_components": {
                "similarity_weight": round(weights["similarity_weight"], 4),
                "entailment_weight": round(weights["entailment_weight"], 4),
                "similarity_score": sim["max"],
                "entailment_score": ent["scores"]["entailment"],
            },
            "strict_mode_applied": strict_mode,
            "strict_mode_source": strict_profile["source"],
            "warnings": warnings,
        }