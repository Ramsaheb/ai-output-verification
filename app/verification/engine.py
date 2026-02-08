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
import re
import time
import unicodedata
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from app.core.config import settings

logger = logging.getLogger(__name__)


class VerificationEngine:
    """Stateless (after init) verification service — thread-safe for FastAPI."""

    def __init__(self) -> None:
        logger.info("Loading verification models …")
        self._embedding = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._nli = CrossEncoder(settings.NLI_MODEL)
        logger.info("Models loaded successfully.")

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
            chunks_norm = [self._normalize(c) for c in context_chunks]

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
            premise = self._truncate(self._normalize(" ".join(context_chunks)))
            hypothesis = self._normalize(answer)
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
        self, context_chunks: List[str], answer: str
    ) -> List[Dict[str, Any]]:
        """Entailment check for every sentence in the answer."""
        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        try:
            premise = self._truncate(self._normalize(" ".join(context_chunks)))
            norm_sents = [self._normalize(s) for s in sentences]
            pairs = [(premise, s) for s in norm_sents]
            raw = self._nli.predict(pairs, apply_softmax=True)

            labels = ["contradiction", "entailment", "neutral"]
            results: List[Dict[str, Any]] = []
            for idx, (sent, scores) in enumerate(zip(sentences, raw)):
                best = labels[int(np.argmax(scores))]
                results.append(
                    {
                        "index": idx,
                        "sentence": sent,
                        "label": best,
                        "entailment_score": round(float(scores[1]), 4),
                        "contradiction_score": round(float(scores[0]), 4),
                        "is_supported": best == "entailment",
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

    # ── hallucination ──────────────────────────────────────

    @staticmethod
    def _is_hallucination(
        entailment_label: str,
        combined_score: float,
        sentence_analysis: List[Dict],
    ) -> bool:
        if entailment_label == "contradiction":
            return True
        if combined_score < 0.40:
            return True
        if any(s.get("label") == "contradiction" for s in sentence_analysis):
            return True
        return False

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

        # 2. NLI (whole answer)
        ent = self.check_entailment(context, answer)

        # 3. NLI (per sentence)
        sent_analysis = self.sentence_level_analysis(context, answer)

        # 4. Weighted confidence
        weights = self._resolve_weights(policy_config)
        score = round(
            weights["similarity_weight"] * sim["max"]
            + weights["entailment_weight"] * ent["scores"]["entailment"],
            4,
        )

        # 5. Coverage & hallucination
        coverage = self._coverage_label(sim["max"])
        hallucinated = self._is_hallucination(ent["label"], score, sent_analysis)

        # 6. Numeric coverage percentage (sentences supported / total)
        supported = sum(1 for s in sent_analysis if s.get("is_supported"))
        total_sents = max(len(sent_analysis), 1)
        coverage_pct = round(supported / total_sents * 100, 1)

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
            "context_coverage": coverage,
            "coverage_percent": coverage_pct,
            "sentence_level_analysis": sent_analysis,
            "chunk_similarities": sim["per_chunk"],
            "inference_time_ms": elapsed,
            "confidence_components": {
                "similarity_weight": round(weights["similarity_weight"], 4),
                "entailment_weight": round(weights["entailment_weight"], 4),
                "similarity_score": sim["max"],
                "entailment_score": ent["scores"]["entailment"],
            },
            "warnings": warnings,
        }