from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════

class Decision(str, Enum):
    """Possible verification decisions."""
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    REFUSE = "REFUSE"


class EntailmentLabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


# ═══════════════════════════════════════════════════════════
#  Request / Response
# ═══════════════════════════════════════════════════════════

class VerificationRequest(BaseModel):
    """Payload sent by the client for output verification."""
    query: str = Field(..., min_length=1, description="Original user query")
    context: List[str] = Field(..., min_length=1, description="Retrieved context chunks from RAG")
    generated_answer: str = Field(..., min_length=1, description="LLM-generated response to verify")
    policy_config: Optional["PolicyConfig"] = Field(
        default=None,
        description="Optional policy overrides (e.g. min_confidence, blocked_keywords)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the capital of France?",
                "context": [
                    "Paris is the capital city of France.",
                    "France is a country in Western Europe.",
                ],
                "generated_answer": "The capital of France is Paris.",
                "policy_config": {"min_confidence": 0.7},
            }
        }
    }


class VerificationResponse(BaseModel):
    """Payload returned to the client after verification."""
    request_id: str = Field(..., description="Unique ID for this request")
    audit_id: str = Field(..., description="Audit log ID — use with GET /audit/{id}")
    decision: Decision
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_detected: bool
    explanation: str
    verification_details: Dict[str, Any]
    policy_results: Dict[str, Any]
    timestamp: datetime


# ═══════════════════════════════════════════════════════════
#  Internal / Supporting schemas
# ═══════════════════════════════════════════════════════════

class EntailmentResult(BaseModel):
    label: EntailmentLabel
    scores: Dict[str, float]
    confidence: float


class SentenceAnalysis(BaseModel):
    sentence_index: int
    sentence: str
    label: str
    entailment_score: float
    contradiction_score: float
    is_supported: bool


class PolicyConfig(BaseModel):
    """Configurable policy rules. Extra/unknown fields are rejected."""
    min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    flag_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    allow_hallucination: bool = False
    require_source_coverage: bool = True
    require_sources: bool = False
    min_coverage_level: str = Field(default="Partial", pattern="^(Low|Partial|Full)$")
    blocked_keywords: List[str] = Field(default_factory=list)
    max_contradiction_sentences: int = Field(default=0, ge=0)
    similarity_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entailment_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_inference_time_ms: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_weights(self) -> "PolicyConfig":
        if (self.similarity_weight is None) ^ (self.entailment_weight is None):
            raise ValueError("Both similarity_weight and entailment_weight must be set together")
        if self.similarity_weight is not None and self.entailment_weight is not None:
            total = self.similarity_weight + self.entailment_weight
            if total <= 0.0:
                raise ValueError("Confidence weights must sum to a positive value")
        return self

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    db_connected: bool
    timestamp: datetime