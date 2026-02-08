# AOVP — API Reference

Base URL: `http://localhost:8000/api/v1`

---

## POST /verify

Verify an LLM-generated answer against provided context.

### Request Body

```json
{
  "query": "string (required) — the original user question",
  "context": ["string array (required) — context chunks from RAG retrieval"],
  "generated_answer": "string (required) — the LLM's response to verify",
  "policy_config": {
    "min_confidence": 0.7,
    "blocked_keywords": ["medical", "diagnosis"],
    "allow_hallucination": false,
    "require_source_coverage": true,
    "min_coverage_level": "Partial",
    "max_contradiction_sentences": 0
  }
}
```

> `policy_config` is optional. When omitted, global defaults are used.

### Response (200)

```json
{
  "request_id": "req_20260208143021_a1b2c3d4",
  "decision": "ALLOW | FLAG | REFUSE",
  "confidence_score": 0.87,
  "hallucination_detected": false,
  "explanation": "Output verified — confidence 87.00%. ...",
  "verification_details": {
    "similarity_score": 0.92,
    "avg_similarity": 0.85,
    "entailment": {
      "label": "entailment",
      "scores": { "contradiction": 0.02, "entailment": 0.91, "neutral": 0.07 },
      "confidence": 0.91
    },
    "context_coverage": "Full",
    "sentence_analysis": [
      {
        "index": 0,
        "sentence": "The capital of France is Paris.",
        "label": "entailment",
        "entailment_score": 0.95,
        "contradiction_score": 0.01,
        "is_supported": true
      }
    ]
  },
  "policy_results": {
    "passed": ["confidence", "hallucination", "coverage", "entailment"],
    "failed": [],
    "flagged": []
  },
  "timestamp": "2026-02-08T14:30:22.123456+00:00"
}
```

### Decision Logic

| Condition | Decision |
|-----------|----------|
| All policy checks pass | `ALLOW` |
| Any check is borderline (None) but none fail | `FLAG` |
| Any check fails (False) | `REFUSE` |

### Policy Checks Performed

| Check | Fails When |
|-------|-----------|
| `confidence` | Score < `flag_threshold` |
| `hallucination` | Hallucination detected and `allow_hallucination` is false |
| `coverage` | Coverage level below `min_coverage_level` |
| `entailment` | NLI label is `contradiction` |
| `sentence_contradictions` | Contradicting sentences exceed `max_contradiction_sentences` |
| `blocked_keywords` | Answer contains any keyword from `blocked_keywords` list |

---

## POST /policies

Create or update the global default policy rules.

### Request Body

```json
{
  "min_confidence": 0.65,
  "allow_hallucination": false,
  "require_source_coverage": true,
  "min_coverage_level": "Partial",
  "blocked_keywords": ["medical advice", "legal opinion"],
  "max_contradiction_sentences": 0
}
```

### Response (200)

```json
{
  "status": "updated",
  "active_policy": { ... }
}
```

---

## GET /policies

Return the currently active global policy defaults.

### Response (200)

```json
{
  "min_confidence": 0.7,
  "flag_threshold": 0.5,
  "allow_hallucination": false,
  "require_source_coverage": true,
  "min_coverage_level": "Partial",
  "blocked_keywords": [],
  "max_contradiction_sentences": 0
}
```

---

## GET /audit/{id}

Retrieve an audit record by `log_id` or `request_id`.

### Response (200)

```json
{
  "log_id": "log_abc123...",
  "request_id": "req_20260208...",
  "timestamp": "2026-02-08T14:30:22+00:00",
  "query_hash": "sha256...",
  "answer_hash": "sha256...",
  "context_hash": "sha256...",
  "decision": "ALLOW",
  "confidence_score": 0.87,
  "hallucination_detected": false,
  "policy_results": { ... },
  "verification_summary": { ... },
  "processing_time_ms": 342.5
}
```

### Response (404)

```json
{ "detail": "Audit record not found: <id>" }
```

---

## GET /audit/recent?limit=50

Return the most recent audit records (default limit: 50).

---

## GET /audit/stats/today

Aggregate verification statistics for the current date.

### Response (200)

```json
{
  "date": "2026-02-08",
  "total": 142,
  "decisions": { "ALLOW": 98, "FLAG": 31, "REFUSE": 13 },
  "avg_confidence": 0.7234,
  "hallucination_count": 9,
  "hallucination_rate": 0.0634
}
```

---

## GET /health

Liveness / readiness probe.

### Response (200)

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "timestamp": "2026-02-08T14:30:22+00:00"
}
```

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `422` | Validation error — missing/invalid fields |
| `404` | Audit record not found |
| `500` | Internal server error (logged with request_id) |
