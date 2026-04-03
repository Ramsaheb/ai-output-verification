# AI Output Verification Platform

A FastAPI service that verifies whether an LLM answer is grounded in the provided context before returning it to users. It combines semantic similarity, NLI entailment checks, sentence-level support analysis, policy gating, and audit logging.

## Overview

This project is designed as a verification layer between LLM output generation and user delivery.

It answers one practical question:

"Is this generated answer sufficiently supported by the retrieved context, according to our policy?"

The platform returns:

- `ALLOW` when answer quality is acceptable.
- `FLAG` when reviewer attention is recommended.
- `REFUSE` when risk is high or policy checks fail.

## Core Capabilities

- Answer grounding checks using embedding similarity and NLI entailment.
- Sentence-level support scoring for coverage and contradiction detection.
- Hallucination severity classification (`none`, `low`, `medium`, `high`).
- Policy engine with configurable thresholds and safety checks.
- Strict mode support, including auto-trigger for sensitive domains.
- Audit trails to SQLite and JSONL for traceability.
- Browser UI served at `/` and `/ui` for manual verification.

## High-Level Flow

`Client -> /api/v1/verify -> VerificationEngine -> PolicyEngine -> Decision + Audit Log`

Processing sequence:

1. Compute answer/context semantic similarity.
2. Run NLI on combined context vs answer.
3. Run sentence-level NLI and support scoring.
4. Compute weighted confidence score.
5. Detect hallucination risk and severity.
6. Evaluate policy checks.
7. Return decision with explanation and details.
8. Persist audit record.

## Decision Logic Summary

`PolicyEngine` applies checks such as:

- confidence threshold
- hallucination risk/severity
- required source coverage
- entailment label
- sentence contradiction limit
- blocked keywords
- source citation requirement (optional)
- inference latency limit

Decision rules:

- any failed check -> `REFUSE`
- no failures and at least one flagged check -> `FLAG`
- all checks pass -> `ALLOW`

## Project Structure

```text
ai-output-verification/
  app/
    api/
    audit/
    core/
    policies/
    ui/
    utils/
    verification/
  models/
  tests/
  docker/
  docs/
  logs/
  requirements.txt
  README.md
```

## Requirements

- Python 3.10+
- pip
- Internet access on first run (model download)

Main dependencies:

- `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`
- `sentence-transformers`, `transformers`, `torch`
- `sqlalchemy`, `numpy`, `python-dotenv`
- `pytest`, `httpx`

## Local Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment

Copy `.env.example` to `.env` and set your values.

Recommended minimum for local run:

- `APP_ENV=development`
- `ENABLE_DOCS=True`
- `HF_TOKEN=` (optional, recommended to avoid HF anonymous rate limits)

### 3) Run API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4) Open UI

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/ui`

## API Endpoints

Base prefix: `/api/v1`

- `POST /verify`
- `GET /policies`
- `POST /policies`
- `GET /audit/{lookup_id}`
- `GET /audit/recent`
- `GET /audit/stats/today`
- `GET /health`

## Verify API

### Request body

```json
{
  "query": "What is the capital of France?",
  "context": [
    "Paris is the capital city of France.",
    "France is a country in Western Europe."
  ],
  "generated_answer": "The capital of France is Paris.",
  "policy_config": {
    "min_confidence": 0.7,
    "strict_mode": false
  }
}
```

### Important request fields

- `query`: original user question.
- `context`: list of supporting chunks (at least one).
- `generated_answer`: model output being verified.
- `policy_config`: optional per-request policy override.

### Example with curl

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is the capital of France?",
    "context":["Paris is the capital of France."],
    "generated_answer":"Paris is the capital of France."
  }'
```

### Example with PowerShell

```powershell
$url = "http://127.0.0.1:8000/api/v1/verify"
$payload = @{
  query = "What is the capital of France?"
  context = @("Paris is the capital of France.")
  generated_answer = "Paris is the capital of France."
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $payload
```

### Response highlights

`POST /verify` returns:

- identifiers: `request_id`, `audit_id`
- decision: `ALLOW`, `FLAG`, or `REFUSE`
- confidence: `confidence_score`
- safety signal: `hallucination_detected`
- explanation: concise reviewer-facing rationale
- details: entailment scores, coverage, sentence analysis, support gaps, strict mode telemetry
- policy results: passed, failed, and flagged checks

## Policy Configuration

Global defaults are managed in `app/policies/rules.py` and can be updated at runtime via `POST /api/v1/policies`.

Common policy fields:

- `min_confidence`
- `flag_threshold`
- `allow_hallucination`
- `require_source_coverage`
- `min_coverage_level` (`Low`, `Partial`, `Full`)
- `blocked_keywords`
- `max_contradiction_sentences`
- `strict_mode`
- `similarity_weight` and `entailment_weight`
- `max_inference_time_ms`

## Environment Variables

Defined in `.env.example` and loaded by `app/core/config.py`.

Application/runtime:

- `APP_ENV`
- `DEBUG`
- `LOG_LEVEL`
- `ENABLE_DOCS`
- `APP_HOST`
- `APP_PORT`
- `WEB_CONCURRENCY`
- `CORS_ALLOW_ORIGINS`
- `ALLOWED_HOSTS`

Models and verification:

- `EMBEDDING_MODEL`
- `NLI_MODEL`
- `HF_TOKEN`
- `CONFIDENCE_ALLOW`
- `CONFIDENCE_FLAG`
- `SIMILARITY_WEIGHT`
- `ENTAILMENT_WEIGHT`
- `MAX_INFERENCE_TIME_MS`
- `STRICT_MODE_DEFAULT`
- `AUTO_STRICT_BY_DOMAIN`
- `STRICT_DOMAIN_KEYWORDS`

Persistence and audit:

- `DATABASE_URL`
- `AUDIT_LOG_DIR`
- `AUDIT_HMAC_KEY`
- `HASH_SALT`

## Production Recommendations

- Set `APP_ENV=production`.
- Keep `DEBUG=False`.
- Keep `ENABLE_DOCS=False` unless explicitly needed.
- Configure `CORS_ALLOW_ORIGINS` to explicit domains.
- Configure `ALLOWED_HOSTS` to explicit hostnames.
- Use strong random values for `AUDIT_HMAC_KEY` and `HASH_SALT`.
- Set `HF_TOKEN` via secure secret management.
- Monitor inference latency and memory usage.
- Persist audit storage according to retention requirements.

## Testing

Run all tests:

```bash
pytest -q
```

Run focused suites:

```bash
pytest tests/test_api.py tests/test_policies.py -q
```

## Troubleshooting

### Server exits with code 1 on startup

If logs show an address bind error on port 8000, another process is already using that port.

Windows PowerShell example:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object LocalAddress, LocalPort, OwningProcess
```

Then either stop that process or run on a different port:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### First request is slow

Model download/loading happens on first startup. Later runs are faster because model artifacts are cached.

### Hugging Face warning about unauthenticated requests

Set `HF_TOKEN` in `.env` to reduce rate-limit risk and avoid anonymous-access warnings.

## Security Notes

- Do not commit `.env` with real secrets.
- Avoid exposing debug endpoints in public environments.
- Restrict CORS and host headers in production.
- Review `blocked_keywords` and strict mode for your risk domain.

## Current Scope and Limits

- This service verifies grounding against provided context; it does not independently fact-check against the open web.
- Performance and threshold tuning should be validated on your domain dataset before high-scale rollout.