# AI Output Verification Platform

A FastAPI middleware that checks whether an LLM answer is grounded in the provided context. It runs similarity and NLI checks, applies policy rules, and logs a decision before any output reaches users.

## What It Does
- Compares answer vs context using embeddings and NLI
- Applies policy gates (confidence, coverage, contradictions, keywords)
- Logs audit trails to SQLite and JSONL

## Tech Used
- Python, FastAPI, Pydantic
- sentence-transformers, CrossEncoder NLI
- SQLAlchemy, SQLite

## Architecture / Flow
Client -> /api/v1/verify -> verification engine -> policy engine -> decision + audit log

## API Endpoints
- POST /api/v1/verify
- GET /api/v1/policies
- POST /api/v1/policies
- GET /api/v1/audit/{id}
- GET /api/v1/health

## Project Structure
```
ai-output-verification/
  app/
  models/
  tests/
  docker/
  docs/
  logs/
```

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Limitations / Notes
- First run downloads models and can take a few minutes.