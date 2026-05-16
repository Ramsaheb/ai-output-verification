# ── Hugging Face Spaces Dockerfile ──────────────────────────
# Deploys the AI Output Verification Platform on HF Spaces.
# HF Spaces requires the app to listen on port 7860.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && echo "torch==2.5.1+cpu" > constraints.txt \
    && grep -vE '^(torch|pytest|httpx)([<>=].*)?$' requirements.txt > requirements.runtime.txt \
    && python -m pip install --prefix=/install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -c constraints.txt \
       -r requirements.runtime.txt

# ── Runtime stage ──────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    ENABLE_DOCS=true \
    APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    WEB_CONCURRENCY=2 \
    CORS_ALLOW_ORIGINS=* \
    ALLOWED_HOSTS=* \
    DATABASE_URL=sqlite:///./aovp_audit.db \
    AUDIT_LOG_DIR=logs/audit

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Hugging Face Spaces requires running as user 1000 to have write permissions
RUN useradd -m -u 1000 user \
    && chown -R user:user /app
USER user

COPY --chown=user:user app/ ./app/
COPY --chown=user:user models/ ./models/

RUN mkdir -p logs/audit

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]
