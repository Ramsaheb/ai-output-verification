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

RUN mkdir -p /root/.cache

ARG PRELOAD_MODELS=false
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-small
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${NLI_MODEL}')"; \
    fi

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    ENABLE_DOCS=false \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    WEB_CONCURRENCY=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache /home/appuser/.cache

COPY app/ ./app/
COPY models/ ./models/
COPY .env.example ./.env.example

RUN mkdir -p logs/audit \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind ${APP_HOST}:${APP_PORT} --workers ${WEB_CONCURRENCY} --timeout 120 --access-logfile - --error-logfile -"]