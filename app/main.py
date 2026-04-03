"""
AI Output Verification Platform — application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import router
from app.core.config import settings

# ── logging ────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"


# ── lifespan (startup / shutdown) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    logger.info("Starting %s v%s …", settings.PROJECT_NAME, settings.VERSION)
    # Models are loaded when `routes` is imported (VerificationEngine.__init__)
    logger.info("Ready in %.2f s", time.perf_counter() - start)
    yield
    logger.info("Shutting down …")


# ── FastAPI app ────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Middleware API that verifies AI-generated outputs before they "
        "reach end users — hallucination detection, policy enforcement, "
        "confidence scoring, and audit logging."
    ),
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENABLE_DOCS else None,
    lifespan=lifespan,
)

# Restrict host headers in production-facing deployments.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)

# Compress responses to reduce bandwidth and improve latency.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=(settings.cors_allow_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes under /api/v1
app.include_router(router, prefix=settings.API_V1_STR, tags=["verification"])

app.mount("/ui/static", StaticFiles(directory=str(UI_DIR)), name="ui-static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(UI_DIR / "index.html")


@app.get("/ui", include_in_schema=False)
async def ui_page():
    return FileResponse(UI_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )