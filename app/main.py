"""
AI Output Verification Platform — application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

# ── logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


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
    lifespan=lifespan,
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes under /api/v1
app.include_router(router, prefix=settings.API_V1_STR, tags=["verification"])


@app.get("/", tags=["root"])
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)