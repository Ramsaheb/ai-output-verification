import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env file."""

    # ── App ────────────────────────────────────────────────
    PROJECT_NAME: str = "AI Output Verification Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    # ── Models ─────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"

    # ── Verification thresholds ────────────────────────────
    CONFIDENCE_ALLOW: float = 0.70   # >= this → ALLOW
    CONFIDENCE_FLAG: float = 0.50    # >= this but < ALLOW → FLAG
    SIMILARITY_WEIGHT: float = 0.40
    ENTAILMENT_WEIGHT: float = 0.60
    MAX_INFERENCE_TIME_MS: float = 30000  # 30s warning threshold

    # ── Database ───────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./aovp_audit.db"

    # ── Audit ──────────────────────────────────────────────
    AUDIT_LOG_DIR: str = "logs/audit"
    AUDIT_HMAC_KEY: Optional[str] = None
    HASH_SALT: Optional[str] = None

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()

# Ensure audit directory exists on import
os.makedirs(settings.AUDIT_LOG_DIR, exist_ok=True)