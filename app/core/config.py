import os
from typing import List
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env file."""

    # ── App ────────────────────────────────────────────────
    PROJECT_NAME: str = "AI Output Verification Platform"
    VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"
    ENABLE_DOCS: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    WEB_CONCURRENCY: int = 1
    CORS_ALLOW_ORIGINS: str = "*"
    ALLOWED_HOSTS: str = "*"

    # ── Models ─────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    HF_TOKEN: Optional[str] = None

    # ── Verification thresholds ────────────────────────────
    CONFIDENCE_ALLOW: float = 0.70   # >= this → ALLOW
    CONFIDENCE_FLAG: float = 0.50    # >= this but < ALLOW → FLAG
    SIMILARITY_WEIGHT: float = 0.40
    ENTAILMENT_WEIGHT: float = 0.60
    MAX_INFERENCE_TIME_MS: float = 30000  # 30s warning threshold
    STRICT_MODE_DEFAULT: bool = False
    AUTO_STRICT_BY_DOMAIN: bool = True
    STRICT_DOMAIN_KEYWORDS: str = (
        "medical,medicine,drug,diagnosis,treatment,prescription,"
        "legal,law,contract,liability,finance,investment,tax"
    )

    # ── Database ───────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./aovp_audit.db"

    # ── Audit ──────────────────────────────────────────────
    AUDIT_LOG_DIR: str = "logs/audit"
    AUDIT_HMAC_KEY: Optional[str] = None
    HASH_SALT: Optional[str] = None

    model_config = {"env_file": ".env", "case_sensitive": True}

    @property
    def cors_allow_origins(self) -> List[str]:
        values = [v.strip() for v in self.CORS_ALLOW_ORIGINS.split(",") if v.strip()]
        return values or ["*"]

    @property
    def allowed_hosts(self) -> List[str]:
        values = [v.strip() for v in self.ALLOWED_HOSTS.split(",") if v.strip()]
        if not values:
            return ["*"]
        if "*" in values:
            return ["*"]

        # Keep local tooling/test hosts working while still enforcing host checks.
        for host in ("127.0.0.1", "localhost", "testserver"):
            if host not in values:
                values.append(host)
        return values

    @property
    def strict_domain_keywords(self) -> List[str]:
        return [v.strip().lower() for v in self.STRICT_DOMAIN_KEYWORDS.split(",") if v.strip()]


settings = Settings()

# Ensure audit directory exists on import
os.makedirs(settings.AUDIT_LOG_DIR, exist_ok=True)