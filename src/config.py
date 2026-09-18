"""
EnterpriseAI Handoff Kit — Configuration
All settings from environment variables with sensible defaults.
No secrets in code. No API keys on disk.
"""

from __future__ import annotations
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Application settings loaded from environment with defaults."""

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR:       Path = BASE_DIR
    DATA_DIR:       Path = BASE_DIR / "data"
    CORPUS_DIR:     Path = BASE_DIR / "data" / "corpus"
    CHROMA_DIR:     Path = BASE_DIR / "data" / "chroma"
    ARTIFACT_DIR:   Path = BASE_DIR / "data" / "artifacts"

    # ── Database ─────────────────────────────────────────────────────────────
    # Default: SQLite for local dev. Production: set DATABASE_URL to Postgres.
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'data' / 'handoff.db'}"
    )

    # ── Vector store ─────────────────────────────────────────────────────────
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "enterprise_kb")
    EMBEDDING_DIM:     int = int(os.getenv("EMBEDDING_DIM", "128"))
    # tfidf_svd (no download, deterministic) | sentence_transformers (needs torch)
    EMBEDDER_BACKEND:  str = os.getenv("EMBEDDER_BACKEND", "tfidf_svd")
    ST_MODEL_NAME:     str = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")

    # ── Retrieval ────────────────────────────────────────────────────────────
    TOP_K:                 int   = int(os.getenv("TOP_K", "3"))
    RELEVANCE_THRESHOLD:   float = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))

    # Out-of-vocabulary gate. Cosine similarity alone cannot reject out-of-scope
    # questions once vectors are L2 normalized, so retrieval also requires the
    # query to carry real IDF-weighted mass against the fitted vocabulary.
    MIN_QUERY_TFIDF_MASS: float = float(os.getenv("MIN_QUERY_TFIDF_MASS", "0.35"))
    MIN_QUERY_COVERAGE:   float = float(os.getenv("MIN_QUERY_COVERAGE", "0.30"))

    # ── Freshness monitoring ─────────────────────────────────────────────────
    STALENESS_THRESHOLD_DAYS: int = int(os.getenv("STALENESS_THRESHOLD_DAYS", "30"))
    FRESH_CUTOFF:             float = 0.70
    STALE_CUTOFF:             float = 0.30

    # How stale chunks are handled at retrieval time.
    #   penalty      multiply similarity by a decaying freshness weight (recommended)
    #   hard_filter  drop stale chunks entirely (naive baseline)
    FRESHNESS_MODE:         str   = os.getenv("FRESHNESS_MODE", "penalty")
    # Confidence lost per multiple of the threshold exceeded.
    FRESHNESS_DECAY:        float = float(os.getenv("FRESHNESS_DECAY", "0.35"))
    # Minimum weight, so a strong stale match is demoted but never erased.
    FRESHNESS_WEIGHT_FLOOR: float = float(os.getenv("FRESHNESS_WEIGHT_FLOOR", "0.45"))
    # Multiplier for chunks whose embedded text no longer matches the source.
    # Harder than the age penalty because drift is verified by hash, not estimated.
    DRIFT_PENALTY:          float = float(os.getenv("DRIFT_PENALTY", "0.40"))

    # ── Integration health thresholds ────────────────────────────────────────
    LATENCY_THRESHOLD_MS:   float = float(os.getenv("LATENCY_THRESHOLD_MS", "500"))
    ERROR_RATE_THRESHOLD:   float = float(os.getenv("ERROR_RATE_THRESHOLD", "2.0"))
    THROUGHPUT_CV_THRESHOLD: float = float(os.getenv("THROUGHPUT_CV_THRESHOLD", "0.20"))

    # ── Snowflake (optional — writes skipped if unset) ───────────────────────
    SNOWFLAKE_ACCOUNT:   str | None = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER:      str | None = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD:  str | None = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_WAREHOUSE: str = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    SNOWFLAKE_DATABASE:  str = os.getenv("SNOWFLAKE_DATABASE", "ENTERPRISE_AI_OPS")
    SNOWFLAKE_SCHEMA:    str = os.getenv("SNOWFLAKE_SCHEMA", "HANDOFF_KIT")

    # ── AWS S3 (optional — falls back to local artifact dir) ─────────────────
    AWS_REGION:     str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET:      str | None = os.getenv("S3_BUCKET")
    S3_PREFIX:      str = os.getenv("S3_PREFIX", "health-reports/")

    # ── Scheduler ────────────────────────────────────────────────────────────
    HEALTH_CHECK_INTERVAL_MINUTES: int = int(os.getenv("HEALTH_CHECK_INTERVAL_MINUTES", "60"))
    SCHEDULER_ENABLED:             bool = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"

    # ── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    PYTHON_API_URL: str = os.getenv("PYTHON_API_URL", "http://localhost:8000")

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Key is NEVER read from env in the dashboard flow — user pastes it per session.
    CLAUDE_MODEL:  str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "1500"))

    @property
    def snowflake_configured(self) -> bool:
        return all([self.SNOWFLAKE_ACCOUNT, self.SNOWFLAKE_USER, self.SNOWFLAKE_PASSWORD])

    @property
    def s3_configured(self) -> bool:
        return self.S3_BUCKET is not None

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.CORPUS_DIR, self.CHROMA_DIR, self.ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
