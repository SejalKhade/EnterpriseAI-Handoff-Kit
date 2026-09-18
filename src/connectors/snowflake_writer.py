"""
EnterpriseAI Handoff Kit — Snowflake Writer
Writes health reports to Snowflake when credentials are configured.
Falls back to local JSONL when they are not, so the pipeline never breaks.

Target schema (created automatically by ensure_tables):
  ENTERPRISE_AI_OPS.HANDOFF_KIT.VECTOR_DB_HEALTH
  ENTERPRISE_AI_OPS.HANDOFF_KIT.EVAL_METRICS_HISTORY
  ENTERPRISE_AI_OPS.HANDOFF_KIT.INTEGRATION_HEALTH

Enable by setting SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.
"""

from __future__ import annotations
from datetime import datetime, timezone
import json

from src.config import settings


DDL_VECTOR_DB_HEALTH = """
CREATE TABLE IF NOT EXISTS VECTOR_DB_HEALTH (
    RUN_TS              TIMESTAMP_NTZ,
    DOC_KEY             VARCHAR(255),
    TITLE               VARCHAR(512),
    CATEGORY            VARCHAR(128),
    CHUNK_COUNT         NUMBER,
    MAX_AGE_DAYS        FLOAT,
    FRESHNESS_SCORE     FLOAT,
    HEALTH_STATUS       VARCHAR(32),
    CONTENT_DRIFT       BOOLEAN,
    AGE_DRIFT           BOOLEAN,
    NEEDS_REINDEX       BOOLEAN
)
"""

DDL_EVAL_METRICS = """
CREATE TABLE IF NOT EXISTS EVAL_METRICS_HISTORY (
    RUN_TS               TIMESTAMP_NTZ,
    RUN_LABEL            VARCHAR(32),
    PRECISION_SCORE      FLOAT,
    RECALL_SCORE         FLOAT,
    F1_SCORE             FLOAT,
    FALSE_POSITIVE_RATE  FLOAT,
    ACCURACY             FLOAT,
    TRUE_POSITIVES       NUMBER,
    FALSE_POSITIVES      NUMBER,
    TRUE_NEGATIVES       NUMBER,
    FALSE_NEGATIVES      NUMBER,
    TOTAL_QUERIES        NUMBER,
    STALENESS_THRESHOLD  NUMBER
)
"""

DDL_INTEGRATION_HEALTH = """
CREATE TABLE IF NOT EXISTS INTEGRATION_HEALTH (
    RUN_TS          TIMESTAMP_NTZ,
    OVERALL_STATUS  VARCHAR(32),
    CHECKS_PASSED   NUMBER,
    AVG_LATENCY_MS  FLOAT,
    P95_LATENCY_MS  FLOAT,
    ERROR_RATE_PCT  FLOAT,
    THROUGHPUT_CV   FLOAT,
    SAMPLE_SIZE     NUMBER
)
"""


class SnowflakeWriter:
    """
    Writes to Snowflake if configured, otherwise appends to a local JSONL file.
    The caller does not need to know which path was taken.
    """

    def __init__(self):
        self.enabled = settings.snowflake_configured
        self.fallback_path = settings.ARTIFACT_DIR / "snowflake_fallback.jsonl"
        self._conn = None
        self._error: str | None = None

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect(self):
        if self._conn is not None:
            return self._conn
        import snowflake.connector  # lazy import
        self._conn = snowflake.connector.connect(
            account=settings.SNOWFLAKE_ACCOUNT,
            user=settings.SNOWFLAKE_USER,
            password=settings.SNOWFLAKE_PASSWORD,
            warehouse=settings.SNOWFLAKE_WAREHOUSE,
            database=settings.SNOWFLAKE_DATABASE,
            schema=settings.SNOWFLAKE_SCHEMA,
        )
        return self._conn

    def ensure_tables(self) -> dict:
        if not self.enabled:
            return {"status": "skipped", "reason": "Snowflake credentials not configured"}
        try:
            cur = self._connect().cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {settings.SNOWFLAKE_DATABASE}")
            cur.execute(f"USE DATABASE {settings.SNOWFLAKE_DATABASE}")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {settings.SNOWFLAKE_SCHEMA}")
            cur.execute(f"USE SCHEMA {settings.SNOWFLAKE_SCHEMA}")
            for ddl in (DDL_VECTOR_DB_HEALTH, DDL_EVAL_METRICS, DDL_INTEGRATION_HEALTH):
                cur.execute(ddl)
            cur.close()
            return {"status": "ok", "tables": 3}
        except Exception as exc:
            self._error = str(exc)
            return {"status": "error", "error": str(exc)}

    # ── Fallback ─────────────────────────────────────────────────────────────

    def _write_fallback(self, table: str, rows: list[dict]) -> dict:
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(self.fallback_path, "a") as f:
            for row in rows:
                f.write(json.dumps({"table": table, "written_at": ts, "row": row},
                                   default=str) + "\n")
        return {
            "status":  "fallback",
            "target":  str(self.fallback_path),
            "table":   table,
            "rows":    len(rows),
            "reason":  self._error or "Snowflake credentials not configured",
        }

    # ── Writes ───────────────────────────────────────────────────────────────

    def write_vector_health(self, freshness: dict) -> dict:
        rows = [{
            "doc_key":         d["doc_key"],
            "title":           d["title"],
            "category":        d["category"],
            "chunk_count":     d["chunk_count"],
            "max_age_days":    d["max_age_days"],
            "freshness_score": d["freshness_score"],
            "health_status":   d["health_status"],
            "content_drift":   d["content_drift"],
            "age_drift":       d["age_drift"],
            "needs_reindex":   d["needs_reindex"],
        } for d in freshness["documents"]]

        if not self.enabled:
            return self._write_fallback("VECTOR_DB_HEALTH", rows)

        try:
            cur = self._connect().cursor()
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
            cur.executemany(
                """INSERT INTO VECTOR_DB_HEALTH
                   (RUN_TS, DOC_KEY, TITLE, CATEGORY, CHUNK_COUNT, MAX_AGE_DAYS,
                    FRESHNESS_SCORE, HEALTH_STATUS, CONTENT_DRIFT, AGE_DRIFT, NEEDS_REINDEX)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(ts, r["doc_key"], r["title"], r["category"], r["chunk_count"],
                  r["max_age_days"], r["freshness_score"], r["health_status"],
                  r["content_drift"], r["age_drift"], r["needs_reindex"]) for r in rows],
            )
            cur.close()
            return {"status": "ok", "table": "VECTOR_DB_HEALTH", "rows": len(rows)}
        except Exception as exc:
            self._error = str(exc)
            return self._write_fallback("VECTOR_DB_HEALTH", rows)

    def write_eval_metrics(self, evaluation: dict) -> dict:
        rows = []
        for label in ("before", "after"):
            m = evaluation[label]
            cm = m["confusion_matrix"]
            rows.append({
                "run_label":           label,
                "precision":           m["precision"],
                "recall":              m["recall"],
                "f1_score":            m["f1_score"],
                "false_positive_rate": m["false_positive_rate"],
                "accuracy":            m["accuracy"],
                "true_positives":      cm["true_positives"],
                "false_positives":     cm["false_positives"],
                "true_negatives":      cm["true_negatives"],
                "false_negatives":     cm["false_negatives"],
                "total_queries":       evaluation["total_queries"],
                "staleness_threshold": evaluation["staleness_threshold"],
            })

        if not self.enabled:
            return self._write_fallback("EVAL_METRICS_HISTORY", rows)

        try:
            cur = self._connect().cursor()
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
            cur.executemany(
                """INSERT INTO EVAL_METRICS_HISTORY
                   (RUN_TS, RUN_LABEL, PRECISION_SCORE, RECALL_SCORE, F1_SCORE,
                    FALSE_POSITIVE_RATE, ACCURACY, TRUE_POSITIVES, FALSE_POSITIVES,
                    TRUE_NEGATIVES, FALSE_NEGATIVES, TOTAL_QUERIES, STALENESS_THRESHOLD)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(ts, r["run_label"], r["precision"], r["recall"], r["f1_score"],
                  r["false_positive_rate"], r["accuracy"], r["true_positives"],
                  r["false_positives"], r["true_negatives"], r["false_negatives"],
                  r["total_queries"], r["staleness_threshold"]) for r in rows],
            )
            cur.close()
            return {"status": "ok", "table": "EVAL_METRICS_HISTORY", "rows": len(rows)}
        except Exception as exc:
            self._error = str(exc)
            return self._write_fallback("EVAL_METRICS_HISTORY", rows)

    def write_integration_health(self, integration: dict) -> dict:
        row = {
            "overall_status": integration["overall_status"],
            "checks_passed":  integration["checks_passed"],
            "avg_latency_ms": integration.get("avg_latency_ms", 0.0),
            "p95_latency_ms": integration.get("p95_latency_ms", 0.0),
            "error_rate_pct": integration.get("error_rate_pct", 0.0),
            "throughput_cv":  integration.get("throughput_cv", 0.0),
            "sample_size":    integration.get("sample_size", 0),
        }

        if not self.enabled:
            return self._write_fallback("INTEGRATION_HEALTH", [row])

        try:
            cur = self._connect().cursor()
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
            cur.execute(
                """INSERT INTO INTEGRATION_HEALTH
                   (RUN_TS, OVERALL_STATUS, CHECKS_PASSED, AVG_LATENCY_MS,
                    P95_LATENCY_MS, ERROR_RATE_PCT, THROUGHPUT_CV, SAMPLE_SIZE)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (ts, row["overall_status"], row["checks_passed"], row["avg_latency_ms"],
                 row["p95_latency_ms"], row["error_rate_pct"], row["throughput_cv"],
                 row["sample_size"]),
            )
            cur.close()
            return {"status": "ok", "table": "INTEGRATION_HEALTH", "rows": 1}
        except Exception as exc:
            self._error = str(exc)
            return self._write_fallback("INTEGRATION_HEALTH", [row])

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def target_info(self) -> dict:
        return {
            "enabled":  self.enabled,
            "database": settings.SNOWFLAKE_DATABASE,
            "schema":   settings.SNOWFLAKE_SCHEMA,
            "tables":   ["VECTOR_DB_HEALTH", "EVAL_METRICS_HISTORY", "INTEGRATION_HEALTH"],
            "fallback": str(self.fallback_path),
        }
