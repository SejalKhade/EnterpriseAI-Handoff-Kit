"""
EnterpriseAI Handoff Kit — Integration Health
Four checks computed from real query_logs, not simulated values.

  1. Retrieval latency      mean measured latency vs threshold
  2. Error rate             logged retrieval errors / total queries
  3. Throughput consistency coefficient of variation on latency
  4. Embedding freshness    mean max embedding age across queries

Every check reports the measured value, the threshold, the formula,
and the number of observations behind it.
"""

from __future__ import annotations
import statistics

from sqlalchemy.orm import Session

from src.config import settings
from src.database import repository as repo


def run_integration_checks(session: Session, sample_size: int = 500) -> dict:
    logs = repo.list_query_logs(session, limit=sample_size)

    if not logs:
        return {
            "module":         "Integration Health",
            "overall_status": "Unknown",
            "checks_passed":  0,
            "total_checks":   4,
            "checks":         [],
            "sample_size":    0,
            "avg_latency_ms": 0.0,
            "error_rate_pct": 0.0,
            "recommendation": "No query logs available. Run the evaluation first.",
        }

    latencies = [l.latency_ms for l in logs]
    errors    = [l for l in logs if l.error]
    ages      = [l.max_embedding_age_days for l in logs]

    avg_latency = round(statistics.mean(latencies), 3)
    p95_latency = round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3) if len(latencies) > 1 else avg_latency
    max_latency = round(max(latencies), 3)

    error_rate = round(len(errors) / len(logs) * 100, 2)

    cv = round(statistics.stdev(latencies) / avg_latency, 4) if len(latencies) > 1 and avg_latency > 0 else 0.0
    avg_age = round(statistics.mean(ages), 2)

    checks = [
        {
            "dimension":   "Retrieval Latency",
            "value":       f"{avg_latency} ms mean, {p95_latency} ms p95, {max_latency} ms max",
            "raw_value":   avg_latency,
            "threshold":   f"mean < {settings.LATENCY_THRESHOLD_MS} ms",
            "status":      "Pass" if avg_latency < settings.LATENCY_THRESHOLD_MS else "Fail",
            "formula":     f"mean(latency_ms over {len(latencies)} queries) = {avg_latency}",
            "source":      "query_logs.latency_ms — measured with perf_counter",
        },
        {
            "dimension":   "Error Rate",
            "value":       f"{error_rate}% ({len(errors)} of {len(logs)})",
            "raw_value":   error_rate,
            "threshold":   f"< {settings.ERROR_RATE_THRESHOLD}%",
            "status":      "Pass" if error_rate < settings.ERROR_RATE_THRESHOLD else "Fail",
            "formula":     f"errors/total = {len(errors)}/{len(logs)} = {error_rate}%",
            "source":      "query_logs.error — non-null values",
        },
        {
            "dimension":   "Throughput Consistency",
            "value":       f"CV = {cv}",
            "raw_value":   cv,
            "threshold":   f"CV < {settings.THROUGHPUT_CV_THRESHOLD}",
            "status":      "Pass" if cv < settings.THROUGHPUT_CV_THRESHOLD else "Fail",
            "formula":     f"stdev/mean = {round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0}/{avg_latency} = {cv}",
            "source":      "Coefficient of variation on query_logs.latency_ms",
        },
        {
            "dimension":   "Embedding Freshness",
            "value":       f"{avg_age} days mean max-age per query",
            "raw_value":   avg_age,
            "threshold":   f"< {settings.STALENESS_THRESHOLD_DAYS} days",
            "status":      "Pass" if avg_age < settings.STALENESS_THRESHOLD_DAYS else "Fail",
            "formula":     f"mean(max_embedding_age_days over {len(ages)} queries) = {avg_age}",
            "source":      "query_logs.max_embedding_age_days",
        },
    ]

    passed = sum(1 for c in checks if c["status"] == "Pass")
    overall = "Healthy" if passed == 4 else "Degraded" if passed >= 2 else "Critical"

    failed_names = [c["dimension"] for c in checks if c["status"] == "Fail"]
    recommendation = (
        "All integration checks within thresholds. No action required."
        if overall == "Healthy" else
        f"Failing checks: {', '.join(failed_names)}. Review within 48 hours."
        if overall == "Degraded" else
        f"Critical: {', '.join(failed_names)} failing. Escalate to the deployment team."
    )

    return {
        "module":          "Integration Health",
        "overall_status":  overall,
        "checks_passed":   passed,
        "total_checks":    4,
        "checks":          checks,
        "sample_size":     len(logs),
        "avg_latency_ms":  avg_latency,
        "p95_latency_ms":  p95_latency,
        "error_rate_pct":  error_rate,
        "throughput_cv":   cv,
        "avg_age_days":    avg_age,
        "thresholds":      {
            "latency_ms":    settings.LATENCY_THRESHOLD_MS,
            "error_rate":    settings.ERROR_RATE_THRESHOLD,
            "throughput_cv": settings.THROUGHPUT_CV_THRESHOLD,
            "freshness_days": settings.STALENESS_THRESHOLD_DAYS,
        },
        "recommendation":  recommendation,
    }
