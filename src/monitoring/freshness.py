"""
EnterpriseAI Handoff Kit — Freshness Monitor
Scores embedding staleness from real database timestamps and detects
drift between source content and what is actually embedded.

Two independent signals:

  1. AGE DRIFT      embedding older than the staleness threshold
  2. CONTENT DRIFT  source content_hash differs from the hash captured
                    at embed time, meaning the source changed after embedding

Content drift is the more serious failure: the vector store is serving
answers from text that no longer exists in the source document.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.config import settings
from src.database import repository as repo
from src.rag.pipeline import age_in_days


def classify(freshness_score: float) -> str:
    if freshness_score > settings.FRESH_CUTOFF:
        return "Fresh"
    if freshness_score > settings.STALE_CUTOFF:
        return "Stale"
    return "Critical"


def score_document(doc, records, threshold: int) -> dict:
    """
    Score one document's embedding health.

    freshness_score = max(0, 1 - (max_chunk_age_days / threshold))
    content_drift   = any(chunk.source_hash_at_embed != doc.content_hash)
    """
    if not records:
        return {
            "doc_key":          doc.doc_key,
            "title":            doc.title,
            "category":         doc.category,
            "chunk_count":      0,
            "max_age_days":     0.0,
            "avg_age_days":     0.0,
            "freshness_score":  0.0,
            "health_status":    "Critical",
            "content_drift":    True,
            "age_drift":        True,
            "needs_reindex":    True,
            "formula":          "no embeddings present",
            "action":           "Document has no embeddings. Index immediately.",
        }

    ages = [age_in_days(r.embedded_at) for r in records]
    max_age = max(ages)
    avg_age = sum(ages) / len(ages)

    score = max(0.0, 1.0 - (max_age / threshold))
    status = classify(score)

    content_drift = any(r.source_hash_at_embed != doc.content_hash for r in records)
    age_drift = max_age > threshold
    needs_reindex = content_drift or age_drift

    if content_drift:
        action = ("Source content changed after embedding. Re-index now — "
                  "the vector store is serving text that no longer exists in the source.")
    elif status == "Critical":
        action = "Embedding exceeds staleness threshold. Re-index immediately."
    elif status == "Stale":
        action = "Schedule re-indexing within 7 days."
    else:
        action = "No action required."

    return {
        "doc_key":          doc.doc_key,
        "title":            doc.title,
        "category":         doc.category,
        "chunk_count":      len(records),
        "max_age_days":     round(max_age, 2),
        "avg_age_days":     round(avg_age, 2),
        "freshness_score":  round(score, 4),
        "health_status":    status,
        "content_drift":    content_drift,
        "age_drift":        age_drift,
        "needs_reindex":    needs_reindex,
        "source_hash":      doc.content_hash[:12],
        "embedded_hash":    records[0].source_hash_at_embed[:12],
        "formula":          f"max(0, 1 - ({max_age:.2f} / {threshold})) = {score:.4f}",
        "action":           action,
    }


def run_freshness_check(session: Session, threshold: int | None = None) -> dict:
    """Score every document in the corpus."""
    threshold = threshold or settings.STALENESS_THRESHOLD_DAYS

    docs = repo.list_documents(session)
    all_records = repo.list_embedding_records(session)

    by_doc: dict[int, list] = {}
    for r in all_records:
        by_doc.setdefault(r.document_id, []).append(r)

    scored = [score_document(d, by_doc.get(d.id, []), threshold) for d in docs]
    scored.sort(key=lambda x: x["freshness_score"])

    critical = [s for s in scored if s["health_status"] == "Critical"]
    stale    = [s for s in scored if s["health_status"] == "Stale"]
    fresh    = [s for s in scored if s["health_status"] == "Fresh"]
    drifted  = [s for s in scored if s["content_drift"]]

    avg_freshness = round(sum(s["freshness_score"] for s in scored) / len(scored), 4) if scored else 0.0

    overall = ("Critical" if critical or drifted else
               "Degraded" if stale else
               "Healthy")

    return {
        "module":              "Freshness Monitor",
        "total_documents":     len(scored),
        "critical_count":      len(critical),
        "stale_count":         len(stale),
        "fresh_count":         len(fresh),
        "content_drift_count": len(drifted),
        "average_freshness":   avg_freshness,
        "staleness_threshold": threshold,
        "overall_health":      overall,
        "documents":           scored,
        "reindex_queue":       [s["doc_key"] for s in scored if s["needs_reindex"]],
        "formula":             "freshness = max(0, 1 - (max_chunk_age_days / threshold))",
        "drift_definition":    "content_drift = source content_hash != hash captured at embed time",
        "snowflake_target":    f"{settings.SNOWFLAKE_DATABASE}.{settings.SNOWFLAKE_SCHEMA}.VECTOR_DB_HEALTH",
        "recommendation": (
            f"{len(drifted)} documents have content drift and must be re-indexed now. "
            f"{len(critical)} exceed the staleness threshold. "
            f"{len(stale)} should be refreshed within 7 days. "
            f"{len(fresh)} are current."
        ),
    }
