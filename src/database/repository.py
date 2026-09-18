"""
EnterpriseAI Handoff Kit — Repository Layer
All database reads and writes go through here.
Keeps SQL out of business logic.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Sequence

from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    Document, EmbeddingRecord, QueryLog, EvalRun, HealthSnapshot, utcnow
)


# ── Documents ────────────────────────────────────────────────────────────────

def upsert_document(session: Session, *, doc_key: str, title: str, content: str,
                    content_hash: str, source_path: str = "",
                    category: str = "general",
                    source_updated_at: datetime | None = None) -> Document:
    doc = session.scalar(select(Document).where(Document.doc_key == doc_key))
    if doc is None:
        doc = Document(doc_key=doc_key, title=title, content=content,
                       content_hash=content_hash, source_path=source_path,
                       category=category,
                       source_updated_at=source_updated_at or utcnow())
        session.add(doc)
    else:
        doc.title = title
        doc.content = content
        doc.content_hash = content_hash
        doc.category = category
        if source_updated_at:
            doc.source_updated_at = source_updated_at
    session.flush()
    return doc


def get_document(session: Session, doc_key: str) -> Document | None:
    return session.scalar(select(Document).where(Document.doc_key == doc_key))


def list_documents(session: Session) -> Sequence[Document]:
    return session.scalars(
        select(Document).options(selectinload(Document.embeddings))
    ).all()


def touch_document(session: Session, doc_key: str,
                   new_content: str, new_hash: str) -> Document | None:
    """Simulate a source document being updated (bumps source_updated_at)."""
    doc = get_document(session, doc_key)
    if doc is None:
        return None
    doc.content = new_content
    doc.content_hash = new_hash
    doc.source_updated_at = utcnow()
    session.flush()
    return doc


# ── Embedding records ────────────────────────────────────────────────────────

def upsert_embedding_record(session: Session, *, document_id: int, chunk_index: int,
                            chunk_id: str, chunk_text: str, backend: str,
                            dim: int, source_hash: str,
                            embedded_at: datetime | None = None) -> EmbeddingRecord:
    rec = session.scalar(select(EmbeddingRecord).where(EmbeddingRecord.chunk_id == chunk_id))
    if rec is None:
        rec = EmbeddingRecord(document_id=document_id, chunk_index=chunk_index,
                              chunk_id=chunk_id, chunk_text=chunk_text,
                              embedder_backend=backend, embedding_dim=dim,
                              source_hash_at_embed=source_hash,
                              embedded_at=embedded_at or utcnow())
        session.add(rec)
    else:
        rec.chunk_text = chunk_text
        rec.embedder_backend = backend
        rec.embedding_dim = dim
        rec.source_hash_at_embed = source_hash
        rec.embedded_at = embedded_at or utcnow()
    session.flush()
    return rec


def backdate_embedding(session: Session, chunk_id: str, days: int) -> EmbeddingRecord | None:
    """Age an embedding artificially — used for seeding realistic staleness."""
    rec = session.scalar(select(EmbeddingRecord).where(EmbeddingRecord.chunk_id == chunk_id))
    if rec is None:
        return None
    rec.embedded_at = utcnow() - timedelta(days=days)
    session.flush()
    return rec


def list_embedding_records(session: Session) -> Sequence[EmbeddingRecord]:
    return session.scalars(
        select(EmbeddingRecord).options(selectinload(EmbeddingRecord.document))
    ).all()


def get_records_for_document(session: Session, doc_key: str) -> Sequence[EmbeddingRecord]:
    return session.scalars(
        select(EmbeddingRecord)
        .join(Document)
        .where(Document.doc_key == doc_key)
    ).all()


# ── Query logs ───────────────────────────────────────────────────────────────

def log_query(session: Session, *, query_text: str, retrieved_chunk_ids: list[str],
              top_score: float, flagged_relevant: bool,
              ground_truth_relevant: bool | None, max_embedding_age_days: float,
              latency_ms: float, error: str | None = None) -> QueryLog:
    row = QueryLog(query_text=query_text, retrieved_chunk_ids=retrieved_chunk_ids,
                   top_score=top_score, flagged_relevant=flagged_relevant,
                   ground_truth_relevant=ground_truth_relevant,
                   max_embedding_age_days=max_embedding_age_days,
                   latency_ms=latency_ms, error=error)
    session.add(row)
    session.flush()
    return row


def list_query_logs(session: Session, limit: int = 500) -> Sequence[QueryLog]:
    return session.scalars(
        select(QueryLog).order_by(desc(QueryLog.created_at)).limit(limit)
    ).all()


def labelled_query_logs(session: Session) -> Sequence[QueryLog]:
    """Only logs that have a ground truth label — used for evaluation."""
    return session.scalars(
        select(QueryLog).where(QueryLog.ground_truth_relevant.isnot(None))
        .order_by(QueryLog.created_at)
    ).all()


def clear_query_logs(session: Session) -> int:
    n = session.query(QueryLog).delete()
    session.flush()
    return n


# ── Eval runs ────────────────────────────────────────────────────────────────

def save_eval_run(session: Session, *, run_label: str, metrics: dict,
                  total_queries: int, staleness_threshold: int) -> EvalRun:
    cm = metrics["confusion_matrix"]
    row = EvalRun(
        run_label=run_label,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        false_positive_rate=metrics["false_positive_rate"],
        accuracy=metrics["accuracy"],
        true_positives=cm["true_positives"],
        false_positives=cm["false_positives"],
        true_negatives=cm["true_negatives"],
        false_negatives=cm["false_negatives"],
        total_queries=total_queries,
        staleness_threshold=staleness_threshold,
    )
    session.add(row)
    session.flush()
    return row


def eval_history(session: Session, limit: int = 60) -> Sequence[EvalRun]:
    return session.scalars(
        select(EvalRun).order_by(desc(EvalRun.created_at)).limit(limit)
    ).all()


# ── Health snapshots ─────────────────────────────────────────────────────────

def save_health_snapshot(session: Session, *, vector_db: dict, integration: dict,
                         payload: dict, snowflake_written: bool = False,
                         s3_written: bool = False) -> HealthSnapshot:
    row = HealthSnapshot(
        vector_db_health=vector_db["overall_health"],
        critical_docs=vector_db["critical_count"],
        stale_docs=vector_db["stale_count"],
        fresh_docs=vector_db["fresh_count"],
        avg_freshness=vector_db["average_freshness"],
        integration_status=integration["overall_status"],
        checks_passed=integration["checks_passed"],
        avg_latency_ms=integration.get("avg_latency_ms", 0.0),
        error_rate_pct=integration.get("error_rate_pct", 0.0),
        payload=payload,
        snowflake_written=snowflake_written,
        s3_written=s3_written,
    )
    session.add(row)
    session.flush()
    return row


def health_history(session: Session, limit: int = 100) -> Sequence[HealthSnapshot]:
    return session.scalars(
        select(HealthSnapshot).order_by(desc(HealthSnapshot.created_at)).limit(limit)
    ).all()


def latest_health(session: Session) -> HealthSnapshot | None:
    return session.scalar(
        select(HealthSnapshot).order_by(desc(HealthSnapshot.created_at)).limit(1)
    )


# ── Stats ────────────────────────────────────────────────────────────────────

def corpus_stats(session: Session) -> dict:
    return {
        "documents":        session.scalar(select(func.count(Document.id))) or 0,
        "embedding_chunks": session.scalar(select(func.count(EmbeddingRecord.id))) or 0,
        "query_logs":       session.scalar(select(func.count(QueryLog.id))) or 0,
        "eval_runs":        session.scalar(select(func.count(EvalRun.id))) or 0,
        "health_snapshots": session.scalar(select(func.count(HealthSnapshot.id))) or 0,
    }
