"""
EnterpriseAI Handoff Kit — Database Models
SQLAlchemy ORM models. SQLite for local dev, PostgreSQL in production.

Tables:
  documents          Source documents with content hash and update tracking
  embedding_records  Vector store state per document chunk
  eval_runs          Historical evaluation metric snapshots
  query_logs         Every RAG query with latency and outcome
  health_snapshots   Periodic system health reports
"""

from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, ForeignKey, Index, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """A source document in the enterprise knowledge base."""
    __tablename__ = "documents"

    id             = Column(Integer, primary_key=True)
    doc_key        = Column(String(255), unique=True, nullable=False, index=True)
    title          = Column(String(512), nullable=False)
    content        = Column(Text, nullable=False)
    content_hash   = Column(String(64), nullable=False)
    source_path    = Column(String(512))
    category       = Column(String(128), default="general")
    source_updated_at = Column(DateTime, default=utcnow, nullable=False)
    created_at     = Column(DateTime, default=utcnow, nullable=False)

    embeddings = relationship("EmbeddingRecord", back_populates="document",
                              cascade="all, delete-orphan")

    __table_args__ = (Index("ix_doc_category_updated", "category", "source_updated_at"),)

    def __repr__(self) -> str:
        return f"<Document {self.doc_key}>"


class EmbeddingRecord(Base):
    """Tracks when each document chunk was embedded into the vector store."""
    __tablename__ = "embedding_records"

    id             = Column(Integer, primary_key=True)
    document_id    = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index    = Column(Integer, nullable=False, default=0)
    chunk_id       = Column(String(255), unique=True, nullable=False, index=True)
    chunk_text     = Column(Text, nullable=False)
    embedded_at    = Column(DateTime, default=utcnow, nullable=False)
    embedder_backend = Column(String(64), nullable=False)
    embedding_dim  = Column(Integer, nullable=False)
    source_hash_at_embed = Column(String(64), nullable=False)

    document = relationship("Document", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<EmbeddingRecord {self.chunk_id}>"


class QueryLog(Base):
    """Every RAG query executed, with retrieval outcome and latency."""
    __tablename__ = "query_logs"

    id                  = Column(Integer, primary_key=True)
    query_text          = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(JSON, nullable=False, default=list)
    top_score           = Column(Float, nullable=False, default=0.0)
    flagged_relevant    = Column(Boolean, nullable=False, default=False)
    ground_truth_relevant = Column(Boolean, nullable=True)
    max_embedding_age_days = Column(Float, nullable=False, default=0.0)
    latency_ms          = Column(Float, nullable=False, default=0.0)
    error               = Column(String(512), nullable=True)
    created_at          = Column(DateTime, default=utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<QueryLog {self.id} score={self.top_score:.3f}>"


class EvalRun(Base):
    """A snapshot of evaluation metrics at a point in time."""
    __tablename__ = "eval_runs"

    id                   = Column(Integer, primary_key=True)
    run_label            = Column(String(64), nullable=False)   # 'before' | 'after'
    precision            = Column(Float, nullable=False)
    recall               = Column(Float, nullable=False)
    f1_score             = Column(Float, nullable=False)
    false_positive_rate  = Column(Float, nullable=False)
    accuracy             = Column(Float, nullable=False)
    true_positives       = Column(Integer, nullable=False)
    false_positives      = Column(Integer, nullable=False)
    true_negatives       = Column(Integer, nullable=False)
    false_negatives      = Column(Integer, nullable=False)
    total_queries        = Column(Integer, nullable=False)
    staleness_threshold  = Column(Integer, nullable=False)
    created_at           = Column(DateTime, default=utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<EvalRun {self.run_label} f1={self.f1_score:.3f}>"


class HealthSnapshot(Base):
    """Periodic full-system health report."""
    __tablename__ = "health_snapshots"

    id                = Column(Integer, primary_key=True)
    vector_db_health  = Column(String(32), nullable=False)
    critical_docs     = Column(Integer, nullable=False, default=0)
    stale_docs        = Column(Integer, nullable=False, default=0)
    fresh_docs        = Column(Integer, nullable=False, default=0)
    avg_freshness     = Column(Float, nullable=False, default=0.0)
    integration_status = Column(String(32), nullable=False)
    checks_passed     = Column(Integer, nullable=False, default=0)
    avg_latency_ms    = Column(Float, nullable=False, default=0.0)
    error_rate_pct    = Column(Float, nullable=False, default=0.0)
    payload           = Column(JSON, nullable=False, default=dict)
    snowflake_written = Column(Boolean, default=False)
    s3_written        = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<HealthSnapshot {self.created_at} {self.vector_db_health}>"
