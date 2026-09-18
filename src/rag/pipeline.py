"""
EnterpriseAI Handoff Kit — RAG Pipeline
Real end-to-end retrieval pipeline:
  chunk -> embed -> store -> retrieve -> score -> log

Freshness enforcement is applied at retrieval time: chunks whose embeddings
are older than the staleness threshold can be filtered out, which is the
core intervention this project measures.
"""

from __future__ import annotations
from datetime import datetime, timezone
import time

from sqlalchemy.orm import Session

from src.config import settings
from src.database import repository as repo
from src.rag.embedder import BaseEmbedder, content_hash
from src.rag.vector_store import VectorStore


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_words: int = 90, overlap: int = 15) -> list[str]:
    """
    Sliding-window word chunking with overlap.
    Overlap prevents answers from being split across a chunk boundary.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    step = max(1, max_words - overlap)
    while start < len(words):
        chunk = " ".join(words[start:start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
        start += step
    return chunks


def age_in_days(ts: datetime) -> float:
    """Age of a timestamp in days, timezone-safe."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    return round(delta.total_seconds() / 86400.0, 4)


class RAGPipeline:
    """Indexing and retrieval over the enterprise knowledge base."""

    def __init__(self, embedder: BaseEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_document(self, session: Session, *, doc_key: str, title: str,
                       content: str, category: str = "general",
                       source_path: str = "",
                       embedded_days_ago: int = 0) -> dict:
        """
        Chunk, embed, store, and record a document.

        embedded_days_ago backdates the embedding timestamp so a realistic
        mix of fresh and stale content can be seeded for demonstration.
        """
        chash = content_hash(content)
        doc = repo.upsert_document(
            session, doc_key=doc_key, title=title, content=content,
            content_hash=chash, source_path=source_path, category=category,
        )

        chunks = chunk_text(content)
        chunk_ids, texts, metadatas = [], [], []

        for idx, chunk in enumerate(chunks):
            cid = f"{doc_key}::chunk::{idx}"
            chunk_ids.append(cid)
            texts.append(chunk)
            metadatas.append({
                "doc_key":      doc_key,
                "title":        title,
                "category":     category,
                "chunk_index":  idx,
                "content_hash": chash,
            })

        written = self.store.upsert_chunks(
            chunk_ids=chunk_ids, texts=texts, metadatas=metadatas
        )

        for idx, cid in enumerate(chunk_ids):
            repo.upsert_embedding_record(
                session, document_id=doc.id, chunk_index=idx, chunk_id=cid,
                chunk_text=texts[idx], backend=self.embedder.name,
                dim=self.embedder.dim, source_hash=chash,
            )
            if embedded_days_ago > 0:
                repo.backdate_embedding(session, cid, embedded_days_ago)

        return {
            "doc_key":     doc_key,
            "chunks":      len(chunks),
            "written":     written,
            "content_hash": chash,
            "embedded_days_ago": embedded_days_ago,
        }

    def reindex_document(self, session: Session, doc_key: str) -> dict:
        """Re-embed a document at the current timestamp — the remediation action."""
        doc = repo.get_document(session, doc_key)
        if doc is None:
            return {"doc_key": doc_key, "status": "not_found"}

        old_ids = [r.chunk_id for r in repo.get_records_for_document(session, doc_key)]
        self.store.delete_chunks(old_ids)

        result = self.index_document(
            session, doc_key=doc.doc_key, title=doc.title, content=doc.content,
            category=doc.category, source_path=doc.source_path or "",
            embedded_days_ago=0,
        )
        result["status"] = "reindexed"
        result["previous_chunks"] = len(old_ids)
        return result

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _chunk_ages(self, session: Session) -> dict[str, float]:
        return {
            r.chunk_id: age_in_days(r.embedded_at)
            for r in repo.list_embedding_records(session)
        }

    def _chunk_state(self, session: Session) -> dict[str, dict]:
        """
        Per-chunk age and content-drift status.

        Drift is not a heuristic: it is a hash comparison between the source
        content now and the content that was actually embedded. When they differ,
        the vector store is provably serving text that no longer exists in the
        source document.
        """
        state: dict[str, dict] = {}
        for r in repo.list_embedding_records(session):
            doc = r.document
            drifted = bool(doc and r.source_hash_at_embed != doc.content_hash)
            state[r.chunk_id] = {
                "age_days":      age_in_days(r.embedded_at),
                "content_drift": drifted,
                "doc_key":       doc.doc_key if doc else None,
            }
        return state

    @staticmethod
    def freshness_weight(age_days: float, threshold: int,
                         content_drift: bool = False) -> tuple[float, str]:
        """
        Confidence multiplier applied to a chunk's similarity score.

        Two independent signals, applied multiplicatively:

          AGE      heuristic. Confidence decays past the staleness threshold and
                   floors at FRESHNESS_WEIGHT_FLOOR, so a strongly matching stale
                   chunk is demoted rather than erased.

                       age <= threshold -> 1.0
                       age  > threshold -> max(floor, 1 - decay * (age/threshold - 1))

          DRIFT    verified fact. The chunk's embedded text hash differs from the
                   current source hash, so the store is provably serving content
                   that no longer exists in the source. This earns a much harder
                   penalty than age alone, because it is not an estimate.

        Returns (weight, reason).
        """
        reasons = []
        weight = 1.0

        if age_days > threshold:
            excess = (age_days / threshold) - 1.0
            age_w = max(settings.FRESHNESS_WEIGHT_FLOOR,
                        1.0 - settings.FRESHNESS_DECAY * excess)
            weight *= age_w
            reasons.append(f"age {age_days:.0f}d>{threshold}d")

        if content_drift:
            weight *= settings.DRIFT_PENALTY
            reasons.append("content drift")

        return round(weight, 6), ", ".join(reasons) if reasons else "current"

    def retrieve(self, session: Session, query: str, *,
                 top_k: int | None = None,
                 apply_freshness_filter: bool = False,
                 staleness_threshold: int | None = None,
                 mode: str | None = None) -> dict:
        """
        Retrieve chunks for a query.

        apply_freshness_filter=True enables staleness-aware scoring, which is the
        intervention whose effect on precision / recall / FPR is measured.

        mode:
          "penalty"     multiply similarity by a freshness weight (default)
          "hard_filter" drop stale chunks entirely (naive baseline for comparison)
        """
        top_k = top_k or settings.TOP_K
        threshold = staleness_threshold or settings.STALENESS_THRESHOLD_DAYS
        mode = mode or settings.FRESHNESS_MODE

        t0 = time.perf_counter()

        # Out-of-vocabulary gate. L2 normalization makes a query matching one
        # incidental term look as confident as a query matching ten specific
        # ones, so cosine alone cannot reject out-of-scope questions. Coverage
        # measures how much IDF-weighted mass the query actually carries.
        cov = self.embedder.coverage(query)
        in_scope = (cov["tfidf_mass"] >= settings.MIN_QUERY_TFIDF_MASS
                    and cov["token_coverage"] >= settings.MIN_QUERY_COVERAGE)

        hits = self.store.query(query, top_k=top_k)
        state = self._chunk_state(session)

        for h in hits:
            st = state.get(h["chunk_id"], {"age_days": 0.0, "content_drift": False})
            age = st["age_days"]
            drift = st["content_drift"]

            h["embedding_age_days"] = age
            h["content_drift"] = drift
            h["stale"] = age > threshold
            h["raw_similarity"] = h["similarity"]

            if apply_freshness_filter:
                w, reason = self.freshness_weight(age, threshold, drift)
                h["freshness_weight"] = w
                h["penalty_reason"] = reason
                h["adjusted_similarity"] = round(h["raw_similarity"] * w, 6)
            else:
                h["freshness_weight"] = 1.0
                h["penalty_reason"] = "not applied"
                h["adjusted_similarity"] = h["raw_similarity"]

        if apply_freshness_filter and mode == "hard_filter":
            kept = [h for h in hits if not (h["stale"] or h["content_drift"])]
            demoted = [h for h in hits if h["stale"] or h["content_drift"]]
        else:
            kept = sorted(hits, key=lambda x: x["adjusted_similarity"], reverse=True)
            demoted = [h for h in hits
                       if apply_freshness_filter and h["freshness_weight"] < 1.0]

        for i, h in enumerate(kept):
            h["rank"] = i + 1

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        top_score = kept[0]["adjusted_similarity"] if kept else 0.0
        max_age = max((h["embedding_age_days"] for h in hits), default=0.0)

        # A query is answerable only if it is in scope AND a chunk clears the bar.
        flagged = (in_scope and bool(kept)
                   and top_score >= settings.RELEVANCE_THRESHOLD)

        reason = ("out_of_vocabulary" if not in_scope else
                  "no_hits" if not kept else
                  "below_threshold" if top_score < settings.RELEVANCE_THRESHOLD else
                  "relevant")

        return {
            "query":            query,
            "hits":             kept,
            "filtered_out":     demoted,
            "all_hits":         hits,
            "top_score":        round(top_score, 6),
            "top_raw_score":    round(kept[0]["raw_similarity"], 6) if kept else 0.0,
            "flagged_relevant": flagged,
            "decision_reason":  reason,
            "in_scope":         in_scope,
            "coverage":         cov,
            "relevance_threshold": settings.RELEVANCE_THRESHOLD,
            "max_embedding_age_days": max_age,
            "latency_ms":       latency_ms,
            "freshness_filter_applied": apply_freshness_filter,
            "freshness_mode":   mode if apply_freshness_filter else "none",
            "staleness_threshold": threshold,
        }

    def query_and_log(self, session: Session, query: str, *,
                      ground_truth_relevant: bool | None = None,
                      apply_freshness_filter: bool = False,
                      staleness_threshold: int | None = None) -> dict:
        """Retrieve and persist the result to query_logs."""
        result = self.retrieve(
            session, query, apply_freshness_filter=apply_freshness_filter,
            staleness_threshold=staleness_threshold,
        )
        repo.log_query(
            session,
            query_text=query,
            retrieved_chunk_ids=[h["chunk_id"] for h in result["hits"]],
            top_score=result["top_score"],
            flagged_relevant=result["flagged_relevant"],
            ground_truth_relevant=ground_truth_relevant,
            max_embedding_age_days=result["max_embedding_age_days"],
            latency_ms=result["latency_ms"],
        )
        result["ground_truth_relevant"] = ground_truth_relevant
        return result

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def fit_embedder_on_corpus(self, session: Session) -> dict:
        """
        Fit the TF-IDF/SVD embedder on all document text.
        Transformer backends are pretrained and skip this.
        """
        docs = repo.list_documents(session)
        corpus: list[str] = []
        for d in docs:
            corpus.extend(chunk_text(d.content))
        if not corpus:
            return {"fitted": False, "reason": "empty corpus"}
        self.embedder.fit(corpus)
        return {
            "fitted":       True,
            "backend":      self.embedder.name,
            "dim":          self.embedder.dim,
            "corpus_chunks": len(corpus),
        }
