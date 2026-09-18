"""
EnterpriseAI Handoff Kit — Vector Store
Real ChromaDB persistent client with explicit embedding control.

We pass precomputed embeddings rather than letting Chroma call an embedding
function, so the backend (TF-IDF/SVD or transformers) is fully under our control
and swappable without touching the store.
"""

from __future__ import annotations
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.rag.embedder import BaseEmbedder


class VectorStore:
    """Persistent ChromaDB collection with precomputed embeddings."""

    def __init__(self, embedder: BaseEmbedder,
                 collection_name: str | None = None,
                 persist_dir: str | None = None):
        self.embedder = embedder
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.persist_dir = persist_dir or str(settings.CHROMA_DIR)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, *, chunk_ids: list[str], texts: list[str],
                      metadatas: list[dict[str, Any]]) -> int:
        """Embed and upsert chunks. Returns number written."""
        if not chunk_ids:
            return 0
        vectors = self.embedder.encode(texts)
        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=[v.tolist() for v in vectors],
            metadatas=metadatas,
        )
        return len(chunk_ids)

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top_k most similar chunks.

        Chroma returns cosine DISTANCE (0 = identical, 2 = opposite).
        We convert to similarity: similarity = 1 - distance.
        """
        top_k = top_k or settings.TOP_K
        count = self.count()
        if count == 0:
            return []

        vector = self.embedder.encode([query_text])[0].tolist()
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []
        ids       = (raw.get("ids")       or [[]])[0]
        docs      = (raw.get("documents") or [[]])[0]
        metas     = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        for i, chunk_id in enumerate(ids):
            distance   = float(distances[i]) if i < len(distances) else 1.0
            similarity = round(1.0 - distance, 6)
            results.append({
                "chunk_id":   chunk_id,
                "text":       docs[i] if i < len(docs) else "",
                "metadata":   metas[i] if i < len(metas) else {},
                "distance":   round(distance, 6),
                "similarity": similarity,
                "rank":       i + 1,
            })
        return results

    def count(self) -> int:
        return self._collection.count()

    def peek(self, limit: int = 5) -> dict:
        return self._collection.peek(limit=limit)

    def all_metadata(self) -> list[dict]:
        """Every chunk's metadata — used by the freshness monitor."""
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["metadatas"])
        ids   = raw.get("ids") or []
        metas = raw.get("metadatas") or []
        return [
            {"chunk_id": ids[i], **(metas[i] or {})}
            for i in range(len(ids))
        ]

    def stats(self) -> dict:
        return {
            "collection":       self.collection_name,
            "chunk_count":      self.count(),
            "persist_dir":      self.persist_dir,
            "embedder_backend": self.embedder.name,
            "embedding_dim":    self.embedder.dim,
            "distance_metric":  "cosine",
        }
