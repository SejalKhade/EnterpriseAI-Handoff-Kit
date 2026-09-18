"""Tests for chunking, embedding, the vector store, and retrieval."""

from __future__ import annotations
import numpy as np
import pytest

from src.config import settings
from src.database import repository as repo
from src.rag.embedder import TfidfSvdEmbedder, content_hash, build_embedder
from src.rag.pipeline import chunk_text, age_in_days, RAGPipeline


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert len(chunk_text("a short sentence")) == 1

    def test_long_text_splits(self):
        text = " ".join(f"word{i}" for i in range(300))
        assert len(chunk_text(text, max_words=90, overlap=15)) > 1

    def test_chunks_overlap(self):
        text = " ".join(f"w{i}" for i in range(200))
        chunks = chunk_text(text, max_words=90, overlap=15)
        first_tail = set(chunks[0].split()[-15:])
        second_head = set(chunks[1].split()[:15])
        assert first_tail & second_head

    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_every_chunk_within_limit(self):
        text = " ".join(f"w{i}" for i in range(500))
        for c in chunk_text(text, max_words=50, overlap=10):
            assert len(c.split()) <= 50


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_differs_on_change(self):
        assert content_hash("hello") != content_hash("hello ")

    def test_sha256_length(self):
        assert len(content_hash("x")) == 64


class TestEmbedder:
    def test_requires_fit(self):
        e = TfidfSvdEmbedder(dim=8)
        with pytest.raises(RuntimeError):
            e.encode(["anything"])

    def test_fit_rejects_empty(self):
        with pytest.raises(ValueError):
            TfidfSvdEmbedder(dim=8).fit([])

    def test_output_shape(self, embedder):
        v = embedder.encode(["refund policy question", "webhook setup"])
        assert v.shape == (2, embedder.dim)

    def test_vectors_normalized(self, embedder):
        v = embedder.encode(["enterprise refund within thirty days"])[0]
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    def test_deterministic(self, embedder):
        a = embedder.encode(["single sign on azure"])
        b = embedder.encode(["single sign on azure"])
        assert np.allclose(a, b)

    def test_dim_capped_by_corpus(self):
        e = TfidfSvdEmbedder(dim=512)
        e.fit(["alpha beta", "gamma delta", "epsilon zeta"])
        assert e.dim <= 2

    def test_save_and_load(self, embedder, tmp_root):
        p = tmp_root / "emb.pkl"
        embedder.save(p)
        fresh = TfidfSvdEmbedder(dim=16)
        assert fresh.load(p) is True
        assert np.allclose(fresh.encode(["refund"]), embedder.encode(["refund"]))

    def test_load_missing_returns_false(self, tmp_root):
        assert TfidfSvdEmbedder(dim=8).load(tmp_root / "nope.pkl") is False

    def test_factory_falls_back(self):
        assert build_embedder("sentence_transformers").is_fitted in (True, False)


class TestCoverage:
    def test_in_vocabulary_query_has_mass(self, embedder):
        c = embedder.coverage("enterprise refund within thirty days")
        assert c["tfidf_mass"] > 0
        assert c["token_coverage"] > 0.5

    def test_out_of_vocabulary_query_has_no_mass(self, embedder):
        c = embedder.coverage("zzzqqq xyzzy plugh frobnicate")
        assert c["tfidf_mass"] == 0.0
        assert c["token_coverage"] == 0.0

    def test_single_incidental_term_has_low_coverage(self, embedder):
        c = embedder.coverage("who is the chief executive requests")
        assert c["token_coverage"] < 0.6


class TestVectorStore:
    def test_starts_empty(self, store):
        assert store.count() == 0

    def test_upsert_then_count(self, store):
        n = store.upsert_chunks(chunk_ids=["a", "b"],
                                texts=["refund policy", "webhook setup"],
                                metadatas=[{"k": 1}, {"k": 2}])
        assert n == 2 and store.count() == 2

    def test_query_empty_store(self, store):
        assert store.query("anything") == []

    def test_similarity_bounded(self, store):
        store.upsert_chunks(chunk_ids=["a"], texts=["refund policy thirty days"],
                            metadatas=[{"k": 1}])
        hit = store.query("refund policy")[0]
        assert -1.0 <= hit["similarity"] <= 1.0

    def test_upsert_is_idempotent(self, store):
        for _ in range(2):
            store.upsert_chunks(chunk_ids=["a"], texts=["x"], metadatas=[{"k": 1}])
        assert store.count() == 1

    def test_delete(self, store):
        store.upsert_chunks(chunk_ids=["a", "b"], texts=["x", "y"],
                            metadatas=[{"k": 1}, {"k": 2}])
        store.delete_chunks(["a"])
        assert store.count() == 1

    def test_metadata_round_trips(self, store):
        store.upsert_chunks(chunk_ids=["a"], texts=["refund"],
                            metadatas=[{"doc_key": "refund_policy"}])
        assert store.all_metadata()[0]["doc_key"] == "refund_policy"

    def test_stats(self, store, embedder):
        s = store.stats()
        assert s["embedder_backend"] == embedder.name
        assert s["distance_metric"] == "cosine"


class TestIndexing:
    def test_creates_document_and_chunks(self, seeded, session):
        doc = repo.get_document(session, "refund_policy")
        assert doc is not None
        assert len(repo.get_records_for_document(session, "refund_policy")) >= 1

    def test_all_docs_indexed(self, seeded, session):
        assert len(repo.list_documents(session)) == 4

    def test_store_populated(self, seeded):
        assert seeded.store.count() >= 4

    def test_backdating_ages_embedding(self, seeded, session):
        recs = repo.get_records_for_document(session, "rate_limits")
        assert age_in_days(recs[0].embedded_at) > 55

    def test_hash_recorded_at_embed_time(self, seeded, session):
        doc = repo.get_document(session, "refund_policy")
        rec = repo.get_records_for_document(session, "refund_policy")[0]
        assert rec.source_hash_at_embed == doc.content_hash

    def test_reindex_resets_age(self, seeded, session):
        seeded.reindex_document(session, "rate_limits")
        recs = repo.get_records_for_document(session, "rate_limits")
        assert age_in_days(recs[0].embedded_at) < 1

    def test_reindex_unknown_doc(self, seeded, session):
        assert seeded.reindex_document(session, "nope")["status"] == "not_found"


class TestRetrieval:
    def test_finds_relevant_document(self, seeded, session):
        r = seeded.retrieve(session, "enterprise refund within thirty days")
        assert r["hits"]
        assert r["flagged_relevant"] is True

    def test_rejects_out_of_vocabulary(self, seeded, session):
        r = seeded.retrieve(session, "zzzqqq xyzzy plugh frobnicate")
        assert r["flagged_relevant"] is False
        assert r["decision_reason"] == "out_of_vocabulary"

    def test_age_is_attached(self, seeded, session):
        r = seeded.retrieve(session, "single sign on azure")
        assert all("embedding_age_days" in h for h in r["all_hits"])

    def test_no_penalty_without_filter(self, seeded, session):
        r = seeded.retrieve(session, "single sign on azure",
                            apply_freshness_filter=False)
        assert all(h["freshness_weight"] == 1.0 for h in r["all_hits"])
        assert all(h["adjusted_similarity"] == h["raw_similarity"] for h in r["all_hits"])

    def test_stale_chunk_is_penalized(self, seeded, session):
        r = seeded.retrieve(session, "single sign on azure identity provider",
                            apply_freshness_filter=True, staleness_threshold=30)
        stale = [h for h in r["all_hits"] if h["stale"]]
        assert stale
        assert all(h["adjusted_similarity"] < h["raw_similarity"] for h in stale)

    def test_fresh_chunk_not_penalized(self, seeded, session):
        r = seeded.retrieve(session, "enterprise refund thirty days",
                            apply_freshness_filter=True, staleness_threshold=30)
        fresh = [h for h in r["all_hits"] if not h["stale"] and not h["content_drift"]]
        assert all(h["freshness_weight"] == 1.0 for h in fresh)

    def test_latency_recorded(self, seeded, session):
        assert seeded.retrieve(session, "refund")["latency_ms"] > 0

    def test_ranks_are_sequential(self, seeded, session):
        r = seeded.retrieve(session, "webhook signature header")
        assert [h["rank"] for h in r["hits"]] == list(range(1, len(r["hits"]) + 1))

    def test_query_and_log_persists(self, seeded, session):
        before = len(repo.list_query_logs(session))
        seeded.query_and_log(session, "refund policy", ground_truth_relevant=True)
        assert len(repo.list_query_logs(session)) == before + 1


class TestFreshnessWeight:
    def test_fresh_is_unpenalized(self):
        w, reason = RAGPipeline.freshness_weight(10, 30, False)
        assert w == 1.0 and reason == "current"

    def test_stale_is_penalized(self):
        w, _ = RAGPipeline.freshness_weight(60, 30, False)
        assert w < 1.0

    def test_penalty_floors(self):
        w, _ = RAGPipeline.freshness_weight(10_000, 30, False)
        assert w >= settings.FRESHNESS_WEIGHT_FLOOR

    def test_drift_penalized_harder_than_age(self):
        age_only, _ = RAGPipeline.freshness_weight(45, 30, False)
        with_drift, _ = RAGPipeline.freshness_weight(45, 30, True)
        assert with_drift < age_only

    def test_drift_alone_penalizes_fresh_chunk(self):
        w, reason = RAGPipeline.freshness_weight(5, 30, True)
        assert w == pytest.approx(settings.DRIFT_PENALTY)
        assert "drift" in reason

    def test_penalty_is_monotonic_in_age(self):
        ws = [RAGPipeline.freshness_weight(a, 30, False)[0] for a in (31, 45, 60, 90)]
        assert ws == sorted(ws, reverse=True)


class TestDriftDetection:
    def test_drift_flagged_after_source_change(self, drifted, session):
        r = drifted.retrieve(session, "enterprise requests per minute limit",
                             apply_freshness_filter=True)
        assert any(h["content_drift"] for h in r["all_hits"])

    def test_drift_lowers_adjusted_score(self, drifted, session):
        r = drifted.retrieve(session, "enterprise requests per minute limit",
                             apply_freshness_filter=True)
        for h in r["all_hits"]:
            if h["content_drift"]:
                assert h["adjusted_similarity"] < h["raw_similarity"]

    def test_reindex_clears_drift(self, drifted, session):
        drifted.reindex_document(session, "rate_limits")
        r = drifted.retrieve(session, "enterprise requests per minute limit",
                             apply_freshness_filter=True)
        assert not any(h["content_drift"] and h["chunk_id"].startswith("rate_limits")
                       for h in r["all_hits"])
