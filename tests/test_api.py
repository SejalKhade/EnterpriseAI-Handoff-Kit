"""API contract tests. Exercises the FastAPI app through its real routes."""

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestRoot:
    def test_root_lists_endpoints(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "/api/evaluate" in r.json()["endpoints"]

    def test_openapi_schema_served(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestStatus:
    def test_returns_operational(self, client):
        body = client.get("/api/status").json()
        assert body["status"] == "operational"

    def test_reports_vector_store(self, client):
        body = client.get("/api/status").json()
        assert "chunk_count" in body["vector_store"]

    def test_reports_connector_targets(self, client):
        body = client.get("/api/status").json()
        assert "enabled" in body["snowflake"]
        assert "enabled" in body["s3"]


class TestHealth:
    def test_returns_both_sections(self, client):
        body = client.get("/api/health").json()
        assert "freshness" in body and "integration" in body


class TestQuery:
    def test_requires_query_field(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_rejects_empty_query(self, client):
        assert client.post("/api/query", json={"query": ""}).status_code == 422

    def test_returns_decision_fields(self, client):
        body = client.post("/api/query", json={"query": "refund policy"}).json()
        for field in ("flagged_relevant", "decision_reason", "top_score", "latency_ms"):
            assert field in body

    def test_strips_bulk_payload(self, client):
        body = client.post("/api/query", json={"query": "refund policy"}).json()
        assert "all_hits" not in body

    def test_truncates_chunk_text(self, client):
        body = client.post("/api/query", json={"query": "refund policy"}).json()
        assert all(len(h["text"]) <= 240 for h in body["hits"])


class TestEvaluate:
    def test_rejects_threshold_below_range(self, client):
        r = client.post("/api/evaluate", json={"staleness_threshold": 0})
        assert r.status_code == 422

    def test_rejects_threshold_above_range(self, client):
        r = client.post("/api/evaluate", json={"staleness_threshold": 500})
        assert r.status_code == 422

    def test_returns_metrics_or_conflict(self, client):
        r = client.post("/api/evaluate", json={"write_external": False})
        assert r.status_code in (200, 409)
        if r.status_code == 200:
            body = r.json()
            assert "before" in body["evaluation"]
            assert "after" in body["evaluation"]


class TestInterpret:
    def test_requires_api_key(self, client):
        assert client.post("/api/interpret", json={}).status_code == 422

    def test_rejects_short_key(self, client):
        r = client.post("/api/interpret", json={"api_key": "short"})
        assert r.status_code == 422


class TestReindex:
    def test_rejects_empty_list(self, client):
        assert client.post("/api/reindex", json={"doc_keys": []}).status_code == 422

    def test_unknown_key_reports_not_found(self, client):
        body = client.post("/api/reindex", json={"doc_keys": ["nope"]}).json()
        assert body["results"][0]["status"] == "not_found"


class TestHistory:
    def test_returns_both_collections(self, client):
        body = client.get("/api/history").json()
        assert "eval_runs" in body and "health_snapshots" in body

    def test_respects_limit(self, client):
        body = client.get("/api/history?limit=2").json()
        assert len(body["eval_runs"]) <= 2
