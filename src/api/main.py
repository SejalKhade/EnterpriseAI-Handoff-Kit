"""
EnterpriseAI Handoff Kit — FastAPI Service
The Python service the Node.js layer and dashboard call.

Endpoints:
  GET  /api/status          service and corpus status
  GET  /api/health          latest freshness and integration health
  POST /api/evaluate        run the direct pipeline, return full metrics
  POST /api/interpret       run the Claude pipeline (key in request body)
  POST /api/reindex         re-embed named documents
  GET  /api/history         historical eval runs and health snapshots
  POST /api/query           single retrieval, for debugging
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import orchestrator
from src.config import settings
from src.database.session import get_session, init_db
from src.database import repository as repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and optionally start the background scheduler."""
    init_db()
    scheduler = None
    if settings.SCHEDULER_ENABLED:
        from src.monitoring.scheduler import start_background
        scheduler = start_background()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="EnterpriseAI Handoff Kit",
    description="Post-deployment observability for enterprise RAG systems",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ───────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    staleness_threshold: int | None = Field(None, ge=1, le=365)
    write_external: bool = True


class InterpretRequest(BaseModel):
    api_key: str = Field(..., min_length=10)
    staleness_threshold: int | None = Field(None, ge=1, le=365)
    write_external: bool = True


class ReindexRequest(BaseModel):
    doc_keys: list[str] = Field(..., min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    apply_freshness_filter: bool = True
    staleness_threshold: int | None = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status() -> dict[str, Any]:
    return orchestrator.system_status()


@app.get("/api/health")
def health() -> dict[str, Any]:
    from src.monitoring.freshness import run_freshness_check
    from src.monitoring.integration import run_integration_checks
    with get_session() as session:
        return {
            "freshness":   run_freshness_check(session),
            "integration": run_integration_checks(session),
        }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    result = orchestrator.run_direct_pipeline(
        staleness_threshold=req.staleness_threshold,
        write_external=req.write_external,
    )
    if result.get("status") != "ok":
        raise HTTPException(status_code=409, detail=result.get("error", "Pipeline failed"))
    return result


@app.post("/api/interpret")
def interpret(req: InterpretRequest) -> dict[str, Any]:
    result = orchestrator.run_claude_pipeline(
        api_key=req.api_key,
        staleness_threshold=req.staleness_threshold,
        write_external=req.write_external,
    )
    if result.get("status") != "ok":
        raise HTTPException(status_code=409, detail=result.get("error", "Pipeline failed"))
    return result


@app.post("/api/reindex")
def reindex(req: ReindexRequest) -> dict[str, Any]:
    return orchestrator.reindex_documents(req.doc_keys)


@app.post("/api/query")
def query(req: QueryRequest) -> dict[str, Any]:
    pipeline = orchestrator.get_pipeline()
    with get_session() as session:
        result = pipeline.retrieve(
            session, req.query,
            apply_freshness_filter=req.apply_freshness_filter,
            staleness_threshold=req.staleness_threshold,
        )
    # Trim chunk text for a readable response
    for h in result["hits"]:
        h["text"] = h["text"][:240]
    result.pop("all_hits", None)
    return result


@app.get("/api/history")
def history(limit: int = 30) -> dict[str, Any]:
    with get_session() as session:
        evals = repo.eval_history(session, limit=limit)
        snaps = repo.health_history(session, limit=limit)
    return {
        "eval_runs": [{
            "run_label":           e.run_label,
            "precision":           e.precision,
            "recall":              e.recall,
            "f1_score":            e.f1_score,
            "false_positive_rate": e.false_positive_rate,
            "accuracy":            e.accuracy,
            "total_queries":       e.total_queries,
            "created_at":          e.created_at.isoformat(),
        } for e in evals],
        "health_snapshots": [{
            "vector_db_health":   h.vector_db_health,
            "critical_docs":      h.critical_docs,
            "stale_docs":         h.stale_docs,
            "fresh_docs":         h.fresh_docs,
            "integration_status": h.integration_status,
            "checks_passed":      h.checks_passed,
            "avg_latency_ms":     h.avg_latency_ms,
            "created_at":         h.created_at.isoformat(),
        } for h in snaps],
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "EnterpriseAI Handoff Kit",
        "version": "2.0.0",
        "docs":    "/docs",
        "endpoints": [
            "/api/status", "/api/health", "/api/evaluate",
            "/api/interpret", "/api/reindex", "/api/query", "/api/history",
        ],
    }
