"""
EnterpriseAI Handoff Kit — Orchestrator
Wires every component into the two pipelines the dashboard and API call.

Pipeline 1 (direct):  RAG retrieval -> evaluation -> freshness -> integration
                      -> Snowflake -> S3 -> database snapshot
Pipeline 2 (claude):  Pipeline 1, then Claude interpretation and the
                      hallucination guard over its output.
"""

from __future__ import annotations
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy.orm import Session

from src.config import settings
from src.database import repository as repo
from src.database.session import get_session, init_db
from src.rag.embedder import build_embedder
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline
from src.monitoring.freshness import run_freshness_check
from src.monitoring.eval_metrics import evaluate_before_after
from src.monitoring.integration import run_integration_checks
from src.connectors.snowflake_writer import SnowflakeWriter
from src.connectors.s3_writer import S3Writer
from src.llm import claude_client
from src.llm.hallucination_guard import evaluate as guard_evaluate

EMBEDDER_CACHE_PATH = settings.DATA_DIR / "embedder.pkl"


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Build the RAG pipeline once per process.
    Loads a fitted TF-IDF/SVD embedder from disk when available, otherwise
    fits it on the corpus currently in the database.
    """
    embedder = build_embedder()

    if hasattr(embedder, "load") and embedder.load(EMBEDDER_CACHE_PATH):
        pass  # restored from disk
    elif not embedder.is_fitted:
        with get_session() as session:
            docs = repo.list_documents(session)
            if docs:
                from src.rag.pipeline import chunk_text
                corpus = []
                for d in docs:
                    corpus.extend(chunk_text(d.content))
                if corpus:
                    embedder.fit(corpus)
                    if hasattr(embedder, "save"):
                        embedder.save(EMBEDDER_CACHE_PATH)

    store = VectorStore(embedder)
    return RAGPipeline(embedder, store)


def reset_pipeline_cache() -> None:
    get_pipeline.cache_clear()


def load_test_set(session: Session) -> list[dict]:
    """
    Labelled evaluation set across three categories.

    ANSWERABLE      backed by fresh, undrifted content. The system should answer.

    OUT OF SCOPE    plausible business questions with no supporting document.
                    The system should decline.

    DRIFT SENSITIVE the only supporting document has been updated since it was
                    embedded, so the vector store holds superseded text. A
                    confident answer here is a wrong answer. Ground truth is
                    therefore False: the correct behaviour is to decline until
                    the document is re-indexed.

    The drift-sensitive cases are what separate a monitored deployment from an
    unmonitored one, and they are the reason the before/after comparison exists.
    """
    return [
        # Answerable from fresh content
        {"query": "What is the refund window for enterprise customers?",            "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I export all my data in bulk?",                           "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I set up webhook notifications?",                         "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "What is the data retention period?",                             "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I add custom branding to generated reports?",             "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I revoke an API token?",                                  "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I run a custom SQL query in analytics?",                  "ground_truth_relevant": True,  "category": "answerable"},
        {"query": "How do I configure single sign on with Azure Active Directory?", "ground_truth_relevant": True,  "category": "answerable"},

        # Drift sensitive: supporting document updated after embedding
        {"query": "What are the current enterprise API rate limits per minute?",    "ground_truth_relevant": False, "category": "drift_sensitive"},
        {"query": "What is the burst capacity on the enterprise tier rate limit?",  "ground_truth_relevant": False, "category": "drift_sensitive"},
        {"query": "What is the request per minute limit on the bulk export endpoint?", "ground_truth_relevant": False, "category": "drift_sensitive"},
        {"query": "Is Salesforce opportunity synchronization bidirectional?",       "ground_truth_relevant": False, "category": "drift_sensitive"},
        {"query": "What is the default Salesforce synchronization interval?",       "ground_truth_relevant": False, "category": "drift_sensitive"},

        # Out of scope
        {"query": "What is the mailing address of the corporate headquarters?",     "ground_truth_relevant": False, "category": "out_of_scope"},
        {"query": "Who is the current chief executive officer?",                    "ground_truth_relevant": False, "category": "out_of_scope"},
        {"query": "What were the quarterly earnings last fiscal year?",             "ground_truth_relevant": False, "category": "out_of_scope"},
        {"query": "How many employees work in the Berlin office?",                  "ground_truth_relevant": False, "category": "out_of_scope"},
        {"query": "Which venture capital firms led the last funding round?",        "ground_truth_relevant": False, "category": "out_of_scope"},
    ]


def run_direct_pipeline(staleness_threshold: int | None = None,
                        write_external: bool = True) -> dict:
    """Pipeline 1 — fully deterministic, no LLM anywhere in the path."""
    threshold = staleness_threshold or settings.STALENESS_THRESHOLD_DAYS
    init_db()
    pipeline = get_pipeline()

    with get_session() as session:
        if pipeline.store.count() == 0:
            return {
                "status": "empty",
                "error":  "Vector store is empty. Run scripts/seed_corpus.py first.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        repo.clear_query_logs(session)
        test_set = load_test_set(session)

        # Populate query_logs so integration checks have real measurements
        for case in test_set:
            pipeline.query_and_log(
                session, case["query"],
                ground_truth_relevant=case["ground_truth_relevant"],
                apply_freshness_filter=False, staleness_threshold=threshold,
            )

        evaluation  = evaluate_before_after(pipeline, session, test_set, threshold)
        freshness   = run_freshness_check(session, threshold)
        integration = run_integration_checks(session)
        vstats      = pipeline.store.stats()
        cstats      = repo.corpus_stats(session)

        report = {
            "status":              "ok",
            "pipeline":            "Direct Pipeline (no LLM)",
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "staleness_threshold": threshold,
            "evaluation":          evaluation,
            "freshness":           freshness,
            "integration":         integration,
            "vector_store":        vstats,
            "corpus_stats":        cstats,
        }

        sf_result = s3_result = None
        if write_external:
            sf = SnowflakeWriter()
            sf.ensure_tables()
            sf_result = {
                "vector_health": sf.write_vector_health(freshness),
                "eval_metrics":  sf.write_eval_metrics(evaluation),
                "integration":   sf.write_integration_health(integration),
                "target":        sf.target_info(),
            }
            sf.close()

            s3 = S3Writer()
            s3_result = s3.write_report(report, name="health_report")
            s3_result["target"] = s3.target_info()

        report["snowflake"] = sf_result
        report["s3"] = s3_result

        repo.save_health_snapshot(
            session, vector_db=freshness, integration=integration,
            payload={
                "evaluation_delta": evaluation["delta"],
                "before": evaluation["before"],
                "after":  evaluation["after"],
            },
            snowflake_written=bool(sf_result and
                                   sf_result["vector_health"]["status"] == "ok"),
            s3_written=bool(s3_result and s3_result["status"] == "ok"),
        )

        return report


def run_claude_pipeline(api_key: str, staleness_threshold: int | None = None,
                        write_external: bool = True) -> dict:
    """Pipeline 2 — direct pipeline, then Claude interpretation plus guard."""
    report = run_direct_pipeline(staleness_threshold, write_external)
    if report.get("status") != "ok":
        return report

    interpretation = claude_client.interpret(report, api_key)
    guard = (guard_evaluate(interpretation["text"], interpretation["payload"])
             if interpretation["success"]
             else {"performed": False, "reason": interpretation["error"]})

    report["pipeline"] = "Claude-Augmented Pipeline"
    report["claude"] = interpretation
    report["hallucination_guard"] = guard
    return report


def reindex_documents(doc_keys: list[str]) -> dict:
    """Re-embed the named documents at the current timestamp."""
    pipeline = get_pipeline()
    results = []
    with get_session() as session:
        for key in doc_keys:
            results.append(pipeline.reindex_document(session, key))
    reset_pipeline_cache()
    return {
        "reindexed":  len([r for r in results if r.get("status") == "reindexed"]),
        "requested":  len(doc_keys),
        "results":    results,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


def system_status() -> dict:
    """Lightweight status for the API health endpoint."""
    init_db()
    pipeline = get_pipeline()
    with get_session() as session:
        stats = repo.corpus_stats(session)
        latest = repo.latest_health(session)
    return {
        "status":       "operational",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "vector_store": pipeline.store.stats(),
        "corpus":       stats,
        "snowflake":    SnowflakeWriter().target_info(),
        "s3":           S3Writer().target_info(),
        "latest_health": {
            "vector_db_health":   latest.vector_db_health,
            "integration_status": latest.integration_status,
            "created_at":         latest.created_at.isoformat(),
        } if latest else None,
    }
