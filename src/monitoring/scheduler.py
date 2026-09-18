"""
EnterpriseAI Handoff Kit — Scheduler
Runs the health pipeline on an interval so the client sees drift appear
without anyone triggering a run by hand. This is the piece that makes the
kit an operating system rather than a report you remember to open.

Run standalone:
    python -m src.monitoring.scheduler

Or enable inside the API process:
    SCHEDULER_ENABLED=true uvicorn src.api.main:app
"""

from __future__ import annotations
from datetime import datetime, timezone
import logging
import signal
import sys

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("handoff.scheduler")


def health_check_job() -> dict:
    """
    One scheduled health run.

    Deliberately imports inside the function so a failure in the pipeline
    cannot prevent the scheduler process from starting.
    """
    started = datetime.now(timezone.utc)
    logger.info("health check starting")

    try:
        from src import orchestrator
        report = orchestrator.run_direct_pipeline(write_external=True)

        if report.get("status") != "ok":
            logger.warning("health check skipped: %s", report.get("error"))
            return {"status": "skipped", "reason": report.get("error")}

        fr = report["freshness"]
        ig = report["integration"]
        ev = report["evaluation"]

        logger.info(
            "health check complete | vector_db=%s (crit=%d stale=%d drift=%d) "
            "| integration=%s (%d/4) | f1 %.4f -> %.4f | %.1fs",
            fr["overall_health"], fr["critical_count"], fr["stale_count"],
            fr["content_drift_count"], ig["overall_status"], ig["checks_passed"],
            ev["before"]["f1_score"], ev["after"]["f1_score"],
            (datetime.now(timezone.utc) - started).total_seconds(),
        )

        if fr["reindex_queue"]:
            logger.warning("documents needing re-index: %s",
                           ", ".join(fr["reindex_queue"]))

        return {
            "status":         "ok",
            "vector_health":  fr["overall_health"],
            "integration":    ig["overall_status"],
            "reindex_queue":  fr["reindex_queue"],
            "duration_s":     (datetime.now(timezone.utc) - started).total_seconds(),
        }

    except Exception:
        logger.exception("health check failed")
        return {"status": "error"}


def build_scheduler(blocking: bool = False):
    """Create a scheduler with the health job registered."""
    scheduler = BlockingScheduler() if blocking else BackgroundScheduler()
    scheduler.add_job(
        health_check_job,
        trigger=IntervalTrigger(minutes=settings.HEALTH_CHECK_INTERVAL_MINUTES),
        id="health_check",
        name="Enterprise AI health check",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


def start_background() -> BackgroundScheduler | None:
    """Start the scheduler inside an existing process. Used by the API."""
    if not settings.SCHEDULER_ENABLED:
        logger.info("scheduler disabled (set SCHEDULER_ENABLED=true to enable)")
        return None
    scheduler = build_scheduler(blocking=False)
    scheduler.start()
    logger.info("background scheduler started, interval=%d min",
                settings.HEALTH_CHECK_INTERVAL_MINUTES)
    return scheduler


def main() -> None:
    """Run the scheduler as its own process."""
    logger.info("starting scheduler process, interval=%d min",
                settings.HEALTH_CHECK_INTERVAL_MINUTES)

    scheduler = build_scheduler(blocking=True)

    def shutdown(signum, frame):
        logger.info("shutdown signal received")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("running initial health check")
    health_check_job()

    scheduler.start()


if __name__ == "__main__":
    main()
