"""
EnterpriseAI Handoff Kit — Claude Client
Claude receives only pre-computed metrics as structured JSON.
It never recalculates a number, so score fabrication is structurally impossible.
The API key is passed per call and never stored.
"""

from __future__ import annotations
import json

from src.config import settings


SYSTEM_PROMPT = """You are a senior AI deployment engineer writing a handover report.

Your reader is a non-technical enterprise operations manager whose vendor
engineering team has just finished a deployment and left. They must now operate
the system themselves.

You will receive a JSON health report with pre-computed metrics. A deterministic
Python pipeline computed every number. You did not compute any of them.

Your job:
1. Explain what changed between the before and after measurements in plain language
2. Name the specific documents that need re-indexing first and say why
3. Explain what the integration checks reveal about day-to-day reliability
4. Give numbered operational actions the reader can take without engineering help
5. Flag anything that needs attention this week

Rules:
- Use only numbers that appear in the JSON. Never estimate, extrapolate, or invent.
- No jargon. Write "the system returned wrong answers" not "elevated FPR".
- Be specific. Name documents, cite the measured values you were given.
- If a number is not in the JSON, do not mention it.

Format your response with these exact headings:

WHAT CHANGED
DOCUMENTS TO RE-INDEX FIRST
INTEGRATION RELIABILITY
WHAT TO DO THIS WEEK
WATCH LIST"""


def build_payload(report: dict) -> dict:
    """Extract the minimal fact set Claude needs. Nothing else is sent."""
    ev    = report["evaluation"]
    fr    = report["freshness"]
    integ = report["integration"]

    return {
        "evaluation": {
            "before": {
                "precision":           ev["before"]["precision"],
                "recall":              ev["before"]["recall"],
                "f1_score":            ev["before"]["f1_score"],
                "false_positive_rate": ev["before"]["false_positive_rate"],
                "accuracy":            ev["before"]["accuracy"],
            },
            "after": {
                "precision":           ev["after"]["precision"],
                "recall":              ev["after"]["recall"],
                "f1_score":            ev["after"]["f1_score"],
                "false_positive_rate": ev["after"]["false_positive_rate"],
                "accuracy":            ev["after"]["accuracy"],
            },
            "stale_chunks_filtered": ev["delta"]["stale_chunks_filtered"],
            "queries_affected":      ev["delta"]["queries_affected"],
            "total_queries":         ev["total_queries"],
            "staleness_threshold_days": ev["staleness_threshold"],
        },
        "vector_db": {
            "overall_health":      fr["overall_health"],
            "total_documents":     fr["total_documents"],
            "critical_count":      fr["critical_count"],
            "stale_count":         fr["stale_count"],
            "fresh_count":         fr["fresh_count"],
            "content_drift_count": fr["content_drift_count"],
            "average_freshness":   fr["average_freshness"],
            "documents_needing_reindex": [
                {
                    "title":         d["title"],
                    "max_age_days":  d["max_age_days"],
                    "status":        d["health_status"],
                    "content_drift": d["content_drift"],
                }
                for d in fr["documents"] if d["needs_reindex"]
            ][:8],
        },
        "integration": {
            "overall_status": integ["overall_status"],
            "checks_passed":  integ["checks_passed"],
            "total_checks":   integ["total_checks"],
            "avg_latency_ms": integ.get("avg_latency_ms", 0),
            "error_rate_pct": integ.get("error_rate_pct", 0),
            "sample_size":    integ.get("sample_size", 0),
            "checks": [
                {"dimension": c["dimension"], "value": c["value"], "status": c["status"]}
                for c in integ["checks"]
            ],
        },
    }


def interpret(report: dict, api_key: str) -> dict:
    """Call Claude with the structured payload. Returns text plus the exact prompt."""
    if not api_key:
        return {"success": False, "error": "No API key provided", "text": "", "payload": {}}

    payload = build_payload(report)
    prompt = (
        "Here is the current system health report. Write the handover explanation.\n\n"
        f"HEALTH REPORT JSON:\n{json.dumps(payload, indent=2)}\n\n"
        "Use only the numbers above."
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=settings.CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "success": True,
            "error":   None,
            "text":    message.content[0].text,
            "model":   settings.CLAUDE_MODEL,
            "payload": payload,
            "prompt":  prompt,
            "usage": {
                "input_tokens":  message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "error":   str(exc),
            "text":    "",
            "model":   settings.CLAUDE_MODEL,
            "payload": payload,
            "prompt":  prompt,
        }
