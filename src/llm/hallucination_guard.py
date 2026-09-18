"""
EnterpriseAI Handoff Kit — Hallucination Guard
Verifies every number Claude writes against the exact JSON payload it received.

Because Claude is given only pre-computed values, any number it produces that
cannot be traced back to that payload is fabricated. This module finds those.

Status meanings:
  VERIFIED       within 5% of a value in the payload
  APPROXIMATION  within 15% — probably rounding or a stated percentage
  UNVERIFIED     not traceable to any payload value
"""

from __future__ import annotations
import re

VERIFIED_TOLERANCE = 0.05
APPROX_TOLERANCE   = 0.15

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)\s*(%|ms|days?|points?|documents?|queries|chunks?)?"
)

# Small integers used as list markers or ordinals produce noise, not claims.
IGNORE_BELOW = 2.0


def extract_claims(text: str) -> list[dict]:
    """Pull every number out of Claude's text with surrounding context."""
    claims = []
    for m in NUMBER_PATTERN.finditer(text):
        value = float(m.group(1))
        if abs(value) < IGNORE_BELOW:
            continue
        start = max(0, m.start() - 80)
        end   = min(len(text), m.end() + 80)
        claims.append({
            "value":   value,
            "unit":    m.group(2) or "",
            "context": text[start:end].replace("\n", " ").strip(),
            "position": m.start(),
        })
    return claims


def flatten_payload(payload: dict) -> list[dict]:
    """Every numeric value in the payload, with its JSON path."""
    found: list[dict] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            found.append({"value": float(node), "path": path})

    walk(payload)
    return found


def verify(value: float, sources: list[dict]) -> dict:
    """
    Match a claimed number against payload values.
    Also checks the percentage form: 0.42 in the payload may be written as 42%.
    """
    best = None

    for src in sources:
        sv = src["value"]
        if sv == 0:
            if abs(value) < 1e-9:
                return {"status": "VERIFIED", "matched": 0.0,
                        "path": src["path"], "diff_pct": 0.0, "form": "direct"}
            continue

        for candidate, form in ((sv, "direct"), (sv * 100, "percentage")):
            if candidate == 0:
                continue
            diff = abs(value - candidate) / abs(candidate)
            if diff <= VERIFIED_TOLERANCE:
                return {"status": "VERIFIED", "matched": round(candidate, 6),
                        "path": src["path"], "diff_pct": round(diff * 100, 2), "form": form}
            if diff <= APPROX_TOLERANCE and (best is None or diff < best["diff"]):
                best = {"diff": diff, "matched": round(candidate, 6),
                        "path": src["path"], "form": form}

    if best:
        return {"status": "APPROXIMATION", "matched": best["matched"],
                "path": best["path"], "diff_pct": round(best["diff"] * 100, 2),
                "form": best["form"]}

    return {"status": "UNVERIFIED", "matched": None, "path": None,
            "diff_pct": None, "form": None}


def evaluate(claude_text: str, payload: dict) -> dict:
    """Full verification pass over Claude's output."""
    if not claude_text or not payload:
        return {"performed": False, "reason": "Missing Claude output or payload"}

    sources = flatten_payload(payload)
    claims  = extract_claims(claude_text)

    results = []
    verified = approx = unverified = 0

    for c in claims:
        v = verify(c["value"], sources)
        results.append({
            "claimed":      c["value"],
            "unit":         c["unit"],
            "context":      c["context"][:110],
            "status":       v["status"],
            "matched":      v["matched"],
            "source_path":  v["path"],
            "diff_pct":     v["diff_pct"],
            "match_form":   v["form"],
        })
        if v["status"] == "VERIFIED":
            verified += 1
        elif v["status"] == "APPROXIMATION":
            approx += 1
        else:
            unverified += 1

    total = len(results)
    reliability = round((verified + approx * 0.5) / total * 100, 1) if total else 100.0
    status = ("RELIABLE" if reliability >= 80 else
              "PARTIALLY RELIABLE" if reliability >= 60 else
              "REVIEW REQUIRED")

    return {
        "performed":         True,
        "total_claims":      total,
        "verified":          verified,
        "approximations":    approx,
        "unverified":        unverified,
        "reliability_score": reliability,
        "overall_status":    status,
        "claims":            results,
        "payload_values":    len(sources),
        "methodology": (
            f"Every number in Claude's output is extracted and compared against the "
            f"{len(sources)} numeric values in the JSON payload it received. "
            f"VERIFIED = within {int(VERIFIED_TOLERANCE*100)}%. "
            f"APPROXIMATION = within {int(APPROX_TOLERANCE*100)}%. "
            f"Both direct and percentage forms are checked. "
            f"Numbers below {IGNORE_BELOW} are skipped as list markers."
        ),
    }
