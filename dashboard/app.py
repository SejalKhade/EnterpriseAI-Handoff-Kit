"""
EnterpriseAI Handoff Kit — Dashboard
Every number shown carries its formula, its inputs, and its source.
No emojis. Nothing inferred that is not computed.
"""

from __future__ import annotations
import sys

# ChromaDB needs SQLite 3.35+. Streamlit Community Cloud and Hugging Face
# Spaces ship an older system sqlite3, so swap in pysqlite3-binary there.
# No-op wherever it isn't installed (e.g. local Windows dev).
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import orchestrator
from src.config import settings

st.set_page_config(
    page_title="EnterpriseAI Handoff Kit",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_seeded() -> None:
    """Seed the bundled demo corpus on a fresh deployment.

    A public deployment has no shell access to run scripts/seed_corpus.py,
    and the container's filesystem is wiped on every cold start, so this
    runs once per process instead of requiring a manual step.
    """
    from src.database.session import init_db
    init_db()
    pipeline = orchestrator.get_pipeline()
    if pipeline.store.count() == 0:
        with st.spinner("Seeding demo corpus (first run only)..."):
            from scripts.seed_corpus import seed
            seed(reset=False)
        orchestrator.reset_pipeline_cache()


_ensure_seeded()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#fafafa; color:#18181b; }
.main .block-container { padding:1.5rem 2.5rem; max-width:1400px; }

.hdr { background:#fff; border:1px solid #e4e4e7; border-radius:8px; padding:1.5rem 2rem; margin-bottom:1.5rem; }
.hdr-t { font-size:1.35rem; font-weight:700; letter-spacing:-.02em; }
.hdr-s { font-size:.76rem; color:#a1a1aa; font-family:'JetBrains Mono',monospace; margin-top:.3rem; }
.chip { display:inline-block; font-size:.64rem; font-family:'JetBrains Mono',monospace; background:#f4f4f5;
        color:#52525b; border:1px solid #e4e4e7; padding:.15rem .55rem; border-radius:4px; margin-right:.35rem; }
.chip-on { background:#ecfdf5; color:#047857; border-color:#a7f3d0; }
.chip-off{ background:#fef2f2; color:#b91c1c; border-color:#fecaca; }

.kpi { background:#fff; border:1px solid #e4e4e7; border-radius:8px; padding:1rem 1.25rem; height:100%; }
.kpi-l { font-size:.64rem; color:#a1a1aa; text-transform:uppercase; letter-spacing:.09em;
         font-weight:600; font-family:'JetBrains Mono',monospace; margin-bottom:.3rem; }
.kpi-v { font-size:1.75rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-s { font-size:.7rem; color:#a1a1aa; margin-top:.3rem; }

.sec-h { font-size:.64rem; font-family:'JetBrains Mono',monospace; color:#a1a1aa; letter-spacing:.1em;
         text-transform:uppercase; padding:.7rem 1rem; background:#fff; border:1px solid #e4e4e7;
         border-bottom:none; border-radius:8px 8px 0 0; margin-top:1.5rem; }
.sec-b { background:#fff; border:1px solid #e4e4e7; border-radius:0 0 8px 8px; padding:1.25rem; }

.fx { background:#fafafa; border-left:3px solid #d4d4d8; border-radius:0 4px 4px 0; padding:.7rem 1rem;
      font-family:'JetBrains Mono',monospace; font-size:.71rem; color:#52525b; margin:.5rem 0; white-space:pre-wrap; }

table.t { width:100%; border-collapse:collapse; font-size:.8rem; }
table.t th { text-align:left; padding:.5rem .7rem; background:#fafafa; color:#71717a; font-weight:500;
             font-size:.64rem; font-family:'JetBrains Mono',monospace; letter-spacing:.06em;
             text-transform:uppercase; border-bottom:1px solid #e4e4e7; }
table.t td { padding:.5rem .7rem; border-bottom:1px solid #f4f4f5; color:#3f3f46; vertical-align:middle; }
table.t tr:last-child td { border-bottom:none; }
.mono { font-family:'JetBrains Mono',monospace; }

.b-fresh{background:#ecfdf5;color:#047857;} .b-stale{background:#fffbeb;color:#b45309;}
.b-crit {background:#fef2f2;color:#b91c1c;} .b-pass {background:#ecfdf5;color:#047857;}
.b-fail {background:#fef2f2;color:#b91c1c;} .b-drift{background:#faf5ff;color:#7e22ce;}
.b-tp{background:#ecfdf5;color:#047857;} .b-fp{background:#fef2f2;color:#b91c1c;}
.b-tn{background:#f4f4f5;color:#52525b;} .b-fn{background:#fffbeb;color:#b45309;}
.b-ver{background:#ecfdf5;color:#047857;} .b-apx{background:#fffbeb;color:#b45309;}
.b-unv{background:#fef2f2;color:#b91c1c;}
.badge{padding:.1rem .45rem;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.63rem;font-weight:600;}

.bar-o{height:5px;background:#f4f4f5;border-radius:3px;width:110px;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:.4rem;}
.bar-i{height:100%;border-radius:3px;}

.claude{background:#fafafa;border:1px solid #e4e4e7;border-left:3px solid #18181b;border-radius:0 8px 8px 0;
        padding:1.25rem 1.5rem;font-size:.84rem;line-height:1.8;white-space:pre-wrap;}
.note{font-size:.7rem;color:#a1a1aa;font-family:'JetBrains Mono',monospace;line-height:1.6;margin-top:.6rem;}
#MainMenu,footer,header{visibility:hidden;} .stDeployButton{display:none;}
</style>
""", unsafe_allow_html=True)


def badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{text}</span>'


def kpi(label: str, value, sub: str, color: str = "#18181b") -> str:
    return (f'<div class="kpi"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v" style="color:{color}">{value}</div>'
            f'<div class="kpi-s">{sub}</div></div>')


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="kpi-l">Pipeline</div>', unsafe_allow_html=True)
    pipeline_choice = st.radio(
        "pipeline", ["Direct (no LLM)", "Claude-augmented"],
        label_visibility="collapsed",
    )

    api_key = None
    if pipeline_choice == "Claude-augmented":
        st.markdown('<div class="kpi-l" style="margin-top:.8rem">Anthropic API key</div>',
                    unsafe_allow_html=True)
        api_key = st.text_input("key", type="password", placeholder="sk-ant-...",
                                label_visibility="collapsed",
                                help="Held in session memory only. Never written to disk.")
        if api_key:
            st.markdown('<div style="font-size:.7rem;color:#047857;'
                        'font-family:JetBrains Mono,monospace">Key held in session only.</div>',
                        unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e4e4e7;margin:.9rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="kpi-l">Parameters</div>', unsafe_allow_html=True)
    threshold = st.slider("Staleness threshold (days)", 7, 90,
                          settings.STALENESS_THRESHOLD_DAYS)
    write_external = st.checkbox("Write to Snowflake and S3", value=True)

    st.markdown("<hr style='border-color:#e4e4e7;margin:.9rem 0'>", unsafe_allow_html=True)
    run = st.button("Run health check", use_container_width=True, type="primary")

    st.markdown("<hr style='border-color:#e4e4e7;margin:.9rem 0'>", unsafe_allow_html=True)
    st.markdown(f"""<div class="note">
    EnterpriseAI Handoff Kit v2<br>
    Sejal Khade<br>
    github.com/SejalKhade<br><br>
    embedder: {settings.EMBEDDER_BACKEND}<br>
    relevance floor: {settings.RELEVANCE_THRESHOLD}<br>
    drift penalty: {settings.DRIFT_PENALTY}<br>
    freshness mode: {settings.FRESHNESS_MODE}
    </div>""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
sf_on = settings.snowflake_configured
s3_on = settings.s3_configured
st.markdown(f"""
<div class="hdr">
  <div class="hdr-t">EnterpriseAI Handoff Kit</div>
  <div style="margin-top:.5rem">
    <span class="chip">ChromaDB</span>
    <span class="chip">SQLAlchemy</span>
    <span class="chip">FastAPI</span>
    <span class="chip">Node.js</span>
    <span class="chip {'chip-on' if sf_on else 'chip-off'}">Snowflake {'live' if sf_on else 'fallback'}</span>
    <span class="chip {'chip-on' if s3_on else 'chip-off'}">S3 {'live' if s3_on else 'fallback'}</span>
  </div>
  <div class="hdr-s" style="margin-top:.6rem">
    What the client can see after the deployment engineers leave
  </div>
</div>
""", unsafe_allow_html=True)


# ── Run ──────────────────────────────────────────────────────────────────────
if run:
    if pipeline_choice == "Claude-augmented" and not api_key:
        st.error("An Anthropic API key is required for the Claude-augmented pipeline.")
        st.stop()
    with st.spinner("Running retrieval, evaluation, and health checks..."):
        try:
            if pipeline_choice == "Direct (no LLM)":
                report = orchestrator.run_direct_pipeline(threshold, write_external)
            else:
                report = orchestrator.run_claude_pipeline(api_key, threshold, write_external)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    if report.get("status") != "ok":
        st.error(report.get("error", "Pipeline did not complete."))
        st.info("Seed the corpus first:  python scripts/seed_corpus.py --reset")
        st.stop()
    st.session_state["report"] = report


if "report" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:4.5rem 2rem;color:#a1a1aa">
      <div style="font-size:1rem;font-weight:500;color:#71717a;margin-bottom:.7rem">
        Run a health check to begin
      </div>
      <div class="note" style="display:inline-block;text-align:left">
        Executes real retrieval against the ChromaDB collection<br>
        Scores every query twice: with and without freshness-aware ranking<br>
        Compares embedded content hashes against current source hashes<br>
        Writes results to Snowflake and S3, or to local fallback
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


report = st.session_state["report"]
ev, fr, ig = report["evaluation"], report["freshness"], report["integration"]


# ── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
fpr_d = ev["delta"]["false_positive_rate_delta"]
f1_d  = ev["delta"]["f1_delta"]

with c1:
    st.markdown(kpi("False positive rate",
                    f'{ev["before"]["false_positive_rate"]:.0%} to {ev["after"]["false_positive_rate"]:.0%}',
                    f'{fpr_d*100:+.1f} points',
                    "#047857" if fpr_d < 0 else "#b91c1c" if fpr_d > 0 else "#18181b"),
                unsafe_allow_html=True)
with c2:
    st.markdown(kpi("F1 score",
                    f'{ev["before"]["f1_score"]:.3f} to {ev["after"]["f1_score"]:.3f}',
                    f'{f1_d*100:+.1f} points',
                    "#047857" if f1_d > 0 else "#b91c1c" if f1_d < 0 else "#18181b"),
                unsafe_allow_html=True)
with c3:
    vc = {"Healthy": "#047857", "Degraded": "#b45309"}.get(fr["overall_health"], "#b91c1c")
    st.markdown(kpi("Vector store", fr["overall_health"],
                    f'{fr["critical_count"]} critical, {fr["content_drift_count"]} drifted',
                    vc), unsafe_allow_html=True)
with c4:
    ic = {"Healthy": "#047857", "Degraded": "#b45309"}.get(ig["overall_status"], "#b91c1c")
    st.markdown(kpi("Integration", ig["overall_status"],
                    f'{ig["checks_passed"]} of {ig["total_checks"]} checks passing',
                    ic), unsafe_allow_html=True)


# ── Evaluation ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-h">Retrieval quality, with and without freshness-aware ranking</div>'
            '<div class="sec-b">', unsafe_allow_html=True)

st.markdown(f"""<div class="fx">Both columns run the same queries against the same ChromaDB collection.
The only difference is whether a chunk's similarity is discounted for being stale or drifted.

  precision           TP/(TP+FP)          of what was answered, how much was correct
  recall              TP/(TP+FN)          of what should have been answered, how much was
  f1                  2PR/(P+R)           harmonic mean
  false positive rate FP/(FP+TN)          of what should have been declined, how much was answered anyway

  relevance floor     {settings.RELEVANCE_THRESHOLD}    a query is answered only above this adjusted similarity
  age penalty         max({settings.FRESHNESS_WEIGHT_FLOOR}, 1 - {settings.FRESHNESS_DECAY} x (age/threshold - 1))
  drift penalty       x {settings.DRIFT_PENALTY}   applied when the embedded text hash differs from the source hash</div>""",
            unsafe_allow_html=True)

rows = ""
for name, key, higher_better in [
    ("Precision", "precision", True),
    ("Recall", "recall", True),
    ("F1 score", "f1_score", True),
    ("False positive rate", "false_positive_rate", False),
    ("Accuracy", "accuracy", True),
]:
    b, a = ev["before"][key], ev["after"][key]
    d = round(a - b, 4)
    good = (d > 0 and higher_better) or (d < 0 and not higher_better)
    col = "#047857" if (good and d != 0) else "#b91c1c" if (d != 0 and not good) else "#a1a1aa"
    rows += (f'<tr><td style="font-weight:500">{name}</td>'
             f'<td class="mono">{ev["before"]["formulas"][key].split("=")[0].strip()}</td>'
             f'<td class="mono" style="text-align:center;color:#71717a">{b:.4f}</td>'
             f'<td class="mono" style="text-align:center;font-weight:600">{a:.4f}</td>'
             f'<td class="mono" style="text-align:center;color:{col}">{d:+.4f}</td></tr>')

st.markdown(f'<table class="t"><thead><tr><th>Metric</th><th>Formula</th>'
            f'<th style="text-align:center">Without</th><th style="text-align:center">With</th>'
            f'<th style="text-align:center">Change</th></tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True)

cb, ca = ev["before"]["confusion_matrix"], ev["after"]["confusion_matrix"]
st.markdown(f"""<div class="note">
Without freshness ranking: TP {cb['true_positives']}, FP {cb['false_positives']},
TN {cb['true_negatives']}, FN {cb['false_negatives']}<br>
With freshness ranking: TP {ca['true_positives']}, FP {ca['false_positives']},
TN {ca['true_negatives']}, FN {ca['false_negatives']}<br><br>
{ev['improvement_summary']}
</div>""", unsafe_allow_html=True)

with st.expander("Per-query outcomes"):
    qrows = ""
    for b, a in zip(ev["before_table"], ev["after_table"]):
        bb = badge(b["cell"], f'b-{b["cell"].lower()}')
        ab = badge(a["cell"], f'b-{a["cell"].lower()}')
        qrows += (f'<tr><td style="font-size:.78rem">{b["query"]}</td>'
                  f'<td class="mono" style="text-align:center">{"yes" if b["ground_truth"] else "no"}</td>'
                  f'<td class="mono" style="text-align:center">{b["top_score"]:.4f}</td><td>{bb}</td>'
                  f'<td class="mono" style="text-align:center">{a["top_score"]:.4f}</td><td>{ab}</td>'
                  f'<td class="mono" style="text-align:center;color:#a1a1aa">{a["max_age_days"]:.0f}d</td></tr>')
    st.markdown(f'<table class="t"><thead><tr><th>Query</th>'
                f'<th style="text-align:center">Should answer</th>'
                f'<th style="text-align:center">Score</th><th>Without</th>'
                f'<th style="text-align:center">Score</th><th>With</th>'
                f'<th style="text-align:center">Age</th></tr></thead>'
                f'<tbody>{qrows}</tbody></table>', unsafe_allow_html=True)
    st.markdown(f'<div class="note">{ev["methodology"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── Vector store health ──────────────────────────────────────────────────────
st.markdown('<div class="sec-h">Vector store health</div><div class="sec-b">',
            unsafe_allow_html=True)

st.markdown(f"""<div class="fx">freshness = max(0, 1 - (oldest chunk age / {fr['staleness_threshold']} days))

  Fresh     score above {settings.FRESH_CUTOFF}
  Stale     score above {settings.STALE_CUTOFF}
  Critical  score at or below {settings.STALE_CUTOFF}

Content drift is separate and is not a heuristic. It compares the SHA256 of the
document now against the SHA256 captured when the chunk was embedded. When they
differ, the store is provably serving text that no longer exists in the source.</div>""",
            unsafe_allow_html=True)

drows = ""
for d in fr["documents"]:
    sb = badge(d["health_status"], {"Fresh": "b-fresh", "Stale": "b-stale"}.get(d["health_status"], "b-crit"))
    db_ = badge("DRIFT", "b-drift") if d["content_drift"] else ""
    w = int(d["freshness_score"] * 110)
    c = "#047857" if d["health_status"] == "Fresh" else "#b45309" if d["health_status"] == "Stale" else "#b91c1c"
    drows += (f'<tr><td style="font-size:.78rem;font-weight:500">{d["title"][:46]}</td>'
              f'<td class="mono" style="color:#a1a1aa">{d["category"]}</td>'
              f'<td class="mono">{d["max_age_days"]:.0f}d</td>'
              f'<td><span class="bar-o"><span class="bar-i" style="width:{w}px;background:{c}"></span></span>'
              f'<span class="mono" style="font-size:.72rem;color:{c}">{d["freshness_score"]:.3f}</span></td>'
              f'<td>{sb} {db_}</td>'
              f'<td style="font-size:.72rem;color:#a1a1aa">{d["action"][:58]}</td></tr>')

st.markdown(f'<table class="t"><thead><tr><th>Document</th><th>Category</th><th>Oldest chunk</th>'
            f'<th>Freshness</th><th>Status</th><th>Action</th></tr></thead>'
            f'<tbody>{drows}</tbody></table>', unsafe_allow_html=True)

if fr["reindex_queue"]:
    st.markdown(f'<div class="note">Re-index queue: {", ".join(fr["reindex_queue"])}</div>',
                unsafe_allow_html=True)
    if st.button("Re-index these documents now"):
        with st.spinner("Re-embedding..."):
            res = orchestrator.reindex_documents(fr["reindex_queue"])
        st.success(f'Re-indexed {res["reindexed"]} of {res["requested"]} documents. '
                   f'Run the health check again to see the effect.')
        st.session_state.pop("report", None)

st.markdown("</div>", unsafe_allow_html=True)


# ── Integration ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-h">Integration checks</div><div class="sec-b">',
            unsafe_allow_html=True)

irows = ""
for c in ig["checks"]:
    sb = badge(c["status"].upper(), "b-pass" if c["status"] == "Pass" else "b-fail")
    irows += (f'<tr><td style="font-weight:500">{c["dimension"]}</td>'
              f'<td class="mono" style="font-size:.76rem">{c["value"]}</td>'
              f'<td class="mono" style="font-size:.72rem;color:#a1a1aa">{c["threshold"]}</td>'
              f'<td class="mono" style="font-size:.7rem;color:#a1a1aa">{c["formula"]}</td>'
              f'<td>{sb}</td></tr>')

st.markdown(f'<table class="t"><thead><tr><th>Check</th><th>Measured</th><th>Threshold</th>'
            f'<th>Computed from</th><th>Status</th></tr></thead>'
            f'<tbody>{irows}</tbody></table>', unsafe_allow_html=True)
st.markdown(f'<div class="note">{ig["recommendation"]}<br>'
            f'Measured over {ig.get("sample_size", 0)} logged queries.</div>',
            unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# ── Destinations ─────────────────────────────────────────────────────────────
if report.get("snowflake") or report.get("s3"):
    st.markdown('<div class="sec-h">Where this report was written</div><div class="sec-b">',
                unsafe_allow_html=True)
    sf, s3 = report.get("snowflake"), report.get("s3")
    a, b = st.columns(2)
    with a:
        if sf:
            st.markdown("**Snowflake**")
            for k, lbl in [("vector_health", "VECTOR_DB_HEALTH"),
                           ("eval_metrics", "EVAL_METRICS_HISTORY"),
                           ("integration", "INTEGRATION_HEALTH")]:
                r = sf[k]
                bd = badge(r["status"].upper(), "b-pass" if r["status"] == "ok" else "b-stale")
                st.markdown(f'<div style="font-size:.76rem;padding:.2rem 0">{bd} '
                            f'<span class="mono">{lbl}</span> '
                            f'<span style="color:#a1a1aa">{r.get("rows", 0)} rows</span></div>',
                            unsafe_allow_html=True)
            st.markdown(f'<div class="note">{sf["target"]["database"]}.{sf["target"]["schema"]}<br>'
                        f'{"Live connection" if sf["target"]["enabled"] else "Fallback: " + sf["target"]["fallback"]}</div>',
                        unsafe_allow_html=True)
    with b:
        if s3:
            st.markdown("**S3**")
            bd = badge(s3["status"].upper(), "b-pass" if s3["status"] == "ok" else "b-stale")
            st.markdown(f'<div style="font-size:.76rem;padding:.2rem 0">{bd} '
                        f'<span class="mono">{s3["bytes"]} bytes</span></div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="note">{s3["target"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Claude ───────────────────────────────────────────────────────────────────
claude = report.get("claude")
if claude and claude.get("success"):
    guard = report.get("hallucination_guard", {})
    st.markdown(f'<div class="sec-h">Plain-language handover, written by {claude["model"]}</div>'
                '<div class="sec-b">', unsafe_allow_html=True)
    st.markdown(f'<div class="claude">{claude["text"]}</div>', unsafe_allow_html=True)

    if guard.get("performed"):
        st.markdown("<hr style='border-color:#e4e4e7;margin:1.1rem 0'>", unsafe_allow_html=True)
        st.markdown('<div class="kpi-l">Verification of every number above</div>',
                    unsafe_allow_html=True)
        rel = guard["reliability_score"]
        col = "#047857" if rel >= 80 else "#b45309" if rel >= 60 else "#b91c1c"
        g1, g2, g3, g4 = st.columns(4)
        for cc, lbl, val, sub, c_ in [
            (g1, "Reliability", f'{rel}%', guard["overall_status"], col),
            (g2, "Traced", guard["verified"], "matched source", "#047857"),
            (g3, "Approximate", guard["approximations"], "within 15%", "#b45309"),
            (g4, "Untraceable", guard["unverified"], "not in payload", "#b91c1c"),
        ]:
            with cc:
                st.markdown(kpi(lbl, val, sub, c_), unsafe_allow_html=True)

        if guard["claims"]:
            with st.expander(f'Claim-by-claim check ({guard["total_claims"]} numbers)'):
                crows = ""
                for cl in guard["claims"]:
                    bd = badge(cl["status"], {"VERIFIED": "b-ver",
                                              "APPROXIMATION": "b-apx"}.get(cl["status"], "b-unv"))
                    crows += (f'<tr><td class="mono">{cl["claimed"]}{cl["unit"]}</td>'
                              f'<td style="font-size:.72rem;color:#a1a1aa">{cl["context"][:75]}</td>'
                              f'<td>{bd}</td>'
                              f'<td class="mono" style="font-size:.7rem">{cl["matched"] if cl["matched"] is not None else "—"}</td>'
                              f'<td class="mono" style="font-size:.68rem;color:#a1a1aa">{cl["source_path"] or "—"}</td></tr>')
                st.markdown(f'<table class="t"><thead><tr><th>Number</th><th>Context</th>'
                            f'<th>Status</th><th>Source value</th><th>Path</th></tr></thead>'
                            f'<tbody>{crows}</tbody></table>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">{guard["methodology"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif claude and not claude.get("success"):
    st.error(f'Claude call failed: {claude.get("error")}')


# ── Raw ──────────────────────────────────────────────────────────────────────
with st.expander("Raw report JSON"):
    trimmed = {k: v for k, v in report.items() if k != "claude"}
    st.code(json.dumps(trimmed, indent=2, default=str)[:60000], language="json")
