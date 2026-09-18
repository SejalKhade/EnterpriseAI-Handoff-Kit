"""Tests for freshness scoring, evaluation metrics, integration checks,
connectors, and the hallucination guard."""

from __future__ import annotations
import pytest

from src.config import settings
from src.database import repository as repo
from src.monitoring.freshness import run_freshness_check, classify
from src.monitoring.eval_metrics import (
    ConfusionMatrix, compute_metrics, confusion_from_results,
    label_results, evaluate_before_after,
)
from src.monitoring.integration import run_integration_checks
from src.llm.hallucination_guard import (
    extract_claims, flatten_payload, verify, evaluate as guard_evaluate,
)
from src.llm.claude_client import build_payload
from src.connectors.snowflake_writer import SnowflakeWriter
from src.connectors.s3_writer import S3Writer


class TestMetricFormulas:
    def test_precision(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        assert m["precision"] == pytest.approx(0.8)

    def test_recall(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        assert m["recall"] == pytest.approx(0.8)

    def test_f1_is_harmonic_mean(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        p, r = m["precision"], m["recall"]
        assert m["f1_score"] == pytest.approx(2 * p * r / (p + r), abs=1e-4)

    def test_false_positive_rate(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        assert m["false_positive_rate"] == pytest.approx(2 / 5)

    def test_accuracy(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        assert m["accuracy"] == pytest.approx(11 / 15, abs=1e-4)

    def test_perfect_classifier(self):
        m = compute_metrics(ConfusionMatrix(10, 0, 10, 0))
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["false_positive_rate"] == 0.0

    def test_no_predictions_is_zero_not_error(self):
        m = compute_metrics(ConfusionMatrix(0, 0, 5, 5))
        assert m["precision"] == 0.0 and m["recall"] == 0.0

    def test_empty_matrix(self):
        m = compute_metrics(ConfusionMatrix(0, 0, 0, 0))
        assert m["accuracy"] == 0.0

    def test_formulas_show_substituted_values(self):
        m = compute_metrics(ConfusionMatrix(8, 2, 3, 2))
        assert "8/(8+2)" in m["formulas"]["precision"]

    def test_confusion_total(self):
        assert ConfusionMatrix(1, 2, 3, 4).total == 10


class TestConfusionFromResults:
    def test_all_four_cells(self):
        results = [
            {"ground_truth_relevant": True,  "flagged_relevant": True},
            {"ground_truth_relevant": False, "flagged_relevant": True},
            {"ground_truth_relevant": False, "flagged_relevant": False},
            {"ground_truth_relevant": True,  "flagged_relevant": False},
        ]
        cm = confusion_from_results(results)
        assert (cm.true_positives, cm.false_positives,
                cm.true_negatives, cm.false_negatives) == (1, 1, 1, 1)

    def test_unlabelled_ignored(self):
        cm = confusion_from_results([{"ground_truth_relevant": None,
                                      "flagged_relevant": True}])
        assert cm.total == 0

    def test_label_results_marks_cells(self):
        rows = label_results([{
            "query": "q", "ground_truth_relevant": True, "flagged_relevant": True,
            "top_score": 0.9, "max_embedding_age_days": 5, "latency_ms": 1.0,
        }])
        assert rows[0]["cell"] == "TP"


class TestFreshness:
    def test_classify_bands(self):
        assert classify(0.9) == "Fresh"
        assert classify(0.5) == "Stale"
        assert classify(0.1) == "Critical"

    def test_scores_every_document(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        assert r["total_documents"] == 4

    def test_counts_reconcile(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        assert r["critical_count"] + r["stale_count"] + r["fresh_count"] == r["total_documents"]

    def test_sorted_worst_first(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        scores = [d["freshness_score"] for d in r["documents"]]
        assert scores == sorted(scores)

    def test_old_document_is_critical(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        doc = next(d for d in r["documents"] if d["doc_key"] == "rate_limits")
        assert doc["health_status"] == "Critical"
        assert doc["age_drift"] is True

    def test_recent_document_is_fresh(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        doc = next(d for d in r["documents"] if d["doc_key"] == "refund_policy")
        assert doc["health_status"] == "Fresh"

    def test_drift_detected(self, drifted, session):
        r = run_freshness_check(session, threshold=30)
        doc = next(d for d in r["documents"] if d["doc_key"] == "rate_limits")
        assert doc["content_drift"] is True

    def test_drift_forces_critical_overall(self, drifted, session):
        assert run_freshness_check(session, threshold=30)["overall_health"] == "Critical"

    def test_reindex_queue_contains_drifted(self, drifted, session):
        assert "rate_limits" in run_freshness_check(session, threshold=30)["reindex_queue"]

    def test_formula_is_exposed(self, seeded, session):
        r = run_freshness_check(session, threshold=30)
        assert "max(0, 1 -" in r["documents"][0]["formula"]

    def test_higher_threshold_improves_scores(self, seeded, session):
        low  = run_freshness_check(session, threshold=30)["average_freshness"]
        high = run_freshness_check(session, threshold=90)["average_freshness"]
        assert high > low


class TestIntegrationChecks:
    def test_no_logs_reports_unknown(self, session):
        repo.clear_query_logs(session)
        assert run_integration_checks(session)["overall_status"] == "Unknown"

    def test_four_checks_returned(self, seeded, session):
        for q in ("refund policy", "webhook setup", "single sign on"):
            seeded.query_and_log(session, q, ground_truth_relevant=True)
        assert len(run_integration_checks(session)["checks"]) == 4

    def test_latency_check_passes_locally(self, seeded, session):
        for q in ("refund policy", "webhook setup"):
            seeded.query_and_log(session, q, ground_truth_relevant=True)
        r = run_integration_checks(session)
        latency = next(c for c in r["checks"] if c["dimension"] == "Retrieval Latency")
        assert latency["status"] == "Pass"

    def test_error_rate_zero_without_errors(self, seeded, session):
        seeded.query_and_log(session, "refund policy", ground_truth_relevant=True)
        assert run_integration_checks(session)["error_rate_pct"] == 0.0

    def test_every_check_has_formula_and_source(self, seeded, session):
        seeded.query_and_log(session, "refund policy", ground_truth_relevant=True)
        for c in run_integration_checks(session)["checks"]:
            assert c["formula"] and c["source"]


class TestBeforeAfter:
    TEST_SET = [
        {"query": "enterprise refund within thirty days", "ground_truth_relevant": True},
        {"query": "webhook signature header HMAC",        "ground_truth_relevant": True},
        {"query": "zzzqqq xyzzy plugh frobnicate",        "ground_truth_relevant": False},
    ]

    def test_returns_both_runs(self, seeded, session):
        r = evaluate_before_after(seeded, session, self.TEST_SET, 30)
        assert "before" in r and "after" in r and "delta" in r

    def test_does_not_lose_true_positives(self, seeded, session):
        r = evaluate_before_after(seeded, session, self.TEST_SET, 30)
        assert r["after"]["recall"] >= r["before"]["recall"] - 1e-9

    def test_false_positives_do_not_increase(self, drifted, session):
        r = evaluate_before_after(drifted, session, self.TEST_SET, 30)
        assert (r["after"]["confusion_matrix"]["false_positives"]
                <= r["before"]["confusion_matrix"]["false_positives"])

    def test_persists_two_eval_runs(self, seeded, session):
        before = len(repo.eval_history(session))
        evaluate_before_after(seeded, session, self.TEST_SET, 30)
        assert len(repo.eval_history(session)) == before + 2

    def test_tables_match_test_set_size(self, seeded, session):
        r = evaluate_before_after(seeded, session, self.TEST_SET, 30)
        assert len(r["before_table"]) == len(self.TEST_SET)

    def test_methodology_present(self, seeded, session):
        r = evaluate_before_after(seeded, session, self.TEST_SET, 30)
        assert "ChromaDB" in r["methodology"]


class TestHallucinationGuard:
    PAYLOAD = {"evaluation": {"before": {"precision": 0.6154},
                              "after": {"precision": 0.8889}},
               "vector_db": {"critical_count": 4, "total_documents": 10}}

    def test_extracts_numbers(self):
        claims = extract_claims("There are 10 documents and 4 are critical.")
        assert {c["value"] for c in claims} >= {10.0, 4.0}

    def test_skips_list_markers(self):
        assert all(c["value"] >= 2.0 for c in extract_claims("1. First item"))

    def test_flatten_finds_nested(self):
        paths = {v["path"] for v in flatten_payload(self.PAYLOAD)}
        assert "vector_db.critical_count" in paths

    def test_flatten_skips_booleans(self):
        assert flatten_payload({"flag": True}) == []

    def test_exact_match_verified(self):
        v = verify(10.0, flatten_payload(self.PAYLOAD))
        assert v["status"] == "VERIFIED"

    def test_percentage_form_verified(self):
        v = verify(88.89, flatten_payload(self.PAYLOAD))
        assert v["status"] == "VERIFIED"
        assert v["form"] == "percentage"

    def test_fabricated_is_unverified(self):
        v = verify(9999.0, flatten_payload(self.PAYLOAD))
        assert v["status"] == "UNVERIFIED"

    def test_reliable_when_all_traced(self):
        r = guard_evaluate("There are 10 documents, 4 critical.", self.PAYLOAD)
        assert r["overall_status"] == "RELIABLE"

    def test_flags_invented_numbers(self):
        r = guard_evaluate("The system processed 7777 records.", self.PAYLOAD)
        assert r["unverified"] >= 1

    def test_no_text_returns_not_performed(self):
        assert guard_evaluate("", self.PAYLOAD)["performed"] is False

    def test_reliability_bounded(self):
        r = guard_evaluate("10 documents, 4 critical, 9999 unknown.", self.PAYLOAD)
        assert 0.0 <= r["reliability_score"] <= 100.0


class TestClaudePayload:
    def test_payload_contains_only_needed_sections(self, seeded, session):
        from src.monitoring.freshness import run_freshness_check
        seeded.query_and_log(session, "refund policy", ground_truth_relevant=True)
        report = {
            "evaluation": evaluate_before_after(
                seeded, session,
                [{"query": "refund policy", "ground_truth_relevant": True}], 30),
            "freshness": run_freshness_check(session, 30),
            "integration": run_integration_checks(session),
        }
        payload = build_payload(report)
        assert set(payload) == {"evaluation", "vector_db", "integration"}

    def test_payload_has_no_raw_text(self, seeded, session):
        from src.monitoring.freshness import run_freshness_check
        seeded.query_and_log(session, "refund policy", ground_truth_relevant=True)
        report = {
            "evaluation": evaluate_before_after(
                seeded, session,
                [{"query": "refund policy", "ground_truth_relevant": True}], 30),
            "freshness": run_freshness_check(session, 30),
            "integration": run_integration_checks(session),
        }
        import json
        assert "chunk_text" not in json.dumps(build_payload(report))


class TestConnectors:
    def test_snowflake_falls_back_without_credentials(self, seeded, session):
        w = SnowflakeWriter()
        r = w.write_vector_health(run_freshness_check(session, 30))
        assert r["status"] in ("ok", "fallback")

    def test_snowflake_reports_targets(self):
        info = SnowflakeWriter().target_info()
        assert "VECTOR_DB_HEALTH" in info["tables"]

    def test_snowflake_ensure_tables_skips_when_unset(self):
        w = SnowflakeWriter()
        if not w.enabled:
            assert w.ensure_tables()["status"] == "skipped"

    def test_s3_writes_something(self):
        r = S3Writer().write_report({"test": 1}, name="unit_test")
        assert r["status"] in ("ok", "fallback")
        assert r["bytes"] > 0

    def test_s3_reports_target(self):
        assert "bucket" in S3Writer().target_info()
