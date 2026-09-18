"""
EnterpriseAI Handoff Kit — Evaluation Metrics
Computes precision, recall, F1, accuracy, and false positive rate from
real RAG retrieval outcomes stored in query_logs.

Every metric returns the formula with substituted values so a reviewer can
verify the arithmetic without reading the source.
"""

from __future__ import annotations
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.config import settings
from src.database import repository as repo


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positives:  int
    false_positives: int
    true_negatives:  int
    false_negatives: int

    @property
    def total(self) -> int:
        return (self.true_positives + self.false_positives
                + self.true_negatives + self.false_negatives)


def compute_metrics(cm: ConfusionMatrix) -> dict:
    """All classification metrics with explicit formulas."""
    tp, fp = cm.true_positives, cm.false_positives
    tn, fn = cm.true_negatives, cm.false_negatives

    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    fpr       = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    accuracy  = round((tp + tn) / cm.total, 4) if cm.total else 0.0

    return {
        "precision":           precision,
        "recall":              recall,
        "f1_score":            f1,
        "false_positive_rate": fpr,
        "accuracy":            accuracy,
        "confusion_matrix": {
            "true_positives":  tp,
            "false_positives": fp,
            "true_negatives":  tn,
            "false_negatives": fn,
            "total":           cm.total,
        },
        "formulas": {
            "precision":           f"TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision}",
            "recall":              f"TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall}",
            "f1_score":            f"2*P*R/(P+R) = 2*{precision}*{recall}/({precision}+{recall}) = {f1}",
            "false_positive_rate": f"FP/(FP+TN) = {fp}/({fp}+{tn}) = {fpr}",
            "accuracy":            f"(TP+TN)/Total = ({tp}+{tn})/{cm.total} = {accuracy}",
        },
    }


def confusion_from_results(results: list[dict]) -> ConfusionMatrix:
    """Build a confusion matrix from retrieval results carrying ground truth."""
    tp = fp = tn = fn = 0
    for r in results:
        gt = r.get("ground_truth_relevant")
        ai = r.get("flagged_relevant", False)
        if gt is None:
            continue
        if gt and ai:
            tp += 1
        elif not gt and ai:
            fp += 1
        elif not gt and not ai:
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)


def label_results(results: list[dict]) -> list[dict]:
    """Attach the confusion-matrix cell to each result for the audit table."""
    labelled = []
    for r in results:
        gt = r.get("ground_truth_relevant")
        ai = r.get("flagged_relevant", False)
        if gt is None:
            cell = "UNLABELLED"
        elif gt and ai:
            cell = "TP"
        elif not gt and ai:
            cell = "FP"
        elif not gt and not ai:
            cell = "TN"
        else:
            cell = "FN"

        labelled.append({
            "query":            r["query"][:70],
            "ground_truth":     gt,
            "flagged_relevant": ai,
            "top_score":        r["top_score"],
            "max_age_days":     round(r["max_embedding_age_days"], 2),
            "filtered_count":   len(r.get("filtered_out", [])),
            "latency_ms":       r["latency_ms"],
            "cell":             cell,
        })
    return labelled


def evaluate_before_after(pipeline, session: Session, test_set: list[dict],
                          staleness_threshold: int | None = None) -> dict:
    """
    Run the labelled test set twice against the real RAG pipeline.

      BEFORE: retrieval with no freshness filter (stale chunks can be returned)
      AFTER:  retrieval with freshness filtering enabled

    Both runs hit the real vector store. The only variable is the filter.
    """
    threshold = staleness_threshold or settings.STALENESS_THRESHOLD_DAYS

    before_results, after_results = [], []
    for case in test_set:
        q  = case["query"]
        gt = case["ground_truth_relevant"]

        rb = pipeline.retrieve(session, q, apply_freshness_filter=False,
                               staleness_threshold=threshold)
        rb["ground_truth_relevant"] = gt
        before_results.append(rb)

        ra = pipeline.retrieve(session, q, apply_freshness_filter=True,
                               staleness_threshold=threshold)
        ra["ground_truth_relevant"] = gt
        after_results.append(ra)

    cm_before = confusion_from_results(before_results)
    cm_after  = confusion_from_results(after_results)
    m_before  = compute_metrics(cm_before)
    m_after   = compute_metrics(cm_after)

    stale_filtered = sum(len(r.get("filtered_out", [])) for r in after_results)
    queries_affected = sum(1 for r in after_results if r.get("filtered_out"))

    avg_latency_before = round(sum(r["latency_ms"] for r in before_results) / len(before_results), 3) if before_results else 0.0
    avg_latency_after  = round(sum(r["latency_ms"] for r in after_results) / len(after_results), 3) if after_results else 0.0

    delta = {
        "precision_delta":           round(m_after["precision"] - m_before["precision"], 4),
        "recall_delta":              round(m_after["recall"] - m_before["recall"], 4),
        "f1_delta":                  round(m_after["f1_score"] - m_before["f1_score"], 4),
        "false_positive_rate_delta": round(m_after["false_positive_rate"] - m_before["false_positive_rate"], 4),
        "accuracy_delta":            round(m_after["accuracy"] - m_before["accuracy"], 4),
        "stale_chunks_filtered":     stale_filtered,
        "queries_affected":          queries_affected,
    }

    repo.save_eval_run(session, run_label="before", metrics=m_before,
                       total_queries=len(test_set), staleness_threshold=threshold)
    repo.save_eval_run(session, run_label="after", metrics=m_after,
                       total_queries=len(test_set), staleness_threshold=threshold)

    fpr_change = abs(delta["false_positive_rate_delta"]) * 100
    prec_change = abs(delta["precision_delta"]) * 100
    summary = (
        f"Freshness filtering removed {stale_filtered} stale chunks across "
        f"{queries_affected} of {len(test_set)} queries. "
        f"False positive rate {'fell' if delta['false_positive_rate_delta'] < 0 else 'rose' if delta['false_positive_rate_delta'] > 0 else 'was unchanged'} "
        f"by {fpr_change:.1f} percentage points. "
        f"Precision {'rose' if delta['precision_delta'] > 0 else 'fell' if delta['precision_delta'] < 0 else 'was unchanged'} "
        f"by {prec_change:.1f} percentage points."
    )

    return {
        "before":              m_before,
        "after":               m_after,
        "delta":               delta,
        "before_table":        label_results(before_results),
        "after_table":         label_results(after_results),
        "total_queries":       len(test_set),
        "staleness_threshold": threshold,
        "avg_latency_before_ms": avg_latency_before,
        "avg_latency_after_ms":  avg_latency_after,
        "relevance_threshold": settings.RELEVANCE_THRESHOLD,
        "improvement_summary": summary,
        "methodology": (
            "Both runs execute real retrieval against the ChromaDB collection. "
            "A query is flagged relevant when the top cosine similarity of the "
            f"returned chunks is at least {settings.RELEVANCE_THRESHOLD}. "
            "The only difference between runs is whether chunks with embeddings "
            f"older than {threshold} days are excluded before scoring."
        ),
    }
