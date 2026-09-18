# CONTEXT.md — Methodology and Sources

## Problem framing

Industry data on enterprise AI deployment outcomes:

- RAND Corporation, 65 practitioner interviews: over 80% of AI projects fail,
  roughly twice the rate of conventional IT projects.
- Gartner, April 2026 survey of 782 infrastructure leaders: 28% of AI
  infrastructure projects delivered the promised return.
- S&P Global, 1,000+ enterprises: 42% abandoned most AI initiatives in 2025,
  up from 17% the year prior.
- Reuters, July 2026: TCS plans 5,900 to 8,900 forward deployed engineers,
  explicitly to address the gap between models that work in a demo and models
  that work in production.

Failures trace to the same causes each time: models built for controlled
environments meeting production data, integrations never properly scoped, and
no observability after handover.

This project addresses the last one.

## Evaluation methodology

A labelled query set in three categories:

- **answerable** — supported by fresh, undrifted content. Correct behaviour is
  to answer.
- **out of scope** — plausible business questions with no supporting document.
  Correct behaviour is to decline.
- **drift sensitive** — the only supporting document was updated after it was
  embedded. The store holds superseded text, so a confident answer is a wrong
  answer. Ground truth is False; correct behaviour is to decline until
  re-indexed.

The drift sensitive category is the reason the before/after comparison exists.
Without it, freshness monitoring has nothing to prove.

Each query runs twice against the same collection. The only variable is whether
freshness-aware scoring is applied.

## Metric definitions

```
precision           = TP / (TP + FP)
recall              = TP / (TP + FN)
f1                  = 2PR / (P + R)
false positive rate = FP / (FP + TN)
accuracy            = (TP + TN) / total
```

A query counts as answered when the top adjusted similarity is at or above
`RELEVANCE_THRESHOLD` and the query passes the vocabulary coverage gate.

## Freshness scoring

```
freshness_score = max(0, 1 - (oldest chunk age / threshold))

Fresh     > 0.70
Stale     > 0.30
Critical  <= 0.30
```

Retrieval-time confidence weight:

```
age <= threshold  ->  1.0
age  > threshold  ->  max(FRESHNESS_WEIGHT_FLOOR,
                          1 - FRESHNESS_DECAY x (age/threshold - 1))
content drift     ->  x DRIFT_PENALTY
```

Defaults: floor 0.45, decay 0.35, drift penalty 0.40, threshold 30 days.

## Content drift

Each embedding record stores `source_hash_at_embed`, the SHA256 of the document
content at the moment it was embedded. Each document stores `content_hash`, the
SHA256 of its content now. When they differ, the vector store is serving text
that no longer exists in the source.

This is a verified fact, not a heuristic, which is why it earns a harder penalty
than age and why it forces the overall vector store status to Critical.

## Query coverage gate

Embeddings are L2 normalized, so vector magnitude carries no information about
match quality. A query matching one incidental surviving token is the same
length as a query matching ten specific ones, and can score above 0.95 cosine
against an unrelated document.

The gate measures, before normalization:

- `tfidf_mass` — L2 norm of the raw TF-IDF vector
- `token_coverage` — in-vocabulary content tokens divided by total

Both must clear their floors for a query to be answerable. Defaults 0.35 and
0.30. Transformer backends report full coverage since subword embeddings handle
unseen tokens natively.

## Integration checks

Computed from `query_logs`, not simulated:

| Check | Source | Default threshold |
|---|---|---|
| Retrieval latency | `latency_ms`, measured with `perf_counter` | mean < 500 ms |
| Error rate | non-null `error` column | < 2% |
| Throughput consistency | stdev/mean of latency | CV < 0.20 |
| Embedding freshness | `max_embedding_age_days` | < 30 days |

## Hallucination guard

Claude receives `build_payload()` output only. Every number it writes is
extracted by regex and matched against that payload in both direct and
percentage form.

```
VERIFIED       within 5%
APPROXIMATION  within 15%
UNVERIFIED     not traceable
reliability    = (verified + 0.5 x approximations) / total x 100
```

Numbers below 2.0 are skipped; they are list markers and ordinals, not claims.

## Connector targets

```
ENTERPRISE_AI_OPS.HANDOFF_KIT.VECTOR_DB_HEALTH
ENTERPRISE_AI_OPS.HANDOFF_KIT.EVAL_METRICS_HISTORY
ENTERPRISE_AI_OPS.HANDOFF_KIT.INTEGRATION_HEALTH

s3://{bucket}/{prefix}{YYYY}/{MM}/{DD}/health_report_{timestamp}.json
```

Both fall back to `data/artifacts/` when unconfigured. DDL is created
automatically by `ensure_tables()`.

## Corpus

Ten documents written for this project, covering policy, integration, API, and
analytics topics. Embedding ages span 4 to 71 days. Two documents are given
content drift during seeding by updating the source after embedding.

Real deployment replaces `scripts/seed_corpus.py` with a loader against actual
content. Nothing downstream changes.
