# EnterpriseAI Handoff Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Node 18+](https://img.shields.io/badge/node-18%2B-green.svg)](package.json)

Post-deployment observability for enterprise RAG systems.

Repo: [github.com/SejalKhade/EnterpriseAI-Handoff-Kit](https://github.com/SejalKhade/EnterpriseAI-Handoff-Kit)

When a forward deployed engineering team finishes an AI project and leaves, the
client is left with a working system and no way to tell when it stops working.
Embeddings go stale. Source documents change underneath them. Retrieval quality
degrades quietly, and the first signal is usually a confidently wrong answer in
front of a customer.

This is the observability layer that ships with the handover.

---

## What it measures

**Retrieval quality.** Runs a labelled query set twice against the same ChromaDB
collection, once with freshness-aware ranking and once without, and reports
precision, recall, F1, false positive rate, and accuracy for both.

**Vector store health.** Scores every document on embedding age, and separately
detects content drift by comparing the SHA256 of the source now against the
SHA256 captured when the chunk was embedded. Drift is not an estimate. When the
hashes differ, the store is provably serving text that no longer exists.

**Integration reliability.** Four checks computed from real logged queries:
retrieval latency, error rate, throughput consistency, and mean embedding age.

---

## Measured result on the bundled corpus

Ten documents, twenty one chunks, eighteen labelled queries. Both columns run
real retrieval against the same collection. The only variable is whether a
chunk's similarity is discounted for being stale or drifted.

| Metric | Without freshness ranking | With | Change |
|---|---|---|---|
| Precision | 0.6154 | 0.8889 | +27.4 points |
| Recall | 1.0000 | 1.0000 | unchanged |
| F1 | 0.7619 | 0.9412 | +17.9 points |
| False positive rate | 0.5000 | 0.1000 | −40.0 points |
| Accuracy | 0.7222 | 0.9444 | +22.2 points |

Confusion matrix moved from TP 8 / FP 5 / TN 5 / FN 0 to TP 8 / FP 1 / TN 9 / FN 0.

Four false positives removed. Every true positive preserved. That second half
matters more than the first, and it is the reason the scoring is a graded
penalty rather than a filter.

---

## Two design decisions worth explaining

### Staleness is a penalty, not a filter

The obvious approach is to drop chunks whose embeddings exceed the staleness
threshold. On this corpus that removes false positives and takes recall from
1.00 down to 0.70, because some questions are only answerable from a document
that happens to be old.

Instead, similarity is multiplied by a confidence weight:

```
age    <= threshold  ->  1.0
age     > threshold  ->  max(0.45, 1 - 0.35 x (age/threshold - 1))
content drift        ->  x 0.40
```

A strongly matching stale chunk is demoted but still reachable. A weakly
matching stale chunk falls below the relevance floor and the system declines to
answer. Drift earns the harder multiplier because it is verified by hash rather
than inferred from a timestamp.

### Cosine similarity alone cannot reject out-of-scope questions

Embeddings are L2 normalized, so a query matching one incidental term is the
same length as a query matching ten specific ones. "How many employees work in
the Berlin office?" matched a single stopword-surviving token and scored 0.97
against an unrelated document.

Retrieval therefore also requires the query to carry real IDF-weighted mass
against the fitted vocabulary before any answer is returned. Adding that gate
took out-of-scope accuracy from 2/6 to 6/6.

---

## Architecture

```
dashboard/app.py          Streamlit UI, every number shows its formula
        |
api_node/server.js        Node.js integration layer, reshaping and TTL cache
        |
src/api/main.py           FastAPI service, seven routes
        |
src/orchestrator.py       Wires both pipelines
        |
   +----+----+-----------+--------------+
   |         |           |              |
  rag/   monitoring/  connectors/     llm/
   |         |           |              |
ChromaDB  freshness   Snowflake      Claude
SQLAlchemy eval       S3             hallucination guard
          integration
```

| Layer | Files | Responsibility |
|---|---|---|
| Config | `src/config.py` | Every threshold env-configurable, no secrets in code |
| Persistence | `src/database/` | Five SQLAlchemy tables, repository pattern |
| Retrieval | `src/rag/` | Chunking, embedding, ChromaDB, drift-aware scoring |
| Monitoring | `src/monitoring/` | Freshness, evaluation, integration, scheduler |
| Connectors | `src/connectors/` | Snowflake and S3 with graceful fallback |
| LLM | `src/llm/` | Structured-payload client, claim verification |
| Services | `src/api/`, `api_node/` | FastAPI and Node.js layers |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker and Docker Compose (only needed for the containerized stack)

## Live demo

Deployable to [Streamlit Community Cloud](https://share.streamlit.io) with
zero configuration: point it at this repo, main file path
`dashboard/app.py`. The dashboard seeds the bundled ten-document corpus
automatically on first load, so "Run health check" works immediately with no
shell access needed. Snowflake and S3 are unconfigured there, so those writes
fall back to local files, exactly as described in
[Optional integrations](#optional-integrations).

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/seed_corpus.py --reset     # builds the index, ~5 seconds
streamlit run dashboard/app.py            # http://localhost:8501
```

Full stack:

```bash
uvicorn src.api.main:app --reload         # :8000, docs at /docs
npm install && npm start                  # :3001
python -m src.monitoring.scheduler        # hourly health checks
```

Containers:

```bash
docker compose up --build
# postgres :5432   api :8000   node :3001   dashboard :8501   scheduler
```

Tests:

```bash
pytest tests/ -v                          # 126 tests
node api_node/smoke.js                    # requires both services running
```

---

## The Claude layer

Claude receives a JSON payload of pre-computed metrics and nothing else. No raw
chunk text, no source documents, no ability to recalculate. It writes the
plain-language handover a non-technical operations manager reads.

Because the payload is fixed and known, every number Claude writes can be traced
back to it. The hallucination guard extracts each figure, matches it against the
payload in both direct and percentage form, and labels it VERIFIED,
APPROXIMATION, or UNVERIFIED. Anything untraceable was invented.

The API key is pasted into the dashboard per session, held in memory, and never
written to disk or read from the environment.

---

## Optional integrations

Snowflake and S3 writes activate when credentials are present and fall back to
`data/artifacts/` when they are not. The pipeline never fails because a connector
is unconfigured.

| Target | Enable with | Fallback |
|---|---|---|
| Snowflake | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` | `snowflake_fallback.jsonl` |
| S3 | `S3_BUCKET` | `data/artifacts/` |
| Transformer embeddings | `EMBEDDER_BACKEND=sentence_transformers` | TF-IDF + SVD |
| Postgres | `DATABASE_URL` | SQLite |

---

## Honest limitations

The default embedder is TF-IDF with truncated SVD. It is real vectorization and
real cosine retrieval, but it has no contextual understanding beyond
co-occurrence statistics in the fitted corpus. Set
`EMBEDDER_BACKEND=sentence_transformers` for transformer embeddings; the vector
store and every monitoring module are backend-agnostic and need no changes.

The bundled corpus is ten documents written for this project. The pipeline reads
any text; swap `scripts/seed_corpus.py` for a loader against real content.

Integration checks measure this system's own retrieval path. Wiring them to an
external service means replacing the `query_logs` source in
`src/monitoring/integration.py`.

The nodal basis impact of retrieval quality on business outcomes is not modelled.
This measures whether the system is answering correctly, not what a wrong answer
costs.

---

## License

MIT — see [LICENSE](LICENSE).

---

Sejal Khade · [github.com/SejalKhade](https://github.com/SejalKhade)
