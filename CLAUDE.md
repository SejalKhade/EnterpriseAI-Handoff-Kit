# CLAUDE.md — EnterpriseAI Handoff Kit

Context for Claude Code when working in this repository.

## What this is

Post-deployment observability for enterprise RAG systems. Measures whether a
deployed AI system is still answering correctly after the engineering team
leaves. Real ChromaDB, real SQLAlchemy, real retrieval — no simulated data.

## Architecture rules

- `src/config.py` owns every threshold. Never hardcode a number elsewhere.
- Database access goes through `src/database/repository.py`. No SQL in business logic.
- The embedder is an interface. `BaseEmbedder` has two implementations and the
  vector store must never depend on which one is active.
- Connectors always degrade gracefully. A missing credential writes to fallback,
  it never raises.
- Claude receives `build_payload()` output and nothing else. If a value is not
  in that payload, Claude cannot legitimately mention it.

## Scoring model

Two independent staleness signals, applied multiplicatively to cosine similarity:

- AGE, a heuristic. Decays past the threshold, floors at `FRESHNESS_WEIGHT_FLOOR`.
- DRIFT, a verified fact. Hash of source now vs hash captured at embed time.
  Earns the harder `DRIFT_PENALTY` because it is not an estimate.

Retrieval also gates on query coverage. Cosine alone cannot reject out-of-scope
questions once vectors are normalized. See the README section on this.

## When changing retrieval

Run `pytest tests/test_rag.py` first. The tests in `TestFreshnessWeight` and
`TestDriftDetection` encode the intended behaviour. If a change makes them fail,
the change is probably wrong.

Re-run the calibration sweep before altering `RELEVANCE_THRESHOLD`. The current
value of 0.55 was chosen because it maximises after-correct without breaking any
answerable query.

## Files

```
src/config.py                  all thresholds
src/database/models.py         5 tables
src/database/repository.py     data access
src/rag/embedder.py            TF-IDF+SVD and transformer backends
src/rag/vector_store.py        ChromaDB wrapper
src/rag/pipeline.py            chunk, index, retrieve, score
src/monitoring/freshness.py    age and drift scoring
src/monitoring/eval_metrics.py precision/recall/F1/FPR
src/monitoring/integration.py  four reliability checks
src/monitoring/scheduler.py    APScheduler
src/connectors/                Snowflake, S3
src/llm/                       Claude client, hallucination guard
src/orchestrator.py            wires both pipelines
src/api/main.py                FastAPI
api_node/server.js             Node.js layer
dashboard/app.py               Streamlit
```

## Commands

```bash
python scripts/seed_corpus.py --reset   # rebuild index
pytest tests/ -q                        # 126 tests
uvicorn src.api.main:app --reload       # :8000
npm start                               # :3001
streamlit run dashboard/app.py          # :8501
docker compose up --build               # all services
```

## Conventions

- No emojis anywhere in output or UI.
- Every metric shown to a user carries its formula and its source.
- Type hints on all public functions.
- Docstrings explain why, not what.
