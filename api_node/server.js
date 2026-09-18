/**
 * EnterpriseAI Handoff Kit — Node.js API Layer
 *
 * This is the integration seam between the Python AI service and the enterprise
 * JavaScript stack. It does three things a raw proxy does not:
 *
 *   1. Reshapes Python responses into the flat, camelCase payloads that
 *      enterprise frontends and BI tools expect
 *   2. Adds a short-lived response cache so dashboard refreshes do not
 *      re-trigger expensive evaluation runs
 *   3. Degrades gracefully when the Python service is unreachable, returning
 *      a structured error rather than a stack trace
 *
 * Run:   npm install && npm start
 * Env:   PYTHON_API_URL (default http://localhost:8000)
 *        PORT           (default 3001)
 */

const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API = process.env.PYTHON_API_URL || 'http://localhost:8000';
const CACHE_TTL_MS = 30_000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ── Request logging ──────────────────────────────────────────────────────────
app.use((req, res, next) => {
  const started = Date.now();
  res.on('finish', () => {
    console.log(
      `${new Date().toISOString()} ${req.method} ${req.path} ` +
      `${res.statusCode} ${Date.now() - started}ms`
    );
  });
  next();
});

// ── Tiny TTL cache ───────────────────────────────────────────────────────────
const cache = new Map();

function cacheGet(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry.value;
}

function cacheSet(key, value) {
  cache.set(key, { at: Date.now(), value });
}

// ── Python service client ────────────────────────────────────────────────────
async function callPython(path, { method = 'GET', body = null } = {}) {
  const url = `${PYTHON_API}${path}`;
  const options = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) options.body = JSON.stringify(body);

  const response = await fetch(url, options);
  const text = await response.text();

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw Object.assign(new Error('Python service returned non-JSON'), {
      statusCode: 502,
      detail: text.slice(0, 400),
    });
  }

  if (!response.ok) {
    throw Object.assign(new Error('Python service error'), {
      statusCode: response.status,
      detail: parsed.detail || parsed,
    });
  }
  return parsed;
}

function fail(res, err) {
  const status = err.statusCode || 503;
  res.status(status).json({
    status: 'error',
    message: err.message,
    detail: err.detail || null,
    pythonApi: PYTHON_API,
    hint: status === 503
      ? 'The Python service is unreachable. Start it with: uvicorn src.api.main:app'
      : undefined,
  });
}

// ── Routes ───────────────────────────────────────────────────────────────────

app.get('/api/status', async (req, res) => {
  try {
    const upstream = await callPython('/api/status');
    res.json({
      status: 'operational',
      service: 'EnterpriseAI Handoff Kit — Node API',
      version: '2.0.0',
      pythonApi: PYTHON_API,
      vectorStore: {
        collection: upstream.vector_store.collection,
        chunkCount: upstream.vector_store.chunk_count,
        embedder: upstream.vector_store.embedder_backend,
        dimensions: upstream.vector_store.embedding_dim,
      },
      corpus: {
        documents: upstream.corpus.documents,
        chunks: upstream.corpus.embedding_chunks,
        queryLogs: upstream.corpus.query_logs,
      },
      connectors: {
        snowflakeEnabled: upstream.snowflake.enabled,
        snowflakeTarget: `${upstream.snowflake.database}.${upstream.snowflake.schema}`,
        s3Enabled: upstream.s3.enabled,
        s3Bucket: upstream.s3.bucket,
      },
    });
  } catch (err) {
    fail(res, err);
  }
});

app.get('/api/health', async (req, res) => {
  const cached = cacheGet('health');
  if (cached) return res.json({ ...cached, cached: true });

  try {
    const upstream = await callPython('/api/health');
    const fr = upstream.freshness;
    const ig = upstream.integration;

    const payload = {
      status: 'success',
      timestamp: new Date().toISOString(),
      vectorDb: {
        overallHealth: fr.overall_health,
        totalDocuments: fr.total_documents,
        critical: fr.critical_count,
        stale: fr.stale_count,
        fresh: fr.fresh_count,
        contentDrift: fr.content_drift_count,
        averageFreshness: fr.average_freshness,
        reindexQueue: fr.reindex_queue,
      },
      integration: {
        overallStatus: ig.overall_status,
        checksPassed: ig.checks_passed,
        totalChecks: ig.total_checks,
        avgLatencyMs: ig.avg_latency_ms,
        errorRatePct: ig.error_rate_pct,
        checks: (ig.checks || []).map((c) => ({
          dimension: c.dimension,
          value: c.value,
          threshold: c.threshold,
          status: c.status,
        })),
      },
      cached: false,
    };
    cacheSet('health', payload);
    res.json(payload);
  } catch (err) {
    fail(res, err);
  }
});

app.post('/api/evaluate', async (req, res) => {
  const threshold = req.body?.stalenessThreshold ?? null;
  const key = `evaluate:${threshold}`;
  const cached = cacheGet(key);
  if (cached) return res.json({ ...cached, cached: true });

  try {
    const upstream = await callPython('/api/evaluate', {
      method: 'POST',
      body: {
        staleness_threshold: threshold,
        write_external: req.body?.writeExternal ?? true,
      },
    });

    const ev = upstream.evaluation;
    const payload = {
      status: 'success',
      timestamp: upstream.timestamp,
      stalenessThreshold: upstream.staleness_threshold,
      totalQueries: ev.total_queries,
      before: {
        precision: ev.before.precision,
        recall: ev.before.recall,
        f1Score: ev.before.f1_score,
        falsePositiveRate: ev.before.false_positive_rate,
        accuracy: ev.before.accuracy,
        confusionMatrix: ev.before.confusion_matrix,
      },
      after: {
        precision: ev.after.precision,
        recall: ev.after.recall,
        f1Score: ev.after.f1_score,
        falsePositiveRate: ev.after.false_positive_rate,
        accuracy: ev.after.accuracy,
        confusionMatrix: ev.after.confusion_matrix,
      },
      delta: {
        precision: ev.delta.precision_delta,
        recall: ev.delta.recall_delta,
        f1Score: ev.delta.f1_delta,
        falsePositiveRate: ev.delta.false_positive_rate_delta,
        accuracy: ev.delta.accuracy_delta,
        chunksDemoted: ev.delta.stale_chunks_filtered,
        queriesAffected: ev.delta.queries_affected,
      },
      summary: ev.improvement_summary,
      cached: false,
    };
    cacheSet(key, payload);
    res.json(payload);
  } catch (err) {
    fail(res, err);
  }
});

app.post('/api/reindex', async (req, res) => {
  const docKeys = req.body?.docKeys;
  if (!Array.isArray(docKeys) || docKeys.length === 0) {
    return res.status(400).json({
      status: 'error',
      message: 'docKeys must be a non-empty array',
    });
  }
  try {
    const upstream = await callPython('/api/reindex', {
      method: 'POST',
      body: { doc_keys: docKeys },
    });
    cache.clear();
    res.json({
      status: 'success',
      reindexed: upstream.reindexed,
      requested: upstream.requested,
      results: upstream.results,
      timestamp: upstream.timestamp,
    });
  } catch (err) {
    fail(res, err);
  }
});

app.post('/api/query', async (req, res) => {
  const q = req.body?.query;
  if (!q) {
    return res.status(400).json({ status: 'error', message: 'query is required' });
  }
  try {
    const upstream = await callPython('/api/query', {
      method: 'POST',
      body: {
        query: q,
        apply_freshness_filter: req.body?.applyFreshnessFilter ?? true,
        staleness_threshold: req.body?.stalenessThreshold ?? null,
      },
    });
    res.json({
      status: 'success',
      query: upstream.query,
      flaggedRelevant: upstream.flagged_relevant,
      decisionReason: upstream.decision_reason,
      topScore: upstream.top_score,
      topRawScore: upstream.top_raw_score,
      inScope: upstream.in_scope,
      latencyMs: upstream.latency_ms,
      hits: (upstream.hits || []).map((h) => ({
        chunkId: h.chunk_id,
        rank: h.rank,
        rawSimilarity: h.raw_similarity,
        adjustedSimilarity: h.adjusted_similarity,
        freshnessWeight: h.freshness_weight,
        penaltyReason: h.penalty_reason,
        embeddingAgeDays: h.embedding_age_days,
        contentDrift: h.content_drift,
        text: h.text,
      })),
    });
  } catch (err) {
    fail(res, err);
  }
});

app.get('/api/history', async (req, res) => {
  try {
    const limit = req.query.limit || 30;
    const upstream = await callPython(`/api/history?limit=${limit}`);
    res.json({
      status: 'success',
      evalRuns: upstream.eval_runs,
      healthSnapshots: upstream.health_snapshots,
    });
  } catch (err) {
    fail(res, err);
  }
});

app.get('/', (req, res) => {
  res.json({
    service: 'EnterpriseAI Handoff Kit — Node.js API Layer',
    version: '2.0.0',
    pythonApi: PYTHON_API,
    endpoints: [
      'GET  /api/status',
      'GET  /api/health',
      'POST /api/evaluate',
      'POST /api/query',
      'POST /api/reindex',
      'GET  /api/history',
    ],
  });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Node API listening on http://localhost:${PORT}`);
    console.log(`Proxying to Python service at ${PYTHON_API}`);
  });
}

module.exports = app;
