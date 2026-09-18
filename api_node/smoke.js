/**
 * Smoke test for the Node API layer.
 * Verifies each route responds and that the Python service is reachable.
 *
 * Run:  node api_node/smoke.js
 */

const BASE = process.env.NODE_API_URL || 'http://localhost:3001';

const checks = [
  { name: 'root',     method: 'GET',  path: '/' },
  { name: 'status',   method: 'GET',  path: '/api/status' },
  { name: 'health',   method: 'GET',  path: '/api/health' },
  { name: 'history',  method: 'GET',  path: '/api/history?limit=5' },
  { name: 'query',    method: 'POST', path: '/api/query',
    body: { query: 'What is the refund window?' } },
  { name: 'evaluate', method: 'POST', path: '/api/evaluate',
    body: { stalenessThreshold: 30, writeExternal: false } },
];

async function run() {
  let passed = 0;
  let failed = 0;

  for (const c of checks) {
    const started = Date.now();
    try {
      const options = { method: c.method, headers: { 'Content-Type': 'application/json' } };
      if (c.body) options.body = JSON.stringify(c.body);

      const res = await fetch(`${BASE}${c.path}`, options);
      const body = await res.json();
      const ms = Date.now() - started;

      if (res.ok) {
        passed++;
        console.log(`PASS  ${c.name.padEnd(9)} ${res.status}  ${String(ms).padStart(5)}ms`);
      } else {
        failed++;
        console.log(`FAIL  ${c.name.padEnd(9)} ${res.status}  ${body.message || ''}`);
        if (body.hint) console.log(`      ${body.hint}`);
      }
    } catch (err) {
      failed++;
      console.log(`ERROR ${c.name.padEnd(9)} ${err.message}`);
    }
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

run();
