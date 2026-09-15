import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

const moduleUrl = new URL('../scripts/sync-artifact-dates.mjs', import.meta.url);
const digest = (x: string | Buffer) => createHash('sha256').update(x).digest('hex');
const at = '2026-09-15T13:27:30Z';
const fact = () => ({ at, precision: 'second', basis: 'First recorded in the notebook Git snapshot.' });
async function implementation() {
  assert.ok(existsSync(moduleUrl), 'The artifact date inventory and validator are not implemented');
  return import(moduleUrl.href);
}
function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'artifact-dates-'));
  const put = (path: string, value: unknown) => {
    const dest = join(root, path); mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, typeof value === 'string' ? value : JSON.stringify(value));
  };
  const body = 'print(1)\n', log = 'Epoch 0, Test Accuracy: 1 %\nRUN_DONE\n';
  put('public/source/s.txt', body); put('public/assets/logs/r.txt', log);
  for (const path of ['figure.webp', 'table.csv', 'report.pdf', 'charts-and-tables.zip']) put(`public/assets/${path}`, path);
  put('public/data/research.json', {
    meta: { snapshotId: 'snapshot', asOf: '2026-09-15', generatedAt: at, downloads: { pdf: '/assets/report.pdf', bundle: '/assets/charts-and-tables.zip' } },
    figures: [{ id: 'f', href: '/assets/figure.webp', experimentIds: ['CVK2'] }],
    tables: [{ id: 't', href: '/assets/table.csv', experimentIds: ['CVK2'] }],
    sources: [{ id: 's', href: '/source/s.txt', originalSha256: digest(body), publicSha256: digest(body), experimentIds: ['CVK2'] }],
    experiments: [{ id: 'CVK2', runIds: ['r'] }],
    warnings: [{ id: 'w', experimentId: 'CVK2', detail: 'Unresolved' }],
    latest: { experimentId: 'CVK2', cells: [{ mean: 1.0, sem: 0.001 }] },
  });
  put('public/data/runs.json', [{ id: 'r', jobId: '1', experimentIds: ['CVK2'], logHref: '/assets/logs/r.txt', parameters: { originalLogSha256: digest(log), publicLogSha256: digest(log) } }]);
  return { root, put, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
function catalog(inventory: any) {
  return { schemaVersion: 1, snapshotId: inventory.snapshotId, snapshotSha256: inventory.snapshotSha256, runsSha256: inventory.runsSha256, publishedAt: inventory.publishedAt, exportedAt: inventory.exportedAt, recordedAt: at,
    records: Object.fromEntries(Object.entries(inventory.records).map(([key, row]: [string, any]) => [key, { kind: row.kind, id: row.id, contentSha256: row.contentSha256, experimentIds: row.experimentIds, firstRecorded: fact() }])) };
}

test('one canonical inventory covers every artifact class, including warnings, chart and downloads', async () => {
  const { buildInventory, canonicalJson } = await implementation(), f = fixture();
  try {
    const inv = buildInventory(f.root);
    assert.deepEqual(Object.keys(inv.records).sort(), ['chart:CVK2', 'download:bundle', 'download:pdf', 'experiment:CVK2', 'figure:f', 'run:r', 'source:s', 'table:t', 'warning:w']);
    assert.equal(inv.records['figure:f'].contentSha256, digest('figure.webp'));
    assert.equal(canonicalJson({ z: [1.0, { b: 2, a: 1 }], a: 0.001 }), '{"a":0.001,"z":[1,{"a":1,"b":2}]}');
    assert.deepEqual(inv.records['chart:CVK2'].runIds, ['r']);
    assert.ok(!Object.keys(inv.records).some(key => key.includes('source.zip')));
  } finally { f.cleanup(); }
});

test('validation rejects omitted, unknown, stale and reassociated artifact records', async () => {
  const { buildInventory, validateCatalog } = await implementation(), f = fixture();
  try {
    const inv = buildInventory(f.root), good = catalog(inv);
    assert.doesNotThrow(() => validateCatalog(good, inv));
    for (const mutate of [
      (x: any) => delete x.records['warning:w'],
      (x: any) => x.records['warning:invented'] = x.records['warning:w'],
      (x: any) => x.records['figure:f'].contentSha256 = '0'.repeat(64),
      (x: any) => x.records['table:t'].experimentIds = [],
      (x: any) => x.snapshotSha256 = '0'.repeat(64),
      (x: any) => x.runsSha256 = '0'.repeat(64),
    ]) { const broken = structuredClone(good); mutate(broken); assert.throws(() => validateCatalog(broken, inv)); }
  } finally { f.cleanup(); }
});

test('facts reject impossible calendar dates, absent timezones and unsafe or missing provenance links', async () => {
  const { buildInventory, validateCatalog } = await implementation(), f = fixture();
  try {
    const inv = buildInventory(f.root);
    for (const bad of [
      { at: '2026-02-30', precision: 'day' }, { at: '2026-09-15T13:27:30', precision: 'second' },
      { at: '2026-13-01', precision: 'day' }, { at: '2026-09-15', precision: 'second' },
      { at, precision: 'second', href: 'javascript:alert(1)' },
      { at, precision: 'second', href: 'https://github.com.evil.test/a' },
      { at, precision: 'second', href: '/assets/../secret' },
      { at, precision: 'second', href: '/assets/not-present.csv' },
    ]) { const c = catalog(inv); c.records['figure:f'].firstRecorded = { basis: 'Recorded', ...bad } as any; assert.throws(() => validateCatalog(c, inv)); }
    for (const good of [{ at: '2024-02-29', precision: 'day' }, { at: '2026-09-16T00:01:00+02:00', precision: 'second' }]) {
      const c = catalog(inv); c.records['figure:f'].firstRecorded = { basis: 'Recorded', ...good }; assert.doesNotThrow(() => validateCatalog(c, inv));
    }
  } finally { f.cleanup(); }
});

test('source bytes and original-provenance changes cannot keep stale content hashes', async () => {
  const { buildInventory, validateCatalog } = await implementation(), f = fixture();
  try {
    const initial = buildInventory(f.root), c = catalog(initial);
    const research = JSON.parse(readFileSync(join(f.root, 'public/data/research.json'), 'utf8'));
    research.sources[0].originalSha256 = 'b'.repeat(64); f.put('public/data/research.json', research);
    const changed = buildInventory(f.root);
    assert.notEqual(changed.records['source:s'].contentSha256, initial.records['source:s'].contentSha256);
    assert.throws(() => validateCatalog(c, changed));
    f.put('public/source/s.txt', 'unexpected content'); assert.throws(() => buildInventory(f.root), /source.*hash|hash.*source/i);
  } finally { f.cleanup(); }
});

test('linked run windows require the real dated-run count and time extrema', async () => {
  const { buildInventory, validateCatalog } = await implementation(), f = fixture();
  try {
    const inv = buildInventory(f.root), c: any = catalog(inv);
    const started = { at: '2026-09-14T14:58:01Z', precision: 'second', basis: 'Slurm start' };
    const finished = { at: '2026-09-14T16:00:35Z', precision: 'second', basis: 'Slurm end' };
    c.records['run:r'].started = started; c.records['run:r'].finished = finished;
    c.records['chart:CVK2'].runWindow = { started, finished, datedRuns: 1, totalRuns: 1 };
    assert.doesNotThrow(() => validateCatalog(c, inv));
    c.records['chart:CVK2'].runWindow.datedRuns = 0; assert.throws(() => validateCatalog(c, inv));
    c.records['chart:CVK2'].runWindow.datedRuns = 1; c.records['chart:CVK2'].runWindow.finished = fact(); assert.throws(() => validateCatalog(c, inv));
  } finally { f.cleanup(); }
});

test('sync copies a valid catalog without regenerating dates, and check rejects drift', async () => {
  const { buildInventory, syncCatalog } = await implementation(), f = fixture();
  try {
    const c = catalog(buildInventory(f.root)); f.put('content/artifact-dates.json', c);
    await syncCatalog(f.root);
    const before = readFileSync(join(f.root, 'public/data/artifact-dates.json'), 'utf8');
    await syncCatalog(f.root); assert.equal(readFileSync(join(f.root, 'public/data/artifact-dates.json'), 'utf8'), before);
    await syncCatalog(f.root, true);
    f.put('public/data/artifact-dates.json', { ...c, recordedAt: '2026-09-16T00:00:00Z' });
    await assert.rejects(async () => syncCatalog(f.root, true), /differs|drift|match/);
  } finally { f.cleanup(); }
});
