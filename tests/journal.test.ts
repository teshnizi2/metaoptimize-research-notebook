import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const project = fileURLToPath(new URL('../', import.meta.url));
const valid = { id: 'J-FIRST', date: '2026-09-15T09:00:00.000Z', kind: 'note', title: 'Metric clarification', detail: 'Use the last five completed epochs.', experimentIds: ['MT014'] };
function fixture(t: { after: (fn: () => void) => void }) {
  const root = mkdtempSync(join(tmpdir(), 'metaopt-journal-'));
  mkdirSync(join(root, 'content'), { recursive: true });
  mkdirSync(join(root, 'public/data'), { recursive: true });
  writeFileSync(join(root, 'content/journal.json'), '[]\n');
  writeFileSync(join(root, 'public/data/journal.json'), '[]\n');
  writeFileSync(join(root, 'public/data/research.json'), JSON.stringify({ experiments: [{ id: 'MT014' }, { id: 'MT098' }, { id: 'CVK2' }], activity: [{ id: 'BASE-EVENT' }] }));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return root;
}
function cli(root: string, args: string[] = [], script = 'journal.mjs') {
  const result = spawnSync(process.execPath, [resolve(project, 'scripts', script), ...args], { cwd: root, encoding: 'utf8', timeout: 30000 });
  assert.ifError(result.error);
  return result;
}
function command(id = 'J-FIRST') { return ['--type', 'note', '--title', valid.title, '--text', valid.detail, '--experiments', 'MT014,CVK2', '--id', id, '--date', valid.date]; }
function journal(root: string) { return JSON.parse(readFileSync(join(root, 'content/journal.json'), 'utf8')); }

test('CLI appends durable entries across processes and preserves existing records', t => {
  const root = fixture(t);
  const first = cli(root, command()); assert.equal(first.status, 0, first.stderr);
  const original = journal(root)[0];
  const second = cli(root, [...command('J-SECOND').map(x => x === 'note' ? 'warning' : x)]); assert.equal(second.status, 0, second.stderr);
  assert.equal(journal(root).length, 2); assert.deepEqual(journal(root)[0], original);
  assert.deepEqual(journal(root)[1].experimentIds, ['MT014', 'CVK2']); assert.equal(journal(root)[1].kind, 'warning');
});
test('CLI rejects unknown experiment IDs without changing stored history', t => {
  const root = fixture(t), before = readFileSync(join(root, 'content/journal.json'), 'utf8');
  const result = cli(root, command().map(x => x === 'MT014,CVK2' ? 'MT999' : x));
  assert.notEqual(result.status, 0); assert.match(result.stderr, /unknown experiment/i);
  assert.equal(readFileSync(join(root, 'content/journal.json'), 'utf8'), before);
});
test('CLI rejects unsupported types and blank content', t => {
  const root = fixture(t);
  for (const [old, replacement, message] of [['note', 'success', /type|kind/i], [valid.title, '  ', /title/i], [valid.detail, '', /text|detail/i]] as const) {
    const result = cli(root, command().map(x => x === old ? replacement : x));
    assert.notEqual(result.status, 0); assert.match(result.stderr, message); assert.deepEqual(journal(root), []);
  }
});
test('CLI rejects duplicate entry IDs in journal and base research history', t => {
  const root = fixture(t); writeFileSync(join(root, 'content/journal.json'), JSON.stringify([valid]));
  for (const id of ['J-FIRST', 'BASE-EVENT']) { const result = cli(root, command(id)); assert.notEqual(result.status, 0); assert.match(result.stderr, /duplicate|already exists/i); }
  assert.deepEqual(journal(root), [valid]);
});
test('CLI rejects invalid dates, duplicate linked IDs and unsupported flags', t => {
  const root = fixture(t);
  for (const args of [command().map(x => x === valid.date ? '2026-02-30' : x), command().map(x => x === 'MT014,CVK2' ? 'MT014,MT014' : x), [...command(), '--publish']]) {
    const result = cli(root, args); assert.notEqual(result.status, 0); assert.match(result.stderr, /invalid|duplicate|unknown|unsupported/i);
  }
  assert.deepEqual(journal(root), []);
});
test('CLI rejects explicitly empty optional IDs and dates instead of silently replacing them', t => {
  const root = fixture(t);
  for (const option of ['--id', '--date']) {
    const args = command(); args[args.indexOf(option) + 1] = '';
    const result = cli(root, args); assert.notEqual(result.status, 0); assert.match(result.stderr, /invalid/i);
    assert.deepEqual(journal(root), []);
  }
});
test('append and sync reject private infrastructure and credentials without exposing or storing them', t => {
  const root = fixture(t);
  const privateValues = [
    '/' + ['home', 'synthetic-review-user', 'results.log'].join('/'),
    '/' + ['Users', 'example-user', 'research'].join('/'),
    ['C:', 'Users', 'example-user', 'research'].join('\\'),
    'API' + '_KEY=synthetic-secret-value',
    '"API' + '_KEY": "example-credential"',
    ['access', 'token'].join('_') + ': "example-credential"',
    ['example-person', 'example.invalid'].join('@'),
    ['192', '168', '33', '10'].join('.'),
    ['2001', 'db8', '', '1'].join(':'),
    'ssh ' + ['example-user', 'example-host'].join('@'),
    'cluster' + '_account: example-login',
  ];
  for (const value of privateValues) {
    const append = cli(root, command().map(x => x === valid.detail ? value : x));
    assert.notEqual(append.status, 0); assert.match(append.stderr, /private|sensitive/i); assert.ok(!append.stderr.includes(value));
    assert.deepEqual(journal(root), []);
    writeFileSync(join(root, 'content/journal.json'), JSON.stringify([{ ...valid, detail: value }]));
    const publish = cli(root, [], 'sync-journal.mjs');
    assert.notEqual(publish.status, 0); assert.match(publish.stderr, /private|sensitive/i); assert.ok(!publish.stderr.includes(value));
    assert.equal(readFileSync(join(root, 'public/data/journal.json'), 'utf8'), '[]\n');
    writeFileSync(join(root, 'content/journal.json'), '[]\n');
  }
});
test('journal privacy guard accepts scientific values and public repository references', t => {
  const root = fixture(t);
  const detail = 'CVK2: mean 70.044%; alpha0=1e-6; beta=-3; cut 19; registered bar 0.957313 pp. Source: https://github.com/example/research/blob/main/score.py';
  const result = cli(root, command().map(x => x === valid.detail ? detail : x));
  assert.equal(result.status, 0, result.stderr); assert.equal(journal(root)[0].detail, detail);
});
test('CLI refuses corrupt existing JSON rather than replacing it', t => {
  const root = fixture(t); writeFileSync(join(root, 'content/journal.json'), '{broken');
  const result = cli(root, command()); assert.notEqual(result.status, 0); assert.match(result.stderr, /journal|JSON/i);
  assert.equal(readFileSync(join(root, 'content/journal.json'), 'utf8'), '{broken');
});
test('CLI generates unique IDs and a real recording timestamp when omitted', t => {
  const root = fixture(t), start = Date.now();
  const args = command().slice(0, 8);
  for (let i = 0; i < 2; i++) { const result = cli(root, args); assert.equal(result.status, 0, result.stderr); }
  const rows = journal(root); assert.notEqual(rows[0].id, rows[1].id);
  for (const row of rows) { assert.ok(Date.parse(row.date) >= start); assert.ok(Date.parse(row.date) <= Date.now()); }
});
test('sync publishes a valid journal and accepts an empty initial journal', t => {
  const root = fixture(t); assert.equal(cli(root, [], 'sync-journal.mjs').status, 0);
  writeFileSync(join(root, 'content/journal.json'), JSON.stringify([valid]));
  const result = cli(root, [], 'sync-journal.mjs'); assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(readFileSync(join(root, 'public/data/journal.json'), 'utf8')), [valid]);
});
test('sync validates links before publishing and leaves public history intact on failure', t => {
  const root = fixture(t); writeFileSync(join(root, 'content/journal.json'), JSON.stringify([{ ...valid, experimentIds: ['MT999'] }]));
  const result = cli(root, [], 'sync-journal.mjs'); assert.notEqual(result.status, 0); assert.match(result.stderr, /unknown experiment/i);
  assert.equal(readFileSync(join(root, 'public/data/journal.json'), 'utf8'), '[]\n');
});
test('sync and append refuse deletion or rewriting of already published entries', t => {
  const root = fixture(t); writeFileSync(join(root, 'public/data/journal.json'), JSON.stringify([valid]));
  for (const rows of [[], [{ ...valid, detail: 'Rewritten history' }]]) {
    writeFileSync(join(root, 'content/journal.json'), JSON.stringify(rows));
    for (const result of [cli(root, [], 'sync-journal.mjs'), cli(root, command('J-NEXT'))]) {
      assert.notEqual(result.status, 0); assert.match(result.stderr, /append.only|published|immutable/i);
    }
    assert.deepEqual(JSON.parse(readFileSync(join(root, 'public/data/journal.json'), 'utf8')), [valid]);
  }
});
test('exclusive journal lock rejects concurrent writers without deleting their lock', t => {
  const root = fixture(t); writeFileSync(join(root, 'content/.journal.lock'), 'another writer');
  const result = cli(root, command()); assert.notEqual(result.status, 0); assert.match(result.stderr, /lock|progress/i);
  assert.deepEqual(journal(root), []); assert.equal(readFileSync(join(root, 'content/.journal.lock'), 'utf8'), 'another writer');
});

test('ledger combines text, technical-state, dataset and experiment filters', async () => {
  const { filterRunLedger } = await import('../src/pages/Ledger');
  const base = { account: '', batch: 'cvk1', seed: '55', architecture: 'VGG11', epochs: 100, testAccuracy: 68, parameters: { meta_stepsize: '1e-3' } };
  const runs = [{ ...base, id: 'a', jobId: '10', status: 'COMPLETED', dataset: 'CIFAR100', experimentIds: ['CVK2'] }, { ...base, id: 'b', jobId: '11', status: 'CANCELLED', dataset: 'CIFAR10', experimentIds: ['MT014'] }];
  assert.deepEqual(filterRunLedger(runs, { q: 'vgg 1e-3', status: 'COMPLETED', dataset: 'CIFAR100', experiment: 'CVK2' }).map(r => r.id), ['a']);
  assert.deepEqual(filterRunLedger(runs, { q: 'missing' }), []);
});
test('ledger pagination clamps invalid and stale page numbers and uses 50 rows', async () => {
  const { ledgerPage } = await import('../src/pages/Ledger');
  const rows = Array.from({ length: 113 }, (_, i) => i);
  assert.deepEqual(ledgerPage(rows, '2').items, rows.slice(50, 100));
  assert.equal(ledgerPage(rows, '999').page, 3); assert.equal(ledgerPage(rows, '-1').page, 1);
  assert.equal(ledgerPage(rows, 'nope').page, 1); assert.equal(ledgerPage([], '2').page, 1);
});
