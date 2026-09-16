import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ActivityEvent, Experiment, Outcome, ResearchData } from '../src/types.ts';

async function helpers() {
  const module = await import('../src/lib/overview.ts').catch(() => null);
  assert.ok(module, 'The general homepage needs an overview aggregation module.');
  return module;
}
const experiment = (id: string, area: string, outcome: Outcome | null, kind: Experiment['kind'] = 'research') => ({ id, area, outcome, kind }) as Experiment;
const event = (id: string, date: string, kind = 'note'): ActivityEvent => ({ id, date, kind, title: id, detail: 'Recorded update', experimentIds: [] });

test('area summaries count research goals by outcome and list method checks separately', async () => {
  const { areaOutcomeCounts } = await helpers();
  const experiments = [experiment('A', 'Optimization', 'success'), experiment('B', 'Optimization', 'fail'), experiment('C', 'Optimization', 'unresolved'), experiment('D', 'Audit', null, 'method-check'), experiment('E', 'Audit', 'mixed')];
  const areas = [{ id: 2, label: 'Audit', count: 500 }, { id: 1, label: 'Optimization', count: 0 }];
  const input = JSON.stringify({ experiments, areas });
  assert.deepEqual(areaOutcomeCounts(experiments, areas), [
    { label: 'Audit', total: 1, counts: { success: 0, fail: 0, mixed: 1, unresolved: 0 }, methodChecks: 1 },
    { label: 'Optimization', total: 3, counts: { success: 1, fail: 1, mixed: 0, unresolved: 1 }, methodChecks: 0 },
  ]);
  assert.equal(JSON.stringify({ experiments, areas }), input);
});

test('empty known areas and unlisted new areas cannot hide or invent research', async () => {
  const { areaOutcomeCounts } = await helpers();
  const summaries = areaOutcomeCounts([experiment('N', 'New area', 'mixed')], [{ id: 1, label: 'Empty area', count: 99 }]);
  assert.deepEqual(summaries.map(row => [row.label, row.total, row.methodChecks]), [['Empty area', 0, 0], ['New area', 1, 0]]);
  assert.equal(Object.values(summaries[0].counts).reduce((sum, n) => sum + n, 0), 0);
  assert.deepEqual(areaOutcomeCounts([], []), []);
});

test('recent updates are genuine dated notes, corrections and warnings, not imported phase windows', async () => {
  const { recentNotebookUpdates } = await helpers();
  const events = [
    event('phase', '2026-09-20', 'phase-completion'),
    event('publication', '2026-09-19', 'publication-snapshot'),
    event('note', '2026-09-14T12:00:00Z'),
    event('correction', '2026-09-15T08:00:00Z', 'correction'),
    event('warning', '2026-09-15T09:00:00Z', 'warning'),
    event('undated', ''), event('impossible', '2026-02-30'),
    event('no-zone', '2026-09-16T00:00:00'),
  ];
  const input = JSON.stringify(events);
  assert.deepEqual(recentNotebookUpdates(events).map(e => e.id), ['warning', 'correction', 'note']);
  assert.equal(JSON.stringify(events), input);
});

test('recent update order uses timestamp offsets with stable ties and respects the limit', async () => {
  const { recentNotebookUpdates } = await helpers();
  const events = [event('b', '2026-09-15T00:30:00+02:00'), event('a', '2026-09-14T22:30:00Z'), event('day', '2026-09-14')];
  assert.deepEqual(recentNotebookUpdates(events).map(e => e.id), ['a', 'b', 'day']);
  assert.deepEqual(recentNotebookUpdates(events, 1).map(e => e.id), ['a']);
  assert.deepEqual(recentNotebookUpdates(events, 0), []);
  assert.deepEqual(recentNotebookUpdates(events, -1), []);
});

test('the published register is fully represented in the general review', async () => {
  const { areaOutcomeCounts } = await helpers();
  const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
  const rows = areaOutcomeCounts(data.experiments, data.areas);
  assert.equal(rows.length, 10);
  assert.equal(rows.reduce((sum, row) => sum + row.total, 0), 136);
  assert.equal(rows.reduce((sum, row) => sum + row.methodChecks, 0), 18);
  for (const row of rows) {
    assert.equal(Object.values(row.counts).reduce((sum, count) => sum + count, 0), row.total);
    assert.equal(row.total + row.methodChecks, data.experiments.filter(e => e.area === row.label).length);
  }
  const audit = rows.find(row => row.label === 'Count-matched partition audit')!;
  assert.deepEqual([audit.total, audit.counts, audit.methodChecks], [34, { success: 12, fail: 5, mixed: 4, unresolved: 13 }, 3]);
  // MASTER-TABLE lines 212-217: five mechanism rows and one baseline row (cuc1).
  assert.deepEqual([rows.find(row => row.label === 'Mechanism and isolation')!.total, rows.find(row => row.label === 'Mechanism and isolation')!.counts], [40, { success: 12, fail: 12, mixed: 8, unresolved: 8 }]);
  assert.deepEqual([rows.find(row => row.label === 'Baseline comparisons')!.total, rows.find(row => row.label === 'Baseline comparisons')!.counts], [8, { success: 6, fail: 1, mixed: 0, unresolved: 1 }]);
});
