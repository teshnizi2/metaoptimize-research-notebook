import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import Papa from 'papaparse';
import type { ActivityEvent, Experiment, ResearchData } from '../src/types.ts';

async function helpers() {
  const module = await import('../src/lib/timeline.ts').catch(() => null);
  assert.ok(module, 'The homepage needs a timeline derived from documented phase events.');
  return module;
}
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const phase = (id: string, date: string, experimentIds: string[] = []): ActivityEvent => ({
  id, date, kind: 'research-phase', title: id, experimentIds,
  detail: 'Documented phase: 18-20 Aug 2026. Test: Compare matched controls. Result: +0.49 pp; overlapping seeds. Next question: Test scope. Date marks the start of the documented phase window, not individual run launch dates.',
});
const experiment = (id: string, outcome: Experiment['outcome'] = 'success') => ({ id, outcome }) as Experiment;
const normalize = (text: string) => text.replace(/\s+/g, ' ').replace(/[.\s]+$/, '').trim();

test('newest-first timeline retains documented windows and every original CSV observation', async () => {
  const { researchTimeline } = await helpers();
  const rows = researchTimeline(data.activity, data.experiments);
  const csv = Papa.parse<Record<string, string>>(readFileSync(new URL('../public/assets/tables/research_timeline.csv', import.meta.url), 'utf8'), { header: true, skipEmptyLines: true }).data;
  assert.equal(rows.length, 8);
  assert.deepEqual(rows.map(row => row.id), Array.from({ length: 8 }, (_, i) => `phase-0${8 - i}`));
  rows.forEach(row => {
    const original = csv.find(entry => entry.phase.split('\n')[1] === row.title)!;
    assert.ok(original, `${row.title} must preserve a recorded CSV phase`);
    const [period, title] = original.phase.split('\n');
    assert.equal(row.period, `${period} 2026`);
    assert.equal(row.title, title);
    assert.equal(normalize(row.test!), normalize(original.test));
    assert.equal(normalize(row.observation!), normalize(original.observed_result));
    assert.equal(normalize(row.nextQuestion!), normalize(original.next_question));
  });
});

test('coverage deduplicates overlaps without dating unlinked records from the snapshot import', async () => {
  const { researchTimeline, filterByResearchPhase } = await helpers();
  const rows = researchTimeline(data.activity, data.experiments);
  const phaseIds = new Set(rows.flatMap(row => row.experiments.map(entry => entry.id)));
  const unlinked = filterByResearchPhase(data.experiments, data.activity, 'unlinked');
  assert.equal(rows.reduce((sum, row) => sum + row.experiments.length, 0), 32);
  assert.equal(phaseIds.size, 31);
  assert.equal(unlinked.length, 80);
  assert.ok(unlinked.every(entry => !phaseIds.has(entry.id)));
  assert.equal(phaseIds.size + unlinked.length, data.experiments.length);
  assert.ok(rows.filter(row => row.experiments.some(entry => entry.id === 'MT098')).length === 2);
  assert.deepEqual(filterByResearchPhase(data.experiments, data.activity, 'phase-08').map(entry => entry.id), ['CVK2']);
});

test('current verdict counts remain separate from favorable descriptive observations', async () => {
  const { researchTimeline } = await helpers();
  const rows = researchTimeline(data.activity, data.experiments);
  assert.deepEqual(rows.find(row => row.id === 'phase-03')!.counts, { success: 0, fail: 1, mixed: 0, unresolved: 0, correction: 0 });
  assert.deepEqual(rows.find(row => row.id === 'phase-06')!.counts, { success: 1, fail: 0, mixed: 1, unresolved: 1, correction: 0 });
  assert.equal(rows.find(row => row.id === 'phase-08')!.counts.unresolved, 1);
  assert.equal(rows.find(row => row.id === 'phase-08')!.counts.success, 0);
});

test('phase filters use exact explicit links, deduplicate links and reject unknown phases', async () => {
  const { researchTimeline, filterByResearchPhase } = await helpers();
  const entries = [experiment('MT1'), experiment('MT10', 'fail'), experiment('Other')];
  const events = [phase('first', '2026-08-18', ['MT1', 'MT1', 'missing']), phase('second', '2026-08-19', ['MT1', 'MT10'])];
  const before = JSON.stringify({ entries, events });
  assert.deepEqual(filterByResearchPhase(entries, events, 'first').map(entry => entry.id), ['MT1']);
  assert.deepEqual(filterByResearchPhase(entries, events, 'second').map(entry => entry.id), ['MT1', 'MT10']);
  assert.deepEqual(filterByResearchPhase(entries, events, 'unlinked').map(entry => entry.id), ['Other']);
  assert.deepEqual(filterByResearchPhase(entries, events, 'missing'), []);
  assert.deepEqual(filterByResearchPhase(entries, events), entries);
  assert.equal(researchTimeline(events, entries).find(row => row.id === 'first')!.experiments.length, 1);
  assert.equal(JSON.stringify({ entries, events }), before);
});

test('non-phase events never create test periods, and incomplete phases stay visible without invented dates', async () => {
  const { researchTimeline } = await helpers();
  const events = [
    { ...phase('publication', '2026-09-15'), kind: 'publication-snapshot' },
    { ...phase('completed', '2026-09-14'), kind: 'completed-experiment' },
    { ...phase('unknown-window', '2026-08-19'), detail: 'Only a partial record survives.' },
    phase('later', '2026-08-20'), phase('earlier', '2026-08-18'),
    phase('invalid-date', '2026-02-30'),
  ];
  const rows = researchTimeline(events, []);
  assert.deepEqual(rows.map(row => row.id), ['later', 'unknown-window', 'earlier', 'invalid-date']);
  assert.equal(rows.find(row => row.id === 'unknown-window')!.period, null);
  assert.equal(rows.find(row => row.id === 'unknown-window')!.test, null);
  assert.equal(rows.find(row => row.id === 'invalid-date')!.startDate, null);
  assert.deepEqual(researchTimeline([], []), []);
});

test('malformed phase detail and ID lists cannot crash pages or disagree with phase filters', async () => {
  const { researchTimeline, filterByResearchPhase } = await helpers();
  const entries = [experiment('MT1')];
  for (const detail of [undefined, null, 42, { text: 'Partial note' }]) {
    for (const experimentIds of [undefined, null, 'MT1', { id: 'MT1' }, ['MT1', null, 2, 'MT1']]) {
      const events = [{ ...phase('partial', '2026-08-18'), detail, experimentIds }] as unknown as ActivityEvent[];
      const rows = researchTimeline(events, entries);
      const expected = Array.isArray(experimentIds) ? ['MT1'] : [];
      assert.equal(rows[0].period, null);
      assert.deepEqual(rows[0].experiments.map(entry => entry.id), expected);
      assert.deepEqual(filterByResearchPhase(entries, events, 'partial').map(entry => entry.id), expected);
      assert.equal(filterByResearchPhase(entries, events, 'unlinked').length, 1 - expected.length);
    }
  }
});

test('explicit phase links remain stable as the log grows and honor excluding filters', async () => {
  const { researchTimeline, resolvePhaseDetail } = await helpers();
  assert.equal(typeof resolvePhaseDetail, 'function');
  const phases = researchTimeline(data.activity, data.experiments);
  const event = data.activity.find(entry => entry.id === 'phase-08')!;
  const note = { ...event, id: 'review-note', kind: 'note', detail: 'Reviewed phase-08 scope' };
  assert.equal(resolvePhaseDetail(phases, [event, note], 'phase-08', 'phase')?.id, 'phase-08', 'Later notes must not break an existing detail link');
  assert.equal(resolvePhaseDetail(phases, [event], 'phase-08', 'phase')?.id, 'phase-08');
  assert.equal(resolvePhaseDetail(phases, [event], 'phase-08'), undefined, 'An ordinary search remains a log search');
  assert.equal(resolvePhaseDetail(phases, [], 'phase-08', 'phase'), undefined, 'Excluding type or experiment filters must not expose a detail');
  assert.equal(resolvePhaseDetail(phases, [event], 'VGG', 'phase'), undefined);
  assert.equal(resolvePhaseDetail(phases, [event], 'phase-0', 'phase'), undefined);
  assert.equal(resolvePhaseDetail(phases, [note], 'phase-08', 'phase'), undefined);
  assert.equal(resolvePhaseDetail(phases, [{ ...event, kind: 'note' }], 'phase-08', 'phase'), undefined);
  assert.equal(resolvePhaseDetail([], [event], 'phase-08', 'phase'), undefined);
});
