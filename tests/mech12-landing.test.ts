import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 247 (crt2, CORRECTIONS 319), appended at campaign commit a32bd86 and imported as MT247 through
// scripts/register_model.py MECH12_ROWS. That step edited header line 3 and appended the row; it amended no earlier row and
// did not touch line 5. The ingest d597d00 appended the batch's 45 exclusion rows, ALL one-kind ARGS_WD_BASE rows at 5e-4 --
// crt2 ran the UNPATCHED pinned tree, so no run prints an ON line and no row is two-axis.
//
// MT247 is OPEN: its primary RETUNE-UNDECIDED licenses neither "beats" nor "ties (a bound)", only the interval and the grid,
// and it carries the registered LOC-PARTIAL clause that no grain's optimum was located. MT241 (crt1), whose sentence it
// supersedes, and MT175 (the audit headline) were NOT amended; the relations live in content/later-evidence.json.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const later = JSON.parse(readFileSync(new URL('../content/later-evidence.json', import.meta.url), 'utf8')) as
  { earlier: string; later: string; relation: string; source: string; quote: string; reason: string }[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const crt2 = byId.get('MT247')!;

test('MT247 carries the outcome the documented rules give it', () => {
  assert.deepEqual(data.experiments.slice(-1).map(e => e.id), ['MT247'], 'appended after every existing record');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [184, 166, 18, 3596]);
  assert.deepEqual([crt2.section, crt2.area, crt2.kind, crt2.outcome, crt2.corrected, crt2.batches],
    [10, 'Count-matched partition audit', 'research', 'unresolved', false, ['crt2']]);
  assert.ok(crt2.reason.startsWith('Verdict: RETUNE-UNDECIDED + LOC-PARTIAL:ch=NOT-LOCATED-MSHI,nd=NOT-LOCATED-MSHI,k01=NOT-LOCATED-A0HI-CEILING + TUNED-UNRESOLVED + ORACLE-PARTITION-TIES-SCALAR'), crt2.reason);
  assert.deepEqual([crt2.figureIds, crt2.eventIds], [['page-8'], ['phase-22']]);
  assert.ok(crt2.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 247), 'MT247 anchors MASTER-TABLE line 247');
  assert.deepEqual(crt2.warningIds, ['warning-MT247-verdict', 'warning-MT247-args-deviation', 'warning-MT247-source-1']);
  assert.ok(!data.warnings.some(w => /-amendment-319$/.test(w.id)), 'the landing amended no row');
});

test('MT247 is Open: nothing is located, and the tuned contrast is an interval containing zero', () => {
  assert.match(crt2.reason, /the primary is unresolved AND nothing is located/);
  assert.match(crt2.reason, /ALL THREE GRAINS SELECTED AT A GRID EDGE/);
  assert.match(crt2.reason, /T_ch -0\.0373 pp and T_nd -0\.1407 pp at the TRAIN-selected arms, interval of min T \[-0\.7164, \+0\.4351\]/);
  assert.match(crt2.reason, /the row may say neither beats nor ties \(a bound\); it states the interval and the grid/);
  assert.match(crt2.reason, /the question after tuning until located is NOT answered for that grain/);
  assert.match(crt2.reason, /the count-matched headline is untouched and is NOT removed/);
  // The sigma disclosure the entry leads with is on the record, not only in the entry.
  assert.match(crt2.reason, /the reading is UNRESOLVED only for sigma above 0\.191226/);
  assert.match(crt2.reason, /17\.5x-98\.9x under-decayed/);
  for (const id of ['MT241', 'MT175', 'MT240', 'MT244']) {
    assert.ok(!byId.get(id)!.warningIds.some(wid => wid.includes('amendment-319')), `${id} is not amended`);
  }
  assert.equal(byId.get('MT241')!.outcome, 'mixed');
  assert.equal(byId.get('MT175')!.outcome, 'success');
});

test('every crt2 run is a one-kind ARGS_WD_BASE row, witnessed on its own ARGS line', () => {
  const crt2Runs = runs.filter(run => run.batch === 'crt2');
  assert.equal(crt2Runs.length, 45);
  assert.deepEqual([...new Set(crt2Runs.flatMap(run => run.experimentIds))], ['MT247']);
  for (const run of crt2Runs) {
    const arm = run.parameters.runLabel.split('-')[1];
    assert.equal(run.parameters.argsDeviation, `${arm}: --weight-decay-base 5e-4`, run.id);
    assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=5e-4', run.id);
    assert.equal(run.parameters.argsDeviationAxes, undefined, `${run.id} is not two-axis: crt2 ran the unpatched tree`);
    assert.equal(run.parameters.intervention, undefined, run.id);
    assert.ok(run.parameters.argsDeviationArgsWitness!.startsWith('ARGS: --'), run.id);
  }
  assert.equal(new Set(crt2Runs.map(run => run.parameters.runLabel.split('-')[1])).size, 15, 'five configurations x three grains');
});

test('phase-22 is its own documented phase and carries one record', () => {
  const phase = data.activity.find(e => e.id === 'phase-22')!;
  assert.deepEqual([phase.experimentIds, phase.kind, phase.date], [['MT247'], 'research-phase', '2026-09-28']);
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 22);
});

test('CORRECTIONS 319.10 seeds exactly two later-evidence relations, and crt1 is superseded rather than amended', () => {
  const seeded = later.filter(r => r.later === 'MT247');
  assert.deepEqual(seeded.map(r => [r.earlier, r.later, r.relation]), [['MT241', 'MT247', 'supersedes'], ['MT175', 'MT247', 'qualifies']]);
  assert.ok(!later.some(r => r.later === 'MT247' && ['MT244', 'MT245', 'MT246'].includes(r.earlier)), 'the entry refuses those bearings as relations');
  assert.equal(byId.get('MT241')!.outcome, 'mixed', 'the superseded row keeps its record and its outcome');
});
