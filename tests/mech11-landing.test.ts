import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 245-246 (crd1, CORRECTIONS 317; cai1, CORRECTIONS 318), appended at campaign commit fc48d02 and
// imported as MT245 and MT246 through scripts/register_model.py MECH11_ROWS. That step edited header line 3 and appended the
// two rows; it amended no earlier row and did not touch line 5. The ONE ingest fecd462 appended the 50 exclusion rows of both
// batches: crd1's 12 shrink-only / trace-only runs are one-kind rows under the NEW DECAY_ROUTE kind (CORRECTIONS 308), and
// crd1's 6 alpha-independent runs and all 32 cai1 runs are TWO-AXIS rows listed by ARGS_WD_BASE with a DECAY_ROUTE clause.
//
// MT245 is GOAL MET: both registered ladders reached registered branches and it is the registered prediction. MT246 is OPEN:
// its primary is unreadable at BOTH doses, the same rule that made MT240, MT243 and MT244 Open. Neither row amends an
// earlier one; the relations they state live in content/later-evidence.json.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const later = JSON.parse(readFileSync(new URL('../content/later-evidence.json', import.meta.url), 'utf8')) as
  { earlier: string; later: string; relation: string; source: string; quote: string; reason: string }[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const crd1 = byId.get('MT245')!, cai1 = byId.get('MT246')!;

test('MT245 and MT246 carry the outcome the documented rules give them', () => {
  assert.deepEqual(data.experiments.slice(-3, -1).map(e => e.id), ['MT245', 'MT246'], 'appended in line order, before the later crt2 row');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [184, 166, 18, 3596]);
  // crd1 answers a mechanism question in its registered direction; cai1's primary is unreadable, so it is Open.
  assert.deepEqual([crd1.section, crd1.area, crd1.kind, crd1.outcome, crd1.corrected, crd1.batches],
    [9, 'Mechanism and isolation', 'research', 'success', false, ['crd1']]);
  assert.deepEqual([cai1.section, cai1.area, cai1.kind, cai1.outcome, cai1.corrected, cai1.batches],
    [10, 'Count-matched partition audit', 'research', 'unresolved', false, ['cai1']]);
  assert.ok(crd1.reason.startsWith('Verdict: ROUTE-IS-WEIGHT-SHRINK + INDEP-NOGAP + S-COLLAPSE+T-NOGAP+I-NOGAP'), crd1.reason);
  assert.ok(cai1.reason.startsWith('Verdict: AI-UNREADABLE + SC-PARTIAL + I5-BOXBOUND+I4-UNHEALTHY'), cai1.reason);
  assert.deepEqual([crd1.figureIds, crd1.eventIds], [['page-19'], ['phase-21']]);
  assert.deepEqual([cai1.figureIds, cai1.eventIds], [['page-8'], ['phase-21']]);
  assert.ok(crd1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 245), 'MT245 anchors MASTER-TABLE line 245');
  assert.ok(cai1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 246), 'MT246 anchors MASTER-TABLE line 246');
  assert.ok(!data.warnings.some(w => /-amendment-31[678]$/.test(w.id)), 'neither landing amended a row');
});

test('MT245 states sufficiency, not necessity, and keeps the undecayed trace term in every route sentence', () => {
  assert.match(crd1.reason, /the alpha-scaled weight shrink with the trace factor removed collapses the scalar step size to at most half its in-batch layerwise level/);
  assert.match(crd1.reason, /with the direct a\*wd\*w term in the trace, undecayed/);
  assert.match(crd1.reason, /SUFFICIENCY, NOT NECESSITY/);
  assert.match(crd1.reason, /TRL 60\.2460, only \+5\.2460 pp above REF_MIN 55/);
  assert.match(crd1.reason, /the alpha-independent arm is NOT a pure route control/);
  assert.match(crd1.reason, /there is no in-batch OFF anchor/);
});

test('MT246 is Open: neither dose is readable, so the headline is neither confirmed nor removed', () => {
  assert.match(cai1.reason, /the primary is unreadable at BOTH rungs/);
  assert.match(cai1.reason, /both LAMBDA rungs are unreadable: no partition reading under alpha-independent decay; the levels themselves are the finding/);
  assert.match(cai1.reason, /no combined scalar sentence/);
  assert.match(cai1.reason, /keeps its at alpha-scaled decay qualifier unchanged and IS NOT REMOVED/);
  // The rows it bears on were NOT amended and keep their outcomes.
  for (const id of ['MT240', 'MT175', 'MT241', 'MT242']) {
    assert.ok(!byId.get(id)!.warningIds.some(wid => wid.includes('amendment-31')), `${id} is not amended`);
  }
  assert.equal(byId.get('MT240')!.outcome, 'unresolved');
});

test('crd1 owns both kinds of exclusion row and cai1 only two-axis rows, each witnessed from its own log', () => {
  const crd1Runs = runs.filter(run => run.batch === 'crd1'), cai1Runs = runs.filter(run => run.batch === 'cai1');
  assert.deepEqual([crd1Runs.length, cai1Runs.length], [18, 32]);
  assert.deepEqual([...new Set(crd1Runs.flatMap(run => run.experimentIds))], ['MT245']);
  assert.deepEqual([...new Set(cai1Runs.flatMap(run => run.experimentIds))], ['MT246']);
  const route = crd1Runs.filter(run => run.parameters.intervention);
  const twoAxis = crd1Runs.filter(run => run.parameters.argsDeviation);
  assert.deepEqual([route.length, twoAxis.length], [12, 6]);
  for (const run of route) {
    const arm = run.parameters.runLabel.split('-')[1];
    const mode = arm.startsWith('SR') ? 'shrink_only' : 'trace_only';
    assert.equal(run.parameters.intervention, `${arm}: DECAY_ROUTE=${mode}`, run.id);
    assert.ok(run.parameters.interventionWitness!.startsWith(`DECAY_ROUTE: on mode=${mode} `), run.id);
    assert.equal(run.parameters.argsDeviation, undefined, run.id);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    assert.ok(log.split('\n').includes(run.parameters.interventionWitness!), `${run.id} log carries its witness line`);
  }
  for (const run of twoAxis) {
    assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=0', run.id);
    assert.equal(run.parameters.argsDeviationAxes, 'DECAY_ROUTE=alpha_indep:3.15e-4', run.id);
    assert.ok(run.parameters.interventionAdditionalWitness!.startsWith('DECAY_ROUTE: on mode=alpha_indep '), run.id);
  }
  for (const run of cai1Runs) {
    const rung = run.parameters.runLabel.split('-')[1].endsWith('I5') ? '5e-5' : '5e-4';
    assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=0', run.id);
    assert.equal(run.parameters.argsDeviationAxes, `DECAY_ROUTE=alpha_indep:${rung}`, run.id);
    assert.equal(run.parameters.intervention, undefined, run.id);
  }
});

test('phase-21 is its own documented phase and carries both records', () => {
  const phase = data.activity.find(e => e.id === 'phase-21')!;
  assert.deepEqual([phase.experimentIds, phase.kind, phase.date], [['MT245', 'MT246'], 'research-phase', '2026-09-22']);
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 22);
});

test('CORRECTIONS 317.10 and 318.10 seed exactly three later-evidence relations, in the vocabulary', () => {
  const seeded = later.filter(r => r.later === 'MT245' || r.later === 'MT246');
  assert.deepEqual(seeded.map(r => [r.earlier, r.later, r.relation]),
    [['MT242', 'MT245', 'qualifies'], ['MT240', 'MT246', 'qualifies'], ['MT175', 'MT246', 'qualifies']]);
  // The bearings the entries explicitly refuse to state as relations are absent.
  assert.ok(!later.some(r => r.later === 'MT245' && r.earlier === 'MT238'), 'MT238 is a premise, not a relation');
  assert.ok(!later.some(r => r.earlier === 'MT245' && r.later === 'MT246'), 'the two rows bear on different cells');
});
