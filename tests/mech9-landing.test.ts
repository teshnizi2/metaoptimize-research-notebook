import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 243 (cvl1, CORRECTIONS 312), appended at campaign commit a3d422a and imported as MT243 through
// scripts/register_model.py MECH9_ROWS. That step edited header line 3 and appended the row; it amended no earlier row and
// did not touch line 5. The ingest 5545d88 touched results/ alone and appended all 32 exclusion rows of the batch, of TWO
// kinds: 16 one-kind VAL_SPLIT rows (weight decay 0.1) and 16 two-axis ARGS_WD_BASE + VAL_SPLIT rows (5e-4).
//
// MT243 is OPEN, as MT240: its primary VAL-DIFFERS-UNRESOLVED licenses "report both states and the paired intervals; no
// sentence that the ranking does or does not hold on validation". The co-reported selection (SELECT-SAME) is resolved and
// carried in the reason. MT240, whose cell it re-reads, was NOT amended and keeps its outcome.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const cvl1 = byId.get('MT243')!;

const CVL1_FINAL = ['VAL-DIFFERS-UNRESOLVED', 'SELECT-SAME', 'W1-TUNDECIDED/VSURVIVES+W4-TUNDECIDED/VVANISHES+SC-TSCALAR-BEATS-BEST/VSCALAR-BEATS-BEST',
  'SELECT-W1-SAME', 'SELECT-W4-SAME', 'TEST-SIGMA-INBATCH', 'VAL-SIGMA-INBATCH', 'TEST-STATE-DIFFERS-FROM-CGW1-W1',
  'TEST-LEVEL-DIFFERS-FROM-CGW1-chW1', 'TEST-LEVEL-DIFFERS-FROM-CGW1-ndW1', 'TEST-LEVEL-DIFFERS-FROM-CGW1-chW4',
  'TEST-LEVEL-DIFFERS-FROM-CGW1-ndW4', 'TEST-LEVEL-DIFFERS-FROM-CGW1-k01W4',
  'TEST-LEVEL-DIFFERS-FROM-CGW1-kLW4 (primary', 'selection', 'per-claim states', '11 stamps; the 7 registered bounds in the next column)'];

test('MT243 carries the outcome the documented rules give it, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-2, -1).map(e => e.id), ['MT243'], 'appended in line order, before the later g3b row');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  // cvl1 belongs to the count-matched partition audit; Open, as MT240, and it corrects no earlier published claim.
  assert.deepEqual([cvl1.section, cvl1.area, cvl1.kind, cvl1.outcome, cvl1.corrected, cvl1.batches],
    [10, 'Count-matched partition audit', 'research', 'unresolved', false, ['cvl1']]);
  assert.ok(cvl1.reason.startsWith(`Verdict: ${CVL1_FINAL.join(' + ')} Open: VAL-DIFFERS-UNRESOLVED: `), cvl1.reason);
  assert.deepEqual([cvl1.figureIds, cvl1.eventIds], [['page-8'], ['phase-19']]);
  assert.ok(cvl1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 243), 'MT243 anchors MASTER-TABLE line 243');
  // The row owns BOTH an intervention note (VAL_SPLIT) and an ARGS-value note (the two-axis 5e-4 rung).
  assert.ok(cvl1.warningIds.includes('warning-MT243-intervention'));
  assert.ok(cvl1.warningIds.includes('warning-MT243-args-deviation'));
  assert.ok(!data.warnings.some(w => /-amendment-31[1-3]$/.test(w.id)), 'the landing amended no row');
  // cvl1's own runner lives in the harness tree on the cluster, not in the repository (as cgw1's and crt1's do), and the
  // record says so rather than hiding it.
  assert.deepEqual(cvl1.warningIds, ['warning-MT243-verdict', 'warning-MT243-intervention', 'warning-MT243-args-deviation', 'warning-MT243-source-1']);
  assert.equal(data.warnings.find(w => w.id === 'warning-MT243-source-1')!.detail, 'Referenced source not available: jobs/run_cifar_cvl1.sh');
});

test('MT243 is Open: the partition readings change state without a resolved gap, and selection agrees', () => {
  assert.match(cvl1.reason, /the primary is undecided/);
  assert.match(cvl1.reason, /W1 D = chunk777 - nodewise is \+0\.2590 pp \(\+1\.48 SE\) UNDECIDED on test and \+0\.4330 pp \(\+2\.40 SE\) SURVIVES on validation/);
  assert.match(cvl1.reason, /W4 D is \+0\.1660 pp UNDECIDED on test and -0\.0920 pp VANISHES \(a bound\) on validation/);
  assert.match(cvl1.reason, /no sentence that the ranking does or does not hold on validation/);
  assert.match(cvl1.reason, /selecting on validation would have picked the same one, among these 8/);
  assert.match(cvl1.reason, /the meta step size and alpha0 were NOT re-selected on validation/);
  assert.match(cvl1.reason, /dropping s187 gives VAL-AGREES/);
  assert.match(cvl1.scope, /MS-ALPHA0-NOT-RESELECTED/);
  assert.match(cvl1.scope, /NOT licensed: "the rankings hold on validation", "the scalar lead does not depend on the reader"/);
  // cgw1's record is untouched: same outcome, no amendment.
  const cgw1 = byId.get('MT240')!;
  assert.equal(cgw1.outcome, 'unresolved');
  assert.ok(!cgw1.warningIds.some(id => id.includes('amendment')));
});

test('the two warnings name their kinds, and every row of the batch is listed once', () => {
  const wi = data.warnings.find(w => w.id === 'warning-MT243-intervention')!;
  const wa = data.warnings.find(w => w.id === 'warning-MT243-args-deviation')!;
  for (const warning of [wi, wa]) assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(wi.detail, /VAL_SPLIT=5000:302/);
  assert.match(wi.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 5545d88; CORRECTIONS 302, 303, 304 and 312\.$/);
  assert.match(wa.detail, /TWO-AXIS rows \(CORRECTIONS 284\)/);
  assert.match(wa.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 5545d88; CORRECTIONS 263, 284, 304 and 312\.$/);
});

test('the 32 runs link to MT243 with sanitized logs; W1 runs carry the VAL_SPLIT mark, W4 runs the two-axis mark', () => {
  assert.equal(runs.length, 3501);
  const linked = runs.filter(run => run.batch === 'cvl1');
  assert.equal(linked.length, 32);
  assert.deepEqual([...cvl1.runIds].sort(), linked.map(run => run.id).sort());
  for (const run of linked) {
    const arm = run.parameters.runLabel.split('-')[1];
    assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
      [['MT243'], 'Account2', 'completed', 'ResNet18', 'CIFAR10', 100], run.id);
    assert.ok(['184', '185', '186', '187'].includes(run.seed), run.id);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    const lines = log.split('\n');
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    assert.equal(lines.filter(line => /^VAL: epoch \d+ /.test(line)).length, 100, `${run.id} keeps every validation line`);
    assert.equal(lines.filter(line => line.startsWith('VAL_SPLIT: on dataset=CIFAR10 n_val=5000 n_train=45000 ')).length, 1, `${run.id} prints its split witness`);
    if (arm.endsWith('W1')) {
      assert.equal(run.parameters.intervention, `${arm}: VAL_SPLIT=5000:302`, run.id);
      assert.ok(run.parameters.interventionWitness.startsWith('VAL_SPLIT: on dataset=CIFAR10 n_val=5000 '), run.id);
      assert.equal(run.parameters.argsDeviation, undefined, `${run.id} deviates on no ARGS flag`);
    } else {
      assert.equal(run.parameters.intervention, undefined, `${run.id} is an ARGS-value row`);
      assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=5e-4', run.id);
      assert.equal(run.parameters.argsDeviationArgsWitness, 'ARGS: --weight-decay-base 5e-4', run.id);
      assert.equal(run.parameters.argsDeviationAxes, 'VAL_SPLIT=5000:302', run.id);
      assert.ok(run.parameters.interventionAdditionalWitness.startsWith('VAL_SPLIT: on dataset=CIFAR10 n_val=5000 '), run.id);
      assert.match(run.parameters.argsDeviationNote, /TWO-AXIS/);
    }
  }
  assert.equal(runs.filter(run => run.parameters.intervention).length, 199, "183 before, plus cvl1's 16 VAL_SPLIT rows");
  assert.equal(runs.filter(run => run.batch !== 'g3b' && run.parameters.argsDeviation).length, 155, "139 before, plus cvl1's 16 two-axis rows");
});

test('the published arm means reproduce every TEST level the row reads', () => {
  const mean = (arm: string) => {
    const armRuns = runs.filter(run => run.batch === 'cvl1' && run.parameters.runLabel.split('-')[1] === arm);
    assert.equal(armRuns.length, 4, arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['chW1', 'ndW1', 'k01W1', 'kLW1', 'chW4', 'ndW4', 'k01W4', 'kLW4'].map(mean),
    ['91.6045', '91.3455', '91.8845', '92.3140', '86.8265', '86.6605', '89.6360', '88.9275']);
  assert.equal((Number(mean('chW1')) - Number(mean('ndW1'))).toFixed(4), '0.2590');
  assert.equal((Number(mean('chW4')) - Number(mean('ndW4'))).toFixed(4), '0.1660');
});

test('phase-19 is its own documented phase and carries one record', () => {
  const phase19 = data.activity.find(e => e.id === 'phase-19')!;
  assert.deepEqual([phase19.experimentIds, phase19.kind, phase19.date], [['MT243'], 'research-phase', '2026-09-22']);
  assert.match(phase19.title, /Do the audit's rankings hold on a held-out validation split/);
  // phase-20 was added later with g3b (tests/mech10-landing.test.ts).
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 20);
});

test('the landing links the scorer, parser, design and audit tools it cites', () => {
  const paths = new Set(cvl1.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  for (const path of ['analysis/cVL1_valsplit_score.py', 'analysis/cvl1_attack_indep.py', 'analysis/cvl1_design.py', 'bin/cVL1_rule20.sh']) {
    assert.ok(paths.has(path), `MT243 links ${path}`);
  }
});
