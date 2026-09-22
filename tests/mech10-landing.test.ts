import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 244 (g3b, CORRECTIONS 315), appended at campaign commit 12a4c5d and imported as MT244 through
// scripts/register_model.py MECH10_ROWS. That step edited header line 3 and appended the row; it amended no earlier row and
// did not touch line 5. The ingest 4f7e191 appended the batch's 16 exclusion rows, all one-kind ARGS_WD_BASE rows at 5e-4.
//
// MT244 is OPEN, as MT240 and MT243: its primary UNRESOLVED-BOXBOUND-W4 licenses only "a primary-rung partition arm is
// box-bound; the contrast is not an audit replication". The co-reported scalar (SCALAR-BEATS-BEST) and denominator
// (DENOM-HOLDS) readings are resolved and carried in the reason. MT240, whose question it asks on CIFAR-100, and MT155, whose
// denominator arm it re-runs at 5e-4, were NOT amended and keep their outcomes.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const later = JSON.parse(readFileSync(new URL('../content/later-evidence.json', import.meta.url), 'utf8')) as
  { earlier: string; later: string; relation: string; source: string; quote: string; reason: string }[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const g3b = byId.get('MT244')!;

const G3B_FINAL = ['UNRESOLVED-BOXBOUND-W4', 'SCALAR-BEATS-BEST', 'DENOM-HOLDS', 'W1-SURVIVES+W4-BOXBOUND', 'SIGMA-INBATCH',
  'ANCHOR-MATCHES-POOL', 'W1-LEVEL-DIFFERS-k01', 'K01W1-COLLAPSED', 'LAYERWISE-BELOW-SCALAR-W4 (three branch words', 'rung states',
  '5 stamps; the 6 registered bounds in the next column)'];

test('MT244 carries the outcome the documented rules give it, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-1).map(e => e.id), ['MT244'], 'appended after every existing record');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  // g3b belongs to the count-matched partition audit; Open, as MT240 and MT243, and it corrects no earlier published claim.
  assert.deepEqual([g3b.section, g3b.area, g3b.kind, g3b.outcome, g3b.corrected, g3b.batches],
    [10, 'Count-matched partition audit', 'research', 'unresolved', false, ['g3b']]);
  assert.ok(g3b.reason.startsWith(`Verdict: ${G3B_FINAL.join(' + ')} Open: UNRESOLVED-BOXBOUND-W4: `), g3b.reason);
  assert.deepEqual([g3b.figureIds, g3b.eventIds], [['page-8'], ['phase-20']]);
  assert.ok(g3b.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 244), 'MT244 anchors MASTER-TABLE line 244');
  // The row owns an ARGS-value note (its 5e-4 rung) and no patch-intervention note; its runner lives in the harness tree on
  // the cluster, not in the repository (as cgw1's, crt1's and cvl1's do), and the record says so.
  assert.deepEqual(g3b.warningIds, ['warning-MT244-verdict', 'warning-MT244-args-deviation', 'warning-MT244-source-1']);
  assert.equal(data.warnings.find(w => w.id === 'warning-MT244-source-1')!.detail, 'Referenced source not available: jobs/run_cifar_g3b.sh');
  assert.ok(!data.warnings.some(w => /-amendment-31[45]$/.test(w.id)), 'the landing amended no row');
});

test('MT244 is Open: the 5e-4 partition contrast is box-bound, and the scalar and denominator readings are resolved', () => {
  assert.match(g3b.reason, /the primary is unreadable/);
  assert.match(g3b.reason, /D = chunk771 - nodewise is \+1\.2170 pp \(\+3\.99 SE\) SURVIVES/);
  assert.match(g3b.reason, /D_W4 \(\+0\.0645 pp, \+\/-2 SE \[-0\.5451, \+0\.6741\]\) is not a registered survive, vanish or reverse reading/);
  assert.match(g3b.reason, /a primary-rung partition arm is box-bound; the contrast is not an audit replication/);
  assert.match(g3b.reason, /\+6\.0345 \/ \+6\.0990 pp \(19\.8 \/ 20\.0 SE, SCALAR-BEATS-BEST\), an UNTUNED reading/);
  assert.match(g3b.reason, /\+14\.5857 pp \(\+44\.61 SE\) below the landed tuned SGD \+ cosine \(DENOM-HOLDS, between batches/);
  assert.match(g3b.scope, /W4-IS-A-NOMINAL-VALUE/);
  assert.match(g3b.scope, /NOT licensed: "survives \/ vanishes at 5e-4" on CIFAR-100/);
  for (const id of ['MT240', 'MT155', 'MT243']) {
    const record = byId.get(id)!;
    assert.ok(!record.warningIds.some(wid => wid.includes('amendment-31')), `${id} is not amended`);
  }
  assert.equal(byId.get('MT240')!.outcome, 'unresolved');
  assert.equal(byId.get('MT155')!.outcome, 'fail');
});

test('the ARGS-value warning names its kind, and every W4 row of the batch is listed once', () => {
  const wa = data.warnings.find(w => w.id === 'warning-MT244-args-deviation')!;
  assert.deepEqual([wa.severity, wa.status], ['caution', 'documented']);
  assert.match(wa.detail, /--weight-decay-base 5e-4/);
  assert.match(wa.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 4f7e191; CORRECTIONS 263, 299 and 315\.$/);
  assert.ok(!data.warnings.some(w => w.id === 'warning-MT244-intervention'), 'no patch ran');
});

test('the 32 runs link to MT244 with sanitized logs; only the W4 runs carry the ARGS-value mark', () => {
  assert.equal(runs.length, 3501);
  const linked = runs.filter(run => run.batch === 'g3b');
  assert.equal(linked.length, 32);
  assert.deepEqual([...g3b.runIds].sort(), linked.map(run => run.id).sort());
  for (const run of linked) {
    const arm = run.parameters.runLabel.split('-')[1];
    assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
      [['MT244'], 'Account2', 'completed', 'ResNet18_c100', 'CIFAR100', 100], run.id);
    assert.ok(['170', '171', '172', '173'].includes(run.seed), run.id);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.equal(log.split('\n').filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    assert.equal(run.parameters.intervention, undefined, `${run.id} ran no patch`);
    if (arm.endsWith('W1')) {
      assert.equal(run.parameters.argsDeviation, undefined, `${run.id} sits at the standard 0.1`);
    } else {
      assert.equal(run.parameters.argsDeviation, `${arm}: --weight-decay-base 5e-4`, run.id);
      assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=5e-4', run.id);
      assert.equal(run.parameters.argsDeviationArgsWitness, 'ARGS: --weight-decay-base 5e-4', run.id);
      assert.equal(run.parameters.argsDeviationAxes, undefined, `${run.id} is a one-axis row`);
    }
  }
  assert.equal(runs.filter(run => run.parameters.intervention).length, 199, 'no patch intervention was added');
  assert.equal(runs.filter(run => run.parameters.argsDeviation).length, 171, "155 before, plus g3b's 16 W4 rows");
});

test('the published arm means reproduce every level the row reads', () => {
  const mean = (arm: string) => {
    const armRuns = runs.filter(run => run.batch === 'g3b' && run.parameters.runLabel.split('-')[1] === arm);
    assert.equal(armRuns.length, 4, arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['chW1', 'ndW1', 'k01W1', 'kLW1', 'chW4', 'ndW4', 'k01W4', 'kLW4'].map(mean),
    ['71.8810', '70.6640', '35.1490', '70.7495', '63.0070', '62.9425', '69.0415', '67.0545']);
  assert.equal((Number(mean('chW1')) - Number(mean('ndW1'))).toFixed(4), '1.2170');
  assert.equal((Number(mean('chW4')) - Number(mean('ndW4'))).toFixed(4), '0.0645');
  assert.equal((Number(mean('k01W4')) - Number(mean('chW4'))).toFixed(4), '6.0345');
  assert.equal((77.5927 - Number(mean('chW4'))).toFixed(4), '14.5857');
});

test('phase-20 is its own documented phase and carries one record', () => {
  const phase20 = data.activity.find(e => e.id === 'phase-20')!;
  assert.deepEqual([phase20.experimentIds, phase20.kind, phase20.date], [['MT244'], 'research-phase', '2026-09-22']);
  assert.match(phase20.title, /Does the audit's weight-decay reading hold on CIFAR-100/);
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 20);
});

test('the landing links the scorer, parser, design and audit tools it cites', () => {
  const paths = new Set(g3b.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  for (const path of ['analysis/cG3B_c100wd_score.py', 'analysis/g3b_attack_indep.py', 'analysis/g3b_design.py', 'bin/cG3B_rule20.sh']) {
    assert.ok(paths.has(path), `MT244 links ${path}`);
  }
});

test('CORRECTIONS 315.10 (e) seeds exactly four later-evidence relations, in the vocabulary, and none for the denominator', () => {
  const mine = later.filter(entry => entry.later === 'MT244');
  assert.deepEqual(mine.map(entry => [entry.earlier, entry.relation, entry.source]), [
    ['MT240', 'replicates', 'CORRECTIONS 315.10'],
    ['MT191', 'replicates', 'CORRECTIONS 315.10'],
    ['MT199', 'replicates', 'CORRECTIONS 315.10'],
    ['MT175', 'qualifies', 'CORRECTIONS 315.10'],
  ]);
  assert.equal(mine[0].quote, "cgw1's reading holds on the audit's second dataset");
  assert.match(mine[0].reason, /untuned/i);
  assert.ok(!later.some(entry => entry.later === 'MT244' && entry.earlier === 'MT155'), 'the denominator bearing is stated in other words');
});
