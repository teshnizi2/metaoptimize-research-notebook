import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 218 at campaign commit 82867bb (CORRECTIONS 230): the cvt1 landing, imported as MT218
// through scripts/register_model.py LANDED_ROWS. MUTE, DOSE and INJECT are vote-weight interventions
// that the run inventory cannot tell apart from plain k01 / HEAD rows (results/CORPUS-EXCLUSIONS.tsv).
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const record = data.experiments.find(e => e.id === 'MT218')!;
const FINAL = ['STEP-SIZE-NEEDED-VOTE-SUFFICES', 'HARNESS-CLEAN', 'PATCH-BITES', 'K-FIXED-691', 'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN',
  'POSITIVE-CONTROL-REPRODUCES', 'DOSE-AT-K01', 'INJECT-PARTIAL', 'MUTE-BELOW-HEAD', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES'];

test('MT218 is the cvt1 row, Mixed under the documented rules, with its FINAL tokens quoted', () => {
  assert.ok(record, 'MT218 must exist');
  assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches], [9, 'Mechanism and isolation', 'research', 'mixed', false, ['cvt1']]);
  assert.ok(record.reason.startsWith(`Verdict: ${FINAL.join(' + ')} Mixed: STEP-SIZE-NEEDED-VOTE-SUFFICES: `), record.reason);
  assert.match(record.reason, /misses its registered 8-30 band by 0\.14 pp/);
  assert.match(record.title, /^On the residual-free ResNet-18, is layer4\.1\.bn2\.weight's rescue the removal of its VOTE/);
  assert.match(record.result, /P_VOTE = HEAD - MUTE = 64\.7940 - 10\.9860 = \+53\.8080 pp = \+94\.90 SE/);
  assert.match(record.result, /P_INJECT = HEAD - INJECT = 64\.7940 - 30\.1400 = \+34\.6540 pp = \+61\.12 SE/);
  assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']]);
  assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes('MT218'));
  const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
  assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 218));
});

test('the record text makes the MUTE, DOSE and INJECT interventions visible', () => {
  assert.match(record.comparison, /PATCH_VOTEWEIGHT/);
  assert.match(record.comparison, /MUTE = scalar grouping with 50's term x0/);
  assert.match(record.comparison, /DOSE = the same x0\.1/);
  assert.match(record.comparison, /INJECT = HEAD's grouping with the twin 47 layer4\.1\.bn1\.weight x691/);
  assert.match(record.scope, /The 9 MUTE \/ DOSE \/ INJECT rows carry k01's \/ HEAD's CSV cell key \(no VOTE_W column\) and are listed in results\/CORPUS-EXCLUSIONS\.tsv; \[SUPERSEDED, not true when written: every cell-pooling reader drops them\.\]/);
  const warning = data.warnings.find(w => w.id === 'warning-MT218-intervention')!;
  assert.ok(warning && record.warningIds.includes(warning.id));
  assert.equal(warning.title, 'MUTE, DOSE and INJECT are vote-weight interventions, not plain arms');
  assert.match(warning.detail, /They are NOT plain scalar \/ plain HEAD measurements\./);
  assert.match(warning.detail, /seed for seed they differ from those arms only in run, job_id, node, wallclock_min and the accuracy columns/);
  assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 82867bb; CORRECTIONS 230\.$/);
});

test('the 15 cvt1 runs link to MT218, and exactly the 9 intervened runs are marked with their VOTE_W witness', () => {
  assert.equal(runs.length, 3469);
  const cvt1 = runs.filter(run => run.batch === 'cvt1');
  assert.equal(cvt1.length, 15);
  assert.deepEqual([...record.runIds].sort(), cvt1.map(run => run.id).sort());
  const witness: Record<string, [string, string]> = {
    MUTE: ['MUTE: VOTE_W=layer4.1.bn2.weight:0', 'VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53'],
    DOSE: ['DOSE: VOTE_W=layer4.1.bn2.weight:0.1', 'VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.1:group=0:groupsize=53'],
    INJECT: ['INJECT: VOTE_W=layer4.1.bn1.weight:691', 'VOTE_W: on type=blockwise items=47:layer4.1.bn1.weight:w=691.0:group=0:groupsize=52'],
  };
  for (const run of cvt1) {
    const arm = run.parameters.runLabel.split('-')[1];
    assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [['MT218'], 'Account2', 'completed', 'PlainNet18_c100', 'CIFAR100', 100], run.id);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    if (arm in witness) {
      assert.deepEqual([run.parameters.intervention, run.parameters.interventionWitness], witness[arm], run.id);
      assert.match(run.parameters.interventionNote, arm === 'INJECT' ? /^Not a plain HEAD / : /^Not a plain k01 \(granularity scalar\) measurement/, run.id);
      assert.ok(log.split('\n').includes(witness[arm][1]), `${run.id} log carries its witness line`);
    } else {
      assert.ok(['k01', 'HEAD'].includes(arm), run.id);
      assert.equal(run.parameters.intervention, undefined, run.id);
      assert.ok(log.split('\n').includes('VOTE_W: off'), run.id);
    }
    // Same redaction as the earlier landed batches.
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.match(log, /cluster\/Account2\/metaopt\/runs\/cvt1\/probe_cvt1-/, run.id);
    assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
  }
  assert.equal(cvt1.filter(run => run.parameters.intervention).length, 9, 'exactly the 9 cvt1 intervention runs are marked');
  // cvt3 and cvt2 add their own 9 and 15 (tests/cvt3-cvt2-landing.test.ts), cvt4 and cvt5 their 12 and 6
  // (tests/cvt4-cvt5-landing.test.ts), cvt6 and cvt7 their 15 and 9 (tests/cvt6-cvt7-landing.test.ts),
  // cvt8 and cvt9 their 15 and 18 (tests/cvt8-cvt9-landing.test.ts), and cvt10, cwd1, csv1 and cwd2 their 15, 6, 12 and 12
  // (tests/mech4-landing.test.ts), cwd3 and cwd4 theirs, and cvl1 its 16 VAL_SPLIT rows (tests/mech9-landing.test.ts); no other batch is marked.
  assert.deepEqual([...new Set(runs.filter(run => run.parameters.intervention).map(run => run.batch))].sort(), ['csv1', 'cvl1', 'cvt1', 'cvt10', 'cvt2', 'cvt3', 'cvt4', 'cvt5', 'cvt6', 'cvt7', 'cvt8', 'cvt9', 'cwd1', 'cwd2', 'cwd3', 'cwd4']);
});
