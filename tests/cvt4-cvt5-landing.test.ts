import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 222 (cvt4) and 223 (cvt5) at campaign commit 643264c (CORRECTIONS 240 and 241), imported as
// MT222 and MT223 through scripts/register_model.py CVT45_ROWS. The landing amended no existing row (only header
// lines 3 and 5). The intervened arms are listed in results/CORPUS-EXCLUSIONS.tsv.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CVT4_FINAL = ['OWN-STEP-MAGNITUDE', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT', 'REPLAY-MAX-RATE-TRIANGLE', 'SHARED-IS-MUTE-ARITHMETIC',
  'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'HOLDLOW-AT-HEAD', 'HOLDSHARED-AT-K01',
  'HOLDHIGH-AT-K01', 'HOLDHEAD-AT-HEAD', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES'];
const CVT5_FINAL = ['K-DEPENDENT-EQUILIBRIUM', 'RESCUE-SURVIVES', 'K33-HOLDS-PINNED', 'HARNESS-CLEAN', 'PATCH-BITES', 'ONE-NETWORK-PLAINNET', 'HORIZON-300',
  'NO-TIME-GATE', 'ONE-CELL-ONLY', 'SIGMA-PRIOR-FROZEN', 'RHO:1.0012', 'TRAIN-AGREES-A', 'TRAIN-AGREES-B', 'K13-LEVEL-HOLDS', 'HEAD-LEVEL-HOLDS',
  'K33-LEVEL-HOLDS', 'K13-COMPLEMENT-PINNED', 'K13-PIN-EPOCH:160.3', 'K13-POST-PIN-WINDOW:138.6', 'K13-WINDOW-MEETS-W100', 'K13-PIN-INSIDE-BOUND',
  'K13-FROZEN-AT-PIN', 'HEAD-COMPLEMENT-PINNED', 'HEAD-PIN-EPOCH:63.3', 'HEAD-POST-PIN-WINDOW:236.0', 'HEAD-WINDOW-MEETS-W100', 'HEAD-FROZEN-AT-PIN',
  'K33-COMPLEMENT-PINNED', 'K33-MAXR-POST100:1.2325', 'K33-EXACT-CLAMP-CONTINUOUS', 'K01-PINNED', 'FLOOR-HOLDS'];

test('MT222 and MT223 are the cvt4 and cvt5 rows, both Goal met under the documented rules, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(158, 160).map(e => e.id), ['MT222', 'MT223'], 'appended after every record that existed before them, in line order');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  assert.equal(CVT4_FINAL.length, 16);
  assert.equal(CVT5_FINAL.length, 32);
  const cases: [string, string, string[], number, string][] = [
    ['MT222', 'cvt4', CVT4_FINAL, 222, 'OWN-STEP-MAGNITUDE'],
    ['MT223', 'cvt5', CVT5_FINAL, 223, 'K-DEPENDENT-EQUILIBRIUM + RESCUE-SURVIVES + K33-HOLDS-PINNED'],
  ];
  for (const [id, batch, final, line, branch] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    // Research questions with a real outcome; neither corrects an earlier claim, so no Corrected badge.
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', 'success', false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} Goal met: ${branch}: `), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
  }
  const cvt4 = byId.get('MT222')!, cvt5 = byId.get('MT223')!;
  assert.match(cvt4.title, /^With layer4\.1\.bn2\.weight's vote already out of the complement's sum \(HEAD's grouping\), is the MAGNITUDE of its own step size the lever/);
  assert.match(cvt4.result, /CO-PRIMARY P_HIGH = HEAD - HOLDHIGH = 64\.7347 - 11\.1267 = \+53\.6080 pp = \+94\.55 SE/);
  assert.match(cvt4.reason, /the registered account hits all 9 of its bands/);
  assert.match(cvt4.reason, /a direct stall is not separated from one through the complement/);
  assert.match(cvt5.title, /^Does PlainNet's HEAD rescue survive 300 epochs past its own complement's pin/);
  assert.match(cvt5.result, /PRIMARY \(a\) G13 = K13@300 - K33@300 = 53\.0807 - 37\.1347 = \+15\.9460 pp = \+23\.04 SE/);
  assert.match(cvt5.result, /PRIMARY \(b\) RHO = D_HEAD@300 \/ D_HEAD@100 = 52\.6447 \/ 52\.5807 = 1\.0012/);
  assert.match(cvt5.reason, /the tested EQUILIBRIUM \+ FREEZE-HOLDS account hits all 8 registered bands/);
  assert.match(cvt5.reason, /its level settled \(epoch 105\) before its pin \(~160\)/);
  // The licence clause that does not fit this batch travels with the record.
  assert.match(cvt5.scope, /The scorer's printed RESCUE-SURVIVES licence says the exact clamp came 'at ~96-100'; in this batch HEAD's exact-clamp dwell starts at 299\.8 \/ never \/ 299\.8/);
  // jobs/run_cifar_cvt4.sh was archived at campaign commit 0ade9cc (CORRECTIONS 244.1): the runner links, no source warning.
  // Both rows were amended in place at CORRECTIONS 244 (tests/c244-amendments.test.ts).
  assert.deepEqual(warningsOf('MT222').map(w => w.id), ['warning-MT222-verdict', 'warning-MT222-amendment-244', 'warning-MT222-intervention']);
  const runner = data.sources.find(s => s.path === 'jobs/run_cifar_cvt4.sh')!;
  assert.ok(runner && cvt4.codeIds.includes(runner.id), 'MT222 links the archived cvt4 runner');
  assert.equal(runner.originalSha256, 'c801fc85115f124d82186078a722904a0e8189b94eff44e0daf0e6a7e1c7145d');
  assert.ok(!data.warnings.some(w => /Referenced source not available: jobs\/run_cifar_cvt4\.sh/.test(w.detail)), 'no record warns that the cvt4 runner is missing');
  assert.deepEqual(warningsOf('MT223').map(w => w.id), ['warning-MT223-verdict', 'warning-MT223-amendment-244', 'warning-MT223-intervention']);
});

test('the intervention warnings name every intervened arm and the exclusion list', () => {
  const cvt4 = data.warnings.find(w => w.id === 'warning-MT222-intervention')!;
  assert.equal(cvt4.title, 'HOLDLOW, HOLDSHARED, HOLDHIGH and HOLDHEAD are step-size hold interventions, not plain HEAD arms');
  assert.match(cvt4.detail, /HOLDSHARED = the complement's live beta \(cvt1's MUTE in exact arithmetic\)/);
  assert.match(cvt4.detail, /They are NOT plain HEAD measurements\./);
  assert.match(cvt4.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 643264c; CORRECTIONS 237, 239 and 240\.$/);
  const cvt5 = data.warnings.find(w => w.id === 'warning-MT223-intervention')!;
  assert.equal(cvt5.title, 'K13 and K33 are vote-weight interventions, not plain HEAD arms');
  assert.match(cvt5.detail, /cvt2's K13 and K33 strings byte for byte, run to 300 epochs/);
  assert.match(cvt5.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 643264c; CORRECTIONS 238 and 241\.$/);
});

test('the 30 runs link to MT222 / MT223 with sanitized logs, and exactly the 18 intervened runs are marked', () => {
  assert.equal(runs.length, 3501);
  const batches: [string, string, string[], number, number, Record<string, [number, string, string]>][] = [
    ['cvt4', 'MT222', ['90', '91', '92'], 18, 100, {
      HOLDLOW: [3, 'BETA_HOLD', 'step-size hold'], HOLDSHARED: [3, 'BETA_HOLD', 'step-size hold'],
      HOLDHIGH: [3, 'BETA_HOLD', 'step-size hold'], HOLDHEAD: [3, 'BETA_HOLD', 'step-size hold'],
    }],
    ['cvt5', 'MT223', ['93', '94', '95'], 12, 300, { K13: [3, 'VOTE_W', 'vote-weight'], K33: [3, 'VOTE_W', 'vote-weight'] }],
  ];
  for (const [batch, id, seeds, count, epochs, intervened] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, count, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      const arm = run.parameters.runLabel.split('-')[1];
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [[id], 'Account2', 'completed', 'PlainNet18_c100', 'CIFAR100', epochs], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      if (arm in intervened) {
        const [, patch, kind] = intervened[arm];
        assert.ok(run.parameters.intervention.startsWith(`${arm}: ${patch}=`), run.id);
        assert.ok(run.parameters.interventionWitness.startsWith(`${patch}: on type=`), run.id);
        assert.ok(lines.includes(run.parameters.interventionWitness), `${run.id} log carries its witness line`);
        assert.ok(run.parameters.interventionNote.startsWith('Not a plain HEAD '), run.id);
        assert.ok(run.parameters.interventionNote.includes(`no column records the ${kind} intervention`), run.id);
      } else {
        assert.ok(['k01', 'HEAD'].includes(arm), run.id);
        assert.equal(run.parameters.intervention, undefined, run.id);
        assert.ok(lines.includes('VOTE_W: off'), run.id);
        if (batch === 'cvt4') assert.ok(lines.includes('BETA_HOLD: off'), run.id);
      }
      // Same redaction as every earlier landed batch.
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.match(log, new RegExp(`cluster/Account2/metaopt/runs/${batch}/probe_${batch}-`), run.id);
      assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, epochs, `${run.id} keeps every epoch line`);
    }
    const marked = linked.filter(run => run.parameters.intervention).map(run => run.parameters.runLabel.split('-')[1]);
    assert.deepEqual(Object.fromEntries(Object.keys(intervened).map(arm => [arm, marked.filter(m => m === arm).length])),
      Object.fromEntries(Object.entries(intervened).map(([arm, [n]]) => [arm, n])), batch);
  }
  assert.equal(runs.filter(run => ['cvt1', 'cvt2', 'cvt3', 'cvt4', 'cvt5'].includes(run.batch) && run.parameters.intervention).length, 51, 'cvt1 9 + cvt3 9 + cvt2 15 + cvt4 12 + cvt5 6');
  // cvt6 and cvt7 add 15 and 9 (tests/cvt6-cvt7-landing.test.ts), cvt8 and cvt9 15 and 18 (tests/cvt8-cvt9-landing.test.ts).
  assert.equal(runs.filter(run => run.batch !== 'cvl1' && run.parameters.intervention).length, 183);
  // The arm means of the published plateau5 values reproduce the rows' levels (cvt5 at 300 epochs).
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'HEAD', 'HOLDLOW', 'HOLDSHARED', 'HOLDHIGH', 'HOLDHEAD'].map(arm => mean('cvt4', arm)), ['11.6947', '64.7347', '65.4607', '11.3940', '11.1267', '64.6660']);
  assert.deepEqual(['k01', 'HEAD', 'K13', 'K33'].map(arm => mean('cvt5', arm)), ['11.6700', '64.3147', '53.0807', '37.1347']);
});

test('the earlier intervened runs keep their vote-weight note wording', () => {
  const earlier = runs.filter(run => ['cvt1', 'cvt2', 'cvt3'].includes(run.batch) && run.parameters.intervention);
  assert.equal(earlier.length, 33);
  assert.ok(earlier.every(run => run.parameters.interventionNote.includes('no column records the vote-weight intervention')));
});
