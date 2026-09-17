import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 224 (cvt6) and 225 (cvt7) at campaign commit dae2a49 (CORRECTIONS 246 and 247), imported as MT224 and
// MT225 through scripts/register_model.py CVT67_ROWS. The landing amended no existing row (header line 3 and the bottom-line
// paragraph, line 5, only), and corrected its own registrations (CORRECTIONS 242 and 243.4) in place with brackets.
// The intervened arms are listed in results/CORPUS-EXCLUSIONS.tsv; cvt6's three forced arms carry two holds at once.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CVT6_FINAL = ['GRADED', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT', 'MUTEPATH-MAX-RATE-TRIANGLE', 'HEADPATH-REPLAY-FILE',
  'FORCED-ARMS-OPEN-LOOP', 'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'HOLDLOW-AT-HEAD',
  'HOLDHIGH-AT-K01', 'HIGHHEADPATH-AT-K01', 'LOWMUTEPATH-BETWEEN', 'LOWHEADPATH-AT-HEAD', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES'];
const CVT7_FINAL = ['TRANSFERS-GRADED', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT', 'REPLAY-MAX-RATE-TRIANGLE', 'HIGH-IS-K01-TRAJECTORY',
  'CARRIER-GROUP-OF-3', 'ONE-NETWORK-RESNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'HOLDLOW-AT-ISO',
  'HOLDHIGH-BETWEEN', 'HOLDISO-AT-ISO', 'TRAIN-AGREES'];

test('MT224 (cvt6) is Mixed and MT225 (cvt7) Goal met under the documented rules, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(160, 162).map(e => e.id), ['MT224', 'MT225'], 'appended after every record that existed before them, in line order');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [164, 146, 18, 3127]);
  assert.equal(CVT6_FINAL.length, 18);
  assert.equal(CVT7_FINAL.length, 15);
  const cases: [string, string, string[], number, string, string][] = [
    ['MT224', 'cvt6', CVT6_FINAL, 224, 'mixed', 'Mixed'],
    ['MT225', 'cvt7', CVT7_FINAL, 225, 'success', 'Goal met'],
  ];
  for (const [id, batch, final, line, outcome, label] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    // Research questions with a real outcome; neither corrects an earlier published claim, so no Corrected badge.
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', outcome, false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} ${label}: ${final[0]}: `), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
  }
  const cvt6 = byId.get('MT224')!, cvt7 = byId.get('MT225')!;
  assert.match(cvt6.title, /^Does `?layer4\.1\.bn2\.weight`?'s \(idx 50\) large step size stall PlainNet by itself, or through the complement's early collapse/);
  assert.match(cvt6.result, /CO-PRIMARY P_50 = LOWHEADPATH - HIGHHEADPATH = 65\.0207 - 11\.4933 = \+53\.5273 pp = \+94\.40 SE/);
  assert.match(cvt6.result, /P_C = LOWHEADPATH - LOWMUTEPATH = \+23\.4653 pp = \+41\.38 SE/);
  // Mixed, as MT218 and MT221: the branch answers, but no registered account fits every band.
  assert.match(cvt6.reason, /no registered account fits every band: GRADED's own account hits 3 of 5/);
  assert.match(cvt6.reason, /no single-route sentence/);
  assert.match(cvt6.scope, /Bounded, and led with \(246\.4\): no bar is near; the forced arms are open-loop/);
  assert.match(cvt6.scope, /NOT licensed: any single-route sentence; necessity;/);
  assert.match(cvt7.title, /^Does PlainNet's 'the rescue needs the carrier on a small step size' \(240, read as sufficiency\) transfer to ResNet18_c100/);
  assert.match(cvt7.result, /PRIMARY P_HIGH = ISO - HOLDHIGH = 70\.0413 - 50\.2273 = \+19\.8140 pp = \+34\.94 SE/);
  // Goal met, as MT219 and MT222: the returned branch's account hits every band, the control reproduces, no prior was stated.
  assert.match(cvt7.reason, /the TRANSFERS-GRADED account hits every registered band/);
  assert.match(cvt7.reason, /dose and held set are confounded with network/);
  assert.match(cvt7.scope, /Bounded, and led with \(247\.4\): \(1\) dose and held set are confounded with network/);
  // The two runners were archived at campaign commit 4ff0891 (CORRECTIONS 250.1, sha256 equal to the registered runners, 242 / 243):
  // both records link them and neither carries a source-unavailable warning.
  assert.deepEqual(warningsOf('MT224').map(w => w.id), ['warning-MT224-verdict', 'warning-MT224-registration-246', 'warning-MT224-intervention']);
  assert.deepEqual(warningsOf('MT225').map(w => w.id), ['warning-MT225-verdict', 'warning-MT225-registration-247', 'warning-MT225-intervention']);
  for (const [record, path, sha] of [[cvt6, 'jobs/run_cifar_cvt6.sh', '9aa2ed5c15b8ecd8c22d5137ab4be169b97f9556ebe67b8460c4971d4f84783f'],
    [cvt7, 'jobs/run_cifar_cvt7.sh', '805267e4cbcc64172d79824d553a2577ef7cdd55fb8f5e7731c0ea3251a04c56']] as const) {
    const runner = data.sources.find(s => s.path === path)!;
    assert.ok(runner && record.codeIds.includes(runner.id), `${record.id} links the archived runner ${path}`);
    assert.equal(runner.originalSha256, sha);
    assert.ok(!data.warnings.some(w => w.detail.includes(`Referenced source not available: ${path}`)), `no record warns that ${path} is missing`);
  }
});

test('the registrations corrected in place at the landing travel with the records, outcome unchanged', () => {
  const cvt6 = data.warnings.find(w => w.id === 'warning-MT224-registration-246')!;
  assert.equal(cvt6.title, 'Registration wording corrected in place');
  assert.match(cvt6.detail, /12 of the 15 landed 100-epoch HEAD runs at this cell/);
  assert.match(cvt6.detail, /at most 0\.1601 log units \(mean 0\.0100\)/);
  assert.match(cvt6.detail, /8 runs had ended/);
  assert.match(cvt6.detail, /epoch 18\.07/);
  assert.match(cvt6.detail, /docs\/CORRECTIONS\.md at dae2a49; CORRECTIONS 246\.9\.$/);
  const cvt7 = data.warnings.find(w => w.id === 'warning-MT225-registration-247')!;
  assert.match(cvt7.detail, /243\.13/);
  assert.match(cvt7.detail, /docs\/CORRECTIONS\.md at dae2a49; CORRECTIONS 247\.12\.$/);
  for (const warning of [cvt6, cvt7]) assert.deepEqual([warning.severity, warning.status], ['limitation', 'documented']);
});

test('the intervention warnings name every intervened arm, both holds of the forced arms, and the exclusion list', () => {
  const cvt6 = data.warnings.find(w => w.id === 'warning-MT224-intervention')!;
  assert.equal(cvt6.title, 'HOLDLOW, HOLDHIGH, HIGHHEADPATH, LOWMUTEPATH and LOWHEADPATH are step-size hold interventions, not plain HEAD arms');
  assert.match(cvt6.detail, /HIGHHEADPATH, LOWMUTEPATH and LOWHEADPATH hold both groups/);
  // All seven arms ran harness_cvt6; COMP_HOLD was set only on the three forced arms (CORRECTIONS 242; cvt6_rule20_full: COMP_HOLD: off x12).
  assert.match(cvt6.detail, /^All seven arms ran harness_cvt6 \(cvt4's tree plus the opt-in PATCH_COMPHOLD\) through jobs\/run_cifar_cvt6\.sh, on HEAD's grouping \(\{50\} \[52,1\]\)\. COMP_HOLD \(=tri:<P> \| rec:<id>\) was set only on the three forced arms, HIGHHEADPATH, LOWMUTEPATH and LOWHEADPATH; HOLDLOW and HOLDHIGH ran with COMP_HOLD off, as k01 and HEAD did\./);
  assert.doesNotMatch(cvt6.detail, /The five held arms ran cvt4's tree plus the opt-in PATCH_COMPHOLD/);
  assert.match(cvt6.detail, /They are NOT plain HEAD measurements\./);
  assert.match(cvt6.detail, /results\/CORPUS-EXCLUSIONS\.tsv at dae2a49; CORRECTIONS 242, 245 and 246\.$/);
  const cvt7 = data.warnings.find(w => w.id === 'warning-MT225-intervention')!;
  assert.equal(cvt7.title, 'HOLDLOW, HOLDHIGH and HOLDISO are group step-size hold interventions, not plain ISO arms');
  assert.match(cvt7.detail, /They are NOT plain ISO measurements\./);
  assert.match(cvt7.detail, /results\/CORPUS-EXCLUSIONS\.tsv at dae2a49; CORRECTIONS 243 and 247\.$/);
});

test('the 36 runs link to MT224 / MT225 with sanitized logs; the 24 intervened runs are marked, two-kind runs with both witnesses', () => {
  assert.equal(runs.length, 3127);
  const batches: [string, string, string[], number, string, string, Record<string, [number, string[], string]>][] = [
    ['cvt6', 'MT224', ['96', '97', '98'], 21, 'PlainNet18_c100', 'HEAD', {
      HOLDLOW: [3, ['BETA_HOLD'], 'step-size hold intervention'], HOLDHIGH: [3, ['BETA_HOLD'], 'step-size hold intervention'],
      HIGHHEADPATH: [3, ['BETA_HOLD', 'COMP_HOLD'], 'step-size hold and complement step-size hold interventions'],
      LOWMUTEPATH: [3, ['BETA_HOLD', 'COMP_HOLD'], 'step-size hold and complement step-size hold interventions'],
      LOWHEADPATH: [3, ['BETA_HOLD', 'COMP_HOLD'], 'step-size hold and complement step-size hold interventions'],
    }],
    ['cvt7', 'MT225', ['99', '100', '101'], 15, 'ResNet18_c100', 'ISO', {
      HOLDLOW: [3, ['GROUP_HOLD'], 'group step-size hold intervention'], HOLDHIGH: [3, ['GROUP_HOLD'], 'group step-size hold intervention'],
      HOLDISO: [3, ['GROUP_HOLD'], 'group step-size hold intervention'],
    }],
  ];
  for (const [batch, id, seeds, count, network, looksLike, intervened] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, count, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      const arm = run.parameters.runLabel.split('-')[1];
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [[id], 'Account2', 'completed', network, 'CIFAR100', 100], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      if (arm in intervened) {
        const [, patches, kind] = intervened[arm];
        assert.ok(run.parameters.intervention.startsWith(`${arm}: ${patches[0]}=`), run.id);
        for (const patch of patches.slice(1)) assert.ok(run.parameters.intervention.includes(` ${patch}=`), run.id);
        assert.ok(run.parameters.interventionWitness.startsWith(`${patches[0]}: on type=`), run.id);
        assert.ok(lines.includes(run.parameters.interventionWitness), `${run.id} log carries its witness line`);
        if (patches.length > 1) {
          // The exclusion list carries only the first hold's witness; the second is read from the run's own log.
          const extra = run.parameters.interventionAdditionalWitness;
          assert.ok(extra && extra.startsWith('COMP_HOLD: on type=blockwise group=0 '), run.id);
          assert.ok(lines.includes(extra), `${run.id} log carries its second witness line`);
          const spec = run.parameters.intervention.split(' COMP_HOLD=')[1];
          assert.ok(spec.startsWith('rec:') ? extra.includes(` mode=rec id=${spec.slice(4)} `) : extra.includes(` mode=tri P=${spec.slice(4)} `), run.id);
        } else {
          assert.equal(run.parameters.interventionAdditionalWitness, undefined, run.id);
        }
        assert.ok(run.parameters.interventionNote.startsWith(`Not a plain ${looksLike} `), run.id);
        assert.ok(run.parameters.interventionNote.includes(`no column records the ${kind}.`), run.id);
      } else {
        assert.ok(['k01', looksLike].includes(arm), run.id);
        assert.equal(run.parameters.intervention, undefined, run.id);
        for (const off of ['VOTE_W: off', 'BETA_HOLD: off', batch === 'cvt6' ? 'COMP_HOLD: off' : 'GROUP_HOLD: off']) assert.ok(lines.includes(off), `${run.id} ${off}`);
      }
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    }
    const marked = linked.filter(run => run.parameters.intervention).map(run => run.parameters.runLabel.split('-')[1]);
    assert.deepEqual(Object.fromEntries(Object.keys(intervened).map(arm => [arm, marked.filter(m => m === arm).length])),
      Object.fromEntries(Object.entries(intervened).map(([arm, [n]]) => [arm, n])), batch);
  }
  assert.equal(runs.filter(run => !['cvt8', 'cvt9'].includes(run.batch) && run.parameters.intervention).length, 75, 'cvt1 9 + cvt3 9 + cvt2 15 + cvt4 12 + cvt5 6 + cvt6 15 + cvt7 9');
  // cvt8 and cvt9 add 15 and 18 (tests/cvt8-cvt9-landing.test.ts).
  assert.equal(runs.filter(run => run.parameters.intervention).length, 108);
  // The arm means of the published plateau5 values reproduce the rows' levels.
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'HEAD', 'HOLDLOW', 'HOLDHIGH', 'HIGHHEADPATH', 'LOWMUTEPATH', 'LOWHEADPATH'].map(arm => mean('cvt6', arm)),
    ['12.0700', '64.9000', '65.6073', '11.6140', '11.4933', '41.5553', '65.0207']);
  assert.deepEqual(['k01', 'ISO', 'HOLDLOW', 'HOLDHIGH', 'HOLDISO'].map(arm => mean('cvt7', arm)), ['22.9527', '70.0413', '70.2980', '50.2273', '70.0580']);
});

test('the 51 earlier intervened runs keep their notes byte for byte and carry no second witness', () => {
  const before = JSON.parse(readFileSync(new URL('./fixtures/intervention-notes-52db8b6.json', import.meta.url), 'utf8')) as Record<string, string>;
  const earlier = runs.filter(run => ['cvt1', 'cvt2', 'cvt3', 'cvt4', 'cvt5'].includes(run.batch) && run.parameters.intervention);
  assert.equal(earlier.length, 51);
  assert.deepEqual(Object.fromEntries(earlier.map(run => [run.id, run.parameters.interventionNote])), before);
  assert.ok(earlier.every(run => run.parameters.interventionAdditionalWitness === undefined));
});
