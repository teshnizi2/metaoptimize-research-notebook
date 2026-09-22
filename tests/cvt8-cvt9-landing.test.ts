import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 226 (cvt8) and 227 (cvt9) at campaign commit ba01f54 (CORRECTIONS 252 and 253), imported as MT226 and
// MT227 through scripts/register_model.py CVT89_ROWS. The landing amended no existing row (header line 3 and the bottom-line
// paragraph, line 5, only) and corrected cvt9's registration (CORRECTIONS 249.3) in place with one bracket (253.9).
// The intervened arms are listed in results/CORPUS-EXCLUSIONS.tsv: cvt8's three forced arms carry GROUP_HOLD + REST_HOLD,
// cvt9's six held arms BETA_HOLD + COMP_HOLD, and EARLY / LATE a third hold, WINDOW_HOLD (CORRECTIONS 251).
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CVT8_FINAL = ['DOSE-FULL+ROUTE-PARTIAL+BIGROUTE-DIRECT', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT', 'REPLAY-MAX-RATE-TRIANGLE',
  'BIG-IS-PLAINNET-DOSE', 'ISOPATH-REPLAY-FILE', 'FORCED-ARMS-OPEN-LOOP', 'CARRIER-GROUP-OF-3', 'ONE-NETWORK-RESNET', 'HORIZON-100-ONLY',
  'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'HOLDHIGH-BETWEEN', 'HOLDBIG-AT-K01', 'HIGHISOPATH-BETWEEN', 'BIGISOPATH-AT-K01',
  'LOWISOPATH-AT-ISO', 'HOLDBIG-BELOW-K01', 'BIGISOPATH-BELOW-K01', 'TRAIN-AGREES'];
const CVT9_FINAL = ['DOSE-GRADED', 'WINDOW-GRADED', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT', 'COMPLEMENT-ON-HEADPATH',
  'FORCED-ARMS-OPEN-LOOP', 'DOSE-IS-TRIANGLE-FAMILY', 'WINDOW-CUT-AT-PEAK', 'ONE-WINDOW-PAIR', 'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY',
  'SIGMA-INBATCH', 'POSITIVE-CONTROL-REPRODUCES', 'HIGHHEADPATH-AT-K01', 'MIDDOSE-BETWEEN', 'RESDOSE-BETWEEN', 'EARLY-PARTIAL',
  'LATE-PARTIAL', 'LATE-ONSET:11.90', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES'];

test('MT226 (cvt8) and MT227 (cvt9) are both Mixed under the documented rules, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(162, 164).map(e => e.id), ['MT226', 'MT227'], 'appended in line order, before the MUST-tier rows MT228-MT231');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [180, 162, 18, 3469]);
  assert.equal(CVT8_FINAL.length, 21);
  assert.equal(CVT9_FINAL.length, 22);
  // Each reason opens with the row's returned branch words, as MT221 and MT223 did.
  const cases: [string, string, string[], number, string][] = [['MT226', 'cvt8', CVT8_FINAL, 226, CVT8_FINAL[0]], ['MT227', 'cvt9', CVT9_FINAL, 227, 'DOSE-GRADED + WINDOW-GRADED']];
  for (const [id, batch, final, line, branch] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    // Research questions with a real outcome; neither corrects an earlier published claim, so no Corrected badge.
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', 'mixed', false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} Mixed: ${branch}: `), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
  }
  const cvt8 = byId.get('MT226')!, cvt9 = byId.get('MT227')!;
  assert.match(cvt8.title, /^On ResNet18_c100, is cvt7's partial loss a DOSE effect/);
  assert.match(cvt8.result, /CO-PRIMARIES P_DOSE = HOLDHIGH - HOLDBIG = 49\.6327 - 19\.3780 = \+30\.2547 pp = \+53\.85 SE/);
  assert.match(cvt8.result, /P_ROUTE = HIGHISOPATH - HOLDHIGH = \+8\.5567 pp = \+15\.23 SE/);
  // Mixed, as MT218 and MT224: the branch answers, but no account fits every band and both big-dose arms sit below k01.
  // 252.3: four DOSE-family accounts, three of them 6 of 7 and DOSE x VIA 5 (the final audit's wording fix; tests/cvt89-audit.test.ts).
  assert.match(cvt8.reason, /no registered account fits every band: of the four DOSE-family accounts, three hit 6 of 7/);
  assert.match(cvt8.reason, /both big-dose arms sit below k01 on every seed/);
  assert.match(cvt8.reason, /two RULE 16 text defects/);
  assert.match(cvt8.scope, /Bounded, and led with \(252\.4\): \(1\) the two big-dose arms sit BELOW k01, not at it/);
  assert.match(cvt8.scope, /NOT licensed: necessity; a dose curve; any single carrier;/);
  assert.match(cvt9.title, /^On PlainNet, how does the stall caused by idx 50's large step size depend on the trajectory's dose/);
  assert.match(cvt9.result, /P_WIN = LATE - EARLY = \+9\.3427 pp = \+15\.31 SE/);
  assert.match(cvt9.reason, /no registered account fits every band/);
  assert.match(cvt9.reason, /one 2\.92 pp move turns the whole FINAL into REPLICATE-FAILED/);
  assert.match(cvt9.scope, /Bounded, and led with \(253\.4\): \(1\) no bar is near an arm, but the whole FINAL is one 2\.92 pp move from REPLICATE-FAILED/);
  assert.match(cvt9.scope, /NOT licensed: necessity of any window or dose;/);
  // Both runners were archived at registration (CORRECTIONS 248, 249): the records link them and carry no source-unavailable warning.
  assert.deepEqual(warningsOf('MT226').map(w => w.id), ['warning-MT226-verdict', 'warning-MT226-intervention']);
  // CORRECTIONS 253.15's in-place correction of row 227 (campaign commit 3cf4201) adds a documented amendment warning to MT227.
  assert.deepEqual(warningsOf('MT227').map(w => w.id), ['warning-MT227-verdict', 'warning-MT227-amendment-253', 'warning-MT227-registration-253', 'warning-MT227-intervention']);
  for (const [record, path, sha] of [[cvt8, 'jobs/run_cifar_cvt8.sh', '1dbe4ff52eccbbb1c9936f4a687ed88172dd41c48b14ae60eb353b45f766f5e0'],
    [cvt9, 'jobs/run_cifar_cvt9.sh', '0fe45b1d1d2cea28740c72e08193af7d5bab3819b397f59f580c71bb9fbf7849']] as const) {
    const runner = data.sources.find(s => s.path === path)!;
    assert.ok(runner && record.codeIds.includes(runner.id), `${record.id} links the archived runner ${path}`);
    assert.equal(runner.originalSha256, sha);
    assert.ok(!data.warnings.some(w => w.detail.includes(`Referenced source not available: ${path}`)), `no record warns that ${path} is missing`);
  }
});

test("cvt9's registration corrected in place at the landing travels with MT227, outcome unchanged", () => {
  const warning = data.warnings.find(w => w.id === 'warning-MT227-registration-253')!;
  assert.equal(warning.title, 'Registration wording corrected in place');
  assert.match(warning.detail, /CORRECTIONS 249\.3/);
  assert.match(warning.detail, /4\.06480e7/);
  assert.match(warning.detail, /The ratio 1\.00103 is correct/);
  assert.match(warning.detail, /docs\/CORRECTIONS\.md at ba01f54; CORRECTIONS 253\.9\.$/);
  assert.deepEqual([warning.severity, warning.status], ['limitation', 'documented']);
  assert.ok(!data.warnings.some(w => w.id.startsWith('warning-MT226-registration')), 'the cvt8 landing corrected no registration text');
});

test('the intervention warnings name every intervened arm, every hold and the exclusion list', () => {
  const cvt8 = data.warnings.find(w => w.id === 'warning-MT226-intervention')!;
  assert.equal(cvt8.title, 'HOLDHIGH, HOLDBIG, HIGHISOPATH, BIGISOPATH and LOWISOPATH are group step-size hold interventions, not plain ISO arms');
  assert.match(cvt8.detail, /REST_HOLD was set only on HIGHISOPATH, BIGISOPATH and LOWISOPATH/);
  assert.match(cvt8.detail, /They are NOT plain ISO measurements\./);
  assert.match(cvt8.detail, /results\/CORPUS-EXCLUSIONS\.tsv at ba01f54; CORRECTIONS 248, 251 and 252\.$/);
  const cvt9 = data.warnings.find(w => w.id === 'warning-MT227-intervention')!;
  assert.equal(cvt9.title, 'LOWHEADPATH, HIGHHEADPATH, MIDDOSE, RESDOSE, EARLY and LATE are step-size hold interventions, not plain HEAD arms');
  assert.match(cvt9.detail, /WINDOW_HOLD was set only on EARLY and LATE/);
  assert.match(cvt9.detail, /updates 9403-9501/);
  assert.match(cvt9.detail, /They are NOT plain HEAD measurements\./);
  assert.match(cvt9.detail, /results\/CORPUS-EXCLUSIONS\.tsv at ba01f54; CORRECTIONS 249, 251 and 253\.$/);
});

test('the 42 runs link to MT226 / MT227 with sanitized logs; the 33 intervened runs are marked, each further hold with its own witness', () => {
  assert.equal(runs.length, 3469);
  const GROUP = 'group step-size hold', REST = 'rest-group step-size hold', BETA = 'step-size hold', COMP = 'complement step-size hold', WINDOW = 'update-window hold';
  const batches: [string, string, string[], string, string, string[], Record<string, [number, string[], string[]]>][] = [
    ['cvt8', 'MT226', ['102', '103', '104'], 'ResNet18_c100', 'ISO', ['VOTE_W: off', 'BETA_HOLD: off', 'GROUP_HOLD: off', 'REST_HOLD: off'], {
      HOLDHIGH: [3, ['GROUP_HOLD'], [GROUP]], HOLDBIG: [3, ['GROUP_HOLD'], [GROUP]],
      HIGHISOPATH: [3, ['GROUP_HOLD', 'REST_HOLD'], [GROUP, REST]], BIGISOPATH: [3, ['GROUP_HOLD', 'REST_HOLD'], [GROUP, REST]],
      LOWISOPATH: [3, ['GROUP_HOLD', 'REST_HOLD'], [GROUP, REST]],
    }],
    ['cvt9', 'MT227', ['105', '106', '107'], 'PlainNet18_c100', 'HEAD', ['VOTE_W: off', 'BETA_HOLD: off', 'COMP_HOLD: off', 'WINDOW_HOLD: off'], {
      LOWHEADPATH: [3, ['BETA_HOLD', 'COMP_HOLD'], [BETA, COMP]], HIGHHEADPATH: [3, ['BETA_HOLD', 'COMP_HOLD'], [BETA, COMP]],
      MIDDOSE: [3, ['BETA_HOLD', 'COMP_HOLD'], [BETA, COMP]], RESDOSE: [3, ['BETA_HOLD', 'COMP_HOLD'], [BETA, COMP]],
      EARLY: [3, ['BETA_HOLD', 'COMP_HOLD', 'WINDOW_HOLD'], [BETA, COMP, WINDOW]], LATE: [3, ['BETA_HOLD', 'COMP_HOLD', 'WINDOW_HOLD'], [BETA, COMP, WINDOW]],
    }],
  ];
  const phrase = (names: string[]) => names.length === 1 ? `${names[0]} intervention` : `${names.slice(0, -1).join(', ')} and ${names.at(-1)} interventions`;
  for (const [batch, id, seeds, network, looksLike, offLines, intervened] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, 21, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      const arm = run.parameters.runLabel.split('-')[1];
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [[id], 'Account2', 'completed', network, 'CIFAR100', 100], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      if (arm in intervened) {
        const [, patches, kinds] = intervened[arm];
        const listed = run.parameters.intervention.slice(`${arm}: `.length).split(' ');
        assert.deepEqual(listed.map(part => part.split('=')[0]), patches, run.id);
        assert.ok(run.parameters.interventionWitness.startsWith(`${patches[0]}: on type=`), run.id);
        assert.ok(lines.includes(run.parameters.interventionWitness), `${run.id} log carries its witness line`);
        if (patches.length > 1) {
          // The exclusion list carries only the first hold's witness; each further hold is read from the run's own log, in order.
          const extra = run.parameters.interventionAdditionalWitness.split(' | ');
          assert.deepEqual(extra.map(line => line.split(': on ')[0]), patches.slice(1), run.id);
          for (const line of extra) assert.ok(lines.includes(line), `${run.id} log carries ${line.slice(0, 20)}`);
          for (const [index, part] of listed.slice(1).entries()) {
            const value = part.split('=')[1];
            const expected = value.startsWith('rec:') ? ` mode=rec id=${value.slice(4)} ` : ` n0=${value.split(':')[0]} n1=${value.split(':')[1]} `;
            assert.ok(extra[index].includes(expected), `${run.id} ${part}`);
          }
        } else {
          assert.equal(run.parameters.interventionAdditionalWitness, undefined, run.id);
        }
        assert.ok(run.parameters.interventionNote.startsWith(`Not a plain ${looksLike} `), run.id);
        assert.ok(run.parameters.interventionNote.includes(`no column records the ${phrase(kinds)}.`), run.id);
      } else {
        assert.ok(['k01', looksLike].includes(arm), run.id);
        assert.equal(run.parameters.intervention, undefined, run.id);
        for (const off of offLines) assert.ok(lines.includes(off), `${run.id} ${off}`);
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
  assert.equal(runs.filter(run => ['cvt8', 'cvt9'].includes(run.batch) && run.parameters.intervention).length, 33, 'cvt8 15 + cvt9 18');
  assert.equal(runs.filter(run => run.batch !== 'cvl1' && run.parameters.intervention).length, 183, 'the 75 earlier + cvt8 15 + cvt9 18 + the 45 of the cvt10 / cwd1 / csv1 / cwd2 landing + cwd3 12 + cwd4 18');
  assert.equal(runs.filter(run => ['cvt8', 'cvt9'].includes(run.batch) && run.parameters.interventionAdditionalWitness?.includes(' | ')).length, 6, 'EARLY and LATE carry two further holds');
  // The arm means of the published plateau5 values reproduce the rows' levels.
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'ISO', 'HOLDHIGH', 'HOLDBIG', 'HIGHISOPATH', 'BIGISOPATH', 'LOWISOPATH'].map(arm => mean('cvt8', arm)),
    ['23.0553', '70.2613', '49.6327', '19.3780', '58.1893', '19.4233', '70.4667']);
  assert.deepEqual(['k01', 'LOWHEADPATH', 'HIGHHEADPATH', 'MIDDOSE', 'RESDOSE', 'EARLY', 'LATE'].map(arm => mean('cvt9', arm)),
    ['12.1347', '65.2080', '11.2107', '55.8553', '21.8387', '18.3713', '27.7140']);
});

test('the 75 earlier intervened runs keep their notes and witnesses byte for byte', () => {
  const before = JSON.parse(readFileSync(new URL('./fixtures/intervention-notes-57b9ad5.json', import.meta.url), 'utf8')) as Record<string, Record<string, string>>;
  const earlier = runs.filter(run => !['cvt8', 'cvt9', 'cvt10', 'cwd1', 'csv1', 'cwd2', 'cwd3', 'cwd4', 'cvl1'].includes(run.batch) && run.parameters.intervention);
  assert.equal(earlier.length, 75);
  assert.deepEqual(Object.keys(before).sort(), earlier.map(run => run.id).sort());
  for (const run of earlier) {
    const { interventionNote, interventionWitness, interventionAdditionalWitness } = run.parameters;
    assert.deepEqual({ interventionNote, interventionWitness, ...(interventionAdditionalWitness === undefined ? {} : { interventionAdditionalWitness }) }, before[run.id], run.id);
  }
});
