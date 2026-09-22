import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 220 (cvt3) and 221 (cvt2) at campaign commit 9c5d72a (CORRECTIONS 235 and 236), imported as
// MT220 and MT221 through scripts/register_model.py CVT23_ROWS. The same pin carries CORRECTIONS 234's rescope of
// row 216 (cgn2, MT216) and CORRECTIONS 236's in-place wording fix of row 213 (cpl1, MT213); CVT23_AMENDMENTS
// applies both without moving an outcome. The intervened arms are listed in results/CORPUS-EXCLUSIONS.tsv.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CVT3_FINAL = ['OWN-STEP-NECESSARY', 'HARNESS-CLEAN', 'PATCH-BITES', 'SETS-FROM-CVT1-MUTE', 'CTL-COUNT-CLASS-NOT-MASS', 'NO-TIME-GATE',
  'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'MUTE50-AT-K01', 'CTL-BELOW-K01',
  'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES'];
const CVT2_FINAL = ['GRADED', 'TOP-ATTENUATED', 'HARNESS-CLEAN', 'PATCH-BITES', 'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'DOSE-ONSET-CONFOUNDED',
  'SIGMA-INBATCH', 'POSITIVE-CONTROL-REPRODUCES', 'K13-BETWEEN', 'K33-BETWEEN', 'K152-AT-PLATEAU', 'TRAIN-AGREES'];

test('MT220 and MT221 are the cvt3 and cvt2 rows, both Mixed under the documented rules, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(156, 158).map(e => e.id), ['MT220', 'MT221'], 'appended after every record that existed before them, in line order');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  const cases: [string, string, string[], number][] = [['MT220', 'cvt3', CVT3_FINAL, 220], ['MT221', 'cvt2', CVT2_FINAL, 221]];
  for (const [id, batch, final, line] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', 'mixed', false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} Mixed: ${final[0]}`), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
    assert.deepEqual(warningsOf(id).map(w => w.id), [`warning-${id}-verdict`, `warning-${id}-intervention`]);
  }
  assert.equal(CVT3_FINAL.length, 14);
  assert.equal(CVT2_FINAL.length, 13);
  const cvt3 = byId.get('MT220')!, cvt2 = byId.get('MT221')!;
  // Mixed, as MT218: the branch answers the question, but a registered expectation was defied.
  assert.match(cvt3.reason, /misses its registered MUTECTL band by 0\.18 pp/);
  assert.match(cvt3.reason, /'no silenced set rescues' is not licensed/);
  assert.match(cvt3.result, /PRIMARY P_COAL = MUTEDOWN - MUTECTL = 11\.3293 - 7\.8213 = \+3\.5080 pp = \+6\.19 SE/);
  assert.match(cvt3.result, /G-RUNAWAY: 0 of 15 runs/);
  assert.match(cvt2.reason, /no registered account predicted this pair of words, the stated TOP-FLAT prior missed by 0\.26 pp/);
  assert.match(cvt2.reason, /K13 is still rising at 100 epochs/);
  assert.match(cvt2.result, /PRIMARY TOP = K691 - K2000 = \+5\.2593 pp = \+7\.39 SE/);
  assert.match(cvt2.title, /^Is cvt1's injected collapse of HEAD's complement a THRESHOLD or GRADED in the twin's weight K/);
});

test('the intervention warnings name every intervened arm and the exclusion list', () => {
  const cvt3 = data.warnings.find(w => w.id === 'warning-MT220-intervention')!;
  assert.equal(cvt3.title, 'MUTE50, MUTEDOWN and MUTECTL are vote-weight interventions, not plain arms');
  assert.match(cvt3.detail, /They are NOT plain scalar measurements\./);
  assert.match(cvt3.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 9c5d72a; CORRECTIONS 232, 235 and 236\.$/);
  const cvt2 = data.warnings.find(w => w.id === 'warning-MT221-intervention')!;
  assert.equal(cvt2.title, 'K13, K33, K152, K691 and K2000 are vote-weight interventions, not plain HEAD arms');
  assert.match(cvt2.detail, /K691 is cvt1's INJECT byte for byte/);
  assert.match(cvt2.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 9c5d72a; CORRECTIONS 233 and 236\.$/);
});

test('the 36 runs link to MT220 / MT221 with sanitized logs, and exactly the 24 intervened runs are marked', () => {
  assert.equal(runs.length, 3501);
  const batches: [string, string, string[], Record<string, [number, string]>][] = [
    ['cvt3', 'MT220', ['84', '85', '86'], { MUTE50: [3, 'k01'], MUTEDOWN: [3, 'k01'], MUTECTL: [3, 'k01'] }],
    ['cvt2', 'MT221', ['87', '88', '89'], { K13: [3, 'HEAD'], K33: [3, 'HEAD'], K152: [3, 'HEAD'], K691: [3, 'HEAD'], K2000: [3, 'HEAD'] }],
  ];
  for (const [batch, id, seeds, intervened] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, batch === 'cvt3' ? 15 : 21, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      const arm = run.parameters.runLabel.split('-')[1];
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [[id], 'Account2', 'completed', 'PlainNet18_c100', 'CIFAR100', 100], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      if (arm in intervened) {
        assert.ok(run.parameters.intervention.startsWith(`${arm}: VOTE_W=`), run.id);
        assert.match(run.parameters.interventionWitness, /^VOTE_W: on type=/, run.id);
        assert.ok(log.split('\n').includes(run.parameters.interventionWitness), `${run.id} log carries its witness line`);
        assert.ok(run.parameters.interventionNote.startsWith(`Not a plain ${intervened[arm][1]} `), run.id);
      } else {
        assert.ok(['k01', 'HEAD'].includes(arm), run.id);
        assert.equal(run.parameters.intervention, undefined, run.id);
        assert.ok(log.split('\n').includes('VOTE_W: off'), run.id);
      }
      // Same redaction as every earlier landed batch.
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.match(log, new RegExp(`cluster/Account2/metaopt/runs/${batch}/probe_${batch}-`), run.id);
      assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
      assert.equal(log.split('\n').filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    }
    const marked = linked.filter(run => run.parameters.intervention).map(run => run.parameters.runLabel.split('-')[1]);
    assert.deepEqual(Object.fromEntries(Object.keys(intervened).map(arm => [arm, marked.filter(m => m === arm).length])),
      Object.fromEntries(Object.entries(intervened).map(([arm, [n]]) => [arm, n])), batch);
  }
  assert.equal(runs.filter(run => ['cvt1', 'cvt2', 'cvt3'].includes(run.batch) && run.parameters.intervention).length, 33, 'cvt1 9 + cvt3 9 + cvt2 15');
  // cvt4 and cvt5 add 12 and 6 (tests/cvt4-cvt5-landing.test.ts), cvt6 and cvt7 15 and 9 (tests/cvt6-cvt7-landing.test.ts).
  // cvt8 and cvt9 add 15 and 18 (tests/cvt8-cvt9-landing.test.ts); cvt10, cwd1, csv1 and cwd2 add 15, 6, 12 and 12 (tests/mech4-landing.test.ts).
  assert.equal(runs.filter(run => run.batch !== 'cvl1' && run.parameters.intervention).length, 183, 'cvt1 9 + cvt3 9 + cvt2 15 + cvt4 12 + cvt5 6 + cvt6 15 + cvt7 9 + cvt8 15 + cvt9 18 + cvt10 15 + cwd1 6 + csv1 12 + cwd2 12 + cwd3 12 + cwd4 18');
  // The arm means of the published plateau5 values reproduce the rows' levels.
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'HEAD', 'MUTE50', 'MUTEDOWN', 'MUTECTL'].map(arm => mean('cvt3', arm)), ['11.5720', '64.5960', '11.2947', '11.3293', '7.8213']);
  assert.deepEqual(['k01', 'HEAD', 'K13', 'K33', 'K152', 'K691', 'K2000'].map(arm => mean('cvt2', arm)), ['11.4767', '64.4187', '52.3493', '35.5933', '34.3420', '30.2773', '25.0180']);
});

test('MT216 carries CORRECTIONS 234: ISO above kL is rescoped to 100 epochs, kept verbatim, outcome unchanged', () => {
  const mt216 = byId.get('MT216')!;
  assert.equal(mt216.outcome, 'success');
  assert.match(mt216.result, /ISO sits \+1\.81 pp above kL; and magnitude is unreachable/, 'the rescoped wording is kept');
  assert.match(mt216.result, /\[RESCOPED at cycle 152, CORRECTIONS 234: 'ISO sits \+1\.81 pp above kL' above, and the ISO-TRACKS-KL stamp, held AT 100 EPOCHS ONLY; both are kept verbatim\. cgn3 \(231\) ran this cell to 430 epochs/);
  assert.match(mt216.result, /at 430 ISO - kL = -3\.0233 pp = -3\.88 SE \(CEIL-BELOW\)/);
  const warning = data.warnings.find(w => w.id === 'warning-MT216-amendment-234')!;
  assert.ok(warning && mt216.warningIds.includes(warning.id));
  assert.equal(warning.title, 'Wording amended by a later result');
  assert.match(warning.detail, /^Outcome unchanged \(Goal met\)\. Rescoped at CORRECTIONS 234/);
  assert.match(warning.detail, /MASTER-TABLE\.md line 216 at 9c5d72a; CORRECTIONS 234\.$/);
  assert.ok(mt216.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 216));
});

test('MT213 carries CORRECTIONS 236: "whoever casts it" is corrected to one other caster, outcome unchanged', () => {
  const mt213 = byId.get('MT213')!;
  assert.equal(mt213.outcome, 'mixed');
  assert.match(mt213.scope, /Amended at CORRECTIONS 231: .* Amended at CORRECTIONS 236: Outcome unchanged \(Mixed\)\. Row 213's CORRECTIONS 231 update said/);
  assert.match(mt213.scope, /one other caster \(layer4\.1\.bn1\.weight\) was run, into one complement \(HEAD's\), at one K in cvt1 \(MT218\) and at five K in cvt2 \(MT221\)/);
  assert.deepEqual(warningsOf('MT213').map(w => w.id), ['warning-MT213-verdict', 'warning-MT213-amendment', 'warning-MT213-amendment-231', 'warning-MT213-amendment-236']);
  assert.match(data.warnings.find(w => w.id === 'warning-MT213-amendment-236')!.detail, /MASTER-TABLE\.md line 213 at 9c5d72a; CORRECTIONS 236\.$/);
  assert.ok(mt213.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 213));
});
