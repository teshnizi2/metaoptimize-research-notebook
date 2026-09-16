import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 219 at campaign commit e3a43da (CORRECTIONS 231): the cgn3 landing, imported as MT219
// through scripts/register_model.py CGN3_ROWS. The same commit amended rows 213 (cpl1, MT213) and
// 218 (cvt1, MT218) in place; CGN3_AMENDMENTS carries both into the records without moving an outcome.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const record = byId.get('MT219')!;
const FINAL = ['RESCUE-SURVIVES', 'RHO:1.2235', 'TRAIN-AGREES', 'COMPLEMENT-PINNED', 'ISO-PIN-EPOCH:224.9', 'POST-PIN-WINDOW:191.6',
  'WINDOW-MEETS-W100', 'PIN-INSIDE-BOUND', 'FROZEN-AT-PIN', 'RHO-ONE:1.6418', 'ONE-SURVIVES', 'ONE-SHARE-RISES', 'ONE-COMPLEMENT-PINNED',
  'HEAD3-NARROWS', 'CEIL-BELOW', 'KL-STILL-TRAINING', 'K01-PINNED', 'FLOOR-HOLDS', 'ISO-TRAIN-MOVES', 'SIGMA-FROZEN-DOMINATES',
  'DIFFERS-FROM-CGN2', 'MAGNITUDE-NOT-SEPARATED', 'NO-IN-BATCH-CTL', 'GN32-ALSO-CHANGES-PER-CHANNEL-INVARIANCE', 'ONE-CELL-ONLY'];
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);

test('MT219 is the cgn3 row, Goal met under the documented rules, with all 25 FINAL tokens quoted', () => {
  assert.ok(record, 'MT219 must exist');
  assert.equal(data.experiments[155].id, 'MT219', 'appended after every record that existed before it');
  assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
    [9, 'Mechanism and isolation', 'research', 'success', false, ['cgn3']]);
  assert.equal(FINAL.length, 25);
  assert.ok(record.reason.startsWith(`Verdict: ${FINAL.join(' + ')} Goal met: RESCUE-SURVIVES: `), record.reason);
  assert.match(record.reason, /every arm lands in the tested FREEZE-HOLDS account's registered band/);
  assert.match(record.reason, /ISO ends 3\.02 pp below kL \(CEIL-BELOW\)/);
  assert.match(record.title, /^Is the GroupNorm isolation rescue \(cgn2, 225\) a rescue or a delayed collapse that 100 epochs could not see/);
  assert.match(record.result, /PRIMARY RHO = D_ISO@430 \/ D_ISO@100 = \+48\.5193 \/ \+39\.6573 = 1\.2235/);
  assert.match(record.result, /ISO's TEST settled \(within 0\.5 pp of its end, for good\) at epoch 162/);
  assert.match(record.scope, /These rows are 430-epoch measurements \(epochs_done 430\) and must not be pooled into a 100-epoch GN cell/);
  assert.match(record.scope, /'the pin freezes the level' is not licensed/);
  assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-09']]);
  assert.ok(data.activity.find(e => e.id === 'phase-09')!.experimentIds.includes('MT219'));
  const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
  assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 219));
  assert.deepEqual(warningsOf('MT219').map(w => w.id), ['warning-MT219-verdict']);
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.runs], [158, 140, 3019]);
});

test('the 12 cgn3 runs link to MT219 as 430-epoch GroupNorm runs with sanitized raw logs', () => {
  assert.equal(runs.length, 3019);
  const cgn3 = runs.filter(run => run.batch === 'cgn3');
  assert.equal(cgn3.length, 12);
  assert.deepEqual([...record.runIds].sort(), cgn3.map(run => run.id).sort());
  const arms = cgn3.map(run => run.parameters.runLabel.split('-')[1]).sort();
  assert.deepEqual(arms, ['ISO', 'ISO', 'ISO', 'ONE', 'ONE', 'ONE', 'k01', 'k01', 'k01', 'kL', 'kL', 'kL']);
  for (const run of cgn3) {
    assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs, run.parameters.epochs_requested],
      [['MT219'], 'Account2', 'completed', 'ResNet18_gn_c100', 'CIFAR100', 430, '430'], run.id);
    assert.ok(['81', '82', '83'].includes(run.seed), run.id);
    assert.equal(run.parameters.intervention, undefined, `${run.id} is an ordinary measurement`);
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.match(log, /--save-directory cluster\/Account2\/metaopt\/runs\/cgn3 /, run.id);
    assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
    assert.equal(log.split('\n').filter(line => /^Epoch \d+,/.test(line)).length, 430, `${run.id} keeps every epoch line`);
  }
  // The four arm means of the published plateau5 values reproduce the row's 430-epoch levels.
  const mean = (arm: string) => cgn3.filter(run => run.parameters.runLabel.includes(`-${arm}-`)).reduce((sum, run) => sum + run.testAccuracy!, 0) / 3;
  assert.deepEqual(['k01', 'kL', 'ISO', 'ONE'].map(arm => mean(arm).toFixed(4)), ['14.8427', '66.3853', '63.3620', '56.6187']);
});

test('MT213 carries CORRECTIONS 231: cvt1 has landed, and the retired pending note is gone', () => {
  const mt213 = byId.get('MT213')!;
  assert.equal(mt213.outcome, 'mixed');
  const text = [mt213.title, mt213.result, mt213.reason, mt213.scope, ...warningsOf('MT213').map(w => w.detail)].join('\n');
  assert.doesNotMatch(text, /cvt1 has not landed/i);
  assert.doesNotMatch(text, /has NOT landed/);
  assert.ok(mt213.scope.includes('HEAD\'s rescue is a delay, and identity vs magnitude is still open. Amended at CORRECTIONS 231: Outcome unchanged (Mixed). cvt1 landed at CORRECTIONS 230 as MT218 (STEP-SIZE-NEEDED-VOTE-SUFFICES)'));
  assert.match(mt213.scope, /MUTE 10\.9860 vs k01 11\.9493/);
  assert.match(mt213.scope, /re-pins that complement at epoch 37\.6 and drops the arm to 30\.1400 \(0\.344 of the gap kept\)/);
  const later = data.warnings.find(w => w.id === 'warning-MT213-amendment-231')!;
  assert.ok(later && mt213.warningIds.includes(later.id));
  assert.equal(later.title, 'Wording amended by a later result');
  assert.match(later.detail, /MASTER-TABLE\.md line 213 at e3a43da; CORRECTIONS 231\.$/);
  // The CORRECTIONS 229 record stays, without the retired clause.
  assert.match(data.warnings.find(w => w.id === 'warning-MT213-amendment')!.detail, /identity vs magnitude is still open\. Amendment record: docs\/MASTER-TABLE\.md line 213 at 6e33fd8/);
  assert.ok(mt213.sourceRefs.some(ref => ref.label.endsWith('CORRECTIONS.md section 231')));
});

test('MT218 carries CORRECTIONS 231: the pooling claim is kept as superseded and the amendment is stated', () => {
  const mt218 = byId.get('MT218')!;
  assert.equal(mt218.outcome, 'mixed');
  assert.ok(mt218.scope.includes('[SUPERSEDED, not true when written: every cell-pooling reader drops them.] [AMENDED at cycle 152, CORRECTIONS 231: no file in analysis/ imports analysis/corpus_exclusions.py'));
  assert.match(mt218.scope, /re-derives SIGMA_PLAIN on the live corpus as 8\.6778 \(df 24\), against 0\.4107 \(df 15\) with the 9 rows dropped/);
  assert.doesNotMatch(mt218.scope, /tsv; every cell-pooling reader drops them/, 'the claim no longer reads as current');
  const later = data.warnings.find(w => w.id === 'warning-MT218-amendment-231')!;
  assert.ok(later && mt218.warningIds.includes(later.id));
  assert.match(later.detail, /^Outcome unchanged \(Mixed\)\. The row said every cell-pooling reader drops the 9 MUTE \/ DOSE \/ INJECT rows; that was not true when written\./);
  assert.match(later.detail, /MASTER-TABLE\.md line 218 at e3a43da; CORRECTIONS 231\.$/);
  assert.ok(mt218.warningIds.includes('warning-MT218-intervention'), 'the intervention warning stays');
  const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
  assert.ok(mt218.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 218));
});
