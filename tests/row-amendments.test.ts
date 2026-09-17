import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// CORRECTIONS 229 (campaign commit 6e33fd8) amended MASTER-TABLE rows 19, 162, 163, 166 and 213 in place.
// Only row 19's verdict moved (OPEN -> CONFIRMED, RESCOPED 229); the notebook applies that through
// scripts/register_model.py ROW_AMENDMENTS and keeps the previous wording in a warning.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warning = (id: string) => data.warnings.find(w => w.id === `warning-${id}-amendment`);

test('MT019, the fairness audit, is Goal met with its limits and its previous Open outcome on record', () => {
  const record = byId.get('MT019')!;
  assert.deepEqual([record.outcome, record.corrected, record.kind], ['success', false, 'research']);
  assert.match(record.result, /\+3\.6173 pp .*\+12\.1780 pp/);
  assert.match(record.scope, /one baseline family/);
  assert.match(record.scope, /alpha0 1e-6 is untested on CIFAR-100/);
  assert.match(record.scope, /one granularity per batch, so meta-side tuning is bounded, not exhausted/);
  const note = warning('MT019')!;
  assert.equal(note.title, 'Outcome moved by a later result');
  assert.match(note.detail, /^CONFIRMED, RESCOPED 229: /);
  assert.match(note.detail, /Outcome before the amendment: Open\. Previous result: Unaugmented CIFAR-10 is measured/);
  assert.match(note.detail, /MASTER-TABLE\.md line 19 at 6e33fd8/);
  assert.ok(record.warningIds.includes(note.id));
  assert.deepEqual(record.batches, ['pp', 'PP', 'ub9', 'cau1', 'cuc1']);
});

test('the four other amended rows keep their outcome and carry the amendment note', () => {
  const expected: Record<string, [string, number, RegExp]> = {
    MT163: ['success', 162, /'ResNet-only' is rescoped/],
    MT164: ['success', 163, /cvi1 isolated bn8\.weight/],
    MT020: ['success', 166, /answered by cuc1 \(MT215/],
    MT213: ['mixed', 213, /overtaken by cpl2 \(MT217\)/],
  };
  for (const [id, [outcome, line, text]] of Object.entries(expected)) {
    const record = byId.get(id)!;
    assert.equal(record.outcome, outcome, id);
    assert.match(record.scope, /Amended at CORRECTIONS 229: Outcome unchanged/, id);
    assert.match(record.scope, text, id);
    const note = warning(id)!;
    assert.equal(note.title, 'Wording amended by a later result', id);
    assert.ok(note.detail.includes(`MASTER-TABLE.md line ${line} at 6e33fd8`), id);
  }
  const amended = data.warnings.filter(w => w.id.endsWith('-amendment')).map(w => w.experimentId).sort();
  assert.deepEqual(amended, ['MT019', 'MT020', 'MT163', 'MT164', 'MT213']);
  assert.deepEqual(data.experiments.filter(e => e.scope.includes('CORRECTIONS 229') || /CORRECTIONS 229/.test(e.result)).map(e => e.id).sort(), ['MT020', 'MT163', 'MT164', 'MT213']);
});

test('the 93 runs of the landed batches link to MT212-MT217 with sanitized raw logs', () => {
  assert.equal(runs.length, 3127);
  assert.equal(data.meta.stats.runs, 3127);
  const batches: Record<string, [string, number]> = { cgn1: ['MT212', 6], cpl1: ['MT213', 15], cvh1: ['MT214', 12], cuc1: ['MT215', 30], cgn2: ['MT216', 15], cpl2: ['MT217', 15] };
  const landed = runs.filter(run => run.batch in batches);
  assert.equal(landed.length, 93);
  for (const [batch, [id, count]] of Object.entries(batches)) {
    const linked = runs.filter(run => run.experimentIds.includes(id));
    assert.equal(linked.length, count, id);
    assert.ok(linked.every(run => run.batch === batch), id);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
  }
  for (const run of landed) {
    assert.equal(run.account, 'Account2', run.id);
    assert.equal(run.status, 'completed', run.id);
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.match(run.parameters.originalLogSha256, /^[0-9a-f]{64}$/);
    assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.match(log, /cluster\/Account2\/metaopt\/runs\//, run.id);
    assert.match(log, /NODE=\[HOST_[0-9a-f]{10}\]/, run.id);
  }
  assert.deepEqual(runs.filter(run => run.batch === 'cuc1').every(run => run.experimentIds.join() === 'MT019,MT215'), true);
  // cvt1 landed at CORRECTIONS 230 and is MT218 (tests/cvt1-landing.test.ts); cgn3 landed at CORRECTIONS 231 and is MT219
  // (tests/cgn3-landing.test.ts); cvt3 and cvt2 landed at CORRECTIONS 235 and 236 and are MT220 and MT221
  // (tests/cvt3-cvt2-landing.test.ts); cvt4 and cvt5 landed at CORRECTIONS 240 and 241 and are MT222 and MT223
  // (tests/cvt4-cvt5-landing.test.ts); cvt6 and cvt7 landed at CORRECTIONS 246 and 247 and are MT224 and MT225
  // (tests/cvt6-cvt7-landing.test.ts); cvt8 and cvt9 landed at CORRECTIONS 252 and 253 and are MT226 and MT227
  // (tests/cvt8-cvt9-landing.test.ts). Each has exactly one record.
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt1')).map(e => e.id), ['MT218']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cgn3')).map(e => e.id), ['MT219']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt3')).map(e => e.id), ['MT220']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt2')).map(e => e.id), ['MT221']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt4')).map(e => e.id), ['MT222']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt5')).map(e => e.id), ['MT223']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt6')).map(e => e.id), ['MT224']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt7')).map(e => e.id), ['MT225']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt8')).map(e => e.id), ['MT226']);
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvt9')).map(e => e.id), ['MT227']);
});
