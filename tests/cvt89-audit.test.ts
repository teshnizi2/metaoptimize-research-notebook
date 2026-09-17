import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData } from '../src/types.ts';

// The final audit of the cvt8 / cvt9 landing (CORRECTIONS 253.15; campaign commit 3cf4201). Wording only: no ID, title,
// kind, outcome or Corrected badge moves (tests/id-stability.test.ts checks the 2969a1c fixture).
// (1) MASTER-TABLE row 227's bound (1) named one single-arm route out of DOSE-GRADED ("only if RESDOSE rises ~39 pp"). The
//     registered decide() has two: DOSE-NONMONOTONE fires when RESDOSE - MIDDOSE > 5, so MIDDOSE falling ~39 pp flips it too.
//     Row 227 was corrected in place with the superseded wording kept in a bracket; MT227's scope takes the corrected cell
//     and gains a documented amendment warning, not a Corrected badge.
// (2) MT226's reason said forcing ISO's complement path recovers "none at PlainNet's dose". The record (CORRECTIONS 252.8
//     item 3) says a recovery under 5 pp cannot be seen there, and that this does not show the free complement plays no role.
// (3) MT226's reason said "the three DOSE accounts hit 6 of 7". 252.3 lists four DOSE-family accounts: three hit 6 of 7 and
//     DOSE x VIA hits 5.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const byId = new Map(data.experiments.map(e => [e.id, e]));
const publishedTextOf = (id: string) => {
  const record = byId.get(id)!;
  return [record.title, record.result, record.reason, record.scope, ...data.warnings.filter(w => w.experimentId === id).map(w => w.detail)];
};

test("MT226 does not claim that forcing ISO's complement path recovers nothing at PlainNet's dose", () => {
  for (const text of publishedTextOf('MT226')) {
    assert.doesNotMatch(text, /none at PlainNet's dose/);
    assert.doesNotMatch(text, /nothing at PlainNet's dose/);
  }
  const record = byId.get('MT226')!;
  assert.match(record.reason, /at PlainNet's dose, with the complement forced, the carriers alone reproduce HOLDBIG's stall \(BIGISOPATH 19\.42, P_ROUTE_BIG \+0\.05 pp\), but both arms sit below k01, so a recovery under 5 pp could not be seen there and this is not evidence that the free complement's collapse plays no role in HOLDBIG;/);
});

test("MT226 counts the DOSE-family accounts exactly: four, three of them hit 6 of 7 and DOSE x VIA hits 5", () => {
  // MASTER-TABLE row 226's own scope says "(three DOSE accounts hit 6/7)", which is true of three of the four; the reason's
  // "the three DOSE accounts" implied there were only three.
  for (const text of publishedTextOf('MT226')) assert.doesNotMatch(text, /the three DOSE accounts/);
  const record = byId.get('MT226')!;
  assert.match(record.reason, /no registered account fits every band: of the four DOSE-family accounts, three hit 6 of 7 \(DOSE x DIRECT misses HIGHISOPATH by 2\.19 pp\) and DOSE x VIA hits 5,/);
  // The source package ships docs/MAINTENANCE_PUBLIC.md in place of the maintainer guide, which alone documents CVT89_ROWS.
  const maintenance = readFileSync(new URL('../docs/MAINTENANCE.md', import.meta.url), 'utf8');
  assert.doesNotMatch(maintenance, /three DOSE accounts hit 6 of 7/);
  if (maintenance.includes('`CVT89_ROWS` imports')) assert.match(maintenance, /of the four DOSE-family accounts three hit 6 of 7 and DOSE x VIA hits 5/);
});

test("MT227's scope carries both single-arm flip routes out of DOSE-GRADED, the superseded wording kept in its bracket", () => {
  const record = byId.get('MT227')!;
  assert.deepEqual([record.outcome, record.kind, record.corrected], ['mixed', 'research', false]);
  assert.match(record.scope, /no single window arm can change WINDOW-GRADED, and DOSE-GRADED changes on one arm only if RESDOSE rises, or MIDDOSE falls, by ~39 pp \(DOSE-NONMONOTONE\) \[CORRECTED IN PLACE at cycle 153, CORRECTIONS 253\.15: /);
  assert.match(record.scope, /SUPERSEDED wording, kept verbatim: 'and DOSE-GRADED changes on one arm only if RESDOSE rises ~39 pp'\]; \(2\) dose is the triangle family/);
  // Outside the bracket only the two-route sentence is left.
  const unbracketed = record.scope.replace(/ \[CORRECTED IN PLACE at cycle 153, CORRECTIONS 253\.15:[^\[\]]*\]/g, '');
  assert.doesNotMatch(unbracketed, /RESDOSE rises ~39 pp/);
  assert.match(unbracketed, /DOSE-GRADED changes on one arm only if RESDOSE rises, or MIDDOSE falls, by ~39 pp \(DOSE-NONMONOTONE\); \(2\)/);
  // The published MASTER-TABLE copy carries the corrected row, and MT227 still anchors line 227.
  const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
  const lines = readFileSync(new URL(`../public${master.href}`, import.meta.url), 'utf8').split('\n');
  assert.match(lines[226], /RESDOSE rises, or MIDDOSE falls, by ~39 pp \(DOSE-NONMONOTONE\) \*\*\[CORRECTED IN PLACE at cycle 153, CORRECTIONS 253\.15: /);
  assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 227));
});

test('the row-227 correction travels with MT227 as a documented amendment warning, outcome and badge unchanged', () => {
  const record = byId.get('MT227')!;
  const warning = data.warnings.find(w => w.id === 'warning-MT227-amendment-253')!;
  assert.ok(warning && record.warningIds.includes(warning.id));
  assert.equal(warning.title, 'Wording amended by a later result');
  assert.deepEqual([warning.severity, warning.status], ['limitation', 'documented']);
  assert.match(warning.detail, /^Outcome unchanged \(Mixed\)\. Amended at CORRECTIONS 253\.15/);
  assert.match(warning.detail, /RESDOSE rises or MIDDOSE falls by more than 39\.02 pp/);
  assert.match(warning.detail, /Amendment record: docs\/MASTER-TABLE\.md line 227 at 3cf4201; CORRECTIONS 253\.15\.$/);
  assert.ok(!data.warnings.some(w => w.experimentId === 'MT227' && w.severity === 'correction'), 'no Corrected badge');
  assert.ok(!data.warnings.some(w => w.id.startsWith('warning-MT226-amendment')), 'row 226 was not amended');
});
