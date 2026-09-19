import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ActivityEvent, ResearchData, SourceFile } from '../src/types.ts';

// Experiment IDs are permanent: journal entries, bookmarks and the GitHub history
// reference them. The fixture is the register as published before the
// four-outcome model and the partition-audit import (research.json at 2590621).
const fixture = JSON.parse(readFileSync(new URL('./fixtures/register-ids-2026-09-15.json', import.meta.url), 'utf8')) as {
  experiments: { id: string; section: number; title: string; outcome: string }[];
};
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const journal = JSON.parse(readFileSync(new URL('../content/journal.json', import.meta.url), 'utf8')) as ActivityEvent[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
// Fixtures before fe542ec keep the ** and backtick marks some titles showed then; the importer now removes them
// (tests/markdown-marks.test.ts), so a title is compared with its marks stripped.
const plain = (title: string) => title.replace(/\*\*|`/g, '').replace(/\s+/g, ' ').trim();
const partitionIds = Array.from({ length: 37 }, (_, i) => `MT${175 + i}`);
// MASTER-TABLE lines 212-217 at campaign commit 64e4f47 (cgn1, cpl1, cvh1, cuc1, cgn2, cpl2).
const appendedIds = Array.from({ length: 6 }, (_, i) => `MT${212 + i}`);
const previous = JSON.parse(readFileSync(new URL('./fixtures/register-ids-2026-09-16.json', import.meta.url), 'utf8')) as {
  experiments: { id: string; section: number; area: string; title: string; kind: string; outcome: string | null; corrected: boolean; outcomeChange?: { from: string; to: string; source: string } }[];
};

test('every previously published experiment keeps its ID, title and section', () => {
  assert.equal(fixture.experiments.length, 111);
  for (const old of fixture.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.equal(current.title, plain(old.title), `${old.id} must not be renumbered onto another row`);
    assert.equal(current.section, old.section, old.id);
  }
});

test('only the 37 partition-audit rows, the 6 appended rows, the cvt1 row, the cgn3 row, the cvt3 and cvt2 rows, the cvt4 and cvt5 rows, the cvt6 and cvt7 rows, the cvt8 and cvt9 rows and the four MUST-tier rows are new, keyed on their line in the pinned MASTER-TABLE', () => {
  const old = new Set(fixture.experiments.map(e => e.id));
  const added = data.experiments.map(e => e.id).filter(id => !old.has(id));
  assert.deepEqual(added.sort(), [...partitionIds, ...appendedIds, 'MT218', 'MT219', 'MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236'].sort());
  assert.equal(new Set(data.experiments.map(e => e.id)).size, data.experiments.length, 'no duplicate IDs');
  assert.ok(fixture.experiments.every(e => !/^MT\d{3}$/.test(e.id) || Number(e.id.slice(2)) < 167), 'new IDs cannot collide with old ones');
  // The cau1 and cvk1 rows already have records (MT020 and CVK2); no duplicates were created for lines 166-167.
  assert.ok(!byId.has('MT167') && !byId.has('MT168'));
  assert.match(byId.get('MT020')!.title, /UNAUGMENTED/);
  assert.ok(partitionIds.every(id => byId.get(id)!.section === 10 && byId.get(id)!.area === 'Count-matched partition audit'));
});

test('MASTER-TABLE anchors resolve by row content, not by the old line numbers', () => {
  const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md') as SourceFile;
  const lines = readFileSync(new URL(`../public${master.href}`, import.meta.url), 'utf8').split('\n');
  const norm = (text: string) => text.replace(/[*`]/g, '').toLowerCase().replace(/[^a-z0-9]/g, '');
  let anchored = 0;
  for (const experiment of data.experiments) {
    for (const ref of experiment.sourceRefs.filter(r => r.sourceId === master.id && r.label.startsWith('Cited research record'))) {
      const cell = lines[ref.line! - 1].split('|')[1] ?? '';
      assert.ok(norm(cell).includes(norm(experiment.title).slice(0, 60)), `${experiment.id} anchor L${ref.line} must be its own row`);
      anchored++;
    }
  }
  assert.ok(anchored >= 140, `expected most records to anchor into MASTER-TABLE, got ${anchored}`);
  const lineOf = (id: string) => byId.get(id)!.sourceRefs.find(r => r.sourceId === master.id && r.label.startsWith('Cited research record'))!.line;
  assert.equal(lineOf('MT014'), 14);
  assert.equal(lineOf('MT026'), 25, 'MT026 was keyed on an uncommitted snapshot one line longer; its row is line 25 in every commit');
  assert.equal(lineOf('MT020'), 166, 'the cau1 row was appended at line 166');
  for (const id of [...partitionIds, ...appendedIds, 'MT218', 'MT219', 'MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']) assert.equal(lineOf(id), Number(id.slice(2)));
});

test('all 148 records published before the lines 212-217 import keep their ID, title, area, kind and outcome', () => {
  assert.equal(previous.experiments.length, 148);
  // Outcome changes are recorded in the fixture on purpose, never absorbed as drift.
  assert.deepEqual(previous.experiments.filter(e => e.outcomeChange).map(e => [e.id, e.outcomeChange!.from, e.outcomeChange!.to]), [['MT019', 'unresolved', 'success']]);
  assert.match(previous.experiments.find(e => e.id === 'MT019')!.outcomeChange!.source, /CORRECTIONS 229/);
  for (const old of previous.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, plain(old.title), old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !previous.experiments.some(e => e.id === id));
  assert.deepEqual(added, [...appendedIds, 'MT218', 'MT219', 'MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236'], 'the imports append exactly MT212-MT217, then MT218, then MT219, then MT220-MT221, then MT222-MT223, then MT224-MT225, then MT226-MT227, then MT228-MT231, then MT232-MT235, then MT236, in line order');
});

test('the appended MASTER-TABLE rows carry their registered verdict, area and no duplicate of cau1 or cvk1', () => {
  const expected: Record<string, [string, string, string]> = {
    MT212: ['cgn1', 'Mechanism and isolation', 'success'],
    MT213: ['cpl1', 'Mechanism and isolation', 'mixed'],
    MT214: ['cvh1', 'Mechanism and isolation', 'success'],
    MT215: ['cuc1', 'Baseline comparisons', 'success'],
    MT216: ['cgn2', 'Mechanism and isolation', 'success'],
    MT217: ['cpl2', 'Mechanism and isolation', 'success'],
  };
  const firstToken: Record<string, string> = { MT212: 'GAP-REPLICATES', MT213: 'HEAD-CARRIES-PLAIN', MT214: 'RESCUE-SURVIVES', MT215: 'DEFICIT-HOLDS', MT216: 'IDENTITY-TRANSFERS-GN', MT217: 'HEAD-CARRIES-ALONE-PLAIN' };
  for (const [id, [batch, area, outcome]] of Object.entries(expected)) {
    const record = byId.get(id)!;
    assert.deepEqual([record.batches, record.area, record.outcome, record.kind, record.corrected], [[batch], area, outcome, 'research', false], id);
    assert.ok(record.reason.startsWith(`Verdict: ${firstToken[id]} + `), `${id} reason quotes the registered FINAL tokens`);
    assert.ok(!/^\[/.test(record.title), `${id} title drops the appended-row provenance label`);
    assert.ok(record.eventIds.includes('phase-09'), id);
    assert.ok(record.figureIds.length && record.tableIds.length && record.sourceRefs.length, id);
  }
  const batches = data.experiments.flatMap(e => e.batches);
  assert.equal(batches.filter(b => b === 'cau1').length, 2, 'cau1 stays on MT019 and MT020 only');
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cvk1')).map(e => e.id), ['CVK2']);
  // cuc1 closed MT019's remaining counts (CORRECTIONS 229), so it links there as cau1 does; every other batch has one record.
  assert.deepEqual(data.experiments.filter(e => e.batches.includes('cuc1')).map(e => e.id), ['MT019', 'MT215']);
  for (const batch of Object.values(expected).map(([b]) => b).filter(b => b !== 'cuc1')) assert.equal(data.experiments.filter(e => e.batches.includes(batch)).length, 1, batch);
});

// The 154 records published at 368a732, before MASTER-TABLE line 218 (cvt1) was imported.
const before218 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-368a732.json', import.meta.url), 'utf8')) as typeof previous;

test('all 154 records published before the line-218 import keep their ID, title, area, kind and outcome; only MT218 is added', () => {
  assert.equal(before218.experiments.length, 154);
  assert.ok(before218.experiments.every(e => !e.outcomeChange), 'the cvt1 import moves no outcome');
  for (const old of before218.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, plain(old.title), old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before218.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT218', 'MT219', 'MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']);
  assert.equal(data.experiments.length, 173);
});

// The 155 records published at acbc2b0, before MASTER-TABLE line 219 (cgn3) was imported and rows 213 and 218
// took CORRECTIONS 231's in-place amendments.
const before219 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-acbc2b0.json', import.meta.url), 'utf8')) as typeof previous;

test('all 155 records published before the line-219 import keep their ID, title, area, kind and outcome; only MT219 is added', () => {
  assert.equal(before219.experiments.length, 155);
  assert.ok(before219.experiments.every(e => !e.outcomeChange), 'the cgn3 import and the CORRECTIONS 231 amendments move no outcome');
  assert.deepEqual(before219.experiments.map(e => e.id), data.experiments.slice(0, 155).map(e => e.id), 'existing records keep their order');
  for (const old of before219.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, plain(old.title), old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before219.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT219', 'MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']);
  assert.equal(data.experiments.length, 173);
});

// The 156 records published at 944fa47, before MASTER-TABLE lines 220-221 (cvt3, cvt2) were imported and rows 213
// and 216 took CORRECTIONS 234's and 236's in-place amendments.
const before220 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-944fa47.json', import.meta.url), 'utf8')) as typeof previous;

test('all 156 records published before the lines 220-221 import keep their ID, title, area, kind and outcome; only MT220 and MT221 are added', () => {
  assert.equal(before220.experiments.length, 156);
  assert.ok(before220.experiments.every(e => !e.outcomeChange), 'the cvt3 / cvt2 import and the CORRECTIONS 234 / 236 amendments move no outcome');
  assert.deepEqual(before220.experiments.map(e => e.id), data.experiments.slice(0, 156).map(e => e.id), 'existing records keep their order');
  for (const old of before220.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, plain(old.title), old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before220.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT220', 'MT221', 'MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']);
  assert.equal(data.experiments.length, 173);
});

// The 158 records published at d2fb38b, before MASTER-TABLE lines 222-223 (cvt4, cvt5) were imported. The landing
// amended no row in place, so no earlier record may change its ID, title, area, kind, outcome or order.
const before222 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-d2fb38b.json', import.meta.url), 'utf8')) as typeof previous;

test('all 158 records published before the lines 222-223 import keep their ID, title, area, kind and outcome; only MT222 and MT223 are added', () => {
  assert.equal(before222.experiments.length, 158);
  assert.ok(before222.experiments.every(e => !e.outcomeChange), 'the cvt4 / cvt5 import moves no outcome');
  assert.deepEqual(before222.experiments.map(e => e.id), data.experiments.slice(0, 158).map(e => e.id), 'existing records keep their order');
  for (const old of before222.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, plain(old.title), old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before222.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT222', 'MT223', 'MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']);
  assert.equal(data.experiments.length, 173);
});

// The 160 records published at 52db8b6, before MASTER-TABLE lines 224-225 (cvt6, cvt7) were imported. That landing amended
// no row in place (only header line 3 and line 5, which feeds no record), so no earlier record may change.
const before224 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-52db8b6.json', import.meta.url), 'utf8')) as typeof previous;

test('all 160 records published before the lines 224-225 import keep their ID, title, area, kind and outcome; only MT224 and MT225 are added', () => {
  assert.equal(before224.experiments.length, 160);
  assert.deepEqual(before224.experiments.map(e => e.id), data.experiments.slice(0, 160).map(e => e.id), 'existing records keep their order');
  for (const old of before224.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before224.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236']);
  assert.equal(data.experiments.length, 173);
});

// The 162 records published at ad152c3. The follow-up fixed two source labels, one intervention note and re-exported sources
// at campaign commit 4ff0891 (CORRECTIONS 250: the cvt6 / cvt7 runners archived); none of that may move a record.
const at225 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-ad152c3.json', import.meta.url), 'utf8')) as typeof previous;

test('all 162 records published at ad152c3 keep their ID, section, area, title, kind, outcome, Corrected badge and order', () => {
  assert.equal(at225.experiments.length, 162);
  // MT226 and MT227 (cvt8, cvt9) were appended after them (the next test).
  assert.deepEqual(at225.experiments.map(e => e.id), data.experiments.slice(0, 162).map(e => e.id), 'no record removed or reordered');
  for (const old of at225.experiments) {
    const current = byId.get(old.id)!;
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
});

// The 162 records published at 57b9ad5, before MASTER-TABLE lines 226-227 (cvt8, cvt9) were imported at campaign commit ba01f54
// (CORRECTIONS 252-253). That landing amended no row in place (header line 3 and line 5 only, which feed no record), so no
// earlier record may move; only MT226 and MT227 are appended.
const before226 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-57b9ad5.json', import.meta.url), 'utf8')) as typeof previous;

test('all 162 records published at 57b9ad5 keep their ID, section, area, title, kind, outcome, Corrected badge and order; only MT226 and MT227 are added', () => {
  assert.equal(before226.experiments.length, 162);
  assert.deepEqual(before226.experiments.map(e => e.id), data.experiments.slice(0, 162).map(e => e.id), 'existing records keep their order');
  for (const old of before226.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  assert.deepEqual(data.experiments.slice(162, 164).map(e => e.id), ['MT226', 'MT227']);
  assert.equal(data.experiments.length, 173);
});

// The 164 records published at 2969a1c. The final audit of the cvt8 / cvt9 landing (CORRECTIONS 253.15; campaign commit 3cf4201)
// corrected MASTER-TABLE row 227's wording in place and fixed MT226's reason; wording only, so no record may move.
const at227 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-2969a1c.json', import.meta.url), 'utf8')) as typeof previous;

test('all 164 records published at 2969a1c keep their ID, section, area, title, kind, outcome, Corrected badge and order', () => {
  assert.equal(at227.experiments.length, 164);
  assert.deepEqual(at227.experiments.map(e => e.id), data.experiments.slice(0, 164).map(e => e.id), 'no earlier record removed or reordered');
  for (const old of at227.experiments) {
    const current = byId.get(old.id)!;
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
});

// The 164 records published at 474fc17, the last data-bearing commit before MASTER-TABLE lines 228-231 (cmo1, cst1, cct1, cmg1)
// were imported at campaign commit 40d29cf (CORRECTIONS 264-266). Main then took two editorial commits, dd33ea4 and 6ca8e80,
// which changed no research data. That landing amended no row in place (header line 3 only, which feeds no record), so no
// earlier record may move; only MT228-MT231 are appended, in line order.
const before228 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-474fc17.json', import.meta.url), 'utf8')) as typeof previous;

test('all 164 records published at 474fc17 keep their ID, section, area, title, kind, outcome, Corrected badge and order; only MT228-MT231 are added', () => {
  assert.equal(before228.experiments.length, 164);
  assert.ok(before228.experiments.every(e => !e.outcomeChange), 'the MUST-tier import moves no outcome');
  assert.deepEqual(before228.experiments.map(e => e.id), data.experiments.slice(0, 164).map(e => e.id), 'existing records keep their order');
  for (const old of before228.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  assert.deepEqual(data.experiments.slice(164, 168).map(e => e.id), ['MT228', 'MT229', 'MT230', 'MT231']);
  // The four records and their outcomes, pinned here so a later import cannot move them silently. MT229 is the one that HAS
  // moved since, and only by the documented in-place amendment of CORRECTIONS 273 (Open -> Mixed, no Corrected badge), which
  // the next fixture pins; the other three are byte-for-byte what the MUST-tier landing published.
  assert.deepEqual(data.experiments.slice(164, 168).map(e => [e.id, e.batches, e.area, e.kind, e.outcome, e.corrected]), [
    ['MT228', ['cmo1'], 'Mechanism and isolation', 'research', 'mixed', false],
    ['MT229', ['cst1'], 'Mechanism and isolation', 'research', 'mixed', false],
    ['MT230', ['cct1'], 'Mechanism and isolation', 'research', 'success', false],
    ['MT231', ['cmg1'], 'Mechanism and isolation', 'research', 'mixed', false],
  ]);
  assert.equal(data.experiments.length, 173);
});

// The 168 records published at notebook commit 2cf35f1, the main this import started from. MASTER-TABLE lines 232-235
// (cvt10, cwd1, csv1, cwd2) were imported at campaign commit 2972d48 (CORRECTIONS 270-273). That landing amended ONE row in
// place, row 229 (MT229), so MT229's outcome moves from Open to Mixed and nothing else about any earlier record may change.
const before232 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-2cf35f1.json', import.meta.url), 'utf8')) as typeof previous;

test('all 168 records published at 2cf35f1 keep their ID, section, area, title, kind and badge; only MT229 moves, and only MT232-MT235 are added', () => {
  assert.equal(before232.experiments.length, 168);
  assert.deepEqual(before232.experiments.map(e => e.id), data.experiments.slice(0, 168).map(e => e.id), 'existing records keep their order');
  for (const old of before232.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    // The Corrected badge stays off everywhere: an outcome moved by later data is not a corrected earlier claim.
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.corrected],
      [old.section, old.area, old.title, old.kind, old.corrected], old.id);
    if (old.id !== 'MT229') assert.equal(current.outcome, old.outcome, `${old.id} outcome must not move`);
  }
  const moved = before232.experiments.filter(old => byId.get(old.id)!.outcome !== old.outcome);
  assert.deepEqual(moved.map(old => [old.id, old.outcome, byId.get(old.id)!.outcome]), [['MT229', 'unresolved', 'mixed']]);
  const warning = data.warnings.find(w => w.id === 'warning-MT229-amendment-273')!;
  assert.ok(warning && byId.get('MT229')!.warningIds.includes(warning.id), 'the move is recorded as an amendment warning');
  assert.equal(warning.title, 'Outcome moved by a later result');
  assert.match(warning.detail, /Outcome before the amendment: Open\./);
  assert.equal(byId.get('MT229')!.corrected, false, 'no Corrected badge: the predecessor was never edited and its reading stands');
  // The four new records and their outcomes, pinned so a later import cannot move them silently.
  assert.deepEqual(data.experiments.slice(168, 172).map(e => [e.id, e.batches, e.area, e.kind, e.outcome, e.corrected]), [
    ['MT232', ['cvt10'], 'Mechanism and isolation', 'research', 'success', false],
    ['MT233', ['cwd1'], 'Mechanism and isolation', 'research', 'success', false],
    ['MT234', ['csv1'], 'Mechanism and isolation', 'research', 'success', false],
    ['MT235', ['cwd2'], 'Mechanism and isolation', 'research', 'success', false],
  ]);
  assert.equal(data.experiments.length, 173);
});

// The 172 records published at notebook commit 98b0e8d, the main this import started from. MASTER-TABLE line 236 (cwd3)
// was imported at campaign commit 97eb049 (CORRECTIONS 278). That landing amended NO row in place -- it edited only header
// line 3 -- so no earlier record may change its ID, section, area, title, kind, outcome, badge or order.
const before236 = JSON.parse(readFileSync(new URL('./fixtures/register-ids-98b0e8d.json', import.meta.url), 'utf8')) as typeof previous;

test('all 172 records published at 98b0e8d keep their ID, section, area, title, kind, outcome, Corrected badge and order; only MT236 is added', () => {
  assert.equal(before236.experiments.length, 172);
  assert.deepEqual(before236.experiments.map(e => e.id), data.experiments.slice(0, 172).map(e => e.id), 'existing records keep their order');
  for (const old of before236.experiments) {
    const current = byId.get(old.id);
    assert.ok(current, `${old.id} must not disappear`);
    assert.deepEqual([current.section, current.area, current.title, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  // The cwd3 landing moved NO outcome anywhere, which is what separates it from the three landings before it.
  assert.deepEqual(before236.experiments.filter(old => byId.get(old.id)!.outcome !== old.outcome).map(old => old.id), []);
  assert.ok(!data.warnings.some(w => w.id.endsWith('-amendment-278')), 'no row was amended at CORRECTIONS 278');
  const added = data.experiments.map(e => e.id).filter(id => !before236.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT236']);
  // The one new record and its outcome, pinned so a later import cannot move it silently.
  assert.deepEqual(data.experiments.slice(172).map(e => [e.id, e.batches, e.area, e.kind, e.outcome, e.corrected]), [
    ['MT236', ['cwd3'], 'Mechanism and isolation', 'research', 'success', false],
  ]);
  assert.equal(data.experiments.length, 173);
});

test('journal entries still resolve to existing experiments', () => {
  for (const entry of journal) for (const id of entry.experimentIds) assert.ok(byId.has(id), `${entry.id} references ${id}`);
});
