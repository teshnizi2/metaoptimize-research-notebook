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
    assert.equal(current.title, old.title, `${old.id} must not be renumbered onto another row`);
    assert.equal(current.section, old.section, old.id);
  }
});

test('only the 37 partition-audit rows, the 6 appended rows, the cvt1 row and the cgn3 row are new, keyed on their line in the pinned MASTER-TABLE', () => {
  const old = new Set(fixture.experiments.map(e => e.id));
  const added = data.experiments.map(e => e.id).filter(id => !old.has(id));
  assert.deepEqual(added.sort(), [...partitionIds, ...appendedIds, 'MT218', 'MT219'].sort());
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
  for (const id of [...partitionIds, ...appendedIds, 'MT218', 'MT219']) assert.equal(lineOf(id), Number(id.slice(2)));
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
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !previous.experiments.some(e => e.id === id));
  assert.deepEqual(added, [...appendedIds, 'MT218', 'MT219'], 'the imports append exactly MT212-MT217, then MT218, then MT219, in line order');
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
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before218.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT218', 'MT219']);
  assert.equal(data.experiments.length, 156);
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
      [old.section, old.area, old.title, old.kind, old.outcome, old.corrected], old.id);
  }
  const added = data.experiments.map(e => e.id).filter(id => !before219.experiments.some(e => e.id === id));
  assert.deepEqual(added, ['MT219']);
  assert.equal(data.experiments.length, 156);
});

test('journal entries still resolve to existing experiments', () => {
  for (const entry of journal) for (const id of entry.experimentIds) assert.ok(byId.has(id), `${entry.id} references ${id}`);
});
