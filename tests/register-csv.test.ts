import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import Papa from 'papaparse';
import type { ResearchData } from '../src/types.ts';

// The downloadable register is regenerated from the notebook register (scripts/register_model.py),
// so it lists every record with the four-outcome model instead of the campaign's 111-row export.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const text = readFileSync(new URL('../public/assets/tables/complete_experiment_register.csv', import.meta.url), 'utf8');
const parsed = Papa.parse<Record<string, string>>(text, { header: true, skipEmptyLines: true });
const rows = parsed.data;
const columns = ['id', 'kind', 'outcome', 'outcome_label', 'corrected', 'correction_note', 'section', 'area', 'goal', 'comparison', 'why', 'result', 'reason', 'scope', 'batches', 'sources', 'original_question', 'register_page', 'mapping_rule', 'master_table_line'];
const labels: Record<string, string> = { success: 'Goal met', fail: 'Goal missed', mixed: 'Mixed', unresolved: 'Open' };

test('the register CSV has one row per published record and the documented columns', () => {
  assert.deepEqual(parsed.errors, []);
  assert.deepEqual(parsed.meta.fields, columns);
  assert.equal(rows.length, data.experiments.length);
  assert.equal(rows.length, 177);
  assert.deepEqual(rows.map(row => row.id), data.experiments.map(e => e.id));
  const table = data.tables.find(entry => entry.href === '/assets/tables/complete_experiment_register.csv')!;
  assert.deepEqual([table.rows, table.columns], [177, columns.length], 'the table catalog reports the regenerated size');
});

test('every CSV row carries its four-way outcome, kind and a separate Corrected column', () => {
  const byId = new Map(data.experiments.map(e => [e.id, e]));
  assert.ok(rows.every(row => row.outcome !== 'correction'), 'the retired correction outcome is gone');
  for (const row of rows) {
    const record = byId.get(row.id)!;
    assert.equal(row.kind, record.kind, row.id);
    assert.equal(row.outcome, record.outcome ?? '', row.id);
    assert.equal(row.outcome_label, record.kind === 'method-check' ? 'Method check (no research outcome)' : labels[record.outcome!], row.id);
    assert.equal(row.corrected, record.corrected ? 'Corrected' : '', row.id);
    assert.equal(row.correction_note, record.correction?.note ?? '', row.id);
    assert.equal(Number(row.section), record.section, row.id);
    assert.equal(row.area, record.area, row.id);
  }
  const count = (key: string, value: string) => rows.filter(row => row[key] === value).length;
  assert.deepEqual([count('kind', 'research'), count('kind', 'method-check'), count('corrected', 'Corrected')], [159, 18, 35]);
  assert.deepEqual(['success', 'fail', 'mixed', 'unresolved'].map(outcome => count('outcome', outcome)), [56, 41, 38, 24]);
  assert.deepEqual(['outcome', 'outcome_label'].map(key => rows.find(row => row.id === 'MT019')![key]), ['success', 'Goal met'], 'MT019 moved from Open at CORRECTIONS 229');
  assert.deepEqual(rows.filter(row => /^MT2(1[2-9]|2\d|3[0-5])$/.test(row.id)).map(row => [row.id, row.master_table_line, row.outcome_label]),
    [['MT212', '212', 'Goal met'], ['MT213', '213', 'Mixed'], ['MT214', '214', 'Goal met'], ['MT215', '215', 'Goal met'], ['MT216', '216', 'Goal met'], ['MT217', '217', 'Goal met'], ['MT218', '218', 'Mixed'], ['MT219', '219', 'Goal met'], ['MT220', '220', 'Mixed'], ['MT221', '221', 'Mixed'], ['MT222', '222', 'Goal met'], ['MT223', '223', 'Goal met'], ['MT224', '224', 'Mixed'], ['MT225', '225', 'Goal met'], ['MT226', '226', 'Mixed'], ['MT227', '227', 'Mixed'], ['MT228', '228', 'Mixed'], ['MT229', '229', 'Mixed'], ['MT230', '230', 'Goal met'], ['MT231', '231', 'Mixed'], ['MT232', '232', 'Goal met'], ['MT233', '233', 'Goal met'], ['MT234', '234', 'Goal met'], ['MT235', '235', 'Goal met']]);
});
