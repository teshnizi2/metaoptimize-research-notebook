import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData } from '../src/types.ts';

// The notebook shows record text as plain text. Bold (**x**, __x__) and code (`x`) marks in MASTER-TABLE cells and in the
// campaign's register export must be removed by the importer (scripts/register_model.py clean()); otherwise titles and
// scopes show literal ** and backticks. Hrefs are file paths (measurement__adaptation_dial__values.csv) and are not text.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as unknown;
const fixture = JSON.parse(readFileSync(new URL('./fixtures/register-ids-fe542ec.json', import.meta.url), 'utf8')) as {
  experiments: { id: string; section: number; area: string; title: string; kind: string; outcome: string | null; corrected: boolean }[];
};

function* strings(value: unknown, path: string): Generator<[string, string]> {
  if (typeof value === 'string') yield [path, value];
  else if (Array.isArray(value)) for (const [index, item] of value.entries()) yield* strings(item, `${path}[${index}]`);
  else if (value && typeof value === 'object')
    for (const [key, item] of Object.entries(value)) if (!/href$/i.test(key)) yield* strings(item, `${path}.${key}`);
}

const marks = (text: string) => [...text.matchAll(/\*\*|__|`/g)];

test('no published record text carries a Markdown bold or code mark (**, __, `)', () => {
  const leaks = [...strings(data, 'research'), ...strings(runs, 'runs')]
    .flatMap(([path, text]) => marks(text).map(match => `${path}: ${text.slice(Math.max(0, match.index! - 30), match.index! + 12)}`));
  assert.deepEqual(leaks, []);
});

test('every record keeps its ID, section, area, kind, outcome and Corrected badge; a title changes only by losing marks', () => {
  assert.equal(fixture.experiments.length, 160);
  const byId = new Map(data.experiments.map(e => [e.id, e]));
  // MT224-MT238 (cvt6, cvt7, cvt8, cvt9, the four MUST-tier batches, then cvt10, cwd1, csv1, cwd2, cwd3, cwd4 and cwd5) were appended after
  // the fix (tests/cvt6-cvt7-landing.test.ts, tests/cvt8-cvt9-landing.test.ts, tests/must-tier-landing.test.ts, tests/mech4-landing.test.ts, tests/mech5-landing.test.ts, tests/mech6-landing.test.ts).
  assert.deepEqual(data.experiments.slice(0, 160).map(e => e.id), fixture.experiments.map(e => e.id));
  assert.deepEqual(data.experiments.slice(160).map(e => e.id), ['MT224', 'MT225', 'MT226', 'MT227', 'MT228', 'MT229', 'MT230', 'MT231', 'MT232', 'MT233', 'MT234', 'MT235', 'MT236', 'MT237', 'MT238', 'MT239', 'MT240', 'MT241', 'MT242']);
  const plain = (text: string) => text.replace(/\*\*|`/g, '').replace(/\s+/g, ' ').trim();
  let retitled = 0;
  for (const old of fixture.experiments) {
    const current = byId.get(old.id)!;
    assert.deepEqual([current.section, current.area, current.kind, current.outcome, current.corrected],
      [old.section, old.area, old.kind, old.outcome, old.corrected], old.id);
    assert.equal(current.title, plain(old.title), old.id);
    if (current.title !== old.title) retitled++;
  }
  assert.equal(retitled, 20, 'the 20 titles that showed marks at fe542ec');
});

test('the records that showed marks read as plain text', () => {
  const byId = new Map(data.experiments.map(e => [e.id, e]));
  assert.equal(byId.get('MT033')!.title, 'Is the granularity ladder flat once EVERY rung sits at its own optimal ms?');
  assert.match(byId.get('MT020')!.title, /when AUGMENT=0 on every arm/);
  assert.equal(byId.get('MT139')!.scope, 'cbl1, 33 runs, CIFAR-100/ResNet18_c100, ms=1e-3, alpha0=1e-6, 100 ep, ONE batch');
  assert.equal(byId.get('MT147')!.title, 'Does the argmax k*=49 move at a converged budget on fresh seeds?', 'a lone star is a literal character');
});
