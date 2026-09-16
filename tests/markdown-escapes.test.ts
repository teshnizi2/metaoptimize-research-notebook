import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData } from '../src/types.ts';

// MASTER-TABLE cells are Markdown, and the importer (scripts/register_model.py clean()) turns them into plain text.
// A backslash escape such as S\* is the literal character S*: the backslash must not reach the published text.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as unknown;

function* strings(value: unknown, path: string): Generator<[string, string]> {
  if (typeof value === 'string') yield [path, value];
  else if (Array.isArray(value)) for (const [index, item] of value.entries()) yield* strings(item, `${path}[${index}]`);
  else if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) yield* strings(item, `${path}.${key}`);
}

test('no published text carries a Markdown escape (\\*, \\_, \\|, \\`)', () => {
  const leaks = [...strings(data, 'research'), ...strings(runs, 'runs')]
    .flatMap(([path, text]) => [...text.matchAll(/\\[*_|`]/g)].map(match => `${path}: ${text.slice(Math.max(0, match.index! - 30), match.index! + 10)}`));
  assert.deepEqual(leaks, []);
});

test("MT215 names the selected cells S* and M*, with every star kept", () => {
  const mt215 = data.experiments.find(e => e.id === 'MT215')!;
  assert.match(mt215.result, /PRIMARY GAP_END = plateau5\(S\*\) - plateau5\(M\*\) = 65\.0573/);
  assert.match(mt215.result, /epoch-to-99 % 18-19 for M\* against 80 for S\*\)\. Selection: S\* and M\* are both max-of-5/);
  assert.match(mt215.result, /-2\.3489 pp at S\*, hence/);
  assert.match(mt215.scope, /it REVERSES at S\*\); RULE 11\./);
});
