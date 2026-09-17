import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData } from '../src/types.ts';

// A MASTER-TABLE ref cell is split into citations on ';'. A ';' inside brackets is part of one citation, not a separator:
// MASTER-TABLE rows 224 and 225 (campaign commit dae2a49) read 'CORRECTIONS 242 (registration and launch; RULE H's ...
// corrected in place at 246), 246 (landing)', and splitting inside the parentheses published the label cut off at
// 'CORRECTIONS 242 (registration and launch' and dropped the rest. Every label must close every bracket it opens.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const links = JSON.parse(readFileSync(new URL('../public/data/source-links.json', import.meta.url), 'utf8')) as
  Record<string, { sourceRefs: { label: string }[] }>;

function unbalanced(text: string) {
  const close: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
  const stack: string[] = [];
  for (const char of text) {
    if ('([{'.includes(char)) stack.push(char);
    else if (char in close && stack.pop() !== close[char]) return true;
  }
  return stack.length > 0;
}

test('the bracket check itself catches an unclosed or crossed bracket', () => {
  assert.equal(unbalanced('CORRECTIONS 242 (registration and launch'), true);
  assert.equal(unbalanced('a (b] c'), true);
  assert.equal(unbalanced("CORRECTIONS 242 (registration and launch; 242.3(2) corrected at 246), 246 (landing)"), false);
});

test('every published source label has balanced brackets', () => {
  const research = data.experiments.flatMap(e => e.sourceRefs.map(ref => `${e.id}: ${ref.label}`));
  const catalog = Object.entries(links).flatMap(([id, entry]) => entry.sourceRefs.map(ref => `${id}: ${ref.label}`));
  assert.ok(research.length > 1000 && catalog.length === research.length);
  assert.deepEqual([...research, ...catalog].filter(unbalanced), []);
});

test('MT224 and MT225 cite their whole MASTER-TABLE ref cell, registration and landing', () => {
  const cited = (id: string) => data.experiments.find(e => e.id === id)!.sourceRefs.filter(ref => /^Cited research record: CORRECTIONS /.test(ref.label)).map(ref => ref.label);
  assert.deepEqual(cited('MT224'), ["Cited research record: CORRECTIONS 242 (registration and launch; RULE H's '12 runs', 242.14 and 242.3(2) corrected in place at 246), 246 (landing)"]);
  assert.deepEqual(cited('MT225'), ["Cited research record: CORRECTIONS 243 (registration and launch; 243.4's cross-reference corrected in place at 247), 247 (landing)"]);
});
