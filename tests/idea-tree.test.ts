import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { countStatuses, findNode, ideaTree, sentenceCount } from '../src/lib/idea-tree.ts';

const research = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8'));
const known = new Set<string>(research.experiments.map((e: { id: string }) => e.id));
const nodes = ideaTree.parts.flatMap(p => [p, ...p.leaves]);

test('the idea tree shows 11 parts and 79 tests, and leaves out the unpublished review paper', () => {
  assert.equal(ideaTree.parts.length, 11);
  assert.equal(ideaTree.parts.reduce((n, p) => n + p.leaves.length, 0), 79);
  assert.ok(!ideaTree.parts.some(p => p.id === 'B11' || p.number === 11));
  assert.ok(!/paperfactory|review paper/i.test(JSON.stringify(ideaTree)));
});

test('every node has a 2-3 sentence summary, a valid colour and a date', () => {
  for (const n of nodes) {
    const s = sentenceCount(n.summary);
    assert.ok(s >= 2 && s <= 3, `${n.id}: ${s} sentences`);
    assert.ok(['g', 'r', 'a', 'u'].includes(n.status), n.id);
  }
  for (const p of ideaTree.parts) for (const l of p.leaves) {
    assert.ok(l.date.length > 0, l.id);
    assert.ok(['primer', 'runs', 'part'].includes(l.dateBasis), l.id);
  }
});

test('every test page link points at a real experiment, and no node shows a made-up one', () => {
  for (const p of ideaTree.parts) for (const l of p.leaves) {
    assert.ok(l.experiments.length <= 3, l.id);
    for (const id of l.experiments) assert.ok(known.has(id), `${l.id} -> ${id}`);
  }
  assert.equal(findNode(ideaTree, 'B4.7')?.kind, 'leaf');
  assert.equal(findNode(ideaTree, 'nope'), null);
});

test('the tree text has no private paths, names or internal reference marks', () => {
  const text = JSON.stringify(ideaTree);
  assert.ok(!/\/Users\/|\/home\/|@|teshnizi|Reza|UNSURE|#\d|D l\.\d/.test(text));
});

test('status counts add up to the number of tests', () => {
  const all = ideaTree.parts.flatMap(p => p.leaves);
  const c = countStatuses(all);
  assert.equal(c.g + c.r + c.a + c.u, all.length);
});
