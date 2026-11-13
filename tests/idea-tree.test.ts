import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { countStatuses, findNode, ideaTree, sentenceCount } from '../src/lib/idea-tree.ts';
import { ancestorsOf, buildChildren, expandableIds, fitView, layoutGraph, wideSizes } from '../src/lib/idea-graph.ts';

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

test('follow-up links stay inside their part, point backwards and give each test one parent', () => {
  let n = 0;
  for (const p of ideaTree.parts) {
    const order = new Map(p.leaves.map((l, i) => [l.id, i]));
    for (const l of p.leaves) {
      if (!l.parent) continue;
      n += 1;
      assert.ok(order.has(l.parent), `${l.id} -> ${l.parent} leaves part ${p.id}`);
      assert.ok(order.get(l.parent)! < order.get(l.id)!, `${l.id}: parent must come earlier`);
      assert.ok(l.parentNote && l.parentNote.length > 0 && l.parentNote.split(' ').length <= 14, `${l.id} note`);
    }
  }
  assert.ok(n > 0);
  // Every test is reachable from the question exactly once.
  const kids = buildChildren(ideaTree);
  const seen = new Set<string>();
  const walk = (id: string) => { for (const c of kids.get(id) ?? []) { assert.ok(!seen.has(c), `${c} reached twice`); seen.add(c); walk(c); } };
  walk('root');
  assert.equal(seen.size, ideaTree.parts.length + ideaTree.parts.reduce((a, p) => a + p.leaves.length, 0));
});

test('the graph shows only what is open, and opening a box adds exactly its children', () => {
  const closed = layoutGraph(ideaTree, new Set(['root']), wideSizes);
  assert.equal(closed.nodes.length, 1 + ideaTree.parts.length);
  const b7 = ideaTree.parts.find(p => p.id === 'B7')!;
  const top = b7.leaves.filter(l => !l.parent).length;
  const opened = layoutGraph(ideaTree, new Set(['root', 'B7']), wideSizes);
  assert.equal(opened.nodes.length, closed.nodes.length + top);
  const all = layoutGraph(ideaTree, new Set(expandableIds(ideaTree)), wideSizes);
  assert.equal(all.nodes.length, 1 + ideaTree.parts.length + ideaTree.parts.reduce((a, p) => a + p.leaves.length, 0));
  assert.equal(all.links.length, all.nodes.length - 1);
  // Columns run left to right and no two cards in a column overlap.
  for (const n of all.nodes) for (const m of all.nodes) {
    if (n.id < m.id && n.depth === m.depth) assert.ok(n.y + n.h <= m.y || m.y + m.h <= n.y, `${n.id} overlaps ${m.id}`);
  }
  const root = all.nodes.find(n => n.id === 'root')!;
  assert.ok(all.nodes.filter(n => n.depth === 1).every(n => n.x > root.x + root.w));
});

test('a shared link opens every box above the selected test; fit keeps the graph inside the view', () => {
  assert.deepEqual(ancestorsOf(ideaTree, 'B7.3'), ['B7.2', 'B7.1', 'B7', 'root']);
  const v = fitView({ minX: 0, minY: 0, maxX: 1000, maxY: 500 }, 500, 400, 0.1, 1);
  assert.ok(v.k <= 0.5 && v.x >= 0 && v.y >= 0);
});
