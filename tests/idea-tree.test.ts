import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { countStatuses, findNode, ideaTree, sentenceCount } from '../src/lib/idea-tree.ts';
import { ancestorsOf, buildChildren, expandableIds, fitView, layoutGraph, narrowRadial, radialLayout, wideRadial, wideSizes } from '../src/lib/idea-graph.ts';

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

test('radial view: question in the centre, parts on a ring, every box shown with everything open and none overlapping', () => {
  for (const sizes of [wideRadial, narrowRadial]) {
    const closed = radialLayout(ideaTree, new Set(['root']), sizes);
    assert.equal(closed.nodes.length, 1 + ideaTree.parts.length);
    const root = closed.nodes.find(n => n.id === 'root')!;
    assert.equal(root.radius, 0);
    const rings = new Set(closed.nodes.filter(n => n.kind === 'part').map(n => Math.round(n.radius!)));
    assert.equal(rings.size, 1, 'all parts sit on one ring');

    const all = radialLayout(ideaTree, new Set(expandableIds(ideaTree)), sizes);
    assert.equal(all.nodes.length, 1 + ideaTree.parts.length + 79);
    for (let i = 0; i < all.nodes.length; i += 1) for (let j = i + 1; j < all.nodes.length; j += 1) {
      const a = all.nodes[i], b = all.nodes[j];
      const apart = a.x + a.w + 2 <= b.x || b.x + b.w + 2 <= a.x || a.y + a.h + 2 <= b.y || b.y + b.h + 2 <= a.y;
      assert.ok(apart, `${a.id} overlaps ${b.id}`);
    }
    // A follow-up grows further out than the test it follows, and every test further out than its part.
    const at = new Map(all.nodes.map(n => [n.id, n.radius!]));
    for (const l of all.links) assert.ok(at.get(l.to)! > at.get(l.from)!, l.id);
  }
});

test('radial view: opening or folding a box never moves another box', () => {
  const one = radialLayout(ideaTree, new Set(['root', 'B7']), wideRadial);
  const more = radialLayout(ideaTree, new Set(['root', 'B7', 'B8', 'B7.1']), wideRadial);
  const at = new Map(more.nodes.map(n => [n.id, `${Math.round(n.x)},${Math.round(n.y)}`]));
  for (const n of one.nodes) assert.equal(at.get(n.id), `${Math.round(n.x)},${Math.round(n.y)}`, n.id);
});
