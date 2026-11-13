import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { learnFileType } from '../api/learn-file.ts';
import { attachedIdeas, branchId, findIdeaNode, futureRows, ideaBranches, IDEAS_URL, IDEA_LEGEND, parseIdeas, type FutureIdea } from '../src/lib/future-ideas.ts';
import { changedRows } from '../src/lib/idea-changed.ts';
import { ancestorsOf, childLabel, expandableIds, layoutGraph, narrowRadial, narrowSizes, radialLayout, wideRadial, wideSizes } from '../src/lib/idea-graph.ts';
import { ideaTree } from '../src/lib/idea-tree.ts';

/** Made-up ideas only: the real ones are unpublished research and never go in this repository. */
const raw = [
  { id: 'F1', title: 'Placeholder idea one', question: 'Placeholder question?', kind: 'continue', attachTo: 'B1.1', whatChanged: { dimension: 'Optimiser', chip: 'Placeholder optimiser' }, why: 'Placeholder reason, see B1.1 and B8.5.', test: 'Placeholder test.', cost: '2 GPU-hours (estimate)', outcomes: ['Placeholder win', 'Placeholder loss'], paperValue: 'Placeholder value', feasibility: 4, priority: 'now', risks: ['Placeholder risk'] },
  { id: 'F2', title: 'Placeholder idea two', question: 'Placeholder question two?', kind: 'check', attachTo: 'B9', whatChanged: { dimension: 'Placeholder dimension', chip: 'Placeholder chip' }, why: 'Placeholder.', test: 'Placeholder.', cost: 'estimate', outcomes: 'Placeholder outcome', paperValue: 'Placeholder', feasibility: 2, priority: 'next', risks: 'Placeholder risk' },
  { id: 'F3', title: 'Placeholder idea three', question: 'Q?', kind: 'new-branch', attachTo: { newBranch: 'Placeholder branch' }, why: 'W', test: 'T', cost: 'C', outcomes: [], paperValue: 'V', feasibility: 3, priority: 'later', risks: [] },
  { id: 'F4', title: 'Placeholder idea four', question: 'Q?', kind: 'new-category', attachTo: { newBranch: 'Placeholder branch' }, why: 'W', test: 'T', cost: 'C', outcomes: [], paperValue: 'V', feasibility: 3, priority: 'later', risks: [] },
];
const ideas: FutureIdea[] = parseIdeas(raw);

test('the ideas file is read defensively', () => {
  assert.equal(ideas.length, 4);
  assert.deepEqual(ideas[1].outcomes, ['Placeholder outcome']);
  assert.deepEqual(ideas[1].risks, ['Placeholder risk']);
  assert.deepEqual(parseIdeas([null, 3, { id: 'X' }, { id: 'F1', title: 'a', attachTo: 'B1' }, { id: 'F1', title: 'dupe', attachTo: 'B1' }, { id: 'F9', title: 'big', attachTo: 'B1', feasibility: 99, priority: 'urgent', kind: 'weird' }]).map(i => [i.id, i.feasibility, i.priority, i.kind]), [['F1', 3, 'later', 'continue'], ['F9', 5, 'later', 'continue']]);
  assert.deepEqual(parseIdeas({ ideas: raw }).map(i => i.id), ['F1', 'F2', 'F3', 'F4']);
  assert.deepEqual(parseIdeas('not a list'), []);
  assert.deepEqual(parseIdeas(undefined), []);
});

test('only ideas that can be drawn are kept', () => {
  const stray = parseIdeas([{ id: 'F5', title: 'x', attachTo: 'B99.9' }, { id: 'B1.1', title: 'clash', attachTo: 'B1' }, { id: 'F6', title: 'on the question', attachTo: 'root' }]);
  assert.deepEqual(attachedIdeas(ideaTree, [...ideas, ...stray]).map(i => i.id), ['F1', 'F2', 'F3', 'F4']);
  assert.equal(findIdeaNode(ideas, 'F1')?.kind, 'idea');
  assert.equal(findIdeaNode(ideas, branchId('Placeholder branch'))?.kind, 'branch');
  assert.equal(findIdeaNode(ideas, 'B1.1'), null);
  assert.deepEqual(ideaBranches(ideas).map(b => [b.name, b.ideas.length]), [['Placeholder branch', 2]]);
});

const all = new Set([...expandableIds(ideaTree, ideas)]);

test('no ideas: both layouts are exactly what they were', () => {
  const open = new Set(expandableIds(ideaTree));
  for (const s of [wideSizes, narrowSizes]) assert.deepEqual(layoutGraph(ideaTree, open, s, []), layoutGraph(ideaTree, open, s));
  for (const s of [wideRadial, narrowRadial]) assert.deepEqual(radialLayout(ideaTree, open, s, []), radialLayout(ideaTree, open, s));
  assert.ok(!layoutGraph(ideaTree, open, wideSizes).nodes.some(n => n.kind === 'idea' || n.kind === 'branch'));
});

test('ideas hang off the part or test they grow from, new branches off the question, in both graphs', () => {
  for (const layout of [layoutGraph(ideaTree, all, wideSizes, ideas), radialLayout(ideaTree, all, wideRadial, ideas)]) {
    const byId = new Map(layout.nodes.map(n => [n.id, n]));
    const parentOf = (id: string) => layout.links.find(l => l.to === id)?.from;
    assert.equal(parentOf('F1'), 'B1.1');
    assert.equal(parentOf('F2'), 'B9');
    assert.equal(parentOf(branchId('Placeholder branch')), 'root');
    assert.equal(parentOf('F3'), branchId('Placeholder branch'));
    assert.equal(parentOf('F4'), branchId('Placeholder branch'));
    assert.equal(byId.get('F1')!.kind, 'idea');
    assert.equal(byId.get('F1')!.status, null, 'a future idea has no verdict');
    assert.equal(byId.get(branchId('Placeholder branch'))!.kind, 'branch');
    assert.equal(byId.get(branchId('Placeholder branch'))!.depth, 1);
    assert.equal(byId.get('F3')!.depth, 2);
    // Real tests are still all there, and no two boxes overlap.
    assert.equal(layout.nodes.filter(n => n.kind === 'leaf').length, ideaTree.parts.flatMap(p => p.leaves).length);
    for (let i = 0; i < layout.nodes.length; i += 1) for (let j = i + 1; j < layout.nodes.length; j += 1) {
      const a = layout.nodes[i], b = layout.nodes[j];
      assert.ok(a.x + a.w <= b.x + 0.5 || b.x + b.w <= a.x + 0.5 || a.y + a.h <= b.y + 0.5 || b.y + b.h <= a.y + 0.5, `${a.id} overlaps ${b.id}`);
    }
  }
});

test('a closed box hides its ideas, Open all and the paths reach them', () => {
  const closed = layoutGraph(ideaTree, new Set(['root']), wideSizes, ideas);
  assert.deepEqual(closed.nodes.filter(n => n.kind === 'idea').map(n => n.id), []);
  assert.deepEqual(closed.nodes.filter(n => n.kind === 'branch').map(n => n.id), [branchId('Placeholder branch')]);
  assert.deepEqual(ancestorsOf(ideaTree, 'F1', ideas), ['B1.1', 'B1.0', 'B1', 'root']);
  assert.deepEqual(ancestorsOf(ideaTree, 'F3', ideas), [branchId('Placeholder branch'), 'root']);
  assert.ok(expandableIds(ideaTree, ideas).includes('B1.1') && expandableIds(ideaTree, ideas).includes(branchId('Placeholder branch')));
  const withIdeas = layoutGraph(ideaTree, all, wideSizes, ideas).nodes.find(n => n.id === 'B9')!;
  assert.equal(withIdeas.ideaCount, 1);
  assert.equal(childLabel('part', withIdeas.childCount, withIdeas.ideaCount), `${withIdeas.childCount - 1} test${withIdeas.childCount - 1 === 1 ? '' : 's'} · 💡 1`);
  assert.equal(childLabel('leaf', 1, 1), '💡 1');
  assert.equal(childLabel('branch', 2), '2 ideas');
  assert.equal(childLabel('root', 13, 1), '12 parts · 1 new branch');
});

test('What we changed: a matching dimension gets dashed chips, a new one a dashed section', () => {
  const rows = futureRows(changedRows, ideas);
  assert.deepEqual(rows.map(r => [r.title, r.isNew, r.chips.map(c => c.label)]), [['Optimiser', false, ['Placeholder optimiser']], ['Placeholder dimension', true, ['Placeholder chip']]]);
  assert.deepEqual(futureRows(changedRows, []), []);
});

test('the private route serves the ideas file and nothing near it', () => {
  assert.match(learnFileType('ideas/ideas.json') ?? '', /json/);
  for (const bad of ['ideas.json', 'ideas/other.json', 'ideas/ideas.json/x', 'ideas/../video/B01.mp4', 'ideas/ideas.JSON', 'data/ideas.json']) assert.equal(learnFileType(bad), null, bad);
  assert.equal(IDEAS_URL, '/learn-files/ideas/ideas.json');
});

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === '.git') continue;
    const file = path.join(dir, name);
    if (statSync(file).isDirectory()) walk(file, out); else out.push(file);
  }
  return out;
}

test('the real ideas are not in the repository', () => {
  const root = path.resolve(import.meta.dirname, '..');
  const files = ['content', 'public', 'src', 'api', 'scripts'].flatMap(d => { try { return walk(path.join(root, d)); } catch { return []; } });
  assert.deepEqual(files.filter(f => /ideas.*\.json$/i.test(f)), [], 'an ideas file must not be committed');
  for (const f of files.filter(f => /\.(ts|tsx|json|mjs|js)$/.test(f))) {
    assert.ok(!/t38-ideas/.test(readFileSync(f, 'utf8')) || f.endsWith('future-ideas.test.ts'), `${f} points at the ideas file`);
  }
});

test('the graph, list and changed views carry the buttons and the legend line', () => {
  const root = path.resolve(import.meta.dirname, '..');
  const read = (f: string) => readFileSync(path.join(root, f), 'utf8');
  assert.equal(IDEA_LEGEND, 'Dashed 💡 = future idea, not run yet');
  for (const f of ['src/components/IdeaGraph.tsx', 'src/components/IdeaTree.tsx']) {
    assert.match(read(f), /IdeasToggle/, f);
    assert.match(read(f), /Show future ideas/, f);
    assert.match(read(f), /IdeaLegendItem/, f);
  }
  assert.match(read('src/components/IdeaChanged.tsx'), /IdeasToggle/);
  assert.match(read('src/components/FutureIdeas.tsx'), /Hide future ideas/);
  assert.match(read('src/components/FutureIdeas.tsx'), /Cost <small>\(estimate\)/);
  assert.match(read('src/pages/IdeaTreePage.tsx'), /useIdeas/);
});
