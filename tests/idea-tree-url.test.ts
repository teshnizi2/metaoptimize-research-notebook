import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { treeAddress, viewOf, withNode, withView } from '../src/lib/idea-tree-url.ts';

const root = new URL('../', import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), 'utf8');
const q = (s: string) => new URLSearchParams(s);

test('choosing a box in the radial, left-to-right and list views keeps that view and adds node', () => {
  for (const view of ['radial', 'tree', 'list']) {
    const next = withNode(q(`view=${view}`), 'B6.2');
    assert.equal(next.get('view'), view, view);
    assert.equal(next.get('node'), 'B6.2', view);
    // Choosing another box replaces node, and closing the panel removes it; the view stays.
    assert.equal(withNode(next, 'B5.1').get('view'), view);
    assert.equal(withNode(next, 'B5.1').get('node'), 'B5.1');
    assert.equal(withNode(next, null).toString(), `view=${view}`);
  }
});

test('with no view in the address, a click keeps it that way (the default view stays the default)', () => {
  assert.equal(withNode(q(''), 'B4.7').toString(), 'node=B4.7');
  assert.equal(viewOf(q('node=B4.7'), false), 'radial');
  assert.equal(viewOf(q('node=B4.7'), true), 'tree');
});

test('Show key results and ?keys=1 do not change the view; switching view drops the one-off keys start', () => {
  const next = withNode(q('view=tree&keys=1&node=B5.1'), null);
  assert.equal(next.get('view'), 'tree');
  assert.equal(withView(q('view=radial&keys=1'), 'list').toString(), 'view=list');
});

test('a chip in "What we changed" opens its test in the tree view the reader came from', () => {
  for (const came of ['radial', 'tree', 'list'] as const) {
    const next = withNode(withView(q('view=changed'), came), 'B8.3');
    assert.equal(next.get('view'), came);
    assert.equal(next.get('node'), 'B8.3');
  }
});

test('links from other pages (achievements, main idea) open the tree in its default view', () => {
  assert.equal(treeAddress('B5.1'), '/idea-tree?node=B5.1');
  assert.equal(treeAddress('B5.1', 'list'), '/idea-tree?view=list&node=B5.1');
  assert.ok(!/view=radial/.test(read('src/pages/Achievements.tsx') + read('src/pages/MainIdea.tsx')), 'no hard-coded radial view');
});

test('the list view, the graph views and the chips all go through withNode, never a bare node-only address', () => {
  const list = read('src/components/IdeaTree.tsx');
  const page = read('src/pages/IdeaTreePage.tsx');
  assert.ok(list.includes('withNode(params, id)'), 'list');
  assert.ok(page.includes('withNode(params, id)'), 'radial and left to right');
  assert.ok(page.includes('withNode(withView(params, lastTree.current), id)'), 'chips');
  assert.ok(!/setParams\(\{ node/.test(list + page), 'no setParams({ node })');
  assert.ok(!/query\.set\('view', phone/.test(page), 'chips do not force a view');
});
