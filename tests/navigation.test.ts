import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { groups, groupOf, navEntryOf, sectionLabel, sections, sectionTabs } from '../src/lib/navigation.ts';

const main = readFileSync(new URL('../src/main.tsx', import.meta.url), 'utf8');
const routePaths = [...main.matchAll(/<Route (?:index |path="([^"]+)")/g)].map(m => m[1] ?? '');
const topRoutes = routePaths.filter(p => p !== '*' && !p.includes(':')).map(p => `/${p}`);

test('the sidebar has 9 entries in 4 groups, in the agreed order', () => {
  assert.deepEqual(groups.map(g => g.label), ['Start', 'Results', 'Progress', 'Research record']);
  assert.deepEqual(groups.flatMap(g => g.entries.map(e => e.label)), [
    'Overview', 'Learn', 'What we achieved', 'The main idea', 'Making it better', 'Weekly record', 'Idea tree', 'Experiments', 'Evidence',
  ]);
});

test('every page route is a section, and every section is reachable from the sidebar (as an entry or a tab)', () => {
  const sectionPaths = sections.map(s => s.to as string);
  for (const r of topRoutes) assert.ok(sectionPaths.includes(r), `route ${r} has no section`);
  const reachable = new Set(groups.flatMap(g => g.entries.flatMap(e => [e.to, ...(e.tabs ?? []).map(t => t.to)])));
  for (const s of sectionPaths) if (s !== '/evidence') assert.ok(reachable.has(s), `${s} is not in the sidebar`);
  for (const to of reachable) assert.ok(topRoutes.includes(to), `sidebar link ${to} has no route`);
});

test('old URLs keep working: every route from before the restructure still exists, and /evidence goes to /figures', () => {
  for (const r of ['/', '/experiments', '/idea-tree', '/main-idea', '/learn', '/achievements', '/better', '/weekly', '/figures', '/code', '/runs', '/warnings', '/activity'])
    assert.ok(topRoutes.includes(r), r);
  for (const r of ['experiments/:id', 'weekly/:id', 'figures/:id', 'code/:id', 'runs/:id']) assert.ok(routePaths.includes(r), r);
  assert.match(main, /<Route path="evidence" element=\{<Navigate to="\/figures" replace\/>\}\/>/);
});

test('each path knows its group, its highlighted sidebar entry and its sub-tabs', () => {
  const cases: [string, string, string, string][] = [
    ['/', 'Start', 'Overview', ''],
    ['/main-idea', 'Results', 'The main idea', ''],
    ['/activity', 'Progress', 'Weekly record', 'Weeks|Research log'],
    ['/weekly/2026-11-13', 'Progress', 'Weekly record', ''],
    ['/warnings', 'Research record', 'Experiments', 'Questions|Warnings & limits'],
    ['/experiments/MT049', 'Research record', 'Experiments', ''],
    ['/runs', 'Research record', 'Evidence', 'Figures & tables|Run ledger|Source code'],
    ['/code/src-x', 'Research record', 'Evidence', ''],
    ['/evidence', 'Research record', 'Evidence', ''],
  ];
  for (const [path, group, entry, tabs] of cases) {
    assert.equal(groupOf(path)?.label, group, path);
    assert.equal(navEntryOf(path)?.label, entry, path);
    assert.equal(sectionTabs(path)?.tabs.map(t => t.label).join('|') ?? '', tabs, path);
  }
  assert.equal(groupOf('/nope'), undefined);
  assert.equal(navEntryOf('/nope'), undefined);
  assert.equal(sectionLabel('/evidence'), 'Evidence');
});

test('page titles match the sidebar names (no slogan titles left)', () => {
  const read = (f: string) => readFileSync(new URL(`../src/pages/${f}`, import.meta.url), 'utf8');
  assert.match(read('Experiments.tsx'), /PageHeading title="Experiments"/);
  assert.match(read('Figures.tsx'), /PageHeading title="Figures & tables"/);
  assert.match(read('Code.tsx'), /PageHeading title="Source code"/);
  assert.match(read('Learn.tsx'), /title="Learn"/);
  assert.match(read('Overview.tsx'), /PageHeading title="Overview"/);
});
