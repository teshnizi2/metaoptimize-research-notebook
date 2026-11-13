import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { findNode, ideaTree } from '../src/lib/idea-tree.ts';
import { achievementStatus, achievements, groups, sourceHolds, sourceText, type Source } from '../src/lib/achievements.ts';

const root = new URL('../', import.meta.url);

function label(s: Source): string {
  return 'partCount' in s ? `count ${s.partCount.part}` : 'test' in s ? `${s.test}: ${s.quote}` : `${s.part}: ${s.quote}`;
}

test('every number on the page is a quote that appears in the idea tree', () => {
  for (const card of achievements) {
    for (const fact of card.facts) assert.ok(sourceHolds(fact.source), `${card.id} fact "${fact.label}" — ${label(fact.source)}`);
    for (const s of card.holdsSources) assert.ok(sourceHolds(s), `${card.id} limit — ${label(s)}`);
  }
});

test('every card names real tests and parts, and has a key number and a limit', () => {
  for (const card of achievements) {
    assert.ok(card.facts.length >= 1, `${card.id} key number`);
    assert.ok(card.holds.length > 10 && card.holdsSources.length >= 1, `${card.id} holds-when line`);
    assert.ok(card.tests.length >= 1, `${card.id} tests`);
    for (const id of [...card.tests, ...(card.limits ?? [])]) assert.equal(findNode(ideaTree, id)?.kind, 'leaf', `${card.id}: ${id}`);
    for (const id of card.parts ?? []) assert.equal(findNode(ideaTree, id)?.kind, 'part', `${card.id}: ${id}`);
    assert.ok(groups.some(g => g.id === card.group), card.id);
  }
});

test('findings that stand are green in the tree; negative results are red or mixed', () => {
  for (const card of achievements.filter(c => c.group === 'stand')) {
    for (const id of card.tests) {
      const node = findNode(ideaTree, id);
      assert.equal(node?.kind === 'leaf' && node.leaf.status, 'g', `${card.id}: ${id} is green`);
    }
    assert.equal(achievementStatus(card), 'g', card.id);
  }
  for (const card of achievements.filter(c => c.group === 'negative')) assert.notEqual(achievementStatus(card), 'g', card.id);
});

test('the pull-toward-one card says 6 of 6 and the tree has six red tests in Part 3', () => {
  const part = findNode(ideaTree, 'B3');
  assert.equal(part?.kind, 'part');
  if (part?.kind !== 'part') return;
  assert.equal(part.part.leaves.length, 6);
  assert.ok(part.part.leaves.every(l => l.status === 'r'));
  const card = achievements.find(c => c.id === 'pull');
  assert.deepEqual(card?.tests, part.part.leaves.map(l => l.id));
});

test('a changed number is caught', () => {
  assert.equal(sourceHolds({ test: 'B5.1', quote: '+0.50 percentage points' }), false);
  assert.equal(sourceHolds({ partCount: { part: 'B3', status: 'r', n: 5 } }), false);
  assert.ok(sourceText({ test: 'B5.1', quote: '' }).length > 0);
});

test('nothing from Part 11 or the review paper appears', () => {
  const text = JSON.stringify(achievements) + readFileSync(new URL('src/pages/Achievements.tsx', root), 'utf8');
  assert.ok(!/PaperFactory|Part 11|review paper|B11/i.test(text));
});

test('the "Rule 11" name, which the site data does not have, is not used', () => {
  assert.ok(!/Rule 11/.test(JSON.stringify(achievements)));
});

test('the notebook counts come from the record', () => {
  const stats = JSON.parse(readFileSync(new URL('public/data/research.json', root), 'utf8')).meta.stats;
  assert.ok(stats.runs > 0 && stats.experiments > 0 && stats.figures > 0);
  const page = readFileSync(new URL('src/pages/Achievements.tsx', root), 'utf8');
  assert.ok(!new RegExp(`\\b(${stats.runs}|3,596|${stats.experiments})\\b`).test(page), 'counts are read at runtime, not typed');
});
