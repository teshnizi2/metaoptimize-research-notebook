import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { findNode, ideaTree } from '../src/lib/idea-tree.ts';
import { achievementStatus, achievements, groups, sectionOf, sections, sourceHolds, sourceText, type Source } from '../src/lib/achievements.ts';
import { firstUses, gloss } from '../src/lib/glossary.ts';

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

test('the audit, the rival comparison and the collapse case study are key cards; the rest are not', async () => {
  const { isKeyAchievement } = await import('../src/lib/achievements.ts');
  const keys = achievements.filter(c => isKeyAchievement(c)).map(c => c.id).sort();
  assert.deepEqual(keys, ['audit', 'collapse', 'rival']);
  // A key card keeps its verdict: the rival card is still red.
  assert.equal(achievementStatus(achievements.find(c => c.id === 'rival')!), 'r');
});

test('the page reads in five sections, in order, and every card lands in one', () => {
  assert.deepEqual(sections.map(s => s.id), ['key', 'stand', 'negative', 'checked', 'built']);
  assert.deepEqual(sections.map(s => s.title), [
    'Key results for the paper', 'Other findings that hold', 'Clean negatives: what we ruled out', 'How the work was checked', 'What we built',
  ]);
  for (const card of achievements) assert.ok(sections.some(s => s.id === sectionOf(card)), card.id);
  assert.deepEqual(achievements.filter(c => sectionOf(c) === 'key').map(c => c.id).sort(), ['audit', 'collapse', 'rival']);
});

test('"At a glance" has 4 to 6 one-line claims, each pointing at a card', () => {
  const lines = achievements.filter(c => c.glance);
  assert.ok(lines.length >= 4 && lines.length <= 6, `${lines.length} lines`);
  for (const c of lines) assert.ok(c.glance!.length < 140 && !c.glance!.includes('\n'), c.id);
  assert.ok(new Set(achievements.map(c => c.id)).size === achievements.length, 'card ids are unique anchors');
});

test('every card says why it matters, and the text says "points", never "pp"', () => {
  for (const card of achievements) {
    assert.ok(card.matters.length > 10, `${card.id} matters`);
    const words = [card.claim, card.holds, card.caveat ?? '', card.matters, card.glance ?? '', ...card.facts.map(f => f.label)].join(' | ');
    assert.ok(!/\bpp\b/i.test(words), `${card.id}: "pp" in the text`);
  }
});

test('a plain term is explained once, at its first use', () => {
  const pieces = gloss('BatchNorm helps, and BatchNorm again', new Set(['batchnorm']));
  assert.equal(pieces.filter(p => p.term).length, 1);
  assert.equal(pieces.map(p => p.text).join(''), 'BatchNorm helps, and BatchNorm again');
  const plan = firstUses([{ key: 'a', text: 'we use SGD' }, { key: 'b', text: 'SGD again' }]);
  assert.ok(plan.get('a')?.has('sgd'));
  assert.ok(!plan.get('b')?.has('sgd'));
});
