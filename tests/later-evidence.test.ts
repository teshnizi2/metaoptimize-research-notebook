import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import type { ResearchData } from '../src/types.ts';
import {
  LATER_EVIDENCE_RELATIONS, correctionsSection, laterEvidenceFor, normaliseRegisteredText, validateLaterEvidence,
} from '../src/lib/later-evidence.ts';
import type { LaterEvidenceRelation } from '../src/lib/later-evidence.ts';
import { LaterEvidence } from '../src/components/LaterEvidence.tsx';

// MASTER-TABLE lines are pinned, so an earlier row that a later batch weakens (row 240, cgw1's SCALAR-BEATS-BEST, weakened
// to a tie by crt1 at row 241) cannot be edited. content/later-evidence.json is the curated, reviewed data-level relation
// instead: (earlier record, later record, relation, CORRECTIONS entry, the entry's own words). Every pair is sourced from a
// CORRECTIONS entry that says the later batch weakens, supersedes, refutes, replicates or qualifies the earlier one.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const relations = JSON.parse(readFileSync(new URL('../content/later-evidence.json', import.meta.url), 'utf8')) as LaterEvidenceRelation[];
const correctionsSource = data.sources.find(source => source.path === 'docs/CORRECTIONS.md')!;
const corrections = readFileSync(new URL(`../public${correctionsSource.href}`, import.meta.url), 'utf8');
const byId = new Map(data.experiments.map(e => [e.id, e]));
const valid = relations.find(r => r.earlier === 'MT240' && r.later === 'MT241' && r.relation === 'weakens')!;

function render(id: string) {
  return renderToStaticMarkup(createElement(MemoryRouter, null,
    createElement(LaterEvidence, { experiment: byId.get(id)!, experiments: data.experiments, relations })));
}

test('the curated list validates against the published register and the published CORRECTIONS copy', () => {
  assert.ok(correctionsSource, 'docs/CORRECTIONS.md is a published source');
  assert.deepEqual(validateLaterEvidence(data.experiments, relations, corrections), []);
  assert.ok(relations.length >= 30, `seeded from the campaign's own statements (${relations.length})`);
});

test('the vocabulary is closed to the five relations the campaign states', () => {
  assert.deepEqual([...LATER_EVIDENCE_RELATIONS], ['weakens', 'supersedes', 'refutes', 'replicates', 'qualifies']);
  for (const r of relations) assert.ok(LATER_EVIDENCE_RELATIONS.includes(r.relation), `${r.earlier}->${r.later}: ${r.relation}`);
});

test('MT240 -> MT241 is present: crt1 weakens cgw1\'s scalar reading, quoting the registered licence at its one cell', () => {
  assert.ok(valid, 'MT240 -> MT241 weakens');
  assert.equal(valid.source, 'CORRECTIONS 309.8');
  assert.equal(valid.quote, "at this cell cgw1's +2.6-2.8 pp was, at least in part, a tuning artefact");
  assert.ok(normaliseRegisteredText(correctionsSection(corrections, 309, 8)!).includes(valid.quote));
  // MT240 keeps its record and its outcome: the relation is data, never an edit of the row.
  assert.equal(byId.get('MT240')!.outcome, 'unresolved');
  assert.match(byId.get('MT240')!.reason, /SCALAR-BEATS-BEST/);
});

test('each pair cites a CORRECTIONS entry that names its later batch, and the earlier record precedes the later one', () => {
  for (const r of relations) {
    const later = byId.get(r.later)!, earlier = byId.get(r.earlier)!;
    const [, entry, sub] = /^CORRECTIONS (\d+)(?:\.(\d+))?$/.exec(r.source)!;
    const text = normaliseRegisteredText(correctionsSection(corrections, Number(entry), sub ? Number(sub) : undefined)!);
    const laterBatches = r.later === 'CVK2' ? ['cvk1'] : later.batches;
    const entryText = normaliseRegisteredText(correctionsSection(corrections, Number(entry))!);
    assert.ok(laterBatches.some(b => entryText.includes(b)), `${r.source} names ${r.later}'s batch`);
    assert.ok(text.includes(r.quote), `${r.source} contains the quote for ${r.earlier}->${r.later}`);
    assert.notEqual(earlier.id, later.id);
    const n = (id: string) => id === 'CVK2' ? 210 : Number(id.slice(2));
    if (r.earlier !== 'MT175') assert.ok(n(r.earlier) < n(r.later) || r.later === 'CVK2', `${r.earlier} precedes ${r.later}`);
  }
});

test('validation rejects an unknown record, a self-link, an open vocabulary, a missing entry, a misquote and a duplicate', () => {
  const issues = (r: Partial<LaterEvidenceRelation>) => validateLaterEvidence(data.experiments, [{ ...valid, ...r }], corrections);
  assert.match(issues({ earlier: 'MT999' }).join('\n'), /MT999.*does not exist/);
  assert.match(issues({ later: 'MT240' }).join('\n'), /links a record to itself/);
  assert.match(issues({ relation: 'undermines' as LaterEvidenceRelation['relation'] }).join('\n'), /closed vocabulary/);
  assert.match(issues({ source: 'CORRECTIONS 9999' }).join('\n'), /no entry CORRECTIONS 9999/);
  assert.match(issues({ source: 'CORRECTIONS 309.99' }).join('\n'), /no entry CORRECTIONS 309\.99/);
  assert.match(issues({ source: 'the draft' }).join('\n'), /must read "CORRECTIONS <n>"/);
  assert.match(issues({ quote: 'cgw1 is refuted' }).join('\n'), /does not contain the quoted words/);
  // A quote from another entry is not enough: it must sit in the cited one.
  assert.match(issues({ source: 'CORRECTIONS 309.3' }).join('\n'), /does not contain the quoted words/);
  assert.match(issues({ quote: 'short' }).join('\n'), /quote is too short/);
  assert.match(issues({ reason: '' }).join('\n'), /reason is required/);
  assert.match(issues({ reason: 'two\nlines' }).join('\n'), /one line/);
  assert.match(validateLaterEvidence(data.experiments, [valid, { ...valid }], corrections).join('\n'), /duplicate/);
});

test('normalisation strips Markdown marks and blockquote breaks but no words', () => {
  assert.equal(normaliseRegisteredText('> **a** `b`\n> c  d'), 'a b c d');
  // The licence in 309.8 is a blockquote broken across two lines; the quote still matches.
  assert.ok(normaliseRegisteredText(correctionsSection(corrections, 309, 8)!).includes('was, at least in part, a tuning artefact'));
  assert.equal(correctionsSection(corrections, 9999), null);
});

test('laterEvidenceFor splits a record\'s relations into later evidence and the records it bears on', () => {
  const mt240 = laterEvidenceFor('MT240', relations), mt241 = laterEvidenceFor('MT241', relations);
  assert.ok(mt240.later.some(r => r.later === 'MT241' && r.relation === 'weakens'));
  assert.ok(mt241.bearsOn.some(r => r.earlier === 'MT240' && r.relation === 'weakens'));
  assert.deepEqual(laterEvidenceFor('MT014', relations), { later: [], bearsOn: [] });
});

test('the earlier record shows a labelled Later evidence note, with the quote and its CORRECTIONS entry', () => {
  const html = render('MT240');
  assert.match(html, /Later evidence/);
  assert.match(html, /Weakened by/);
  assert.match(html, /href="\/experiments\/MT241"/);
  assert.ok(html.includes('at this cell cgw1&#x27;s +2.6-2.8 pp was, at least in part, a tuning artefact'), 'quote rendered verbatim');
  assert.match(html, /CORRECTIONS 309\.8/);
  assert.match(html, /registered text is unchanged/);
});

test('the later record shows what it bears on', () => {
  const html = render('MT241');
  assert.match(html, /Bears on/);
  assert.match(html, /Weakens/);
  assert.match(html, /href="\/experiments\/MT240"/);
  assert.doesNotMatch(html, /Later evidence/);
});

test('a record with no curated relation renders nothing', () => {
  assert.equal(render('MT014'), '');
});

test('copy:check, which runs before every build, also validates the later-evidence list', async () => {
  const { spawnSync } = await import('node:child_process');
  const result = spawnSync(process.execPath, ['--import', 'tsx', 'scripts/check-experiment-copy.ts'], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, new RegExp(`Later evidence check passed: ${relations.length} curated relations, every quote found in its cited CORRECTIONS entry\\.`));
});
