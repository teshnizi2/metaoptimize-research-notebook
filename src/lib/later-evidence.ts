import type { Experiment } from '../types';

import laterEvidenceSource from '../../content/later-evidence.json';

/**
 * Later evidence: a curated, reviewed relation between an earlier record and a later one, sourced from the campaign's own
 * CORRECTIONS entries. MASTER-TABLE rows are pinned and cannot be edited, so when a later batch weakens, supersedes,
 * refutes, replicates or qualifies an earlier row the relation is recorded here, as data, and neither record's registered
 * text changes. Each relation quotes the cited entry's own words; `validateLaterEvidence` checks that they are there.
 */
export const LATER_EVIDENCE_RELATIONS = ['weakens', 'supersedes', 'refutes', 'replicates', 'qualifies'] as const;
export type LaterEvidenceKind = typeof LATER_EVIDENCE_RELATIONS[number];

export interface LaterEvidenceRelation {
  earlier: string;
  later: string;
  relation: LaterEvidenceKind;
  /** `CORRECTIONS <entry>` or `CORRECTIONS <entry>.<subsection>`. */
  source: string;
  /** Verbatim words of the cited entry (Markdown marks removed, whitespace collapsed). */
  quote: string;
  /** One reader-facing line: what the later record does to the earlier one. */
  reason: string;
}

export const laterEvidence = laterEvidenceSource as LaterEvidenceRelation[];

/** Labels read on the earlier record ("Weakened by MT241") and on the later one ("Weakens MT240"). */
export const laterEvidenceLabels: Record<LaterEvidenceKind, { earlier: string; later: string }> = {
  weakens: { earlier: 'Weakened by', later: 'Weakens' },
  supersedes: { earlier: 'Superseded by', later: 'Supersedes' },
  refutes: { earlier: 'Refuted by', later: 'Refutes' },
  replicates: { earlier: 'Replicated by', later: 'Replicates' },
  qualifies: { earlier: 'Qualified by', later: 'Qualifies' },
};

/** Removes Markdown bold/emphasis/code marks and blockquote markers, and collapses whitespace. No word is changed. */
export function normaliseRegisteredText(text: string) {
  return text.replace(/^[ \t]*>[ \t]?/gm, '').replace(/[*`]/g, '').replace(/\s+/g, ' ').trim();
}

/**
 * The text of `## <entry>.` (up to the next `## ` heading), or of its `### <entry>.<sub>` subsection (up to the next
 * heading of either level). Returns null when the entry or subsection does not exist.
 */
export function correctionsSection(corrections: string, entry: number, subsection?: number): string | null {
  const lines = corrections.split('\n');
  const entryStart = lines.findIndex(line => line.startsWith(`## ${entry}. `));
  if (entryStart < 0) return null;
  let end = lines.findIndex((line, i) => i > entryStart && /^## /.test(line));
  if (end < 0) end = lines.length;
  if (subsection === undefined) return lines.slice(entryStart, end).join('\n');
  const subStart = lines.findIndex((line, i) => i > entryStart && i < end && line.startsWith(`### ${entry}.${subsection} `));
  if (subStart < 0) return null;
  let subEnd = lines.findIndex((line, i) => i > subStart && /^#{2,3} /.test(line));
  if (subEnd < 0 || subEnd > end) subEnd = end;
  return lines.slice(subStart, subEnd).join('\n');
}

const MIN_QUOTE = 12, MAX_REASON = 320;

export function validateLaterEvidence(experiments: Pick<Experiment, 'id'>[], relations: LaterEvidenceRelation[], corrections: string) {
  const issues: string[] = [], ids = new Set(experiments.map(e => e.id)), seen = new Set<string>();
  relations.forEach((r, index) => {
    const label = `later-evidence[${index}] ${r.earlier} -> ${r.later}`;
    for (const id of [r.earlier, r.later]) if (!ids.has(id)) issues.push(`${label}: record ${id} does not exist in research.json.`);
    if (r.earlier === r.later) issues.push(`${label}: links a record to itself.`);
    if (!(LATER_EVIDENCE_RELATIONS as readonly string[]).includes(r.relation)) {
      issues.push(`${label}: relation "${r.relation}" is outside the closed vocabulary (${LATER_EVIDENCE_RELATIONS.join(', ')}).`);
    }
    const key = `${r.earlier}|${r.later}|${r.relation}`;
    if (seen.has(key)) issues.push(`${label}: duplicate ${r.relation} relation.`);
    seen.add(key);
    if (typeof r.reason !== 'string' || !r.reason.trim()) issues.push(`${label}: a reason is required.`);
    else {
      if (/\n/.test(r.reason)) issues.push(`${label}: the reason must be one line.`);
      if (r.reason.length > MAX_REASON) issues.push(`${label}: the reason exceeds ${MAX_REASON} characters.`);
    }
    const quote = typeof r.quote === 'string' ? normaliseRegisteredText(r.quote) : '';
    if (quote.length < MIN_QUOTE) issues.push(`${label}: the quote is too short to identify the registered words.`);
    else if (quote !== r.quote) issues.push(`${label}: store the quote normalised (no Markdown marks, single spaces).`);
    const source = /^CORRECTIONS (\d+)(?:\.(\d+))?$/.exec(r.source ?? '');
    if (!source) {
      issues.push(`${label}: source must read "CORRECTIONS <n>" or "CORRECTIONS <n>.<m>".`);
      return;
    }
    const section = correctionsSection(corrections, Number(source[1]), source[2] === undefined ? undefined : Number(source[2]));
    if (section === null) issues.push(`${label}: the published CORRECTIONS has no entry ${r.source}.`);
    else if (quote.length >= MIN_QUOTE && !normaliseRegisteredText(section).includes(quote)) {
      issues.push(`${label}: ${r.source} does not contain the quoted words.`);
    }
  });
  return issues;
}

/** A record's relations: later records that bear on it, and earlier records it bears on. */
export function laterEvidenceFor(id: string, relations: LaterEvidenceRelation[] = laterEvidence) {
  return { later: relations.filter(r => r.earlier === id), bearsOn: relations.filter(r => r.later === id) };
}
