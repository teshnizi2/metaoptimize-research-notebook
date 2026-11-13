import { useEffect, useState } from 'react';
import { findNode, type IdeaTreeData } from './idea-tree';
import type { ChangedRow } from './idea-changed';

/**
 * Future ideas are unpublished research, so they are NOT part of this repository. The site fetches them at runtime from
 * the private route below (behind the Vercel sign-in); a copy that cannot fetch them simply shows none.
 */
export const IDEAS_URL = '/learn-files/ideas/ideas.json';
export const IDEA_LEGEND = 'Dashed 💡 = future idea, not run yet';

export type IdeaKind = 'continue' | 'new-branch' | 'check' | 'new-category';
export type IdeaPriority = 'now' | 'next' | 'later';

export const kindName: Record<IdeaKind, string> = {
  continue: 'Continues a branch',
  'new-branch': 'New branch',
  check: 'Check of a weak spot',
  'new-category': 'New kind of question',
};
export const priorityName: Record<IdeaPriority, string> = { now: 'Do now', next: 'Do next', later: 'Later' };

export interface FutureIdea {
  id: string;
  title: string;
  question: string;
  kind: IdeaKind;
  /** The part or test it grows from, or the name of a new branch off the question. */
  attachTo: string | { newBranch: string };
  whatChanged?: { dimension: string; chip: string };
  why: string;
  test: string;
  /** An estimate, never a measurement. */
  cost: string;
  outcomes: string[];
  paperValue: string;
  /** 1 (unlikely) to 5 (sure). */
  feasibility: number;
  priority: IdeaPriority;
  risks: string[];
}

const KINDS: IdeaKind[] = ['continue', 'new-branch', 'check', 'new-category'];
const PRIORITIES: IdeaPriority[] = ['now', 'next', 'later'];

/** Any plain value as readable text: strings as they are, lists joined, objects as "key: value". */
function plain(v: unknown): string {
  if (v == null) return '';
  if (typeof v === 'string') return v.trim();
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return v.map(plain).filter(Boolean).join('; ');
  if (typeof v === 'object') return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}: ${plain(x)}`).join('; ');
  return '';
}

function lines(v: unknown): string[] {
  if (v == null) return [];
  if (Array.isArray(v)) return v.map(plain).filter(Boolean);
  if (typeof v === 'object') return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}: ${plain(x)}`).filter(Boolean);
  const s = plain(v);
  return s ? [s] : [];
}

/** Reads the ideas file defensively: a wrong entry is skipped, never thrown. */
export function parseIdeas(raw: unknown): FutureIdea[] {
  const list = Array.isArray(raw) ? raw : raw && typeof raw === 'object' && Array.isArray((raw as { ideas?: unknown }).ideas) ? (raw as { ideas: unknown[] }).ideas : [];
  const seen = new Set<string>();
  const out: FutureIdea[] = [];
  for (const item of list) {
    if (!item || typeof item !== 'object') continue;
    const r = item as Record<string, unknown>;
    const id = plain(r.id);
    const title = plain(r.title);
    const at = r.attachTo;
    const attachTo = typeof at === 'string' && at.trim() ? at.trim()
      : at && typeof at === 'object' && plain((at as { newBranch?: unknown }).newBranch) ? { newBranch: plain((at as { newBranch?: unknown }).newBranch) }
      : null;
    if (!id || !title || !attachTo || seen.has(id)) continue;
    seen.add(id);
    const wc = r.whatChanged as { dimension?: unknown; chip?: unknown } | undefined;
    const feas = Number(r.feasibility);
    out.push({
      id, title,
      question: plain(r.question),
      kind: KINDS.includes(r.kind as IdeaKind) ? (r.kind as IdeaKind) : 'continue',
      attachTo,
      whatChanged: wc && plain(wc.dimension) && plain(wc.chip) ? { dimension: plain(wc.dimension), chip: plain(wc.chip) } : undefined,
      why: plain(r.why),
      test: plain(r.test),
      cost: plain(r.cost),
      outcomes: lines(r.outcomes),
      paperValue: plain(r.paperValue),
      feasibility: Number.isFinite(feas) ? Math.min(5, Math.max(1, Math.round(feas))) : 3,
      priority: PRIORITIES.includes(r.priority as IdeaPriority) ? (r.priority as IdeaPriority) : 'later',
      risks: lines(r.risks),
    });
  }
  return out;
}

const slug = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'new';

/** Id of the box a new branch shows as; several ideas with the same branch name share it. */
export const branchId = (name: string) => `branch:${slug(name)}`;

/** The id of the box an idea hangs from: a part, a test, or a new branch. */
export function ideaParent(idea: FutureIdea): string {
  return typeof idea.attachTo === 'string' ? idea.attachTo : branchId(idea.attachTo.newBranch);
}

/** Only the ideas that can be drawn: they grow from a part or test the tree has (or a named new branch), and their ids are unique. */
export function attachedIdeas(data: IdeaTreeData, ideas: readonly FutureIdea[]): FutureIdea[] {
  const seen = new Set<string>();
  return ideas.filter(i => {
    if (seen.has(i.id) || findNode(data, i.id)) return false;
    const node = typeof i.attachTo === 'string' ? findNode(data, i.attachTo) : true;
    if (!node || (typeof i.attachTo === 'string' && i.attachTo === 'root')) return false;
    seen.add(i.id);
    return true;
  });
}

export interface IdeaBranch { id: string; name: string; ideas: FutureIdea[] }

/** The new branches, in the order their first idea appears. */
export function ideaBranches(ideas: readonly FutureIdea[]): IdeaBranch[] {
  const map = new Map<string, IdeaBranch>();
  for (const i of ideas) {
    if (typeof i.attachTo === 'string') continue;
    const id = branchId(i.attachTo.newBranch);
    const b = map.get(id) ?? { id, name: i.attachTo.newBranch, ideas: [] };
    b.ideas.push(i);
    map.set(id, b);
  }
  return [...map.values()];
}

/** Ideas that hang directly from a part or test. */
export function ideasOn(ideas: readonly FutureIdea[], id: string): FutureIdea[] {
  return ideas.filter(i => i.attachTo === id);
}

export type IdeaNode =
  | { kind: 'idea'; id: string; idea: FutureIdea }
  | { kind: 'branch'; id: string; name: string; ideas: FutureIdea[] };

export function findIdeaNode(ideas: readonly FutureIdea[], id: string | null): IdeaNode | null {
  if (!id) return null;
  const idea = ideas.find(i => i.id === id);
  if (idea) return { kind: 'idea', id, idea };
  const branch = ideaBranches(ideas).find(b => b.id === id);
  return branch ? { kind: 'branch', id, name: branch.name, ideas: branch.ideas } : null;
}

/** Whether an id names something in the tree or among the ideas. */
export const isKnown = (data: IdeaTreeData, ideas: readonly FutureIdea[], id: string | null) => !!(findNode(data, id) ?? findIdeaNode(ideas, id));

/** One dashed chip of "What we changed": a setting no test has run yet, with the ideas that would test it. */
export interface FutureChip { label: string; ideas: FutureIdea[] }
export interface FutureRow { title: string; question?: string; isNew: boolean; chips: FutureChip[] }

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim();

/** Groups the ideas that name a setting by dimension: a row the page already has gets dashed chips, any other dimension becomes a dashed section. */
export function futureRows(rows: readonly ChangedRow[], ideas: readonly FutureIdea[]): FutureRow[] {
  const out: FutureRow[] = [];
  for (const i of ideas) {
    if (!i.whatChanged) continue;
    const known = rows.find(r => norm(r.title) === norm(i.whatChanged!.dimension));
    const title = known ? known.title : i.whatChanged.dimension;
    let row = out.find(r => norm(r.title) === norm(title));
    if (!row) { row = { title, question: known?.question, isNew: !known, chips: [] }; out.push(row); }
    let chip = row.chips.find(c => norm(c.label) === norm(i.whatChanged!.chip));
    if (!chip) { chip = { label: i.whatChanged.chip, ideas: [] }; row.chips.push(chip); }
    chip.ideas.push(i);
  }
  return out;
}

/** The test ids that a piece of text mentions (B1.2, B8.5 …), so the detail panel can link them. */
export const TEST_ID = /\bB\d{1,2}\.\d{1,2}\b/g;

let cached: Promise<FutureIdea[]> | null = null;

/** Fetches the private ideas once per page load; any failure (not signed in, not uploaded yet, not a copy of the site) gives an empty list. */
export function loadIdeas(): Promise<FutureIdea[]> {
  cached ??= fetch(IDEAS_URL, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
    .then(res => (res.ok && (res.headers.get('content-type') ?? '').includes('json') ? res.json() : []))
    .then(parseIdeas)
    .catch(() => []);
  return cached;
}

/** The future ideas, once fetched. `ready` is false until the fetch has finished, so a shared link to an idea can wait for it. */
export function useIdeas(): { ideas: FutureIdea[]; ready: boolean } {
  const [state, setState] = useState<{ ideas: FutureIdea[]; ready: boolean }>({ ideas: [], ready: false });
  useEffect(() => {
    let live = true;
    loadIdeas().then(ideas => { if (live) setState({ ideas, ready: true }); });
    return () => { live = false; };
  }, []);
  return state;
}
