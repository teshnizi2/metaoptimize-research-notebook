import { useEffect, useState } from 'react';

/**
 * The "Making it better" page and the Weekly record are unpublished research, so their text, numbers and chart images are
 * NOT part of this repository. The site fetches them at runtime from the private route below (behind the Vercel sign-in);
 * a copy that cannot fetch them shows an empty state. Adding a week or a result is a data change in the private store
 * (see docs/MAINTENANCE.md), never a code change.
 */
export const BETTER_URL = '/learn-files/better/better.json';
export const WEEKLY_INDEX_URL = '/learn-files/weekly/index.json';
export const WEEK_ID = /^\d{4}-\d{2}-\d{2}$/;
export const CHART_NAME = /^[a-z0-9-]+$/;
export const weekUrl = (id: string) => `/learn-files/weekly/${id}.json`;
export const chartUrl = (name: string) => `/learn-files/weekly/charts/${name}.png`;

export type Tone = 'hit' | 'miss' | 'mixed' | 'plain';
const TONES: Tone[] = ['hit', 'miss', 'mixed', 'plain'];

export interface Table { caption: string; columns: string[]; rows: string[][] }
export interface ChartItem { label: string; value: number; lo: number; hi: number; tone: Tone; note: string }
export interface IntervalChartData { title: string; axis: string; band: { from: number; label: string } | null; zero: boolean; items: ChartItem[] }
export interface Verdict { tone: Tone; text: string }
export interface Round {
  id: string; name: string; dates: string; tried: string[]; why: string;
  chart: IntervalChartData | null; table: Table | null; verdicts: Verdict[]; misses: string[]; reading: string;
}
/** One picture with a short text, from the private chart store (the same images the weekly pages use). */
export interface Figure { title: string; text: string; image: string; caption: string }
export interface Better {
  title: string; updated: string; intro: string[]; words: { term: string; meaning: string }[];
  /** Optional section of pictures shown before the rounds; files without it read exactly as before. */
  figures: { headline: string; items: Figure[] } | null;
  rounds: Round[];
  standing: { headline: string; bullets: string[]; next: string[]; note: string } | null;
}
export interface WeekChart { name: string; caption: string }
export interface WeekItem {
  id: string; title: string; said: { when: string; text: string }[];
  question: string; answer: string; why: string; decisions: string; outcome: string; tables: Table[]; charts: WeekChart[];
  /** The closed "Details" part: longer text, tables and charts. */
  details: string; detailTables: Table[]; detailCharts: WeekChart[];
}
export interface WeekSummary { id: string; title: string; range: string; oneline: string; updated: string; open: boolean; questions: number; tasks: number; achievements: number }
export interface Week extends Omit<WeekSummary, 'questions' | 'tasks' | 'achievements'> { summary: string[]; achievements: string[]; questions: WeekItem[]; tasks: WeekItem[] }

/* ---------- defensive reading: anything malformed is skipped, never thrown ---------- */

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const str = (v: unknown): string => (typeof v === 'string' ? v.trim() : typeof v === 'number' && Number.isFinite(v) ? String(v) : '');
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.map(str).filter(Boolean) : []);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const tone = (v: unknown): Tone => (TONES as unknown[]).includes(v) ? v as Tone : 'plain';
const each = <T,>(v: unknown, f: (x: Record<string, unknown>) => T | null): T[] =>
  Array.isArray(v) ? v.flatMap(x => (isObj(x) ? [f(x)] : [])).filter((x): x is T => x !== null) : [];

function table(v: unknown): Table | null {
  if (!isObj(v)) return null;
  const columns = strs(v.columns);
  const rows = Array.isArray(v.rows) ? v.rows.filter(Array.isArray).map(r => columns.map((_, i) => str((r as unknown[])[i]))) : [];
  return columns.length && rows.length ? { caption: str(v.caption), columns, rows } : null;
}

function chart(v: unknown): IntervalChartData | null {
  if (!isObj(v)) return null;
  const items = each(v.items, x => {
    const value = num(x.value), lo = num(x.lo), hi = num(x.hi), label = str(x.label);
    return value === null || lo === null || hi === null || !label ? null : { label, value, lo: Math.min(lo, value), hi: Math.max(hi, value), tone: tone(x.tone), note: str(x.note) };
  });
  if (!items.length) return null;
  const from = isObj(v.band) ? num(v.band.from) : null;
  return {
    title: str(v.title), axis: str(v.axis), zero: v.zero === true, items,
    band: isObj(v.band) && from !== null ? { from, label: str(v.band.label) } : null,
  };
}

export function parseBetter(raw: unknown): Better | null {
  if (!isObj(raw)) return null;
  const rounds = each(raw.rounds, r => {
    const id = str(r.id), name = str(r.name);
    if (!id || !name) return null;
    return {
      id, name, dates: str(r.dates), tried: strs(r.tried), why: str(r.why), chart: chart(r.chart), table: table(r.table),
      verdicts: each(r.verdicts, x => (str(x.text) ? { tone: tone(x.tone), text: str(x.text) } : null)),
      misses: strs(r.misses), reading: str(r.reading),
    };
  });
  if (!rounds.length) return null;
  const s = raw.standing;
  const f = raw.figures;
  const items = isObj(f) ? each(f.items, x => (str(x.title) && CHART_NAME.test(str(x.image))
    ? { title: str(x.title), text: str(x.text), image: str(x.image), caption: str(x.caption) } : null)) : [];
  return {
    title: str(raw.title) || 'Making it better', updated: str(raw.updated), intro: strs(raw.intro), rounds,
    figures: isObj(f) && items.length ? { headline: str(f.headline) || 'Pictures', items } : null,
    words: each(raw.words, w => (str(w.term) && str(w.meaning) ? { term: str(w.term), meaning: str(w.meaning) } : null)),
    standing: isObj(s) ? { headline: str(s.headline) || 'Where things stand now', bullets: strs(s.bullets), next: strs(s.next), note: str(s.note) } : null,
  };
}

const tables = (v: unknown): Table[] => (Array.isArray(v) ? v.map(table).filter((t): t is Table => t !== null) : []);
const weekCharts = (v: unknown): WeekChart[] => each(v, c => (CHART_NAME.test(str(c.name)) ? { name: str(c.name), caption: str(c.caption) } : null));

function weekItem(x: Record<string, unknown>, i: number, prefix: string): WeekItem | null {
  const title = str(x.title);
  if (!title) return null;
  return {
    id: str(x.id) || `${prefix}${i + 1}`, title,
    said: each(x.said, s => (str(s.text) ? { when: str(s.when), text: str(s.text) } : null)),
    question: str(x.question), answer: str(x.answer), why: str(x.why), decisions: str(x.decisions), outcome: str(x.outcome),
    tables: tables(x.tables), charts: weekCharts(x.charts),
    details: str(x.details), detailTables: tables(x.detailTables), detailCharts: weekCharts(x.detailCharts),
  };
}

export function parseWeek(raw: unknown): Week | null {
  if (!isObj(raw)) return null;
  const id = str(raw.id), title = str(raw.title);
  if (!WEEK_ID.test(id) || !title) return null;
  const items = (v: unknown, prefix: string) => (Array.isArray(v) ? v.flatMap((x, i) => (isObj(x) ? [weekItem(x, i, prefix)] : [])).filter((x): x is WeekItem => x !== null) : []);
  return {
    id, title, range: str(raw.range), oneline: str(raw.oneline), updated: str(raw.updated), open: raw.open === true,
    summary: strs(raw.summary), achievements: strs(raw.achievements), questions: items(raw.questions, 'q'), tasks: items(raw.tasks, 't'),
  };
}

export function parseWeekIndex(raw: unknown): WeekSummary[] {
  const list = isObj(raw) ? raw.weeks : raw;
  return each(list, w => {
    const id = str(w.id), title = str(w.title);
    return WEEK_ID.test(id) && title ? { id, title, range: str(w.range), oneline: str(w.oneline), updated: str(w.updated), open: w.open === true, questions: num(w.questions) ?? 0, tasks: num(w.tasks) ?? 0, achievements: num(w.achievements) ?? 0 } : null;
  }).sort((a, b) => (a.id < b.id ? 1 : -1));
}

/** "Wed 18 Nov 22:15 (week still open)" shows as "Wed 18 Nov 22:15". */
export const stamp = (updated: string) => updated.replace(/\s*\([^)]*\)\s*$/, '');

/* ---------- small markdown: paragraphs, "- " lists, **bold**, `code` (never raw HTML) ---------- */

export type Inline = { kind: 'text' | 'bold' | 'code'; text: string };
export type Block = { kind: 'p'; text: string } | { kind: 'ul'; items: string[] };

export function inline(s: string): Inline[] {
  const out: Inline[] = [];
  for (const part of s.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (!part) continue;
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) out.push({ kind: 'bold', text: part.slice(2, -2) });
    else if (part.startsWith('`') && part.endsWith('`') && part.length > 2) out.push({ kind: 'code', text: part.slice(1, -1) });
    else out.push({ kind: 'text', text: part });
  }
  return out;
}

export function blocks(s: string): Block[] {
  const out: Block[] = [];
  for (const line of s.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('- ')) {
      const last = out[out.length - 1];
      if (last?.kind === 'ul') last.items.push(t.slice(2));
      else out.push({ kind: 'ul', items: [t.slice(2)] });
    } else out.push({ kind: 'p', text: t });
  }
  return out;
}

/* ---------- interval chart geometry ---------- */

const MINUS = '−';
/** A signed number with a real minus sign, two decimals: +0.57, −0.33, 0.00. */
export function fmt(v: number): string {
  const s = Math.abs(v).toFixed(2);
  return v < 0 || Object.is(v, -0) ? MINUS + s : v > 0 ? '+' + s : s;
}
export const fmtInterval = (i: { value: number; lo: number; hi: number }) => `${fmt(i.value)} [${fmt(i.lo)}, ${fmt(i.hi)}]`;

export interface Axis { min: number; max: number; ticks: number[] }
/** A shared axis for all rows: every interval, zero and the band edge fit, with round tick marks. */
export function axisFor(c: IntervalChartData): Axis {
  const xs = c.items.flatMap(i => [i.lo, i.hi]).concat(0, c.band ? [c.band.from] : []);
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const span = hi - lo || 1;
  const step = [0.25, 0.5, 1, 2, 5, 10].find(s => span / s <= 7) ?? 20;
  const min = Math.floor((lo - span * 0.04) / step) * step, max = Math.ceil((hi + span * 0.04) / step) * step;
  const ticks: number[] = [];
  for (let t = min; t <= max + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
  return { min, max, ticks };
}
export const pct = (v: number, a: Axis) => ((v - a.min) / (a.max - a.min)) * 100;

export type Reading = 'within' | 'mean-only' | 'misses' | 'above' | 'below' | 'includes-zero';
export const READING_WORD: Record<Reading, string> = {
  within: 'inside the target', 'mean-only': 'by the mean only', misses: 'misses',
  above: 'above 0', below: 'below 0', 'includes-zero': 'interval includes 0',
};
/** Against the band: inside = the whole interval above the edge; by the mean only = only the mean is. Otherwise against zero. */
export function readItem(i: ChartItem, c: IntervalChartData): Reading | null {
  if (c.band) return i.lo >= c.band.from ? 'within' : i.value >= c.band.from ? 'mean-only' : 'misses';
  if (c.zero) return i.lo > 0 ? 'above' : i.hi < 0 ? 'below' : 'includes-zero';
  return null;
}

/* ---------- fetching, with an empty state for everything that can go wrong ---------- */

const cache = new Map<string, Promise<unknown>>();
/** The JSON at a private route, or null when it is missing, blocked, not JSON or the network failed. Failures are not cached. */
export function fetchJson(url: string): Promise<unknown> {
  let p = cache.get(url);
  if (!p) {
    p = fetch(url, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
      .then(res => (res.ok && (res.headers.get('content-type') ?? '').includes('json') ? res.json() : null))
      .catch(() => null);
    cache.set(url, p);
    p.then(v => { if (v === null) cache.delete(url); });
  }
  return p;
}

/** `ready` is false until the fetch has finished; `data` stays null when nothing usable came back. */
export function usePrivate<T>(url: string | null, parse: (raw: unknown) => T | null): { data: T | null; ready: boolean } {
  const [state, setState] = useState<{ url: string | null; data: T | null; ready: boolean }>({ url: null, data: null, ready: false });
  useEffect(() => {
    let live = true;
    if (url === null) { setState({ url, data: null, ready: true }); return; }
    fetchJson(url).then(raw => { if (live) setState({ url, data: raw === null ? null : parse(raw), ready: true }); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);
  return state.url === url ? state : { data: null, ready: false };
}
export const useBetter = () => usePrivate(BETTER_URL, parseBetter);
export const useWeekIndex = () => {
  const r = usePrivate<WeekSummary[]>(WEEKLY_INDEX_URL, parseWeekIndexOrNull);
  return { weeks: r.data ?? [], ready: r.ready };
};
export const useWeek = (id: string) => usePrivate(WEEK_ID.test(id) ? weekUrl(id) : null, parseWeek);
function parseWeekIndexOrNull(raw: unknown): WeekSummary[] | null { const w = parseWeekIndex(raw); return w.length ? w : null; }
