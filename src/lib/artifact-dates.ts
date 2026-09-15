export type ArtifactKind = 'figure' | 'table' | 'source' | 'experiment' | 'run' | 'warning' | 'chart' | 'download';
export interface DateFact { at: string; precision: 'day' | 'second'; basis: string; href?: string }
export interface ArtifactDateRecord {
  kind: ArtifactKind; id: string; contentSha256: string; experimentIds: string[];
  created?: DateFact | null; firstRecorded?: DateFact | null; updated?: DateFact | null;
  exported?: DateFact | null; submitted?: DateFact | null; started?: DateFact | null; finished?: DateFact | null;
  runWindow?: { started: DateFact; finished: DateFact; datedRuns: number; totalRuns: number } | null;
}
export interface ArtifactDateCatalog {
  schemaVersion: 1; snapshotId: string; snapshotSha256: string; runsSha256: string;
  publishedAt: string; exportedAt: string; recordedAt: string;
  records: Record<string, ArtifactDateRecord>;
}
export function artifactDateKey(kind: ArtifactKind, id: string) { return `${kind}:${id}`; }

const dayPattern = /^(\d{4})-(\d{2})-(\d{2})$/;
const timePattern = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(Z|[+-]\d{2}:\d{2})$/;
const dateFormatter = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
const hashPattern = /^[a-f0-9]{64}$/;
const kinds = new Set<ArtifactKind>(['figure', 'table', 'source', 'experiment', 'run', 'warning', 'chart', 'download']);
const dateFields = ['created', 'firstRecorded', 'updated', 'exported', 'submitted', 'started', 'finished'] as const;

function dateValue(fact?: DateFact | null): number | null {
  if (!fact || typeof fact.at !== 'string') return null;
  const match = fact.precision === 'day' ? dayPattern.exec(fact.at) : fact.precision === 'second' ? timePattern.exec(fact.at) : null;
  if (!match) return null;
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const lengths = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (year < 1 || month < 1 || month > 12 || day < 1 || day > lengths[month - 1]) return null;
  if (fact.precision === 'second') {
    if (Number(match[4]) > 23 || Number(match[5]) > 59 || Number(match[6]) > 59) return null;
    const zone = match[7];
    if (zone !== 'Z') {
      const hours = Number(zone.slice(1, 3)), minutes = Number(zone.slice(4, 6));
      if (hours > 14 || minutes > 59 || (hours === 14 && minutes !== 0)) return null;
    }
  }
  const value = Date.parse(fact.precision === 'day' ? `${fact.at}T00:00:00Z` : fact.at);
  return Number.isFinite(value) ? value : null;
}

export function formatDateFact(fact?: DateFact | null): string {
  const value = dateValue(fact);
  return value === null ? 'Not recorded' : dateFormatter.format(value);
}

export function ageLabel(fact?: DateFact | null, now: Date | string = new Date()): string | null {
  const value = dateValue(fact), current = typeof now === 'string' ? Date.parse(now) : now.getTime();
  if (value === null || !Number.isFinite(current)) return null;
  if (value > current) return 'Date is in the future';
  const days = Math.floor(current / 86400000) - Math.floor(value / 86400000);
  return days === 0 ? 'Today' : `${days.toLocaleString('en-GB')} day${days === 1 ? '' : 's'} ago`;
}

export function safeDateHref(href?: string): string | null {
  if (typeof href !== 'string' || !href || /[\\\s\u0000-\u001f\u007f]/.test(href)) return null;
  let decoded: string;
  try { decoded = decodeURIComponent(href); } catch { return null; }
  if (/[\\\u0000-\u001f\u007f]/.test(decoded) || /(?:^|\/)\.\.(?:\/|$|[?#])/.test(decoded)) return null;
  if (href.startsWith('/')) return decoded.startsWith('//') ? null : href;
  try {
    const url = new URL(href);
    if (url.origin !== 'https://github.com' || url.username || url.password) return null;
    return /^\/teshnizi2\/(?:metaoptimize-research-notebook|hierarchical-metaoptimize)(?:\/|$)/.test(url.pathname) ? href : null;
  } catch { return null; }
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireFact(value: unknown) {
  if (!object(value) || dateValue(value as unknown as DateFact) === null || typeof value.basis !== 'string' || !value.basis.trim()
      || (value.href !== undefined && (typeof value.href !== 'string' || !safeDateHref(value.href)))) {
    throw new Error('A recorded date or its provenance is invalid.');
  }
}

export function parseArtifactDateCatalog(value: unknown, snapshotId: string): ArtifactDateCatalog {
  if (!object(value) || value.schemaVersion !== 1 || !object(value.records) || !Object.keys(value.records).length) throw new Error('Date metadata is incomplete.');
  if (value.snapshotId !== snapshotId) throw new Error('Date metadata belongs to a different research snapshot.');
  for (const name of ['snapshotSha256', 'runsSha256']) {
    if (typeof value[name] !== 'string' || !hashPattern.test(value[name])) throw new Error('Date metadata has an invalid snapshot fingerprint.');
  }
  for (const name of ['publishedAt', 'exportedAt', 'recordedAt']) {
    const at = value[name];
    requireFact({ at, precision: typeof at === 'string' && dayPattern.test(at) ? 'day' : 'second', basis: 'Catalog date' });
  }
  for (const [key, entry] of Object.entries(value.records)) {
    if (!object(entry) || !kinds.has(entry.kind as ArtifactKind) || typeof entry.id !== 'string' || !entry.id
        || key !== artifactDateKey(entry.kind as ArtifactKind, entry.id)
        || typeof entry.contentSha256 !== 'string' || !hashPattern.test(entry.contentSha256)
        || !Array.isArray(entry.experimentIds) || !entry.experimentIds.every(id => typeof id === 'string')) {
      throw new Error('An artifact date record is invalid.');
    }
    for (const name of dateFields) if (entry[name] !== undefined && entry[name] !== null) requireFact(entry[name]);
    if (entry.runWindow !== undefined && entry.runWindow !== null) {
      const window = entry.runWindow;
      if (!object(window) || !Number.isInteger(window.datedRuns) || !Number.isInteger(window.totalRuns)
          || (window.datedRuns as number) < 1 || (window.datedRuns as number) > (window.totalRuns as number)) {
        throw new Error('The dated-run coverage is invalid.');
      }
      requireFact(window.started); requireFact(window.finished);
      if (dateValue(window.started as unknown as DateFact)! > dateValue(window.finished as unknown as DateFact)!) throw new Error('The recorded run window is reversed.');
    }
  }
  return value as unknown as ArtifactDateCatalog;
}
