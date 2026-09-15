import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const HASH = /^[a-f0-9]{64}$/;
const FIELDS = ['created', 'firstRecorded', 'updated', 'exported', 'submitted', 'started', 'finished'];
const KINDS = new Set(['figure', 'table', 'source', 'experiment', 'run', 'warning', 'chart', 'download']);
const hash = value => createHash('sha256').update(value).digest('hex');
const insist = (value, message) => { if (!value) throw new Error(message); };
const unique = values => [...new Set(values)].sort();

/** Single canonical implementation, also used by the Python maintainer importer. */
export function canonicalJson(value) {
  if (Array.isArray(value)) return '[' + value.map(canonicalJson).join(',') + ']';
  if (value !== null && typeof value === 'object') {
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonicalJson(value[key])).join(',') + '}';
  }
  insist(value !== undefined && (typeof value !== 'number' || Number.isFinite(value)), 'Non-JSON canonical value');
  return JSON.stringify(value);
}

function publicPath(href) {
  insist(typeof href === 'string' && /^\/(?:assets|source|data)\//.test(href), 'Invalid public evidence path');
  let decoded;
  try { decoded = decodeURIComponent(href.split('#')[0]); } catch { throw new Error('Invalid public path encoding'); }
  insist(!/[\\?\x00-\x1f\x7f]/.test(decoded) && !decoded.split('/').some(x => x === '..' || x === '.'), 'Unsafe public evidence path');
  insist(!href.includes('#') || /#L[1-9]\d*$/.test(href), 'Invalid public evidence fragment');
  return decoded;
}

/** All current entities and bytes; the date catalog and source ZIP are not inputs. */
export function buildInventory(root = process.cwd()) {
  const publicRoot = realpathSync(resolve(root, 'public'));
  const bytes = href => {
    const path = realpathSync(resolve(publicRoot, '.' + publicPath(href)));
    insist(path.startsWith(publicRoot + sep), 'Public evidence path escapes the public directory');
    return readFileSync(path);
  };
  const researchBytes = bytes('/data/research.json'), runBytes = bytes('/data/runs.json');
  const research = JSON.parse(researchBytes), runs = JSON.parse(runBytes);
  const experiments = new Map(research.experiments.map(x => [x.id, x]));
  const runIds = new Set(runs.map(x => x.id));
  insist(experiments.size === research.experiments.length && runIds.size === runs.length, 'Duplicate experiment or run ID');
  const records = {};
  const linkedRuns = ids => unique(ids.flatMap(id => experiments.get(id).runIds));
  const add = (kind, row, experimentIds, options = {}) => {
    const key = `${kind}:${row.id}`;
    insist(!records[key] && typeof row.id === 'string' && row.id.length > 0, `Duplicate or invalid artifact: ${key}`);
    const ids = unique(experimentIds);
    insist(ids.every(id => experiments.has(id)), `Unknown experiment on ${key}`);
    const linked = options.runIds ?? linkedRuns(ids);
    insist(linked.every(id => runIds.has(id)), `Unknown linked run on ${key}`);
    records[key] = { kind, id: row.id, contentSha256: options.file ? hash(bytes(row.href)) : hash(canonicalJson(row)), experimentIds: ids, runIds: unique(linked), ...(row.href ? { href: row.href } : {}), ...(options.evidence ? { evidence: options.evidence } : {}) };
  };
  for (const row of research.figures) add('figure', row, row.experimentIds, { file: true });
  for (const row of research.tables) add('table', row, row.experimentIds, { file: true });
  for (const row of research.sources) {
    insist(HASH.test(row.originalSha256) && hash(bytes(row.href)) === row.publicSha256, `Source hash mismatch: ${row.id}`);
    add('source', row, row.experimentIds, { evidence: { originalSha256: row.originalSha256, publicSha256: row.publicSha256 } });
  }
  for (const row of research.experiments) add('experiment', row, [row.id]);
  for (const row of runs) {
    const evidence = { jobId: row.jobId, originalLogSha256: row.parameters.originalLogSha256, publicLogSha256: row.parameters.publicLogSha256 };
    insist(HASH.test(evidence.originalLogSha256) && hash(bytes(row.logHref)) === evidence.publicLogSha256, `Run log hash mismatch: ${row.id}`);
    add('run', row, row.experimentIds, { runIds: [row.id], evidence });
  }
  for (const row of research.warnings) add('warning', row, row.experimentId ? [row.experimentId] : []);
  add('chart', { ...research.latest, id: research.latest.experimentId }, [research.latest.experimentId]);
  for (const id of ['pdf', 'bundle']) add('download', { id, href: research.meta.downloads[id] }, [...experiments.keys()], { file: true, runIds: [...runIds] });
  const publicHrefs = [];
  const walk = path => {
    for (const entry of readdirSync(path, { withFileTypes: true })) {
      const child = resolve(path, entry.name);
      if (entry.isDirectory()) walk(child);
      else if (entry.isFile()) publicHrefs.push('/' + relative(publicRoot, child).split(sep).join('/'));
    }
  };
  walk(publicRoot);
  return { schemaVersion: 1, snapshotId: research.meta.snapshotId, snapshotSha256: hash(researchBytes), runsSha256: hash(runBytes), publishedAt: research.meta.asOf, exportedAt: research.meta.generatedAt, records: Object.fromEntries(Object.entries(records).sort(([a], [b]) => a.localeCompare(b))), publicHrefs: publicHrefs.sort() };
}

function timeValue(at, precision) {
  insist(typeof at === 'string', 'Missing date value');
  const day = /^(\d{4})-(\d{2})-(\d{2})$/;
  const second = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(Z|[+-]\d{2}:\d{2})$/;
  const parts = (precision === 'day' ? day : precision === 'second' ? second : /$a/).exec(at);
  insist(parts, `Malformed ${precision} date: ${at}`);
  const year = Number(parts[1]), month = Number(parts[2]), date = Number(parts[3]);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  insist(year >= 1 && month >= 1 && month <= 12 && date >= 1 && date <= days[month - 1], `Impossible calendar date: ${at}`);
  if (precision === 'second') {
    insist(Number(parts[4]) < 24 && Number(parts[5]) < 60 && Number(parts[6]) < 60, `Impossible clock time: ${at}`);
    if (parts[7] !== 'Z') insist(Number(parts[7].slice(1, 3)) <= 14 && Number(parts[7].slice(4)) < 60 && (Number(parts[7].slice(1, 3)) < 14 || parts[7].slice(4) === '00'), `Invalid timezone offset: ${at}`);
  }
  const ms = Date.parse(precision === 'day' ? at + 'T00:00:00Z' : at);
  insist(Number.isFinite(ms), `Invalid date: ${at}`);
  return ms;
}

function validateFact(fact, inventory) {
  insist(fact && typeof fact === 'object' && !Array.isArray(fact), 'Malformed date fact');
  timeValue(fact.at, fact.precision);
  insist(typeof fact.basis === 'string' && fact.basis.trim().length > 0, 'Missing date basis');
  if (fact.href != null) {
    if (fact.href.startsWith('/')) {
      const path = publicPath(fact.href);
      insist(inventory.publicHrefs.includes(path), `Missing date provenance link: ${path}`);
    } else {
      let url;
      try { url = new URL(fact.href); } catch { throw new Error('Invalid date provenance URL'); }
      insist(url.protocol === 'https:' && url.hostname === 'github.com' && !url.username && !url.password && !url.port && !url.search && /^\/teshnizi2\/(?:hierarchical-metaoptimize|metaoptimize-research-notebook)\/(?:commit\/[a-f0-9]{40}|blob\/[a-f0-9]{40}\/[^?#]+)$/.test(url.pathname), 'Unsafe date provenance URL');
      insist(!decodeURIComponent(url.pathname).split('/').some(x => x === '.' || x === '..'), 'Unsafe GitHub evidence path');
    }
  }
}

export function validateCatalog(catalog, inventory) {
  insist(catalog && catalog.schemaVersion === 1 && catalog.records && !Array.isArray(catalog.records), 'Invalid date catalog schema');
  for (const key of ['snapshotId', 'snapshotSha256', 'runsSha256', 'publishedAt', 'exportedAt']) insist(catalog[key] === inventory[key], `Date catalog ${key} mismatch`);
  insist(HASH.test(catalog.snapshotSha256) && HASH.test(catalog.runsSha256), 'Invalid snapshot hash');
  for (const key of ['publishedAt', 'exportedAt', 'recordedAt']) timeValue(catalog[key], /^\d{4}-\d{2}-\d{2}$/.test(catalog[key]) ? 'day' : 'second');
  const expected = Object.keys(inventory.records).sort(), actual = Object.keys(catalog.records).sort();
  insist(canonicalJson(expected) === canonicalJson(actual), 'Date catalog artifact coverage mismatch (missing or unknown IDs)');
  for (const key of expected) {
    const row = catalog.records[key], wanted = inventory.records[key];
    insist(row && KINDS.has(row.kind) && row.kind === wanted.kind && row.id === wanted.id, `Invalid date record identity: ${key}`);
    insist(HASH.test(row.contentSha256) && row.contentSha256 === wanted.contentSha256, `Date content hash mismatch: ${key}`);
    insist(canonicalJson(row.experimentIds) === canonicalJson(wanted.experimentIds), `Date experiment associations mismatch: ${key}`);
    for (const field of FIELDS) if (row[field] != null) validateFact(row[field], inventory);
    const order = ['submitted', 'started', 'finished'].filter(field => row[field] != null);
    for (let n = 1; n < order.length; n++) insist(timeValue(row[order[n - 1]].at, row[order[n - 1]].precision) <= timeValue(row[order[n]].at, row[order[n]].precision), `Run date order mismatch: ${key}`);
    if (row.runWindow != null) {
      const window = row.runWindow;
      validateFact(window.started, inventory); validateFact(window.finished, inventory);
      const dated = wanted.runIds.map(id => catalog.records[`run:${id}`]).filter(run => run?.started && run?.finished);
      insist(Number.isInteger(window.datedRuns) && Number.isInteger(window.totalRuns) && window.totalRuns === wanted.runIds.length && window.datedRuns === dated.length && dated.length > 0, `Run window coverage mismatch: ${key}`);
      const start = Math.min(...dated.map(run => timeValue(run.started.at, run.started.precision)));
      const finish = Math.max(...dated.map(run => timeValue(run.finished.at, run.finished.precision)));
      insist(timeValue(window.started.at, window.started.precision) === start && timeValue(window.finished.at, window.finished.precision) === finish, `Run window extrema mismatch: ${key}`);
    }
  }
  return { records: expected.length, counts: Object.fromEntries([...KINDS].map(kind => [kind, expected.filter(key => key.startsWith(kind + ':')).length])) };
}

export function syncCatalog(root = process.cwd(), checkOnly = false) {
  const source = resolve(root, 'content/artifact-dates.json'), destination = resolve(root, 'public/data/artifact-dates.json');
  const content = readFileSync(source, 'utf8'), catalog = JSON.parse(content);
  const result = validateCatalog(catalog, buildInventory(root));
  if (checkOnly) insist(existsSync(destination) && readFileSync(destination, 'utf8') === content, 'Public date catalog differs from the versioned content catalog');
  else if (!existsSync(destination) || readFileSync(destination, 'utf8') !== content) {
    mkdirSync(dirname(destination), { recursive: true });
    const temporary = destination + '.tmp'; writeFileSync(temporary, content); renameSync(temporary, destination);
  }
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const args = process.argv.slice(2), position = args.indexOf('--root');
    const root = position >= 0 ? resolve(args[position + 1]) : process.cwd();
    if (args.includes('--inventory')) console.log(JSON.stringify(buildInventory(root)));
    else {
      const result = syncCatalog(root, args.includes('--check'));
      console.log(`Validated ${result.records} artifact date records${args.includes('--check') ? '' : ' and synchronized the public catalog'}.`);
    }
  } catch (error) { console.error(`Artifact dates: ${error.message}`); process.exitCode = 1; }
}
