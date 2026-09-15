import { mkdir, open, readFile, rename, unlink } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { isIP } from 'node:net';

const kinds = new Set(['note', 'correction', 'warning']);
const fields = new Set(['id', 'date', 'kind', 'title', 'detail', 'experimentIds']);
const help = `Append a versioned research-log entry:
  npm run log -- --type note --title "Short title" --text "What changed and why" --experiments MT014,CVK2
Optional: --id J-UNIQUE-ID --date 2026-09-15T09:00:00.000Z
Types: note, correction, warning. Existing entries are immutable.
The command writes content/journal.json; build and deploy publish the update.`;

async function readJson(path, optional = false) {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch (error) {
    if (optional && error.code === 'ENOENT') return [];
    throw new Error(`Cannot read valid JSON from ${path}: ${error.message}`);
  }
}
function validDate(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?Z)?$/.test(value)) return false;
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime())) return false;
  const iso = parsed.toISOString();
  return value.length === 10 ? iso.slice(0, 10) === value : iso.slice(0, 19) === value.slice(0, 19);
}
export function requirePublicJournal(entry) {
  const text = Object.values(entry).flat().filter(value => typeof value === 'string').join('\n');
  const sensitive = [
    /\/(?:Users|home|data1|scratch)\/|\/zfsstore\/user\/|\/private\/var\/|[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+|~[\\/]/i,
    /\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|passwd|secret|private[_ -]?key)["']?\s*[:=]\s*\S+/i,
    /[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/,
    /\b(?:ssh|scp|sftp)\s+(?:-\S+\s+)*[\w.-]+@[\w.-]+/i,
    /\b(?:cluster|ssh|slurm|hpc|infrastructure)[_ -](?:account|user(?:name)?|login|host)\s*[:=]\s*\S+/i,
    /\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----/,
  ];
  const address = (text.match(/[0-9A-Fa-f:.]+/g) || []).some(value => isIP(value.replace(/\.$/, '')) !== 0);
  if (address || sensitive.some(pattern => pattern.test(text))) {
    throw new Error('Private or sensitive content is not allowed in the public journal. Remove infrastructure identifiers, private paths, contact addresses and credentials before appending.');
  }
}
export function validateJournal(entries, research) {
  if (!Array.isArray(research.experiments) || !research.experiments.length) throw new Error('Research snapshot has no valid experiment index. Import it first.');
  if (!Array.isArray(entries)) throw new Error('Journal must be a JSON array.');
  const experimentIds = new Set(research.experiments.map(e => e.id));
  const used = new Set((research.activity || []).map(e => e.id));
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) throw new Error('Every journal entry must be an object.');
    if (Object.keys(entry).some(key => !fields.has(key))) throw new Error('Journal entry contains an unsupported field.');
    requirePublicJournal(entry);
    if (typeof entry.id !== 'string' || !/^[A-Za-z][A-Za-z0-9_-]{0,79}$/.test(entry.id)) throw new Error('Invalid journal ID: use letters, numbers, underscores and hyphens.');
    if (used.has(entry.id)) throw new Error(`Duplicate journal ID: ${entry.id} already exists.`);
    used.add(entry.id);
    if (!kinds.has(entry.kind)) throw new Error(`Unsupported journal type: ${entry.kind}. Use note, correction or warning.`);
    for (const [field, max] of [['title', 160], ['detail', 20000]]) {
      if (typeof entry[field] !== 'string' || !entry[field].trim() || entry[field].length > max) throw new Error(`Journal ${field} must contain 1–${max} characters.`);
    }
    if (!validDate(entry.date)) throw new Error(`Invalid journal date: ${entry.date}. Use YYYY-MM-DD or a UTC ISO timestamp.`);
    if (!Array.isArray(entry.experimentIds) || !entry.experimentIds.length) throw new Error('Link at least one experiment ID.');
    if (new Set(entry.experimentIds).size !== entry.experimentIds.length) throw new Error('Duplicate linked experiment IDs.');
    for (const id of entry.experimentIds) if (!experimentIds.has(id)) throw new Error(`Unknown experiment ID: ${String(id)}.`);
  }
  return entries;
}
function requirePublishedPrefix(entries, published) {
  if (published.length > entries.length || !published.every((entry, i) => isDeepStrictEqual(entry, entries[i]))) {
    throw new Error('Journal is append-only: published entries are immutable. Restore previous entries and append a correction.');
  }
}
export async function readJournalState(root = process.cwd()) {
  const research = await readJson(resolve(root, 'public/data/research.json'));
  const entries = validateJournal(await readJson(resolve(root, 'content/journal.json')), research);
  const published = validateJournal(await readJson(resolve(root, 'public/data/journal.json'), true), research);
  requirePublishedPrefix(entries, published);
  return { research, entries };
}
export async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.tmp`;
  let file;
  try {
    file = await open(temporary, 'wx', 0o644);
    await file.writeFile(JSON.stringify(value, null, 2) + '\n', 'utf8');
    await file.sync();
    await file.close(); file = null;
    await rename(temporary, path);
  } finally {
    if (file) await file.close();
    await unlink(temporary).catch(error => { if (error.code !== 'ENOENT') throw error; });
  }
}
export async function withJournalLock(root, operation) {
  const lock = resolve(root, 'content/.journal.lock');
  await mkdir(dirname(lock), { recursive: true });
  let handle;
  try { handle = await open(lock, 'wx', 0o644); }
  catch (error) {
    if (error.code === 'EEXIST') throw new Error('A journal write is already in progress (.journal.lock). If interrupted, confirm no writer is running before removing its lock.');
    throw error;
  }
  try { await handle.writeFile(`pid=${process.pid}\n`); return await operation(); }
  finally { await handle.close(); await unlink(lock); }
}
export async function appendJournal(entry, root = process.cwd()) {
  return withJournalLock(root, async () => {
    const { entries, research } = await readJournalState(root);
    const next = [...entries, entry];
    validateJournal(next, research);
    await atomicJson(resolve(root, 'content/journal.json'), next);
    return entry;
  });
}
export function parseArguments(args) {
  const allowed = new Set(['type', 'title', 'text', 'experiments', 'id', 'date']);
  const values = {};
  for (let i = 0; i < args.length; i += 2) {
    const name = args[i].startsWith('--') ? args[i].slice(2) : '';
    if (!allowed.has(name)) throw new Error(`Unknown option: ${args[i]}. Use --help for usage.`);
    if (name in values) throw new Error(`Duplicate option: --${name}.`);
    if (args[i + 1] === undefined || args[i + 1].startsWith('--')) throw new Error(`Missing value for --${name}.`);
    values[name] = args[i + 1];
  }
  for (const name of ['type', 'title', 'text', 'experiments']) if (!(name in values)) throw new Error(`Missing --${name}. Use --help for usage.`);
  return { id: values.id ?? `J-${randomUUID()}`, date: values.date ?? new Date().toISOString(),
    kind: values.type, title: values.title.trim(), detail: values.text.trim(),
    experimentIds: values.experiments.split(',').map(id => id.trim()) };
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.slice(2).length === 1 && process.argv[2] === '--help') console.log(help);
  else {
    try { const entry = await appendJournal(parseArguments(process.argv.slice(2))); console.log(`Appended ${entry.id} to content/journal.json. Build and deploy to publish it.`); }
    catch (error) { console.error(`Journal: ${error.message}`); process.exitCode = 1; }
  }
}
