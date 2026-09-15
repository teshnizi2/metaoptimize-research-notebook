import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { atomicJson, readJournalState, withJournalLock } from './journal.mjs';

export async function syncJournal(root = process.cwd()) {
  return withJournalLock(root, async () => {
    const { entries } = await readJournalState(root);
    await atomicJson(resolve(root, 'public/data/journal.json'), entries);
    return entries.length;
  });
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { const count = await syncJournal(); console.log(`Validated and synchronized ${count} journal entries.`); }
  catch (error) { console.error(`Journal sync: ${error.message}`); process.exitCode = 1; }
}
