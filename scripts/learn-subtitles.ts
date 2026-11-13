/** Usage: tsx scripts/learn-subtitles.ts <folder-with-srt> <output-folder>
 *  Writes one <id>.vtt per srt file; the id is the first word of the file name (B01, 01 …). Output stays out of the repo. */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { srtToVtt } from '../src/lib/srt-to-vtt.ts';

const [from, to] = process.argv.slice(2);
if (!from || !to) throw new Error('Usage: tsx scripts/learn-subtitles.ts <srt-folder> <output-folder>');
mkdirSync(to, { recursive: true });
for (const name of readdirSync(from).filter(f => f.endsWith('.srt'))) {
  const id = name.split(' ')[0];
  writeFileSync(join(to, `${id}.vtt`), srtToVtt(readFileSync(join(from, name), 'utf8')));
}
