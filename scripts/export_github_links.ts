/** Verify immutable GitHub commits and local Git blobs before publishing file links.
 * Run with: npx tsx scripts/export_github_links.ts --research-repo /path/to/research/repo
 * No fetch, checkout, working-tree write, or push is performed in the research repository.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, renameSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { NOTEBOOK_GITHUB, RESEARCH_GITHUB, sourceGitHubLink } from '../src/lib/github.ts';
import type { SourceGitHubEvidence, GitHubFileEvidence } from '../src/lib/github.ts';
import type { SourceFile } from '../src/types.ts';

const portal = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: {
  'research-repo': { type: 'string' },
  'research-revision': { type: 'string', default: '627d69ffd624d768178719b4c5b52b0e3b0e9ed5' },
  'archive-revision': { type: 'string', default: 'e58287a96e59b9d6533023a39cebdfd7db21ff1e' },
} });
if (!values['research-repo']) throw new Error('--research-repo is required; the original repository is read only.');
const research = resolve(values['research-repo']);
const researchRevision = values['research-revision']!, archiveRevision = values['archive-revision']!;
const git = (cwd: string, args: string[]) => execFileSync('git', args, { cwd, maxBuffer: 16 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'] });
const sha256 = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

function verifiedRemoteCommit(cwd: string, repository: string, revision: string) {
  if (!/^[0-9a-f]{40}$/.test(revision)) throw new Error('Use a full immutable commit SHA.');
  const local = git(cwd, ['rev-parse', '--verify', `${revision}^{commit}`]).toString().trim();
  const localTree = git(cwd, ['rev-parse', `${revision}^{tree}`]).toString().trim();
  const slug = repository.replace('https://github.com/', '');
  let remote;
  try {
    remote = JSON.parse(execFileSync('gh', ['api', `repos/${slug}/git/commits/${revision}`],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }));
  } catch (error) {
    if (String((error as { stderr?: Buffer }).stderr).includes('HTTP 404')) return false;
    throw error;
  }
  if (remote.sha !== local || remote.tree?.sha !== localTree) throw new Error(`GitHub commit/tree verification failed for ${slug}.`);
  return true;
}

const originalAvailable = verifiedRemoteCommit(research, RESEARCH_GITHUB, researchRevision);
const archiveAvailable = verifiedRemoteCommit(portal, NOTEBOOK_GITHUB, archiveRevision);
if (!archiveAvailable) throw new Error('The archived-copy commit is not available on GitHub.');

function fileEvidence(cwd: string, revision: string, path: string, expectedHash: string, remoteVerified: boolean): GitHubFileEvidence | undefined {
  if (!remoteVerified) return undefined;
  let bytes;
  try { bytes = git(cwd, ['show', `${revision}:${path}`]); }
  catch { return undefined; }
  return sha256(bytes) === expectedHash
    ? { revision, path, sha256: expectedHash, remoteCommitVerified: true }
    : undefined;
}

const sources: SourceFile[] = JSON.parse(readFileSync(resolve(portal, 'public/data/source-index.json'), 'utf8'));
// The source exporter derives IDs from original paths before redacting display paths.
// Resolve those IDs against the pinned Git tree, never by guessing a similar filename.
const originalPaths = new Map(git(research, ['ls-tree', '-r', '--name-only', '-z', researchRevision])
  .toString().split('\0').filter(Boolean)
  .map(path => ['src-' + sha256(Buffer.from(path)).slice(0, 16), path]));
const links: Record<string, SourceGitHubEvidence> = {};
const counts = { original: 0, archived: 0 };
for (const source of sources) {
  const originalPath = originalPaths.get(source.id);
  const original = originalPath
    ? fileEvidence(research, researchRevision, originalPath, source.originalSha256, originalAvailable)
    : undefined;
  if (original && original.path !== source.path) original.displayPath = source.path;
  const archived = fileEvidence(portal, archiveRevision, `public/source/${source.id}.txt`, source.publicSha256, archiveAvailable);
  const evidence = { ...(original ? { original } : {}), ...(archived ? { archived } : {}) };
  const link = sourceGitHubLink(source, evidence);
  if (!link) throw new Error(`No exact GitHub file verified for ${source.path}; the catalog was not changed.`);
  links[source.id] = evidence;
  counts[link.kind]++;
}
const catalog = {
  schemaVersion: 1,
  verification: 'Each commit SHA and Git tree were checked against GitHub; file bytes were SHA-256 checked against the source catalog. This records file identity, not historical runtime provenance.',
  repositories: {
    research: { url: RESEARCH_GITHUB, revision: researchRevision, remoteCommitVerified: originalAvailable },
    notebook: { url: NOTEBOOK_GITHUB, revision: archiveRevision, remoteCommitVerified: archiveAvailable },
  },
  counts,
  sources: links,
};
const output = resolve(portal, 'public/data/github-links.json');
writeFileSync(`${output}.tmp`, JSON.stringify(catalog, null, 2) + '\n');
renameSync(`${output}.tmp`, output);
process.stdout.write(JSON.stringify({ status: 'PASS', sources: sources.length, ...counts, researchRevision, archiveRevision }) + '\n');
