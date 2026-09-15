import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const moduleUrl = new URL('../src/lib/github.ts', import.meta.url);
const originalRevision = '627d69ffd624d768178719b4c5b52b0e3b0e9ed5';
const archiveRevision = 'e58287a96e59b9d6533023a39cebdfd7db21ff1e';
const source = { id: 'src-example', path: 'analysis/score.py', originalSha256: 'a'.repeat(64), publicSha256: 'b'.repeat(64), lines: 120 };
const evidence = () => ({
  original: { revision: originalRevision, path: source.path, sha256: source.originalSha256, remoteCommitVerified: true },
  archived: { revision: archiveRevision, path: 'public/source/src-example.txt', sha256: source.publicSha256, remoteCommitVerified: true },
});
async function implementation() {
  assert.ok(existsSync(moduleUrl), 'Verified GitHub source mapping has not been implemented');
  return import(moduleUrl.href);
}

test('an exact original hash and verified remote revision link to the research repository', async () => {
  const { sourceGitHubLink } = await implementation();
  const result = sourceGitHubLink(source, evidence(), 81);
  assert.equal(result.kind, 'original');
  assert.equal(result.label, 'Original on GitHub');
  assert.equal(result.href, `https://github.com/teshnizi2/hierarchical-metaoptimize/blob/${originalRevision}/analysis/score.py#L81`);
});

test('uncommitted content uses the verified archived copy, never an original-file claim', async () => {
  const { sourceGitHubLink } = await implementation();
  const record = evidence();
  record.original.sha256 = 'c'.repeat(64);
  const result = sourceGitHubLink(source, record);
  assert.equal(result.kind, 'archived');
  assert.equal(result.label, 'Archived copy on GitHub');
  assert.equal(result.href, `https://github.com/teshnizi2/metaoptimize-research-notebook/blob/${archiveRevision}/public/source/src-example.txt`);
});

test('a local-only original revision cannot produce an original GitHub link', async () => {
  const { sourceGitHubLink } = await implementation();
  const record = evidence();
  record.original.remoteCommitVerified = false;
  assert.equal(sourceGitHubLink(source, record).kind, 'archived');
  record.archived.remoteCommitVerified = false;
  assert.equal(sourceGitHubLink(source, record), null);
});

test('changed or misattributed archive bytes produce no invented fallback link', async () => {
  const { sourceGitHubLink } = await implementation();
  const record = evidence();
  record.original.remoteCommitVerified = false;
  record.archived.sha256 = 'c'.repeat(64);
  assert.equal(sourceGitHubLink(source, record), null);
  record.archived.sha256 = source.publicSha256;
  record.archived.path = 'public/source/another-source.txt';
  assert.equal(sourceGitHubLink(source, record), null);
  assert.equal(sourceGitHubLink(source, undefined), null);
});

test('mutable branch refs and mismatched original paths are never accepted as exact revisions', async () => {
  const { sourceGitHubLink } = await implementation();
  const record = evidence();
  record.original.revision = 'master';
  assert.equal(sourceGitHubLink(source, record).kind, 'archived');
  record.original.revision = originalRevision;
  record.original.path = 'analysis/different.py';
  assert.equal(sourceGitHubLink(source, record).kind, 'archived');
});

test('a verified original can retain its real filename when the displayed archive path was redacted', async () => {
  const { sourceGitHubLink } = await implementation();
  const file = { ...source, path: 'bin/cluster_a.sh' };
  const record = { original: { ...evidence().original, path: 'bin/original_cluster.sh', displayPath: file.path } };
  const link = sourceGitHubLink(file, record);
  assert.ok(link, 'The verified original filename should be retained');
  assert.equal(link.kind, 'original');
  assert.ok(link.href.endsWith('/bin/original_cluster.sh'));
  record.original.displayPath = 'bin/different.sh';
  assert.equal(sourceGitHubLink(file, record), null);
});

test('source paths are encoded and line links stay within the archived line count', async () => {
  const { sourceGitHubLink } = await implementation();
  const file = { ...source, path: 'analysis/a file#1.py' }, record = evidence();
  record.original.path = file.path;
  assert.ok(sourceGitHubLink(file, record, 999).href.endsWith('/analysis/a%20file%231.py#L120'));
  assert.ok(sourceGitHubLink(file, record, -1).href.endsWith('#L1'));
  assert.ok(sourceGitHubLink(file, record, Number.NaN).href.endsWith('#L1'));
  const unsafe = { ...file, path: '../outside.py' };
  record.original.path = unsafe.path;
  assert.equal(sourceGitHubLink(unsafe, { original: record.original }), null);
});

test('the generated GitHub catalog covers the current source snapshot without stale hashes', async () => {
  const { sourceGitHubLink } = await implementation();
  const catalogUrl = new URL('../public/data/github-links.json', import.meta.url);
  assert.ok(existsSync(catalogUrl), 'The verified GitHub source catalog has not been generated');
  const catalog = JSON.parse(readFileSync(catalogUrl, 'utf8'));
  const sources = JSON.parse(readFileSync(new URL('../public/data/source-index.json', import.meta.url), 'utf8'));
  assert.equal(catalog.schemaVersion, 1);
  assert.deepEqual(Object.keys(catalog.sources).sort(), sources.map((s: { id: string }) => s.id).sort());
  for (const file of sources) {
    const record = catalog.sources[file.id];
    assert.ok(sourceGitHubLink(file, record, file.lines), file.path);
    if (record.original) {
      assert.equal('src-' + createHash('sha256').update(record.original.path).digest('hex').slice(0, 16), file.id);
    }
  }
  const scorer = sources.find((s: { path: string }) => s.path === 'analysis/cVK2_vggcut_score.py');
  assert.equal(sourceGitHubLink(scorer, catalog.sources[scorer.id]).kind, 'original');
  const runtime = sources.find((s: { path: string }) => s.path === 'runtime/cifar10/train.py');
  assert.equal(sourceGitHubLink(runtime, catalog.sources[runtime.id]).kind, 'archived');
});
