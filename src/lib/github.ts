import type { SourceFile } from '../types';

export const RESEARCH_GITHUB = 'https://github.com/teshnizi2/hierarchical-metaoptimize';
export const NOTEBOOK_GITHUB = 'https://github.com/teshnizi2/metaoptimize-research-notebook';

export interface GitHubFileEvidence {
  revision: string;
  path: string;
  displayPath?: string;
  sha256: string;
  remoteCommitVerified: boolean;
}
export interface SourceGitHubEvidence {
  original?: GitHubFileEvidence;
  archived?: GitHubFileEvidence;
}
export interface GitHubCatalog {
  schemaVersion: 1;
  sources: Record<string, SourceGitHubEvidence>;
}
type SourceIdentity = Pick<SourceFile, 'id' | 'path' | 'originalSha256' | 'publicSha256' | 'lines'>;

function verifiedFile(evidence: GitHubFileEvidence | undefined, sha256: string) {
  return evidence?.remoteCommitVerified === true && /^[0-9a-f]{40}$/.test(evidence.revision)
    && evidence.sha256 === sha256 && /^[0-9a-f]{64}$/.test(sha256)
    && !/[\\\u0000-\u001f\u007f]/.test(evidence.path)
    && evidence.path.split('/').every(part => part !== '' && part !== '.' && part !== '..');
}

/** Exact file identity only; a GitHub link does not assert historical runtime provenance. */
export function sourceGitHubLink(source: SourceIdentity, evidence?: SourceGitHubEvidence, line?: number) {
  let kind: 'original' | 'archived', file: GitHubFileEvidence;
  if (verifiedFile(evidence?.original, source.originalSha256)
      && (evidence?.original?.displayPath ?? evidence?.original?.path) === source.path) {
    kind = 'original'; file = evidence!.original!;
  } else if (verifiedFile(evidence?.archived, source.publicSha256)
      && evidence?.archived?.path === `public/source/${source.id}.txt`) {
    kind = 'archived'; file = evidence!.archived!;
  } else return null;
  const repository = kind === 'original' ? RESEARCH_GITHUB : NOTEBOOK_GITHUB;
  const anchor = line === undefined ? '' : `#L${Math.max(1, Math.min(source.lines,
    Number.isFinite(line) ? Math.floor(line) : 1))}`;
  return {
    kind,
    revision: file.revision,
    href: `${repository}/blob/${file.revision}/${file.path.split('/').map(encodeURIComponent).join('/')}${anchor}`,
    label: kind === 'original' ? 'Original on GitHub' : 'Archived copy on GitHub',
    description: kind === 'original'
      ? `Original file hash verified at ${file.revision.slice(0, 12)}.`
      : `Archived notebook copy at ${file.revision.slice(0, 12)}.`,
  };
}
