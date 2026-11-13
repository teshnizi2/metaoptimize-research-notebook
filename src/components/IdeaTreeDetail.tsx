import type { ReactNode } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { ideaTree, statusInfo, type IdeaTreeData, type TreeNode, type TreeStatus } from '../lib/idea-tree';

export function TreeIcon({ status }: { status: TreeStatus }) {
  return <span className="tree-icon" aria-hidden="true">{statusInfo[status].icon}</span>;
}

/** How a page turns an experiment id into a link: the site uses its router, the standalone copy a plain address. */
export type TestLink = (id: string, className: string | undefined, children: ReactNode) => ReactNode;

export function TreeDetail({ node, inline, data = ideaTree, pageIds, testLink, onSelect }: {
  node: TreeNode | null;
  inline?: boolean;
  data?: IdeaTreeData;
  pageIds: ReadonlySet<string>;
  testLink: TestLink;
  /** Lets "Followed from" jump to the parent test. */
  onSelect?: (id: string) => void;
}) {
  if (!node) {
    return <div className="tree-detail tree-detail-empty" aria-live="polite">Select a part or a test to see what was tried, what came out, and where to read more.</div>;
  }
  const cls = `tree-detail${inline ? ' tree-detail-inline' : ''}`;
  if (node.kind === 'root') {
    return <article className={cls} aria-live="polite">
      <p className="tree-where">The project question</p>
      <h2>{node.title}</h2>
      <p className="tree-summary">{node.summary}</p>
    </article>;
  }
  if (node.kind === 'part') {
    const { part } = node;
    return <article className={`${cls} tree-${part.status}`} aria-live="polite">
      <header><span className="tree-verdict"><TreeIcon status={part.status}/>{statusInfo[part.status].name}</span></header>
      <p className="tree-where">Part {part.number}</p>
      <h2>{part.title}</h2>
      <p className="tree-headline">{part.verdict}</p>
      <p className="tree-summary">{part.summary}</p>
      <p className="tree-date"><strong>When:</strong> {part.dates}. {part.leaves.length} tests below.</p>
    </article>;
  }
  const { part, leaf } = node;
  const pages = leaf.experiments.filter(id => pageIds.has(id));
  const dateLabel = leaf.dateBasis === 'runs' ? 'Runs recorded' : leaf.dateBasis === 'part' ? `Part ${part.number} ran` : 'Date';
  const parentLeaf = leaf.parent ? part.leaves.find(l => l.id === leaf.parent) : undefined;
  const followUps = part.leaves.filter(l => l.parent === leaf.id);
  return <article className={`${cls} tree-${leaf.status}`} aria-live="polite">
    <header><span className="tree-verdict"><TreeIcon status={leaf.status}/>{statusInfo[leaf.status].name}</span></header>
    <p className="tree-where">Part {part.number} · {part.title}</p>
    <h2>{leaf.title}</h2>
    <p className="tree-headline">{leaf.result}</p>
    <p className="tree-summary">{leaf.summary}</p>
    <p className="tree-date"><strong>{dateLabel}:</strong> {leaf.date}</p>
    {(parentLeaf || followUps.length > 0) && <p className="tree-chain">
      {parentLeaf && <span>Followed from: {onSelect ? <button type="button" onClick={() => onSelect(parentLeaf.id)}>{parentLeaf.title}</button> : parentLeaf.title}{leaf.parentNote ? ` (${leaf.parentNote.replace(/\.$/, '')})` : ''}.</span>}
      {followUps.length > 0 && <span> Led to {followUps.length} follow-up test{followUps.length === 1 ? '' : 's'}.</span>}
    </p>}
    {pages.length > 0 && <div className="tree-links">
      {testLink(pages[0], 'button primary', <>Open the test page · {pages[0]}<ArrowUpRight size={16}/></>)}
      {pages.length > 1 && <span className="tree-more">Also: {pages.slice(1).map((id, i) => <span key={id}>{i > 0 && ', '}{testLink(id, undefined, id)}</span>)}</span>}
    </div>}
  </article>;
}
