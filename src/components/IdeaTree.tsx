import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ChevronRight, Star } from 'lucide-react';
import { useResearch } from '../data';
import { countStatuses, findNode, ideaTree, keyCount, keyIds, statusInfo, statusOrder, type TreeNode } from '../lib/idea-tree';
import { KeyLegendItem, KeyStar, TreeDetail, TreeIcon, testPath, type TestLink } from './IdeaTreeDetail';
import './IdeaTree.css';

const WIDE_QUERY = '(min-width: 1000px)';

function useWide() {
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.matchMedia(WIDE_QUERY).matches);
  useEffect(() => {
    const query = window.matchMedia(WIDE_QUERY);
    const update = () => setWide(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  return wide;
}

const Icon = TreeIcon;

const siteLink: TestLink = (id, className, children) => <Link className={className} to={`/${testPath(id)}`}>{children}</Link>;

function Detail({ node, inline }: { node: TreeNode | null; inline?: boolean }) {
  const data = useResearch();
  const pageIds = useMemo(() => new Set(data.experiments.map(e => e.id)), [data]);
  return <TreeDetail node={node} inline={inline} pageIds={pageIds} testLink={siteLink}/>;
}

export function IdeaTree() {
  const [params, setParams] = useSearchParams();
  const wide = useWide();
  const selectedId = params.get('node');
  const node = findNode(ideaTree, selectedId);
  const [open, setOpen] = useState<Set<string>>(() => {
    const start = findNode(ideaTree, selectedId);
    return new Set(start && start.kind !== 'root' ? [start.kind === 'part' ? start.id : start.part.id] : []);
  });

  useEffect(() => {
    // A shared link such as ?node=B4.7 opens at the selected box.
    if (!selectedId) return;
    const timer = window.setTimeout(() => document.querySelector('.tree-node[aria-current="true"]')?.scrollIntoView({ block: 'center' }), 120);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const select = (id: string) => setParams({ node: id }, { replace: true });
  const toggle = (id: string) => setOpen(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const allOpen = open.size === ideaTree.parts.length;
  const allLeaves = ideaTree.parts.flatMap(p => p.leaves);
  const totals = countStatuses(allLeaves);

  const keyParts = ideaTree.parts.filter(p => keyCount(p) > 0).map(p => p.id);
  const showKeys = () => {
    setOpen(new Set(keyParts));
    window.setTimeout(() => document.querySelector('.tree-part-row .tree-star')?.scrollIntoView({ block: 'center' }), 80);
  };

  const detailBelow = (id: string) => !wide && node && node.id === id ? <Detail node={node} inline/> : null;

  return <div className={wide ? 'tree-wide' : 'tree-narrow'}>
    <ul className="tree-legend" aria-label="Legend">
      {statusOrder.map(s => <li key={s} className={`tree-${s}`}><Icon status={s}/><span>{statusInfo[s].name}<small>{totals[s]} tests</small></span></li>)}
      <KeyLegendItem total={keyIds(ideaTree).length}/>
    </ul>
    <div className="tree-toolbar">
      <p>Colour shows what the project concluded. Select a box for a short summary; the summary links to that test’s page when the notebook has one.</p>
      <button className="button" type="button" onClick={showKeys}><Star size={15}/> Show key results</button>
      <button className="button" type="button" onClick={() => setOpen(allOpen ? new Set() : new Set(ideaTree.parts.map(p => p.id)))}>{allOpen ? 'Collapse all parts' : 'Open all parts'}</button>
    </div>
    <div className="tree-layout">
      <div>
        <button type="button" className="tree-node tree-root" aria-current={selectedId === 'root'} onClick={() => select('root')}>
          <span className="tree-node-text"><span className="tree-node-kicker">The question</span><strong>{ideaTree.root.title}</strong></span>
        </button>
        {detailBelow('root')}
        <ul className="tree-list tree-parts">
          {ideaTree.parts.map(part => {
            const expanded = open.has(part.id);
            const counts = countStatuses(part.leaves);
            return <li key={part.id}>
              <div className="tree-part-row">
                <button type="button" className={`tree-node tree-${part.status}`} aria-current={selectedId === part.id} onClick={() => { select(part.id); if (!expanded) toggle(part.id); }}>
                  <Icon status={part.status}/>
                  <span className="tree-node-text">
                    <span className="tree-node-kicker">Part {part.number} · {part.dates}{keyCount(part) > 0 && <> <KeyStar count={keyCount(part)}/></>}</span>
                    <strong>{part.title}</strong>
                    <small>{part.verdict}</small>
                    <span className="tree-dots" aria-hidden="true">{part.leaves.map(l => <i key={l.id} className={`tree-${l.status}`}/>)}</span>
                    <span className="tree-sr">{statusOrder.filter(s => counts[s]).map(s => `${counts[s]} ${statusInfo[s].short.toLowerCase()}`).join(', ')}</span>
                  </span>
                </button>
                <button type="button" className="tree-toggle" aria-expanded={expanded} aria-label={`${expanded ? 'Hide' : 'Show'} the ${part.leaves.length} tests of part ${part.number}`} onClick={() => toggle(part.id)}>
                  <ChevronRight size={20}/>
                </button>
              </div>
              {detailBelow(part.id)}
              {expanded && <ul className="tree-list tree-leaves">
                {part.leaves.map(leaf => <li key={leaf.id}>
                  <button type="button" className={`tree-node tree-leaf tree-${leaf.status}${leaf.key ? ' tree-key' : ''}`} aria-current={selectedId === leaf.id} onClick={() => select(leaf.id)}>
                    <Icon status={leaf.status}/>
                    <span className="tree-node-text"><strong>{leaf.title}{leaf.key && <> <KeyStar/></>}</strong><small>{leaf.result}</small></span>
                  </button>
                  {detailBelow(leaf.id)}
                </li>)}
              </ul>}
            </li>;
          })}
        </ul>
      </div>
      {wide && <aside className="tree-side" aria-label="Summary of the selected box"><Detail node={node}/></aside>}
    </div>
    <p className="tree-foot">The tree shows 11 of the project’s main parts, in the order they happened, and {allLeaves.length} tests in them. Colours are the project’s own reading of each result, not a count of wins. A test’s experiment page uses the notebook’s own four outcome labels, which can differ from the colour here.</p>
  </div>;
}
