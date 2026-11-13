import { useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { PageHeading } from '../components/common';
import { IdeaTree } from '../components/IdeaTree';
import { IdeaGraph } from '../components/IdeaGraph';
import type { TestLink } from '../components/IdeaTreeDetail';
import { useResearch } from '../data';

const siteLink: TestLink = (id, className, children) => <Link className={className} to={`/experiments/${id}`}>{children}</Link>;

export function IdeaTreePage() {
  const [params, setParams] = useSearchParams();
  const research = useResearch();
  const pageIds = useMemo(() => new Set(research.experiments.map(e => e.id)), [research]);
  const list = params.get('view') === 'list';
  const setView = (view: 'graph' | 'list') => {
    const next = new URLSearchParams(params);
    if (view === 'list') next.set('view', 'list'); else next.delete('view');
    setParams(next, { replace: true });
  };
  const onSelectChange = (id: string | null) => {
    const next = new URLSearchParams(params);
    if (id) next.set('node', id); else next.delete('node');
    setParams(next, { replace: true });
  };
  return <>
    <PageHeading
      eyebrow="Research map"
      title="Idea tree"
      description="The project’s question on the left, and every main idea it tested, coloured by whether it worked. Select a box to open what comes next and to read a short summary."
    />
    <div className="ig-modes" role="group" aria-label="View">
      <button type="button" aria-pressed={!list} onClick={() => setView('graph')}>Tree graph</button>
      <button type="button" aria-pressed={list} onClick={() => setView('list')}>List view</button>
    </div>
    {list
      ? <IdeaTree/>
      : <IdeaGraph pageIds={pageIds} testLink={siteLink} initialNode={params.get('node')} onSelectChange={onSelectChange}/>}
  </>;
}
