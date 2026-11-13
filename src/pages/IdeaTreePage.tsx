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
  const modes = <div className="ig-modes" role="group" aria-label="View">
    <button type="button" aria-pressed={!list} onClick={() => setView('graph')}>Tree graph</button>
    <button type="button" aria-pressed={list} onClick={() => setView('list')}>List view</button>
  </div>;
  if (list) {
    return <div className="ig-page-list">
      <PageHeading
        eyebrow="Research map"
        title="Idea tree"
        description="Every main idea the project tested, in the order it happened, coloured by whether it worked. Select a box for a short summary, then open that test’s page."
      />
      {modes}
      <IdeaTree/>
    </div>;
  }
  return <div className="ig-page">
    <IdeaGraph
      header={<div className="ig-titlebar"><h1>Idea tree</h1>{modes}</div>}
      pageIds={pageIds} testLink={siteLink} initialNode={params.get('node')} onSelectChange={onSelectChange}/>
  </div>;
}
