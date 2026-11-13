import { useMemo, useState } from 'react';
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
  // Radial is the default; a phone is too narrow to read it at first sight, so it starts left to right.
  const [phone] = useState(() => typeof window !== 'undefined' && window.matchMedia('(max-width: 700px)').matches);
  const asked = params.get('view');
  const view: 'radial' | 'tree' | 'list' = asked === 'list' || asked === 'tree' || asked === 'radial' ? asked : phone ? 'tree' : 'radial';
  const list = view === 'list';
  const setView = (next: 'radial' | 'tree' | 'list') => {
    const query = new URLSearchParams(params);
    query.set('view', next);
    setParams(query, { replace: true });
  };
  const onSelectChange = (id: string | null) => {
    const next = new URLSearchParams(params);
    if (id) next.set('node', id); else next.delete('node');
    setParams(next, { replace: true });
  };
  const modes = <div className="ig-modes" role="group" aria-label="View">
    <button type="button" aria-pressed={view === 'radial'} onClick={() => setView('radial')}>Radial</button>
    <button type="button" aria-pressed={view === 'tree'} onClick={() => setView('tree')}>Left to right</button>
    <button type="button" aria-pressed={list} onClick={() => setView('list')}>List</button>
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
      key={view} mode={view}
      header={<div className="ig-titlebar"><h1>Idea tree</h1>{modes}</div>}
      pageIds={pageIds} testLink={siteLink} initialNode={params.get('node')} onSelectChange={onSelectChange}/>
  </div>;
}
