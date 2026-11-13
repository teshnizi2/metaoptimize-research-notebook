import { useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { PageHeading } from '../components/common';
import { IdeaTree } from '../components/IdeaTree';
import { IdeaGraph } from '../components/IdeaGraph';
import { IdeaChanged } from '../components/IdeaChanged';
import { testPath, type TestLink } from '../components/IdeaTreeDetail';
import { useResearch } from '../data';
import { useIdeas } from '../lib/future-ideas';
import { viewOf, withNode, withView, type PageView, type TreeView } from '../lib/idea-tree-url';

const siteLink: TestLink = (id, className, children) => <Link className={className} to={`/${testPath(id)}`}>{children}</Link>;

export function IdeaTreePage() {
  const [params, setParams] = useSearchParams();
  const research = useResearch();
  // Future ideas come from the private route after sign-in; an empty or missing list changes nothing on the page.
  const { ideas, ready } = useIdeas();
  const [showIdeas, setShowIdeas] = useState(true);
  const pageIds = useMemo(() => new Set(research.experiments.map(e => e.id)), [research]);
  // Radial is the default; a phone is too narrow to read it at first sight, so it starts left to right.
  const [phone] = useState(() => typeof window !== 'undefined' && window.matchMedia('(max-width: 700px)').matches);
  const view: PageView = viewOf(params, phone);
  const list = view === 'list';
  // The tree view the reader was last in, so a chip in "What we changed" opens its test there.
  const lastTree = useRef<TreeView>(phone ? 'tree' : 'radial');
  if (view !== 'changed') lastTree.current = view;
  const asked = params.get('node');
  // A shared link to a future idea waits for the ideas to arrive, so the box can open.
  const waiting = !ready && !!asked && !/^B\d/.test(asked) && asked !== 'root';
  const setView = (next: PageView) => setParams(withView(params, next), { replace: true });
  const onSelectChange = (id: string | null) => setParams(withNode(params, id), { replace: true });
  const modes = <div className="ig-modes" role="group" aria-label="View">
    <button type="button" aria-pressed={view === 'radial'} onClick={() => setView('radial')}>Radial</button>
    <button type="button" aria-pressed={view === 'tree'} onClick={() => setView('tree')}>Left to right</button>
    <button type="button" aria-pressed={list} onClick={() => setView('list')}>List</button>
    <button type="button" aria-pressed={view === 'changed'} onClick={() => setView('changed')}>What we changed</button>
  </div>;
  if (view === 'changed') {
    // A chip opens its test in the tree view the reader came from (the default if there was none).
    const open = (id: string) => {
      setParams(withNode(withView(params, lastTree.current), id));
      window.scrollTo(0, 0);
    };
    return <div className="ig-page-list">
      <div className="ig-titlebar"><h1>Idea tree</h1>{modes}</div>
      <IdeaChanged onOpen={open} testLink={siteLink} ideas={ideas} showIdeas={showIdeas} onShowIdeasChange={setShowIdeas}/>
    </div>;
  }
  if (list) {
    return <div className="ig-page-list">
      <PageHeading
        eyebrow="Research map"
        title="Idea tree"
        description="Every main idea the project tested, in the order it happened, coloured by whether it worked. Select a box for a short summary, then open that test’s page."
      />
      {modes}
      {waiting ? null : <IdeaTree ideas={ideas} showIdeas={showIdeas} onShowIdeasChange={setShowIdeas}/>}
    </div>;
  }
  if (waiting) return <div className="ig-page"><div className="ig-titlebar"><h1>Idea tree</h1>{modes}</div></div>;
  return <div className="ig-page">
    <IdeaGraph
      key={view} mode={view}
      header={<div className="ig-titlebar"><h1>Idea tree</h1>{modes}</div>}
      pageIds={pageIds} testLink={siteLink} initialNode={params.get('node')} initialKeys={params.get('keys') === '1'} onSelectChange={onSelectChange} ideas={ideas} showIdeas={showIdeas} onShowIdeasChange={setShowIdeas}/>
  </div>;
}
