import type { ReactNode } from 'react';
import { Lightbulb } from 'lucide-react';
import { findNode, type IdeaTreeData } from '../lib/idea-tree';
import { IDEA_LEGEND, TEST_ID, ideaParent, kindName, priorityName, type FutureIdea, type IdeaNode } from '../lib/future-ideas';

/** The small 💡 on every future idea. A future idea is never a result, so it carries no verdict icon. */
export function IdeaMark() {
  return <span className="idea-mark" aria-hidden="true">💡</span>;
}

/** Legend entry: dashed, light, never one of the four verdict colours. */
export function IdeaLegendItem({ total }: { total?: number }) {
  return <li className="idea-legend"><IdeaMark/><span>{IDEA_LEGEND}{total !== undefined && <small>{total}</small>}</span></li>;
}

/** "Hide future ideas" / "Show them again". Shows nothing when there are no ideas. */
export function IdeasToggle({ count, shown, onChange }: { count: number; shown: boolean; onChange: (shown: boolean) => void }) {
  if (count === 0) return null;
  return <button type="button" className="button" aria-pressed={!shown} onClick={() => onChange(!shown)}><Lightbulb size={15}/> {shown ? 'Hide future ideas' : 'Bring back future ideas'}</button>;
}

/** Text that mentions test ids (B1.2 …): the ids the tree has become buttons that open that test. */
function withTests(text: string, data: IdeaTreeData, onSelect?: (id: string) => void): ReactNode {
  if (!onSelect) return text;
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TEST_ID)) {
    const at = m.index ?? 0;
    const node = findNode(data, m[0]);
    if (!node || node.kind !== 'leaf') continue;
    out.push(text.slice(last, at));
    out.push(<button type="button" className="idea-testid" key={`${m[0]}-${at}`} onClick={() => onSelect(m[0])}>{m[0]}</button>);
    last = at + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

function parentLabel(idea: FutureIdea, data: IdeaTreeData): string {
  if (typeof idea.attachTo !== 'string') return `A new branch: ${idea.attachTo.newBranch}`;
  const n = findNode(data, ideaParent(idea));
  if (!n || n.kind === 'root') return 'The question';
  return n.kind === 'part' ? `Part ${n.part.number} · ${n.part.title}` : `${n.leaf.id} · ${n.leaf.title}`;
}

/** The summary panel of a future idea (or of a new branch): what is asked, why, how it would be tested, and what it would cost. */
export function IdeaDetail({ node, data, inline, onSelect }: { node: IdeaNode; data: IdeaTreeData; inline?: boolean; onSelect?: (id: string) => void }) {
  const cls = `tree-detail idea-detail${inline ? ' tree-detail-inline' : ''}`;
  if (node.kind === 'branch') {
    return <article className={cls} aria-live="polite">
      <header><span className="idea-pill"><IdeaMark/>Future idea · new branch</span></header>
      <p className="tree-where">A new branch off the question</p>
      <h2>{node.name}</h2>
      <p className="tree-summary">Not run yet. It holds {node.ideas.length} idea{node.ideas.length === 1 ? '' : 's'}:</p>
      <ul className="idea-list">{node.ideas.map(i => <li key={i.id}>{onSelect ? <button type="button" onClick={() => onSelect(i.id)}>{i.id} · {i.title}</button> : `${i.id} · ${i.title}`}</li>)}</ul>
    </article>;
  }
  const { idea } = node;
  const link = (text: string) => withTests(text, data, onSelect);
  return <article className={cls} aria-live="polite">
    <header>
      <span className="idea-pill"><IdeaMark/>Future idea · not run yet</span>
      <span className="idea-tag">{priorityName[idea.priority]}</span>
      <span className="idea-tag" title="How likely it is to work as planned: 5 is sure, 1 is not possible as planned">Feasibility {idea.feasibility} / 5</span>
    </header>
    <p className="tree-where">{idea.id} · {kindName[idea.kind]} · grows from {parentLabel(idea, data)}</p>
    <h2>{idea.title}</h2>
    {idea.question && <p className="tree-headline">{idea.question}</p>}
    <dl className="idea-facts">
      {idea.why && <><dt>Why</dt><dd>{link(idea.why)}</dd></>}
      {idea.test && <><dt>The test</dt><dd>{link(idea.test)}</dd></>}
      {idea.cost && <><dt>Cost <small>(estimate)</small></dt><dd>{idea.cost}</dd></>}
      {idea.outcomes.length > 0 && <><dt>What could come out</dt><dd>{idea.outcomes.length === 1 ? link(idea.outcomes[0]) : <ul>{idea.outcomes.map((o, i) => <li key={i}>{link(o)}</li>)}</ul>}</dd></>}
      {idea.paperValue && <><dt>Value for the paper</dt><dd>{idea.paperValue}</dd></>}
      {idea.risks.length > 0 && <><dt>Risks</dt><dd>{idea.risks.length === 1 ? idea.risks[0] : <ul>{idea.risks.map((r, i) => <li key={i}>{r}</li>)}</ul>}</dd></>}
    </dl>
  </article>;
}
