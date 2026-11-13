import { useMemo, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, ArrowUpRight, Compass, Scale } from 'lucide-react';
import { PageHeading } from '../components/common';
import { KeyStar, TreeIcon } from '../components/IdeaTreeDetail';
import { achievementStatus, achievements, isKeyAchievement, notebookCard, sectionOf, sections, type Achievement } from '../lib/achievements';
import { firstUses, gloss } from '../lib/glossary';
import { findNode, ideaTree, statusInfo } from '../lib/idea-tree';
import { useResearch } from '../data';
import { treeAddress } from '../lib/idea-tree-url';
import '../components/IdeaTree.css';
import '../components/Achievements.css';

type Plan = Map<string, Set<string>>;
const NONE: ReadonlySet<string> = new Set();

/** Text with the first use of each term on the page explained by a tooltip (and a dotted underline). */
function Gloss({ text, plan, at }: { text: string; plan: Plan; at: string }) {
  return <>{gloss(text, plan.get(at) ?? NONE).map((p, i) => p.term
    ? <abbr key={i} className="ach-term" tabIndex={0} title={p.term.meaning}>{p.text}</abbr>
    : <span key={i}>{p.text}</span>)}</>;
}

function TestRow({ id, limit }: { id: string; limit?: boolean }) {
  const node = findNode(ideaTree, id);
  if (!node || node.kind !== 'leaf') return null;
  const { leaf } = node;
  const mt = leaf.experiments[0];
  return <li className={`tree-${leaf.status}`}>
    <span className="ach-test-icon"><TreeIcon status={leaf.status}/><span className="tree-sr">{statusInfo[leaf.status].name}</span></span>
    <Link className="ach-test-name" to={treeAddress(id)}><b>{id}</b> {leaf.title}</Link>
    {leaf.key && <KeyStar/>}
    {limit && <span className="ach-test-limit">sets the limit</span>}
    {mt && <Link to={`/experiments/${mt}`} className="ach-test-page">Test page<ArrowUpRight size={13}/></Link>}
  </li>;
}

function PartRow({ id }: { id: string }) {
  const node = findNode(ideaTree, id);
  if (!node || node.kind !== 'part') return null;
  return <li className={`tree-${node.part.status}`}>
    <span className="ach-test-icon"><TreeIcon status={node.part.status}/><span className="tree-sr">{statusInfo[node.part.status].name}</span></span>
    <Link className="ach-test-name" to={treeAddress(id)}><b>Part {node.part.number}</b> {node.part.title}</Link>
  </li>;
}

const verdictWord = { stand: 'Stands', negative: 'Ruled out' } as const;

function Card({ card, plan }: { card: Achievement; plan: Plan }) {
  const status = achievementStatus(card);
  const judged = card.group === 'stand' || card.group === 'negative';
  const [key, ...rest] = card.facts;
  const total = card.tests.length + (card.limits ?? []).length + (card.parts ?? []).length;
  return <article className={`ach-card tree-${judged ? status : 'u'}`} id={card.id}>
    <h3>
      {judged && <span className="ach-verdict"><TreeIcon status={status}/><span className="tree-sr">{verdictWord[card.group as 'stand' | 'negative']}: </span></span>}
      <span><Gloss text={card.claim} plan={plan} at={`${card.id}:claim`}/>{isKeyAchievement(card) && <> <KeyStar/></>}</span>
    </h3>
    <p className="ach-key"><Gloss text={key.label} plan={plan} at={`${card.id}:key`}/></p>
    <p className="ach-line"><Scale size={17} aria-hidden="true"/><span><span className="ach-label">Holds when</span> <Gloss text={card.holds} plan={plan} at={`${card.id}:holds`}/></span></p>
    {card.caveat && <p className="ach-caveat"><Gloss text={card.caveat} plan={plan} at={`${card.id}:caveat`}/></p>}
    <p className="ach-line"><Compass size={17} aria-hidden="true"/><span><span className="ach-label">Why it matters</span> {card.matters}</span></p>
    <details className="ach-more">
      <summary>Show the {total} test{total === 1 ? '' : 's'}{rest.length > 0 && ' and more numbers'}</summary>
      {rest.length > 0 && <ul className="ach-facts">{rest.map(f => <li key={f.label}>{f.label}</li>)}</ul>}
      <ul className="ach-tests" aria-label="Tests behind this">
        {card.tests.map(id => <TestRow key={id} id={id}/>)}
        {(card.limits ?? []).map(id => <TestRow key={id} id={id} limit/>)}
        {(card.parts ?? []).map(id => <PartRow key={id} id={id}/>)}
      </ul>
    </details>
    {card.link && <Link className="text-link" to={card.link.to}>{card.link.label}<ArrowRight size={16}/></Link>}
  </article>;
}

function Section({ id, title, intro, children }: { id: string; title: string; intro: string; children: ReactNode }) {
  return <section className="ach-section" aria-labelledby={`ach-${id}`}>
    <h2 id={`ach-${id}`}>{title}{id === 'key' && <> <KeyStar/></>}</h2>
    <p className="ach-intro">{intro}</p>
    <div className="ach-cards">{children}</div>
  </section>;
}

export function AchievementsPage() {
  const { meta } = useResearch();
  const stats = meta.stats;
  const bySection = useMemo(() => sections.map(s => ({ ...s, cards: achievements.filter(c => sectionOf(c) === s.id) })), []);
  // Terms are explained at their first use in reading order, not on every card.
  const plan = useMemo<Plan>(() => firstUses(bySection.flatMap(s => s.cards.flatMap(c => [
    { key: `${c.id}:claim`, text: c.claim },
    { key: `${c.id}:key`, text: c.facts[0].label },
    { key: `${c.id}:holds`, text: c.holds },
    { key: `${c.id}:caveat`, text: c.caveat ?? '' },
  ]))), [bySection]);
  const glance = achievements.filter(c => c.glance).sort((a, b) => Number(isKeyAchievement(b)) - Number(isKeyAchievement(a)));
  return <div className="ach">
    <PageHeading
      eyebrow="Results"
      title="What we achieved"
      description="Results the project can stand behind, each with the condition it holds under. For what didn’t work, see the Idea tree."
    />
    <p className="ach-links"><Link className="text-link" to="/idea-tree">Idea tree<ArrowRight size={16}/></Link><Link className="text-link" to="/main-idea">The main idea<ArrowRight size={16}/></Link></p>
    <nav className="ach-glance" aria-labelledby="ach-glance-title">
      <h2 id="ach-glance-title">At a glance</h2>
      <ul>
        {glance.map(c => <li key={c.id}>
          <span className="ach-glance-mark">{isKeyAchievement(c) ? <KeyStar/> : <span aria-hidden="true">·</span>}</span>
          <a href={`#${c.id}`}>{c.glance}</a>
        </li>)}
      </ul>
      <p className="ach-glance-note"><KeyStar/> means a key result for the paper. Points are points of test accuracy.</p>
    </nav>
    {bySection.map(s => <Section key={s.id} id={s.id} title={s.title} intro={s.intro}>
      {s.cards.map(c => <Card key={c.id} card={c} plan={plan}/>)}
      {s.id === 'built' && <article className="ach-card tree-u" id={notebookCard.id}>
        <h3>{notebookCard.claim}</h3>
        <p className="ach-key">{stats.runs.toLocaleString('en-US')} runs in the ledger</p>
        <p className="ach-line"><Scale size={17} aria-hidden="true"/><span><span className="ach-label">Holds when</span> {notebookCard.holdsText}</span></p>
        <ul className="ach-facts ach-facts-open">
          <li>{stats.experiments} experiment pages</li>
          <li>{stats.figures} figures and tables</li>
        </ul>
        <p className="ach-where">{notebookCard.links.map(l => <Link key={l.to} className="text-link" to={l.to}>{l.label}<ArrowUpRight size={14}/></Link>)}</p>
      </article>}
    </Section>)}
  </div>;
}
