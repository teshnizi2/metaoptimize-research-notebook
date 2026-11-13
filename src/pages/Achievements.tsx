import { Link } from 'react-router-dom';
import { ArrowRight, ArrowUpRight } from 'lucide-react';
import { PageHeading } from '../components/common';
import { KeyLegendItem, KeyStar, TreeIcon } from '../components/IdeaTreeDetail';
import { achievementStatus, achievements, isKeyAchievement, groups, notebookCard, type Achievement, type Fact } from '../lib/achievements';
import { findNode, ideaTree, statusInfo } from '../lib/idea-tree';
import { useResearch } from '../data';
import '../components/IdeaTree.css';
import '../components/Achievements.css';

function Chip({ id }: { id: string }) {
  const node = findNode(ideaTree, id);
  if (!node || node.kind !== 'leaf') return null;
  const { leaf } = node;
  const mt = leaf.experiments[0];
  return <span className={`ach-chip tree-${leaf.status}${leaf.key ? ' tree-key' : ''}`} title={statusInfo[leaf.status].name}>
    <Link to={`/idea-tree?view=radial&node=${id}`}><TreeIcon status={leaf.status}/><span><strong>{id}</strong> · {leaf.title}{leaf.key && <> <KeyStar/></>}</span></Link>
    {mt && <Link to={`/experiments/${mt}`} className="ach-chip-page">Test page <ArrowUpRight size={14}/></Link>}
  </span>;
}

function FactLine({ fact }: { fact: Fact }) {
  return <li>{fact.label}</li>;
}

function Card({ card }: { card: Achievement }) {
  const status = achievementStatus(card);
  const judged = card.group === 'stand' || card.group === 'negative';
  const [key, ...rest] = card.facts;
  const isKey = isKeyAchievement(card);
  return <article className={`ach-card tree-${judged ? status : 'u'}${isKey ? ' tree-key' : ''}`} id={card.id}>
    <header>
      {judged
        ? <span className="tree-verdict"><TreeIcon status={status}/>{card.group === 'stand' ? 'Stands' : statusInfo[status].name}</span>
        : <span className="ach-tag">{card.group === 'checked' ? 'Checked' : 'Built'}</span>}
      {isKey && <KeyStar/>}
    </header>
    <h3>{card.claim}</h3>
    <p className="ach-key">{key.label}</p>
    {rest.length > 0 && <ul className="ach-facts">{rest.map(f => <FactLine key={f.label} fact={f}/>)}</ul>}
    <p className="ach-holds"><strong>Holds when:</strong> {card.holds}</p>
    <div className="ach-tests" aria-label="Tests behind this">
      {card.tests.map(id => <Chip key={id} id={id}/>)}
      {(card.limits ?? []).map(id => <Chip key={id} id={id}/>)}
      {(card.parts ?? []).map(id => {
        const node = findNode(ideaTree, id);
        return node?.kind === 'part' ? <span key={id} className={`ach-chip tree-${node.part.status}`}>
          <Link to={`/idea-tree?view=radial&node=${id}`}><TreeIcon status={node.part.status}/><span><strong>Part {node.part.number}</strong> · {node.part.title}</span></Link>
        </span> : null;
      })}
    </div>
    {card.link && <Link className="text-link" to={card.link.to}>{card.link.label}<ArrowRight size={16}/></Link>}
  </article>;
}

export function AchievementsPage() {
  const { meta } = useResearch();
  const stats = meta.stats;
  return <div className="ach">
    <PageHeading
      eyebrow="Results"
      title="What we achieved"
      description="These are results the project can stand behind, each with the condition it holds under. For what didn’t work, see the Idea tree."
    />
    <ul className="ach-legend" aria-label="Legend"><KeyLegendItem/></ul>
    <p className="ach-links"><Link className="text-link" to="/idea-tree">Idea tree<ArrowRight size={16}/></Link><Link className="text-link" to="/main-idea">The main idea<ArrowRight size={16}/></Link></p>
    {groups.map(group => {
      const cards = achievements.filter(c => c.group === group.id);
      return <section className="ach-section" key={group.id} aria-labelledby={`ach-${group.id}`}>
        <h2 id={`ach-${group.id}`}>{group.title}</h2>
        <p className="ach-intro">{group.intro}</p>
        <div className="ach-grid">
          {cards.map(c => <Card key={c.id} card={c}/>)}
          {group.id === 'built' && <article className="ach-card tree-u" id={notebookCard.id}>
            <header><span className="ach-tag">Built</span></header>
            <h3>{notebookCard.claim}</h3>
            <p className="ach-key">{stats.runs.toLocaleString('en-US')} runs in the ledger</p>
            <ul className="ach-facts">
              <li>{stats.experiments} experiment pages</li>
              <li>{stats.figures} figures and tables</li>
              <li>This notebook, which holds them together</li>
            </ul>
            <p className="ach-holds"><strong>Holds when:</strong> {notebookCard.holdsText}</p>
            <div className="ach-tests">{notebookCard.links.map(l => <Link key={l.to} className="ach-chip-link" to={l.to}>{l.label}<ArrowUpRight size={14}/></Link>)}</div>
          </article>}
        </div>
      </section>;
    })}
  </div>;
}
