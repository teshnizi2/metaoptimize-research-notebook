import { Link } from 'react-router-dom';
import { ArrowRight, CalendarDays } from 'lucide-react';
import { PageHeading } from '../components/common';
import { DataTable, IntervalChart, RichText } from '../components/NotebookParts';
import { PrivateEmpty } from './Weekly';
import { useBetter, type Round, type Tone } from '../lib/private-pages';
import '../components/Notebook.css';

const TONE_WORD: Record<Tone, string> = { hit: 'Worked', miss: 'Missed', mixed: 'Mixed', plain: 'Note' };

function RoundSection({ round }: { round: Round }) {
  return <section className="nb-section nb-round" id={round.id} aria-labelledby={`${round.id}-h`}>
    <h2 id={`${round.id}-h`}>{round.name}</h2>
    {round.dates && <p className="nb-dates"><CalendarDays size={15} aria-hidden="true"/>{round.dates}</p>}
    {round.tried.length > 0 && <>
      <h3>What we tried and why</h3>
      <ul className="nb-list">{round.tried.map((t, i) => <li key={i}><RichText text={t}/></li>)}</ul>
    </>}
    {(round.chart || round.table) && <>
      <h3>The main result</h3>
      {round.chart && <IntervalChart chart={round.chart}/>}
      {round.table && <DataTable table={round.table}/>}
    </>}
    {round.verdicts.length > 0 && <>
      <h3>Verdicts</h3>
      <ul className="nb-verdicts">{round.verdicts.map((v, i) => <li key={i} className={`nb-tone-${v.tone}`}>
        <span className="nb-badge">{TONE_WORD[v.tone]}</span><RichText text={v.text}/>
      </li>)}</ul>
    </>}
    {round.misses.length > 0 && <>
      <h3>Misses and caveats</h3>
      <ul className="nb-list nb-misses">{round.misses.map((t, i) => <li key={i}><RichText text={t}/></li>)}</ul>
    </>}
    {round.reading && <div className="nb-reading"><RichText text={round.reading}/></div>}
  </section>;
}

export function BetterPage() {
  const { data, ready } = useBetter();
  return <div className="nb">
    <PageHeading
      title="Making it better"
      description="What each round of improving MetaOptimize tried and why, what it found, where it missed, and where things stand."
    />
    <p className="nb-links">
      <Link className="text-link" to="/achievements">What we achieved<ArrowRight size={16}/></Link>
      <Link className="text-link" to="/weekly">Weekly record<ArrowRight size={16}/></Link>
    </p>
    {!data ? <PrivateEmpty ready={ready} what="This page"/> : <>
      {data.updated && <p className="nb-updated">Written {data.updated}. Points are points of test accuracy; every number comes from the round’s own record.</p>}
      {data.intro.length > 0 && <section className="nb-section">{data.intro.map((p, i) => <RichText key={i} text={p}/>)}</section>}
      <nav className="nb-glance" aria-label="Rounds">
        <ol>{data.rounds.map(r => <li key={r.id}><a href={`#${r.id}`}>{r.name}</a></li>)}<li><a href="#standing">Where things stand now</a></li></ol>
      </nav>
      {data.words.length > 0 && <details className="nb-words">
        <summary>Words used on this page</summary>
        <dl>{data.words.map(w => <div key={w.term}><dt>{w.term}</dt><dd>{w.meaning}</dd></div>)}</dl>
      </details>}
      {data.rounds.map(r => <RoundSection key={r.id} round={r}/>)}
      {data.standing && <section className="nb-section nb-standing" id="standing" aria-labelledby="standing-h">
        <h2 id="standing-h">{data.standing.headline}</h2>
        <ul className="nb-list">{data.standing.bullets.map((t, i) => <li key={i}><RichText text={t}/></li>)}</ul>
        {data.standing.next.length > 0 && <>
          <h3>Next, not started</h3>
          <ul className="nb-list">{data.standing.next.map((t, i) => <li key={i}><RichText text={t}/></li>)}</ul>
          {data.standing.note && <p className="nb-note">{data.standing.note}</p>}
        </>}
      </section>}
    </>}
  </div>;
}
