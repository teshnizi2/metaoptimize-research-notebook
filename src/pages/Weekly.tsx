import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, CalendarDays, MessageSquareQuote } from 'lucide-react';
import { Empty, PageHeading } from '../components/common';
import { ChartImage, DataTable, Inline, RichText } from '../components/NotebookParts';
import { stamp, useWeek, useWeekIndex, WEEK_ID, type WeekItem } from '../lib/private-pages';
import '../components/Notebook.css';

/** The state of a private page when its content is missing: still loading, or not available in this copy of the site. */
export function PrivateEmpty({ ready, what }: { ready: boolean; what: string }) {
  if (!ready) return <p className="nb-loading" role="status">Loading…</p>;
  return <Empty title={`${what} has no content to show`} detail="Its text and numbers are kept out of the public repository. They load for people signed in to the workspace; this copy could not fetch them."/>;
}

export function WeeklyIndexPage() {
  const { weeks, ready } = useWeekIndex();
  return <div className="nb">
    <PageHeading
      eyebrow="Weekly record"
      title="Weekly record"
      description="One page for each week, Friday 06:00 to Friday 06:00 (Amsterdam time): what we did, what was asked and the answers, and the questions that came up with the tests behind them."
    />
    <p className="nb-links"><Link className="text-link" to="/better">Making it better<ArrowRight size={16}/></Link></p>
    {weeks.length === 0 ? <PrivateEmpty ready={ready} what="The weekly record"/> : <ol className="nb-weeks">
      {weeks.map(w => <li key={w.id}>
        <Link to={`/weekly/${w.id}`} className="nb-week-card">
          <strong>{w.title}</strong>
          <span className="nb-week-range"><Inline text={w.range}/></span>
          <span className="nb-week-line"><Inline text={w.oneline}/></span>
          <span className="nb-week-meta">
            <span>{w.questions} questions · {w.tasks} results</span>
            <span className={w.open ? 'nb-open' : ''}>{w.open ? `Open, covers up to ${stamp(w.updated)}` : `Closed ${stamp(w.updated)}`}</span>
          </span>
        </Link>
      </li>)}
    </ol>}
  </div>;
}

function Item({ item, asked }: { item: WeekItem; asked: boolean }) {
  const detail = item.why || item.decisions || item.outcome || item.tables.length > 0 || item.charts.length > 0;
  return <article className="nb-item" id={item.id}>
    <h3>{item.title}</h3>
    {asked && item.said.length > 0 && <div className="nb-said">{item.said.map((s, i) => <blockquote key={i}>
      <MessageSquareQuote size={15} aria-hidden="true"/><p>{s.text}</p>{s.when && <footer>Reza, {s.when}</footer>}
    </blockquote>)}</div>}
    {item.question && <div className="nb-field"><h4>The question</h4><RichText text={item.question}/></div>}
    {item.answer && <div className="nb-field nb-answer"><h4>{asked ? 'Answer' : 'The solution'}</h4><RichText text={item.answer}/></div>}
    {detail && <details className="nb-more">
      <summary>{item.tables.length > 0 || item.charts.length > 0 ? 'The tests, tables and charts' : 'The reasoning and what came of it'}</summary>
      {item.why && <div className="nb-field"><h4>Why, and what else was weighed</h4><RichText text={item.why}/></div>}
      {item.decisions && <div className="nb-field"><h4>Decisions</h4><RichText text={item.decisions}/></div>}
      {item.outcome && <div className="nb-field"><h4>What came of it</h4><RichText text={item.outcome}/></div>}
      {item.tables.map((t, i) => <DataTable key={i} table={t}/>)}
      {item.charts.map(c => <ChartImage key={c.name} name={c.name} caption={c.caption}/>)}
    </details>}
  </article>;
}

export function WeekPage() {
  const { id = '' } = useParams();
  const { data: week, ready } = useWeek(id);
  const valid = WEEK_ID.test(id);
  return <div className="nb">
    <p className="nb-back"><Link className="text-link" to="/weekly"><ArrowLeft size={16}/>All weeks</Link></p>
    {!week ? (valid
      ? <><PageHeading eyebrow="Weekly record" title="Week not available"/><PrivateEmpty ready={ready} what="This week"/></>
      : <Empty title="No such week" detail="Weeks are named by the Friday they start on, like 2026-11-13." action={<Link className="button primary" to="/weekly">See all weeks</Link>}/>)
    : <>
      <PageHeading eyebrow="Weekly record" title={week.title} description={week.oneline}/>
      <p className="nb-range"><CalendarDays size={15} aria-hidden="true"/><Inline text={week.range}/></p>
      {week.open
        ? <p className="nb-openbar" role="note">This week is still open. It covers up to {stamp(week.updated)}; later work is added when the week closes.</p>
        : week.updated && <p className="nb-updated">Closed {stamp(week.updated)}.</p>}
      <nav className="nb-glance" aria-label="On this page"><ol>
        <li><a href="#did">What we did this week</a></li>
        {week.questions.length > 0 && <li><a href="#asked">What was asked, with the answers ({week.questions.length})</a></li>}
        {week.tasks.length > 0 && <li><a href="#along">What came up along the way ({week.tasks.length})</a></li>}
      </ol></nav>
      <section className="nb-section" id="did" aria-labelledby="did-h">
        <h2 id="did-h">What we did this week</h2>
        <ul className="nb-list">{week.summary.map((s, i) => <li key={i}><RichText text={s}/></li>)}</ul>
      </section>
      {week.questions.length > 0 && <section className="nb-section" id="asked" aria-labelledby="asked-h">
        <h2 id="asked-h">The main questions and changes asked for, with the answers</h2>
        {week.questions.map(q => <Item key={q.id} item={q} asked/>)}
      </section>}
      {week.tasks.length > 0 && <section className="nb-section" id="along" aria-labelledby="along-h">
        <h2 id="along-h">What came up along the way: solutions and the tests behind them</h2>
        {week.tasks.map(t => <Item key={t.id} item={t} asked={false}/>)}
      </section>}
    </>}
  </div>;
}
