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
      title="Weekly record"
      description="One page for each week, Friday 09:00 to Friday 09:00 (Amsterdam time): where the project stands and the week's achievements, then each question that set or ruled out a direction for the research or the paper, with the answer, the evidence and the reason. The details fold out."
    />
    <p className="nb-links"><Link className="text-link" to="/better">Making it better<ArrowRight size={16}/></Link></p>
    {weeks.length === 0 ? <PrivateEmpty ready={ready} what="The weekly record"/> : <ol className="nb-weeks">
      {weeks.map(w => <li key={w.id}>
        <Link to={`/weekly/${w.id}`} className="nb-week-card">
          <strong>{w.title}</strong>
          <span className="nb-week-range"><Inline text={w.range}/></span>
          <span className="nb-week-line"><Inline text={w.oneline}/></span>
          <span className="nb-week-meta">
            <span>{w.questions} direction {w.questions === 1 ? 'question' : 'questions'} · {w.achievements} {w.achievements === 1 ? 'achievement' : 'achievements'}</span>
            <span className={w.open ? 'nb-open' : ''}>{w.open ? `Open, covers up to ${stamp(w.updated)}` : `Closed ${stamp(w.updated)}`}</span>
          </span>
        </Link>
      </li>)}
    </ol>}
  </div>;
}

function Field({ label, text, answer }: { label: string; text: string; answer?: boolean }) {
  return text ? <div className={answer ? 'nb-field nb-answer' : 'nb-field'}><h4>{label}</h4><RichText text={text}/></div> : null;
}

/** One direction question, always in this order: the question, the answer, the evidence (charts, then tables), why, then the closed details. */
function Item({ item }: { item: WeekItem }) {
  const hasDetails = Boolean(item.decisions || item.outcome || item.details || item.detailTables.length || item.detailCharts.length);
  return <article className="nb-item" id={item.id}>
    <h3>{item.title}</h3>
    {item.said.length > 0 && <div className="nb-said">{item.said.map((s, i) => <blockquote key={i}>
      <MessageSquareQuote size={15} aria-hidden="true"/><p>{s.text}</p>{s.when && <footer>Reza, {s.when}</footer>}
    </blockquote>)}</div>}
    <Field label="The question" text={item.question}/>
    <Field label="The answer" text={item.answer} answer/>
    {(item.tables.length > 0 || item.charts.length > 0) && <div className="nb-field"><h4>The evidence</h4>
      {item.charts.map(c => <ChartImage key={c.name} name={c.name} caption={c.caption}/>)}
      {item.tables.map((t, i) => <DataTable key={i} table={t}/>)}
    </div>}
    <Field label="Why" text={item.why} answer/>
    {hasDetails && <details className="nb-more">
      <summary>Details</summary>
      <Field label="The options on the table" text={item.decisions}/>
      <Field label="What it changed next" text={item.outcome}/>
      <Field label="More detail" text={item.details}/>
      {item.detailTables.map((t, i) => <DataTable key={i} table={t}/>)}
      {item.detailCharts.map(c => <ChartImage key={c.name} name={c.name} caption={c.caption}/>)}
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
      <section className="nb-section" id="summary" aria-labelledby="summary-h">
        <h2 id="summary-h">Week summary</h2>
        <ul className="nb-list">{week.summary.map((s, i) => <li key={i}><RichText text={s}/></li>)}</ul>
        {week.achievements.length > 0 && <>
          <h3>Achievements this week ({week.achievements.length})</h3>
          <ul className="nb-list">{week.achievements.map((a, i) => <li key={i}><RichText text={a}/></li>)}</ul>
        </>}
      </section>
      <section className="nb-section" id="asked" aria-labelledby="asked-h">
        <h2 id="asked-h">Questions that set the direction ({week.questions.length})</h2>
        {week.questions.length > 0
          ? <>
            <nav className="nb-glance" aria-label="The questions"><ol>
              {week.questions.map(q => <li key={q.id}><a href={`#${q.id}`}>{q.title}</a></li>)}
            </ol></nav>
            {week.questions.map(q => <Item key={q.id} item={q}/>)}
          </>
          : <p>No question this week set or ruled out a direction for the research or the paper.</p>}
      </section>
      {week.tasks.length > 0 && <section className="nb-section" id="along" aria-label="The other work this week">
        <details className="nb-more">
          <summary>The other work this week ({week.tasks.length})</summary>
          <ul className="nb-list">{week.tasks.map(t => <li key={t.id}><Inline text={t.title}/></li>)}</ul>
        </details>
      </section>}
    </>}
  </div>;
}
