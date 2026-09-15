import { Link } from 'react-router-dom';
import { ArrowUpRight, BookOpen, Braces, FlaskConical, Images, Terminal, Download } from 'lucide-react';
import { useResearch } from '../data';
import { compactDate, outcomeCounts, outcomeLabels, outcomeOrder } from '../lib/research';
import { areaOutcomeCounts, recentNotebookUpdates } from '../lib/overview';
import { ArtifactDates, SnapshotDate } from '../components/ArtifactDates';
import { SectionTitle } from '../components/common';

export function Overview() {
  const data = useResearch();
  const counts = outcomeCounts(data.experiments);
  const areas = areaOutcomeCounts(data.experiments, data.areas);
  const updates = recentNotebookUpdates(data.activity, 2);

  return <div className="overview-notebook">
    <header className="notebook-header">
      <div>
        <h1>Research notebook</h1>
        <SnapshotDate/>
      </div>
      <div className="notebook-actions">
        <div className="notebook-download-buttons"><a className="button primary" href={data.meta.downloads.pdf} target="_blank" rel="noreferrer">
          <BookOpen size={17}/>Report PDF<ArrowUpRight size={16}/>
        </a><a className="button" href={data.meta.downloads.bundle} download>
          <Download size={16}/>Download evidence
        </a></div>
        <details className="home-download-dates"><summary>Download dates</summary>
          <strong>Report PDF</strong><ArtifactDates kind="download" id="pdf" variant="detail"/>
          <strong>Evidence bundle</strong><ArtifactDates kind="download" id="bundle" variant="detail"/>
        </details>
      </div>
    </header>

    <div className="snapshot-stats" aria-label="Snapshot contents">
      {[
        {value: data.meta.stats.experiments, label: 'questions & audits', icon: FlaskConical, to: '/experiments'},
        {value: data.meta.stats.runs.toLocaleString(), label: 'run logs', icon: Terminal, to: '/runs'},
        {value: data.meta.stats.figures, label: 'report pages', icon: Images, to: '/figures'},
        {value: data.sources.length, label: 'source files', icon: Braces, to: '/code'},
      ].map(item => <Link to={item.to} key={item.label}>
        <item.icon size={17}/><strong>{item.value}</strong><span>{item.label}</span>
      </Link>)}
    </div>

    <section className="home-outcomes" aria-labelledby="home-outcomes-title">
      <div className="section-title">
        <h2 id="home-outcomes-title">Research outcomes</h2>
        <Link className="text-link" to="/experiments">All {data.experiments.length} questions & audits<ArrowUpRight size={16}/></Link>
      </div>
      <p className="home-section-note">Outcomes reflect each stated research goal or audit.</p>
      <div className="home-outcome-bar" role="img" aria-label={outcomeOrder.map(outcome => `${outcomeLabels[outcome]}: ${counts[outcome]}`).join(', ')}>
        {outcomeOrder.map(outcome => <span key={outcome} className={`segment ${outcome}`} style={{flexGrow: counts[outcome]}} hidden={counts[outcome] === 0} aria-hidden="true"/>)}
      </div>
      <nav className="home-outcome-counts" aria-label="Research outcome filters">
        {outcomeOrder.map(outcome => <Link to={`/experiments?outcome=${outcome}`} key={outcome}>
          <span><i className={`outcome-dot ${outcome}`} aria-hidden="true"/>{outcomeLabels[outcome]}</span>
          <strong>{counts[outcome]}</strong>
        </Link>)}
      </nav>
    </section>

    <section className="home-area-review">
      <SectionTitle title="Research areas" aside={<span className="home-section-meta">{areas.length} areas</span>}/>
      <p className="home-table-scroll-hint" id="home-area-scroll-hint">Scroll the table sideways for all five outcomes.</p>
      <div className="home-area-table-wrap" tabIndex={0} role="region" aria-label="Research area outcome counts" aria-describedby="home-area-scroll-hint">
        <table className="home-area-table">
          <caption className="sr-only">Questions and audits by research area and outcome</caption>
          <thead><tr><th scope="col">Research area</th><th scope="col">Total</th>{outcomeOrder.map(outcome => <th scope="col" key={outcome}><span><i className={`outcome-dot ${outcome}`} aria-hidden="true"/>{outcomeLabels[outcome]}</span></th>)}</tr></thead>
          <tbody>{areas.map(area => <tr key={area.label}>
            <th scope="row"><Link to={`/experiments?area=${encodeURIComponent(area.label)}`}>{area.label || 'Area not recorded'}</Link></th>
            <td><Link className="home-area-total" to={`/experiments?area=${encodeURIComponent(area.label)}`} aria-label={`${area.label}: all ${area.total} questions and audits`}>{area.total}</Link></td>
            {outcomeOrder.map(outcome => <td key={outcome}><Link className={`home-area-count ${area.counts[outcome] ? `status-${outcome}` : 'is-zero'}`} to={`/experiments?area=${encodeURIComponent(area.label)}&outcome=${outcome}`} aria-label={`${area.label}: ${area.counts[outcome]} ${outcomeLabels[outcome]}`}>{area.counts[outcome]}</Link></td>)}
          </tr>)}</tbody>
        </table>
      </div>
    </section>

    <section className="home-updates">
      <SectionTitle title="Notebook updates" to="/activity" label="All updates"/>
      {updates.length ? <ul className="home-update-list">{updates.map(event => <li key={event.id}>
        <Link to={`/activity?q=${encodeURIComponent(event.id)}`}>
          <div className="home-update-meta"><time dateTime={event.date}>{compactDate(event.date)}</time><span>{event.kind}</span></div>
          <strong>{event.title}</strong><ArrowUpRight size={17}/>
        </Link>
      </li>)}</ul> : <p className="home-section-note">No dated notebook updates are recorded.</p>}
    </section>
  </div>;
}
