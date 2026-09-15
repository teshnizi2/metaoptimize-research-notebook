import { Link } from 'react-router-dom';
import { ArrowRight, ArrowUpRight, BookOpen, Braces, FlaskConical, Images, Terminal, Download } from 'lucide-react';
import { useResearch } from '../data';
import { outcomeCounts, outcomeLabels, outcomeOrder } from '../lib/research';
import { ArtifactDates, SnapshotDate } from '../components/ArtifactDates';
import { ExperimentTable, SectionTitle, Status } from '../components/common';
import { CutChart } from '../components/CutChart';

export function Overview() {
  const data = useResearch();
  const counts = outcomeCounts(data.experiments);
  const latest = data.experiments.find(e => e.id === data.latest.experimentId)!;
  const recent = ['CVK2', 'MT166', 'MT165', 'MT163', 'MT155']
    .map(id => data.experiments.find(e => e.id === id)).filter(e => !!e);

  return <div className="overview-notebook">
    <header className="notebook-header">
      <div>
        <h1>Research notebook</h1>
        <SnapshotDate/>
      </div>
      <div className="notebook-actions">
        <div className="notebook-download"><a className="button primary" href={data.meta.downloads.pdf} target="_blank" rel="noreferrer">
          <BookOpen size={17}/>Report PDF<ArrowUpRight size={16}/>
        </a><ArtifactDates kind="download" id="pdf"/><details className="download-date-details"><summary>Report dates</summary><ArtifactDates kind="download" id="pdf" variant="detail"/></details></div>
        <div className="notebook-download"><a className="button" href={data.meta.downloads.bundle} download>
          <Download size={16}/>Download evidence
        </a><ArtifactDates kind="download" id="bundle"/><details className="download-date-details"><summary>Evidence bundle dates</summary><ArtifactDates kind="download" id="bundle" variant="detail"/></details></div>
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

    <section className="notebook-results">
      <SectionTitle title="Recent results" to="/experiments" label={`All ${data.meta.stats.experiments} questions & audits`}/>
      <p className="table-scroll-hint">Scroll the table sideways for outcomes and evidence.</p>
      <div className="panel"><ExperimentTable experiments={recent}/></div>
    </section>

    <div className="overview-grid notebook-latest">
      <section className="panel latest-panel">
        <div className="panel-topline">
          <h2>CVK2 · VGG cut comparison</h2>
          <Status outcome={latest.outcome} label="Prediction open"/>
        </div>
        <p className="chart-subtitle">Seven cuts and two anchors · 27 of 27 runs completed</p>
        <CutChart/>
        <div className="notebook-chart-result">
          <p><strong>Cuts 16, 19 and 22 share the registered peak set.</strong> Best observed mean: cut 19. Predicted cut: 22.</p>
          <nav className="notebook-evidence-links" aria-label="CVK2 evidence">
            {[
              {label: 'Result & scope', tab: 'overview'},
              {label: 'Figures', tab: 'figures'},
              {label: 'Code', tab: 'code'},
              {label: 'Runs & logs', tab: 'runs'},
            ].map(item => <Link className="text-link" key={item.tab} to={`/experiments/${latest.id}?tab=${item.tab}`}>
              {item.label}<ArrowUpRight size={14}/>
            </Link>)}
          </nav>
        </div>
      </section>

      <aside className="panel outcome-panel">
        <div className="panel-topline"><h2>Research outcomes</h2></div>
        <div className="outcome-bar" aria-label="Outcome distribution">
          {outcomeOrder.map(outcome => <div key={outcome} className={`segment ${outcome}`} style={{flex: counts[outcome]}} title={`${outcomeLabels[outcome]}: ${counts[outcome]}`}/>)}
        </div>
        <div className="outcome-counts">
          {outcomeOrder.map(outcome => <Link to={`/experiments?outcome=${outcome}`} key={outcome}>
            <span><i className={`outcome-dot ${outcome}`}/>{outcomeLabels[outcome]}</span><strong>{counts[outcome]}</strong>
          </Link>)}
        </div>
        <p>Outcomes refer to each stated research goal. Completed runs can still leave a prediction open.</p>
        <Link className="text-link" to="/warnings">Warnings & limits<ArrowRight size={16}/></Link>
      </aside>
    </div>

    <section className="notebook-areas">
      <SectionTitle title="Research areas" aside={<span className="small muted">{data.areas.length} areas</span>}/>
      <nav className="area-links" aria-label="Research areas">
        {data.areas.map(area => <Link key={area.id} to={`/experiments?area=${encodeURIComponent(area.label)}`}>
          <strong>{area.label}</strong><span>{area.count}</span><ArrowUpRight size={16}/>
        </Link>)}
      </nav>
    </section>
    <p className="notebook-scope">The run ledger retains reruns, incomplete runs and superseded history. Read each result with its linked scope and corrections.</p>
  </div>;
}
