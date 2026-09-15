import { Link } from 'react-router-dom';
import { ArrowLeft, ArrowUpRight, Download } from 'lucide-react';
import { useResearch } from '../data';
import { outcomeLabels, outcomeOrder } from '../lib/research';
import type { ResearchPhase } from '../lib/timeline';
import { ArtifactDates } from './ArtifactDates';
import { CopyLink, ExperimentTable } from './common';

// Historical scope notes qualify the recorded observations. Current verdicts
// remain attached to the individual question records below.
const scopeNotes: Record<string, string> = {
  'phase-01': 'A local additive gain does not validate true pooling.',
  'phase-03': 'The positive contrasts do not rescue the failed accuracy-prediction rule.',
  'phase-04': 'The rescaling explanation failed; a local additive gain remains.',
  'phase-05': 'Tuned gap at α₀ = 10⁻³; the other stratum remains at the search edge.',
  'phase-06': 'Separate initial and 250-epoch batches. Required pinning gates remain unmet.',
  'phase-07': 'Different comparisons and datasets; the fairness audit remains open.',
};

export function ResearchPhaseDetail({ phase }: { phase: ResearchPhase }) {
  const data = useResearch();
  const timeline = data.tables.find(table => table.href === '/assets/tables/research_timeline.csv');
  const recordsHref = `/experiments?phase=${encodeURIComponent(phase.id)}`;

  return <article className="research-phase-detail" aria-labelledby="phase-detail-title">
    <nav className="phase-detail-navigation" aria-label="Research phase navigation">
      <Link className="text-link" to="/"><ArrowLeft size={17} aria-hidden="true"/>Overview</Link>
      <Link className="text-link" to="/activity">Full research log<ArrowUpRight size={17} aria-hidden="true"/></Link>
    </nav>

    <header className="phase-detail-header">
      <div>
        <p className="phase-detail-period"><strong>{phase.period?.replace(/-/g, '–') || 'Window not recorded'}</strong><span className="mono">{phase.id}</span></p>
        <h1 id="phase-detail-title">{phase.title}</h1>
        <p className="phase-detail-context">Documented research period; phases can overlap. Individual run dates are in the linked records.</p>
      </div>
      <CopyLink/>
    </header>

    <dl className="phase-detail-record">
      <div>
        <dt>What we tested</dt>
        <dd>{phase.test || 'Test not recorded'}</dd>
      </div>
      <div>
        <dt>Recorded observation</dt>
        <dd>
          <p>{phase.observation || 'Observation not recorded'}</p>
          {scopeNotes[phase.id] && <p className="phase-detail-scope"><strong>Scope:</strong> {scopeNotes[phase.id]}</p>}
        </dd>
      </div>
      <div>
        <dt>Next step at the time</dt>
        <dd>{phase.nextQuestion || 'Next question not recorded'}</dd>
      </div>
    </dl>

    <section className="phase-detail-questions" aria-labelledby="phase-detail-questions-title">
      <div className="phase-detail-section-heading">
        <h2 id="phase-detail-questions-title">Current linked question verdicts</h2>
        <Link className="text-link" to={recordsHref}>{phase.experiments.length} linked {phase.experiments.length === 1 ? 'record' : 'records'}<ArrowUpRight size={17} aria-hidden="true"/></Link>
      </div>
      <p className="phase-detail-context">Verdicts apply to each question’s goal. There is no single verdict for the whole phase. Method checks are listed but not counted as outcomes.</p>
      {phase.experiments.length > 0 ? <>
        <nav className="phase-detail-outcomes" aria-label="Filter linked questions by current verdict">
          {outcomeOrder.filter(outcome => phase.counts[outcome] > 0).map(outcome => <Link key={outcome}
            className={`status status-${outcome}`} to={`${recordsHref}&outcome=${outcome}`}
            aria-label={`${phase.counts[outcome]} linked questions: ${outcomeLabels[outcome]}`}>
            <span className="status-dot" aria-hidden="true"/>{phase.counts[outcome]} {outcomeLabels[outcome]}
          </Link>)}
          {phase.methodChecks > 0 && <Link className="status status-method-check" to={`${recordsHref}&kind=method-check`}
            aria-label={`${phase.methodChecks} linked method ${phase.methodChecks === 1 ? 'check' : 'checks'}`}>
            <span className="status-dot" aria-hidden="true"/>{phase.methodChecks} method {phase.methodChecks === 1 ? 'check' : 'checks'}
          </Link>}
        </nav>
        <p className="phase-detail-scroll-hint" id="phase-detail-scroll-hint">Scroll sideways to see outcomes and evidence links.</p>
        <div className="phase-detail-table-region" role="region" tabIndex={0} aria-label="Linked questions, results and evidence" aria-describedby="phase-detail-scroll-hint">
          <ExperimentTable experiments={phase.experiments}/>
        </div>
      </> : <p className="phase-detail-empty">No question associations are recorded for this phase.</p>}
    </section>

    <section className="phase-detail-source" aria-labelledby="phase-detail-source-title">
      <div className="phase-detail-section-heading">
        <h2 id="phase-detail-source-title">Original timeline record</h2>
        {timeline && <a className="text-link" href={timeline.href} download><Download size={17} aria-hidden="true"/>Timeline CSV</a>}
      </div>
      {timeline ? <details className="phase-detail-dates">
        <summary>Timeline table dates and provenance</summary>
        <ArtifactDates kind="table" id={timeline.id} variant="detail"/>
      </details> : <p className="phase-detail-context">The original timeline CSV is not included in this snapshot.</p>}
    </section>
  </article>;
}
