import { Link } from 'react-router-dom';
import { ArrowUpRight, Download } from 'lucide-react';
import { useResearch } from '../data';
import { researchTimeline, filterByResearchPhase } from '../lib/timeline';
import { outcomeLabels, outcomeOrder } from '../lib/research';
import { ArtifactDates } from './ArtifactDates';
import { Status } from './common';

// Short reading notes for the documented historical phases. Current verdicts
// and individual goals below are always read from the linked experiment records.
const scopeNotes: Record<string, string> = {
  'phase-01': 'A local additive gain does not validate true pooling.',
  'phase-03': 'The positive contrasts do not rescue the failed accuracy-prediction rule.',
  'phase-04': 'The rescaling explanation failed; a local additive gain remains.',
  'phase-05': 'Tuned gap at α₀ = 10⁻³; the other stratum remains at the search edge.',
  'phase-06': 'Separate initial and 250-epoch batches. Required pinning gates remain unmet.',
  'phase-07': 'Different comparisons and datasets; the fairness audit remains open.',
};

export function ResearchTimeline() {
  const data = useResearch();
  const phases = researchTimeline(data.activity, data.experiments);
  const unlinked = filterByResearchPhase(data.experiments, data.activity, 'unlinked');
  const linkedCount = new Set(phases.flatMap(phase => phase.experiments.map(experiment => experiment.id))).size;
  const asset = data.tables.find(table => table.href === '/assets/tables/research_timeline.csv');

  return <section className="home-research-timeline" aria-labelledby="home-timeline-title" id="research-timeline">
    <div className="section-title">
      <h2 id="home-timeline-title">Research timeline</h2>
      {asset && <a className="text-link" href={asset.href} download><Download size={16}/>Timeline CSV</a>}
    </div>
    <div className="home-timeline-intro">
      <p className="home-section-note"><strong>{phases.length} documented phases</strong> · Overlapping research periods, not individual run dates.</p>
      {asset && <details className="home-timeline-dates"><summary>Table dates</summary><ArtifactDates kind="table" id={asset.id} variant="detail"/></details>}
    </div>
    <p className="home-table-scroll-hint" id="home-timeline-scroll-hint">Scroll sideways to see the results and next questions.</p>
    <div className="home-timeline-table-wrap" tabIndex={0} role="region" aria-label="Research timeline table" aria-describedby="home-timeline-scroll-hint">
      <table className="home-timeline-table">
        <caption className="sr-only">What we tested in each documented research period. Outcome counts refer to the current verdicts of linked questions, not one verdict for a whole phase.</caption>
        <colgroup><col className="phase-column"/><col className="test-column"/><col className="result-column"/><col className="next-column"/></colgroup>
        <thead><tr><th scope="col">Period / phase</th><th scope="col">What we tested</th><th scope="col">Observation / current verdicts</th><th scope="col">Next step at the time</th></tr></thead>
        <tbody>{phases.map(phase => <tr key={phase.id}>
          <th scope="row">
            <span className="home-phase-period">{phase.period?.replace(/-/g, '–') || 'Window not recorded'}</span>
            <strong className="home-phase-title">{phase.title}</strong>
            <Link className="text-link" to={`/activity?q=${encodeURIComponent(phase.id)}`} aria-label={`${phase.title}: phase log`}>Phase log<ArrowUpRight size={14}/></Link>
          </th>
          <td>
            <strong className="home-phase-test">{phase.test || 'See the phase log'}</strong>
            {phase.experiments.length > 0 && <details className="home-phase-goals">
              <summary>{phase.experiments.length} {phase.experiments.length === 1 ? 'question & goal' : 'questions & goals'}</summary>
              <ul>{phase.experiments.map(experiment => <li key={experiment.id}>
                <Link to={`/experiments/${experiment.id}`}><span className="mono">{experiment.id}</span><strong>{experiment.goal}</strong></Link>
                <Status outcome={experiment.outcome}/>
              </li>)}</ul>
            </details>}
          </td>
          <td>
            <strong className="home-phase-observation">{phase.observation || 'Observation not recorded'}</strong>
            {scopeNotes[phase.id] && <p className="home-phase-scope">{scopeNotes[phase.id]}</p>}
            {phase.experiments.length > 0 && <nav className="home-phase-outcomes" aria-label={`${phase.title}: current question verdicts`}>
              {outcomeOrder.filter(outcome => phase.counts[outcome] > 0).map(outcome => <Link key={outcome} className={`status status-${outcome}`} to={`/experiments?phase=${encodeURIComponent(phase.id)}&outcome=${outcome}`} aria-label={`${phase.title}: ${phase.counts[outcome]} ${outcomeLabels[outcome]}`}>
                <span className="status-dot" aria-hidden="true"/>{phase.counts[outcome]} {outcomeLabels[outcome]}
              </Link>)}
            </nav>}
          </td>
          <td>
            <p className="home-phase-next">{phase.nextQuestion || 'Next question not recorded'}</p>
            <Link className="text-link home-phase-records" to={`/experiments?phase=${encodeURIComponent(phase.id)}`} aria-label={`${phase.title}: all ${phase.experiments.length} linked ${phase.experiments.length === 1 ? 'record' : 'records'}`}>Open records<ArrowUpRight size={14}/></Link>
          </td>
        </tr>)}</tbody>
      </table>
    </div>
    <div className="home-timeline-coverage">
      <span><strong>{linkedCount}</strong> unique linked records · Shared records can appear in more than one phase.</span>
      <Link className="text-link" to="/experiments?phase=unlinked">{unlinked.length} other records · no phase link<ArrowUpRight size={16}/></Link>
    </div>
  </section>;
}
