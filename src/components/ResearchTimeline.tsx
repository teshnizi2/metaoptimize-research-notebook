import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { useResearch } from '../data';
import { researchTimeline, filterByResearchPhase } from '../lib/timeline';
import { outcomeLabels, outcomeOrder } from '../lib/research';

export function ResearchTimeline() {
  const data = useResearch();
  const phases = researchTimeline(data.activity, data.experiments);
  const unlinked = filterByResearchPhase(data.experiments, data.activity, 'unlinked');

  return <section className="home-research-timeline" aria-labelledby="home-timeline-title" id="research-timeline">
    <div className="section-title">
      <h2 id="home-timeline-title">Research timeline</h2>
      <span className="home-section-meta">Newest first</span>
    </div>
    <div className="home-review-feed" id="research-history" role="region" aria-label="Research phases, newest first"
      aria-describedby="research-history-hint">
      <ul className="home-review-list">{phases.map(phase => <li key={phase.id}>
        <Link className="home-review-row" to={`/activity?q=${encodeURIComponent(phase.id)}&view=phase`}>
          <span className="home-review-period">{phase.period?.replace(/-/g, '–') || 'Window not recorded'}</span>
          <span className="home-review-topic"><strong>{phase.title}</strong><span>{phase.test || 'Recorded research phase'}</span></span>
          <span className="home-review-verdicts"><span className="sr-only">Current question outcomes: </span>
            {outcomeOrder.filter(outcome => phase.counts[outcome] > 0).map(outcome => <span key={outcome} className={`status status-${outcome}`}>
              <span className="status-dot" aria-hidden="true"/>{phase.counts[outcome]} {outcomeLabels[outcome]}
            </span>)}
          </span>
          <ChevronRight size={17} className="home-review-arrow" aria-hidden="true"/>
        </Link>
      </li>)}</ul>
    </div>
    <div className="home-review-footer">
      <span id="research-history-hint">{phases.length} documented research periods</span>
      {unlinked.length > 0 && <Link className="text-link" to="/experiments?phase=unlinked">{unlinked.length} records without a phase link</Link>}
    </div>
  </section>;
}
