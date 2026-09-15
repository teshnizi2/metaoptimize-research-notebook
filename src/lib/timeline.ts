import type { ActivityEvent, Experiment, Outcome } from '../types';
import { formatDateFact } from './artifact-dates';
import { outcomeCounts } from './research';

export interface ResearchPhase {
  id: string;
  title: string;
  startDate: string | null;
  period: string | null;
  test: string | null;
  observation: string | null;
  nextQuestion: string | null;
  experiments: Experiment[];
  counts: Record<Outcome, number>;
}

const phaseEvents = (activity: ActivityEvent[]) => activity.filter(event => event.kind === 'research-phase');
const linkedIds = (event: ActivityEvent): string[] => Array.isArray(event.experimentIds) ? event.experimentIds.filter((id): id is string => typeof id === 'string') : [];
const clean = (text: string) => text.replace(/\s+/g, ' ').replace(/[.\s]+$/, '').trim();

export function researchTimeline(activity: ActivityEvent[], experiments: Experiment[]): ResearchPhase[] {
  return phaseEvents(activity).map(event => {
    // These labeled fields are part of the published phase record. Never substitute
    // a publication date or a single run's timestamp for a missing phase window.
    const fields = typeof event.detail === 'string' ? event.detail.match(/^Documented phase: (.+?)\. Test: ([\s\S]+?)\. Result: ([\s\S]+?)\. Next question: ([\s\S]+?)\. Date marks\b/) : null;
    const validDate = formatDateFact({ at: event.date, precision: 'day', basis: 'Documented phase start' }) !== 'Not recorded';
    const ids = new Set(linkedIds(event));
    const linked = experiments.filter(experiment => ids.has(experiment.id));
    return {
      id: event.id, title: event.title, startDate: validDate ? event.date : null,
      period: fields ? clean(fields[1]) : null,
      test: fields ? clean(fields[2]) : null,
      observation: fields ? clean(fields[3]) : null,
      nextQuestion: fields ? clean(fields[4]) : null,
      experiments: linked, counts: outcomeCounts(linked),
    };
  }).sort((a, b) => (b.startDate || '').localeCompare(a.startDate || '') || a.id.localeCompare(b.id));
}

export function resolvePhaseDetail(phases: ResearchPhase[], filteredEntries: ActivityEvent[], query: string, view = ''): ResearchPhase | undefined {
  if (view !== 'phase' || !filteredEntries.some(entry => entry.kind === 'research-phase' && entry.id === query)) return undefined;
  return phases.find(phase => phase.id === query);
}

export function filterByResearchPhase(experiments: Experiment[], activity: ActivityEvent[], phaseId = ''): Experiment[] {
  if (!phaseId) return experiments;
  const phases = phaseEvents(activity);
  const ids = new Set(phases.filter(event => phaseId === 'unlinked' || event.id === phaseId).flatMap(linkedIds));
  return experiments.filter(experiment => phaseId === 'unlinked' ? !ids.has(experiment.id) : ids.has(experiment.id));
}
