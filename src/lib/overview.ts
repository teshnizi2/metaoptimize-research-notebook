import type { ActivityEvent, Experiment, Outcome, ResearchData } from '../types';
import { formatDateFact } from './artifact-dates';
import { outcomeCounts } from './research';

export interface AreaOutcomeSummary {
  label: string;
  total: number;
  counts: Record<Outcome, number>;
}

export function areaOutcomeCounts(experiments: Experiment[], areas: ResearchData['areas']): AreaOutcomeSummary[] {
  const groups = new Map<string, Experiment[]>(areas.map(area => [area.label, []]));
  for (const experiment of experiments) {
    if (!groups.has(experiment.area)) groups.set(experiment.area, []);
    groups.get(experiment.area)!.push(experiment);
  }
  return [...groups].map(([label, entries]) => ({ label, total: entries.length, counts: outcomeCounts(entries) }));
}

export function recentNotebookUpdates(activity: ActivityEvent[], limit = 4): ActivityEvent[] {
  const kinds = new Set(['note', 'correction', 'warning']);
  return activity.filter(entry => kinds.has(entry.kind) && formatDateFact({
    at: entry.date,
    precision: /^\d{4}-\d{2}-\d{2}$/.test(entry.date) ? 'day' : 'second',
    basis: 'Recorded notebook update',
  }) !== 'Not recorded')
    .sort((a, b) => Date.parse(b.date) - Date.parse(a.date) || a.id.localeCompare(b.id))
    .slice(0, Math.max(0, Math.floor(limit)));
}
