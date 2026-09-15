import { useArtifactDates } from '../artifact-dates';
import { ageLabel, formatDateFact, safeDateHref } from '../lib/artifact-dates';
import type { ArtifactKind, DateFact } from '../lib/artifact-dates';

const fields = [
  ['created', 'Creation'], ['firstRecorded', 'First recorded'], ['updated', 'Updated'],
  ['exported', 'Exported'], ['submitted', 'Submitted'], ['started', 'Started'], ['finished', 'Finished'],
] as const;
type FactField = typeof fields[number][0];
const compactOrder: Record<ArtifactKind, readonly FactField[]> = {
  figure: ['created', 'firstRecorded', 'exported', 'updated'],
  table: ['created', 'firstRecorded', 'exported', 'updated'],
  source: ['updated', 'firstRecorded', 'created', 'exported'],
  experiment: ['updated', 'firstRecorded', 'created', 'exported'],
  run: ['finished', 'started', 'submitted', 'firstRecorded'],
  warning: ['updated', 'firstRecorded', 'created', 'exported'],
  chart: ['firstRecorded', 'created', 'exported', 'updated'],
  download: ['exported', 'firstRecorded', 'created', 'updated'],
};

function DateValue({ fact }: { fact?: DateFact | null }) {
  const value = formatDateFact(fact);
  return <strong>{fact && value !== 'Not recorded' ? <time dateTime={fact.at}>{value}</time> : value}</strong>;
}

function DateRow({ label, fact }: { label: string; fact?: DateFact | null }) {
  const age = ageLabel(fact);
  return <div className="artifact-date-row" title={fact ? `${fact.basis} · ${fact.at}` : undefined}>
    <dt>{label}</dt>
    <dd><DateValue fact={fact}/>{age && <span className="artifact-date-age">{age}</span>}</dd>
  </div>;
}

export function SnapshotDate() {
  const { catalog, loading, error } = useArtifactDates('download', 'pdf');
  if (loading) return <div className="artifact-dates-state" aria-busy="true">Snapshot date loading…</div>;
  if (error || !catalog) return <div className="artifact-dates-state artifact-dates-error">Snapshot date unavailable</div>;
  const published: DateFact = {
    at: catalog.publishedAt,
    precision: catalog.publishedAt.length === 10 ? 'day' : 'second',
    basis: 'Publication of this notebook snapshot; not the creation of its research evidence.',
  };
  return <div className="artifact-dates artifact-dates-snapshot"><dl><DateRow label="Snapshot published" fact={published}/></dl></div>;
}

export function ArtifactDates({ kind, id, variant = 'compact' }: {
  kind: ArtifactKind; id: string; variant?: 'compact' | 'detail';
}) {
  const { record, loading, error } = useArtifactDates(kind, id);
  if (loading) return <div className="artifact-dates-state" aria-busy="true">Dates loading…</div>;
  if (error || !record) return <div className="artifact-dates-state artifact-dates-error" title={error || 'No date metadata is recorded for this item.'}>Date metadata unavailable</div>;

  const selected = compactOrder[kind].find(field => record[field]);
  const visible = variant === 'detail'
    ? fields.filter(([field]) => field === 'created' || record[field])
    : fields.filter(([field]) => field === selected);
  const provenance = fields.filter(([field]) => record[field]);
  const window = record.runWindow;
  const lastFinishAge = ageLabel(window?.finished);

  return <div className={`artifact-dates artifact-dates-${variant}`}>
    <dl aria-label={`Dates for ${id}`}>
      {visible.length ? visible.map(([field, label]) => <DateRow key={field} label={variant === 'compact' && field === 'created' ? 'Created' : label} fact={record[field]}/>) : <DateRow label="Date"/>}
      {variant === 'compact' && window && ['experiment', 'figure', 'table', 'chart'].includes(kind) &&
        <DateRow label="Last linked run" fact={{ ...window.finished, basis: `Latest completion among linked runs. ${window.finished.basis}` }}/>}
      {variant === 'detail' && window && <div className="artifact-date-row artifact-date-window">
        <dt>Linked run window</dt>
        <dd><span className="artifact-date-range"><DateValue fact={window.started}/><span aria-hidden="true">→</span><span className="sr-only">to</span><DateValue fact={window.finished}/></span><span className="artifact-date-coverage">{window.datedRuns.toLocaleString()} of {window.totalRuns.toLocaleString()} linked runs dated</span>{lastFinishAge && <span className="artifact-date-age">Last finish: {lastFinishAge}</span>}</dd>
      </div>}
    </dl>
    {variant === 'detail' && <div className="artifact-date-note">Timestamp dates shown in UTC</div>}
    {variant === 'detail' && (provenance.length > 0 || window) && <details className="artifact-date-provenance">
      <summary>Date provenance</summary>
      <ul>{provenance.map(([field, label]) => {
        const fact = record[field]!, href = safeDateHref(fact.href);
        return <li key={field}><strong>{label}:</strong> <time dateTime={fact.at}>{fact.at}</time> · {fact.basis}{href && <> <a href={href} target="_blank" rel="noopener noreferrer">Source<span className="sr-only"> for {label.toLowerCase()}</span></a></>}</li>;
      })}{window && (['started', 'finished'] as const).map(field => {
        const fact = window[field], href = safeDateHref(fact.href);
        return <li key={`window-${field}`}><strong>Linked runs {field}:</strong> <time dateTime={fact.at}>{fact.at}</time> · {fact.basis}{href && <> <a href={href} target="_blank" rel="noopener noreferrer">Source<span className="sr-only"> for linked runs {field}</span></a></>}</li>;
      })}</ul>
    </details>}
  </div>;
}
