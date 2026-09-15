import { useEffect, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft, ArrowUpRight, ChevronLeft, ChevronRight, Download, History, Search, Terminal, TriangleAlert, X } from 'lucide-react';
import { useResearch, useRuns } from '../data';
import { CopyLink, Empty, ExperimentChips, PageHeading, Status } from '../components/common';
import { compactDate, safeHref } from '../lib/research';
import { ArtifactDates } from '../components/ArtifactDates';
import { ResearchPhaseDetail } from '../components/ResearchPhaseDetail';
import { researchTimeline, resolvePhaseDetail } from '../lib/timeline';
import type { Run } from '../types';

type Filters = Record<string, string | undefined>;
const size = 50;
const words = (q = '') => q.toLowerCase().trim().split(/\s+/).filter(Boolean);
const matches = (q: string, values: unknown[]) => words(q).every(term => values.join(' ').toLowerCase().includes(term));
const pretty = (s: string) => s.replace(/[_-]/g, ' ').replace(/\b\w/g, x => x.toUpperCase());
const accuracy = (n: number | null) => n !== null && Number.isFinite(n) ? `${n.toFixed(3)}%` : 'Not reported';
const dateValue = (s: string) => Number.isFinite(Date.parse(s)) ? Date.parse(s) : -Infinity;

export function filterRunLedger(runs: Run[], f: Filters = {}) {
  return runs.filter(r => (!f.status || r.status === f.status) && (!f.model || r.architecture === f.model)
    && (!f.dataset || r.dataset === f.dataset) && (!f.batch || r.batch === f.batch)
    && (!f.experiment || r.experimentIds.includes(f.experiment))
    && matches(f.q || '', [r.id, r.jobId, r.batch, r.seed, r.architecture, r.dataset, r.status,
      ...r.experimentIds, ...Object.entries(r.parameters).flat()]));
}
export function ledgerPage<T>(items: T[], requested: string | number = 1) {
  const pages = Math.max(1, Math.ceil(items.length / size));
  const n = Number(requested), page = Number.isFinite(n) ? Math.max(1, Math.min(pages, Math.floor(n))) : 1;
  return { page, pages, items: items.slice((page - 1) * size, page * size), start: items.length ? (page - 1) * size + 1 : 0, end: Math.min(page * size, items.length) };
}
function useFilters() {
  const [params, setParams] = useSearchParams();
  const set = (name: string, value: string) => {
    const next = new URLSearchParams(params);
    value ? next.set(name, value) : next.delete(name);
    if (name !== 'page') next.delete('page');
    setParams(next, { replace: name === 'q' });
  };
  return { get: (name: string) => params.get(name) || '', set, reset: () => setParams({}) };
}
function Query({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder: string }) {
  return <label className="filter-search"><Search size={18} aria-hidden="true"/><span className="sr-only">{placeholder}</span>
    <input type="search" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}/></label>;
}
function Select({ label, value, values, onChange }: { label: string; value: string; values: { value: string; label: string }[]; onChange: (v: string) => void }) {
  return <label className="filter-field"><span>{label}</span><select value={value} onChange={e => onChange(e.target.value)}>
    <option value="">All {label.toLowerCase()}</option>{value && !values.some(v => v.value === value) && <option value={value}>{value} (not in snapshot)</option>}{values.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
  </select></label>;
}
function Pagination({ page, pages, total, start, end, set }: { page: number; pages: number; total: number; start: number; end: number; set: (p: string) => void }) {
  return <nav className="pagination" aria-label="Pagination"><span aria-live="polite">{start.toLocaleString()}–{end.toLocaleString()} of {total.toLocaleString()}</span>
    <div><button className="button" disabled={page <= 1} onClick={() => set(String(page - 1))}><ChevronLeft size={16}/>Previous</button>
      <span className="mono">Page {page} / {pages}</span><button className="button" disabled={page >= pages} onClick={() => set(String(page + 1))}>Next<ChevronRight size={16}/></button></div></nav>;
}
function Reset({ onClick }: { onClick: () => void }) { return <button className="button" onClick={onClick}><X size={15}/>Reset filters</button>; }
function RunLoading({ error, retry }: { error: string; retry: () => void }) {
  return error ? <Empty title="The run ledger could not be loaded" detail={error} action={<button className="button" onClick={retry}>Try again</button>}/>
    : <div className="empty-state" role="status" aria-busy="true"><Terminal size={28}/><h3>Loading recorded runs…</h3></div>;
}
function ExecutionState({ value }: { value: string }) { return <Status outcome="neutral" label={value || 'Not recorded'}/>; }
function experimentOptions(experiments: { id: string; goal: string }[]) { return experiments.map(e => ({ value: e.id, label: `${e.id} · ${e.goal}` })); }

export function RunsPage() {
  const data = useResearch(), { runs, error, retry } = useRuns(), f = useFilters();
  const filters = { q: f.get('q'), status: f.get('status'), model: f.get('model'), dataset: f.get('dataset'), batch: f.get('batch'), experiment: f.get('experiment') };
  const filtered = useMemo(() => filterRunLedger(runs || [], filters), [runs, ...Object.values(filters)]);
  const page = ledgerPage(filtered, f.get('page') || 1);
  const options = (key: 'status' | 'architecture' | 'dataset' | 'batch') => [...new Set((runs || []).map(r => r[key]).filter(Boolean))].sort().map(v => ({ value: v, label: v }));
  return <><PageHeading eyebrow="Execution evidence" title="Run ledger" description={`${data.meta.stats.runs.toLocaleString()} recorded jobs, with parameters and connections to the questions they inform.`} actions={<CopyLink/>}/>
    <p className="notice"><Terminal size={17}/>Execution states describe jobs. Scientific outcomes belong to the linked experiments. This is a published snapshot.</p>
    {!runs ? <RunLoading error={error} retry={retry}/> : <><div className="filter-bar">
      <Query value={filters.q} onChange={v => f.set('q', v)} placeholder="Search run, batch, model or parameter…"/>
      <Select label="States" value={filters.status} values={options('status')} onChange={v => f.set('status', v)}/>
      <Select label="Models" value={filters.model} values={options('architecture')} onChange={v => f.set('model', v)}/>
      <Select label="Datasets" value={filters.dataset} values={options('dataset')} onChange={v => f.set('dataset', v)}/>
      <Select label="Batches" value={filters.batch} values={options('batch')} onChange={v => f.set('batch', v)}/>
      <Select label="Experiments" value={filters.experiment} values={experimentOptions(data.experiments)} onChange={v => f.set('experiment', v)}/>
      {Object.values(filters).some(Boolean) && <Reset onClick={f.reset}/>}</div>
      <div className="results-summary"><span>{filtered.length.toLocaleString()} matching runs</span><span>Test accuracy: last five epoch records</span></div>
      {filtered.length ? <><div className="table-wrap"><table className="ledger-table"><thead><tr><th>Run / batch</th><th>Model / dataset</th><th>Completed epochs</th><th>Test accuracy</th><th>Execution state</th><th>Evidence</th></tr></thead>
        <tbody>{page.items.map(r => <tr key={r.id}><td><Link className="experiment-title mono" to={`/runs/${encodeURIComponent(r.id)}`}>{r.id}</Link><div className="small muted">{r.batch || 'Batch not recorded'} · Seed {r.seed || '—'}</div><ArtifactDates kind="run" id={r.id}/></td>
          <td><strong>{r.architecture || 'Not recorded'}</strong><div className="small muted">{r.dataset || 'Not recorded'}</div></td><td className="mono">{r.epochs ?? '—'}</td><td className="mono">{accuracy(r.testAccuracy)}</td>
          <td><ExecutionState value={r.status}/></td><td><Link className="text-link" to={`/runs/${encodeURIComponent(r.id)}`}>{r.logHref ? 'Log + details' : 'Parameters'}<ArrowUpRight size={16}/></Link><div className="small muted">{r.experimentIds.length} linked questions</div></td></tr>)}</tbody></table></div>
        <Pagination {...page} total={filtered.length} set={p => f.set('page', p)}/></> : <Empty action={<Reset onClick={f.reset}/>}/>}</>}
  </>;
}

export function RunDetailPage() {
  const { id = '' } = useParams(), data = useResearch(), { runs, error, retry } = useRuns();
  const run = runs?.find(r => r.id === id), [log, setLog] = useState<string | null>(null), [logError, setLogError] = useState(''), [attempt, setAttempt] = useState(0), [parameterQuery, setParameterQuery] = useState('');
  const href = run?.logHref;
  useEffect(() => {
    const controller = new AbortController(); setLog(null); setLogError('');
    if (!href) return;
    if (safeHref(href) !== href) { setLogError('The recorded log link is invalid.'); return; }
    fetch(href, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error('The published log could not be loaded.'); return r.text(); })
      .then(setLog).catch(e => { if (e.name !== 'AbortError') setLogError(e.message); });
    return () => controller.abort();
  }, [href, attempt]);
  if (!runs) return <RunLoading error={error} retry={retry}/>;
  if (!run) return <Empty title="Run not found" detail="This run ID is not in the published ledger." action={<Link className="button" to="/runs">Browse recorded runs</Link>}/>;
  const parameters = Object.entries(run.parameters).filter(([key, value]) => matches(parameterQuery, [key, value]));
  const connected = data.experiments.filter(e => run.experimentIds.includes(e.id));
  return <><Link className="text-link" to="/runs"><ArrowLeft size={16}/>Run ledger</Link>
    <PageHeading eyebrow={`${run.batch || 'Recorded run'} · Job ${run.jobId || 'ID unavailable'}`} title={run.id} description={`${run.architecture} · ${run.dataset} · Seed ${run.seed}`} actions={<CopyLink/>}/>
    <ArtifactDates kind="run" id={run.id} variant="detail"/>
    <div className="run-metrics"><div className="metric-card"><span className="eyebrow">Execution state</span><ExecutionState value={run.status}/></div>
      <div className="metric-card"><span className="eyebrow">Test accuracy</span><strong>{accuracy(run.testAccuracy)}</strong><span className="small muted">Mean of the last five completed epochs; valid windows only</span></div>
      <div className="metric-card"><span className="eyebrow">Completed epochs</span><strong>{run.epochs ?? '—'}</strong><span className="small muted">Requested budget is recorded below</span></div></div>
    <p className="notice">A completed job does not establish a successful research result. Accuracy is reported as stored; validity and comparison limits belong to the experiment.</p>
    <section><div className="section-title"><h2>Connected experiments</h2><span className="muted small">{connected.length} questions</span></div>
      {connected.length ? <div className="warning-list">{connected.map(e => <article className="warning-card" key={e.id}><div><Link className="text-link" to={`/experiments/${e.id}`}><span className="mono">{e.id}</span>{e.goal}<ArrowUpRight size={15}/></Link><p>{e.result}</p></div><Status outcome={e.outcome}/></article>)}</div>
        : <Empty title="No experiment association recorded" detail="The job remains in the ledger without an invented experiment link."/>}</section>
    <section><div className="section-title"><h2>Recorded parameters</h2><span className="small muted">{Object.keys(run.parameters).length} fields</span></div>
      <Query value={parameterQuery} onChange={setParameterQuery} placeholder="Find a parameter or value…"/>
      {parameters.length ? <div className="table-wrap"><table className="ledger-table parameter-table"><thead><tr><th>Parameter</th><th>Recorded value</th></tr></thead><tbody>{parameters.map(([key, value]) => <tr key={key}><th scope="row" className="mono">{key}</th><td className="mono">{value || <span className="muted">Not recorded</span>}</td></tr>)}</tbody></table></div>
        : <Empty title={parameterQuery ? 'No matching parameters' : 'No parameters recorded'} action={parameterQuery ? <Reset onClick={() => setParameterQuery('')}/> : undefined}/>}</section>
    <section><div className="section-title"><h2>Raw execution log</h2>{href && safeHref(href) === href && <a className="button" href={href} download><Download size={16}/>Download log</a>}</div>
      {!href ? <Empty title="Raw log not included in this snapshot" detail="The ledger row and its parameters are available. No log has been reconstructed or invented."/>
        : logError ? <Empty title="Log unavailable" detail={logError} action={<button className="button" onClick={() => setAttempt(x => x + 1)}>Try again</button>}/>
          : log === null ? <p role="status" aria-busy="true">Loading the published log…</p> : <pre className="log-viewer" tabIndex={0} aria-label={`Execution log for ${run.id}`}><code>{log || 'The published log file is empty.'}</code></pre>}
    </section></>;
}

export function WarningsPage() {
  const data = useResearch(), f = useFilters();
  const entries = [...data.warnings.map(w => ({ ...w, experimentIds: [w.experimentId], journal: false, date: '' })),
    ...data.activity.filter(e => e.kind === 'warning').map(e => ({ ...e, id: `journal-${e.id}`, experimentId: '', severity: 'journal', status: 'journal', journal: true }))];
  const q = f.get('q'), severity = f.get('severity'), state = f.get('status'), experiment = f.get('experiment');
  const filtered = entries.filter(w => (!severity || w.severity === severity) && (!state || w.status === state)
    && (!experiment || w.experimentIds.includes(experiment)) && matches(q, [w.id, w.title, w.detail, w.date, ...w.experimentIds]));
  const page = ledgerPage(filtered, f.get('page') || 1);
  return <><PageHeading eyebrow="Interpretation boundaries" title="Warnings & limits" description="Read corrections, unresolved questions, and limits alongside the evidence they qualify." actions={<CopyLink/>}/>
    <div className="filter-bar"><Query value={q} onChange={v => f.set('q', v)} placeholder="Search warnings, scope or experiment IDs…"/>
      <Select label="Severities" value={severity} values={[...new Set(entries.map(w => w.severity))].sort().map(v => ({ value: v, label: v === 'journal' ? 'Journal warning' : pretty(v) }))} onChange={v => f.set('severity', v)}/>
      <Select label="States" value={state} values={[...new Set(entries.map(w => w.status))].sort().map(v => ({ value: v, label: v === 'journal' ? 'Maintainer entry' : pretty(v) }))} onChange={v => f.set('status', v)}/>
      <Select label="Experiments" value={experiment} values={experimentOptions(data.experiments)} onChange={v => f.set('experiment', v)}/>
      {(q || severity || state || experiment) && <Reset onClick={f.reset}/>}</div>
    <div className="results-summary"><span>{filtered.length.toLocaleString()} matching warnings and limits</span><span>These qualify conclusions; they are not job failures.</span></div>
    {filtered.length ? <><div className="warning-list">{page.items.map(w => <article className="warning-card" key={w.id} id={w.id}><TriangleAlert size={20} aria-hidden="true"/>
      <div><div className="record-id">{w.id}<span>{w.journal ? `Journal · ${compactDate(w.date)}` : pretty(w.status)}</span></div><h3>{w.title}</h3>{!w.journal && <ArtifactDates kind="warning" id={w.id}/>}<p>{w.detail}</p><ExperimentChips ids={w.experimentIds} limit={w.experimentIds.length}/></div>
      <Status outcome={w.severity === 'correction' ? 'correction' : w.severity === 'open' ? 'unresolved' : 'mixed'} label={w.journal ? 'Journal warning' : pretty(w.severity)}/></article>)}</div>
      <Pagination {...page} total={filtered.length} set={p => f.set('page', p)}/></> : <Empty title="No matching warnings" action={<Reset onClick={f.reset}/>}/>}</>;
}

export function ActivityPage() {
  const data = useResearch(), f = useFilters(), q = f.get('q'), kind = f.get('kind'), experiment = f.get('experiment');
  const entries = useMemo(() => [...data.activity].sort((a, b) => dateValue(b.date) - dateValue(a.date) || a.id.localeCompare(b.id)), [data.activity]);
  const filtered = entries.filter(e => (!kind || e.kind === kind) && (!experiment || e.experimentIds.includes(experiment))
    && matches(q, [e.id, e.date, compactDate(e.date), e.kind, e.title, e.detail, ...e.experimentIds]));
  const page = ledgerPage(filtered, f.get('page') || 1);
  const phase = resolvePhaseDetail(researchTimeline(data.activity, data.experiments), filtered, q, f.get('view'));
  if (phase) return <ResearchPhaseDetail phase={phase}/>;
  return <><PageHeading eyebrow="The research record" title="Research log" description="Dated findings, corrections, and maintainer notes connected to their experiments." actions={<CopyLink/>}/>
    <p className="notice"><History size={18}/>Versioned research log · Updates published by the maintainer.</p>
    <div className="filter-bar"><Query value={q} onChange={v => f.set('q', v)} placeholder="Search entries, dates or experiment IDs…"/>
      <Select label="Types" value={kind} values={[...new Set(entries.map(e => e.kind))].sort().map(v => ({ value: v, label: pretty(v) }))} onChange={v => f.set('kind', v)}/>
      <Select label="Experiments" value={experiment} values={experimentOptions(data.experiments)} onChange={v => f.set('experiment', v)}/>
      {(q || kind || experiment) && <Reset onClick={f.reset}/>}</div>
    <div className="results-summary"><span>{filtered.length.toLocaleString()} matching entries</span><span>Newest recorded dates first</span></div>
    {filtered.length ? <><div className="activity-list">{page.items.map(e => <article className="activity-item" key={e.id} id={`event-${e.id}`}>
      <div className="activity-date">{dateValue(e.date) !== -Infinity ? <time dateTime={e.date}>{compactDate(e.date)}</time> : <span>Date not recorded</span>}<span className="mono small">{pretty(e.kind)}</span></div>
      <div><span className="record-id">{e.id}</span><h3>{e.kind === 'research-phase' ? <Link to={`/activity?q=${encodeURIComponent(e.id)}&view=phase`}>{e.title}</Link> : e.title}</h3><p>{e.detail}</p><ExperimentChips ids={e.experimentIds} limit={e.experimentIds.length}/></div>
    </article>)}</div><Pagination {...page} total={filtered.length} set={p => f.set('page', p)}/></> : <Empty title="No matching history" detail="Try another search, or remove a filter to see the published log." action={<Reset onClick={f.reset}/>}/>}</>;
}
