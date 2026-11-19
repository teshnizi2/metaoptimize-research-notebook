import { useState } from 'react';
import { axisFor, blocks, chartUrl, fmt, inline, pct, readItem, READING_WORD, type IntervalChartData, type Table } from '../lib/private-pages';
import './Notebook.css';

/** Text with **bold** and `code`; paragraphs and "- " lists. Built as React nodes, never as HTML. */
export function RichText({ text, className }: { text: string; className?: string }) {
  return <div className={`nb-rich ${className ?? ''}`}>{blocks(text).map((b, i) => b.kind === 'p'
    ? <p key={i}><Inline text={b.text}/></p>
    : <ul key={i}>{b.items.map((it, j) => <li key={j}><Inline text={it}/></li>)}</ul>)}</div>;
}

export function Inline({ text }: { text: string }) {
  return <>{inline(text).map((p, i) => p.kind === 'bold' ? <strong key={i}>{p.text}</strong> : p.kind === 'code' ? <code key={i}>{p.text}</code> : <span key={i}>{p.text}</span>)}</>;
}

export function DataTable({ table }: { table: Table }) {
  return <figure className="nb-table">
    {table.caption && <figcaption><Inline text={table.caption}/></figcaption>}
    <div className="nb-table-wrap" tabIndex={0} role="region" aria-label={table.caption || 'Table'}>
      <table>
        <thead><tr>{table.columns.map((c, i) => <th key={i} scope="col"><Inline text={c}/></th>)}</tr></thead>
        <tbody>{table.rows.map((r, i) => <tr key={i}>{r.map((c, j) => j === 0 ? <th key={j} scope="row"><Inline text={c}/></th> : <td key={j}><Inline text={c}/></td>)}</tr>)}</tbody>
      </table>
    </div>
  </figure>;
}

/** One image from the private store; if it cannot be fetched, the caption is shown alone. */
export function ChartImage({ name, caption }: { name: string; caption: string }) {
  const [failed, setFailed] = useState(false);
  return <figure className="nb-chart-image">
    {!failed && <img src={chartUrl(name)} alt={caption || 'Chart'} loading="lazy" onError={() => setFailed(true)}/>}
    {caption && <figcaption><Inline text={caption}/></figcaption>}
  </figure>;
}

const tick = (t: number) => (t === 0 ? '0' : `${t < 0 ? '\u2212' : '+'}${Math.abs(t)}`);

/**
 * Dots with whiskers on one shared axis, drawn with plain elements (no chart library). Each row says its numbers in words,
 * so the picture is a help, not the only way to read it.
 */
export function IntervalChart({ chart }: { chart: IntervalChartData }) {
  const axis = axisFor(chart);
  const zero = pct(0, axis);
  return <figure className="nb-interval">
    {chart.title && <figcaption>{chart.title}</figcaption>}
    <ul className="nb-interval-rows">
      {chart.items.map(it => {
        const reading = readItem(it, chart);
        return <li key={it.label} className={reading ? `nb-is-${reading}` : `nb-tone-${it.tone}`}>
          <div className="nb-interval-text">
            <span className="nb-interval-label">{it.label}</span>
            <span className="nb-interval-num"><b>{fmt(it.value)}</b> <span>[{fmt(it.lo)}, {fmt(it.hi)}]</span>{reading && <em>{READING_WORD[reading]}</em>}</span>
          </div>
          <div className="nb-interval-track" aria-hidden="true">
            {chart.band && <span className="nb-interval-band" style={{ left: `${pct(chart.band.from, axis)}%`, right: 0 }}/>}
            <span className="nb-interval-zero" style={{ left: `${zero}%` }}/>
            <span className="nb-interval-whisker" style={{ left: `${pct(it.lo, axis)}%`, width: `${pct(it.hi, axis) - pct(it.lo, axis)}%` }}/>
            <span className="nb-interval-dot" style={{ left: `${pct(it.value, axis)}%` }}/>
          </div>
          {it.note && <p className="nb-interval-note">{it.note}</p>}
        </li>;
      })}
    </ul>
    <div className="nb-interval-axis" aria-hidden="true">
      {axis.ticks.map(t => <span key={t} style={{ left: `${pct(t, axis)}%` }}>{tick(t)}</span>)}
    </div>
    <p className="nb-interval-legend">{chart.axis}{chart.band && <> · shaded: {chart.band.label}</>} · dot = mean, line = 95 % interval. The numbers are written above each line.</p>
  </figure>;
}
