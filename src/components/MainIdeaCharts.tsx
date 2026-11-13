import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { findNode, ideaTree, statusInfo } from '../lib/idea-tree';
import type { BarChartData, LineChartData } from '../lib/main-idea';
import { TreeIcon } from './IdeaTreeDetail';

/** Link chip to the test, coloured by its status in the tree. */
export function TestChip({ id }: { id: string }) {
  const node = findNode(ideaTree, id);
  if (!node || node.kind !== 'leaf') return null;
  const mt = node.leaf.experiments[0];
  const body = <><TreeIcon status={node.leaf.status}/><span><strong>{id}</strong> · {node.leaf.title}</span></>;
  return <span className={`mi-chip tree-${node.leaf.status}`} title={statusInfo[node.leaf.status].name}>
    {body}
    {mt && <Link to={`/experiments/${mt}`} className="mi-chip-link">Open test page <ArrowUpRight size={14}/></Link>}
  </span>;
}

export function BarChart({ chart }: { chart: BarChartData }) {
  const values = [0, ...chart.rows.map(r => r.value), ...(chart.reference ? [chart.reference.value] : [])];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  const at = (v: number) => ((v - lo) / span) * 100;
  return <div className="mi-bars" role="img" aria-label={`${chart.question} ${chart.rows.map(r => `${r.label}: ${r.display}`).join('; ')}. Unit: ${chart.unit}.`}>
    {chart.rows.map(r => <div className="mi-bar-row" key={r.label}>
      <div className="mi-bar-label"><span>{r.label}</span><strong className="mi-bar-value">{r.display}</strong></div>
      <div className="mi-bar-track">
        <span className={`mi-bar mi-tone-bar-${r.tone}`} style={{ left: `${at(Math.min(0, r.value))}%`, width: `${(Math.abs(r.value) / span) * 100}%` }}/>
        <span className="mi-zero" style={{ left: `${at(0)}%` }}/>
        {chart.reference && <span className="mi-ref" style={{ left: `${at(chart.reference.value)}%` }}/>}
      </div>
    </div>)}
    {chart.reference && <p className="mi-ref-note"><span className="mi-ref-key" aria-hidden="true"/>{chart.reference.label}</p>}
    <p className="mi-unit">Unit: {chart.unit}.</p>
  </div>;
}

const W = 440, H = 300, L = 52, R = 14, T = 32, B = 56;

export function LineChart({ chart }: { chart: LineChartData }) {
  const all = [...chart.series.flatMap(s => s.points), ...chart.references.map(r => r.value)];
  const step = chart.id === 'cifar100' ? 20 : 1;
  const lo = Math.floor((Math.min(...all) - 0.2) / step) * step;
  const hi = Math.ceil((Math.max(...all) + 0.2) / step) * step;
  const xs = (i: number) => L + (i / (chart.x.length - 1)) * (W - L - R);
  const ys = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(Math.round(v * 100) / 100);
  return <figure className="mi-line">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${chart.question} ${chart.yLabel} against ${chart.xLabel}.`}>
      {ticks.map(t => <g key={t}>
        <line x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} className="mi-grid"/>
        <text x={L - 8} y={ys(t) + 4} textAnchor="end" className="mi-axis">{t}</text>
      </g>)}
      {chart.x.map((x, i) => <text key={x} x={xs(i)} y={H - B + 18} textAnchor="middle" className="mi-axis">{x}</text>)}
      <text x={(L + W - R) / 2} y={H - B + 38} textAnchor="middle" className="mi-axis-title">{chart.xLabel}</text>
      <text x={4} y={14} textAnchor="start" className="mi-axis-title">{chart.yLabel}</text>
      {chart.references.map(r => <line key={r.label} x1={L} x2={W - R} y1={ys(r.value)} y2={ys(r.value)} className={`mi-refline mi-tone-${r.tone}`}/>)}
      {chart.series.map(s => <g key={s.label} className={`mi-tone-${s.tone}`}>
        <polyline points={s.points.map((p, i) => `${xs(i)},${ys(p)}`).join(' ')} className="mi-path"/>
        {s.points.map((p, i) => <circle key={i} cx={xs(i)} cy={ys(p)} r={4} className="mi-dot"/>)}
      </g>)}
    </svg>
    <ul className="mi-key">
      {chart.series.map(s => <li key={s.label} className={`mi-tone-${s.tone}`}><span className="mi-key-line"/>{s.label}</li>)}
      {chart.references.map(r => <li key={r.label} className={`mi-tone-${r.tone}`}><span className="mi-key-dash"/>{r.label}: {r.value}</li>)}
    </ul>
    <details className="mi-numbers">
      <summary>Show the numbers</summary>
      <table>
        <thead><tr><th>r</th>{chart.series.map(s => <th key={s.label}>{s.label}</th>)}</tr></thead>
        <tbody>{chart.x.map((x, i) => <tr key={x}><td>{x}</td>{chart.series.map(s => <td key={s.label}>{s.points[i]}</td>)}</tr>)}</tbody>
      </table>
    </details>
  </figure>;
}
