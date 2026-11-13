import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { linkHorizontal } from 'd3-shape';
import { select } from 'd3-selection';
import { zoom, zoomIdentity, zoomTransform, type ZoomBehavior } from 'd3-zoom';
import { Maximize2, Minus, Plus, X } from 'lucide-react';
import { countStatuses, findNode, ideaTree, statusInfo, statusOrder, type IdeaTreeData, type TreeNode } from '../lib/idea-tree';
import { ancestorsOf, childLabel, expandableIds, fitView, layoutGraph, narrowSizes, wideSizes, type GraphNode, type View } from '../lib/idea-graph';
import { TreeDetail, TreeIcon, type TestLink } from './IdeaTreeDetail';
import './IdeaTree.css';
import './IdeaGraph.css';

const MIN_K = 0.15, MAX_K = 2.5;
const link = linkHorizontal<{ s: [number, number]; t: [number, number] }, [number, number]>().source(d => d.s).target(d => d.t).x(p => p[0]).y(p => p[1]);

function useWidth(ref: React.RefObject<HTMLElement | null>) {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setWidth(el.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

export function IdeaGraph({ data = ideaTree, pageIds, testLink, initialNode = null, onSelectChange }: {
  data?: IdeaTreeData;
  pageIds: ReadonlySet<string>;
  testLink: TestLink;
  initialNode?: string | null;
  onSelectChange?: (id: string | null) => void;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<ZoomBehavior<HTMLDivElement, unknown> | null>(null);
  const anchor = useRef<{ id: string; sx: number; sy: number; reveal: boolean } | null>(null);
  const first = useRef(true);
  const pendingFit = useRef<number | null>(null);
  const frameWidth = useWidth(frameRef);
  const narrow = frameWidth > 0 && frameWidth < 700;
  const sheet = frameWidth < 1000;
  const sizes = narrow ? narrowSizes : wideSizes;

  const [selectedId, setSelectedId] = useState<string | null>(initialNode && findNode(data, initialNode) ? initialNode : null);
  const [open, setOpen] = useState<Set<string>>(() => new Set(['root', ...(initialNode && findNode(data, initialNode) ? ancestorsOf(data, initialNode) : [])]));
  const layout = useMemo(() => layoutGraph(data, open, sizes), [data, open, sizes]);
  const byId = useMemo(() => new Map(layout.nodes.map(n => [n.id, n])), [layout]);
  const node: TreeNode | null = findNode(data, selectedId);

  const apply = useCallback((v: View) => {
    const view = viewRef.current, z = zoomRef.current;
    if (!view || !z) return;
    const t = zoomIdentity.translate(v.x, v.y).scale(v.k);
    select(view).call(z.transform, t);
  }, []);

  // Pan and zoom: drag, wheel and pinch. The stage is moved with a CSS transform, so cards stay real, selectable HTML.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const z = zoom<HTMLDivElement, unknown>().scaleExtent([MIN_K, MAX_K]).on('zoom', e => {
      if (stageRef.current) stageRef.current.style.transform = `translate(${e.transform.x}px, ${e.transform.y}px) scale(${e.transform.k})`;
    });
    zoomRef.current = z;
    select(view).call(z).on('dblclick.zoom', null);
    return () => { select(view).on('.zoom', null); };
  }, []);

  const fit = useCallback((minK: number) => {
    const view = viewRef.current;
    if (!view) return;
    apply(fitView(layout.bounds, view.clientWidth, view.clientHeight, minK, 1.1));
  }, [apply, layout]);

  // First paint: show the whole graph, or the selected box when a shared link names one.
  useLayoutEffect(() => {
    if (!frameWidth || !first.current) return;
    first.current = false;
    const view = viewRef.current;
    if (!view) return;
    const start = selectedId ? byId.get(selectedId) : null;
    if (start) {
      const k = narrow ? 0.9 : 1;
      apply({ k, x: view.clientWidth / 2 - (start.x + start.w / 2) * k, y: view.clientHeight * (sheet ? 0.24 : 0.5) - (start.y + start.h / 2) * k });
    } else {
      const box = { ...layout.bounds, maxX: layout.bounds.maxX };
      apply(fitView(box, view.clientWidth, view.clientHeight, narrow ? 0.85 : 0.6, 1));
    }
  }, [frameWidth, narrow, sheet, layout, byId, selectedId, apply]);

  // After a box opens or folds, keep it where it was on screen, and bring its new children into view.
  useLayoutEffect(() => {
    const a = anchor.current, view = viewRef.current;
    if (!view) return;
    if (pendingFit.current !== null) {
      apply(fitView(layout.bounds, view.clientWidth, view.clientHeight, pendingFit.current, 1.1));
      pendingFit.current = null;
      anchor.current = null;
      return;
    }
    if (!a) return;
    anchor.current = null;
    const n = byId.get(a.id);
    if (!n) return;
    const t = zoomTransform(view);
    let x = a.sx - (n.x + n.w / 2) * t.k, y = a.sy - (n.y + n.h / 2) * t.k;
    if (a.reveal) {
      const kids = layout.links.filter(l => l.from === a.id).map(l => byId.get(l.to)).filter((c): c is GraphNode => !!c);
      if (kids.length) {
        const pad = 16, vw = view.clientWidth, vh = view.clientHeight;
        const right = Math.max(...kids.map(c => c.x + c.w)) * t.k + x, left = n.x * t.k + x;
        if (right > vw - pad) x -= Math.min(right - (vw - pad), Math.max(0, left - pad));
        const top = Math.min(...kids.map(c => c.y)) * t.k + y, bottom = Math.max(...kids.map(c => c.y + c.h)) * t.k + y;
        if (!sheet && bottom - top <= vh - 2 * pad) {
          if (top < pad) y += pad - top;
          else if (bottom > vh - pad) y -= bottom - (vh - pad);
        }
      }
    }
    apply({ x, y, k: t.k });
  }, [layout, byId, apply, sheet]);

  const select_ = (id: string | null) => { setSelectedId(id); onSelectChange?.(id); };

  const onNode = (n: GraphNode) => {
    select_(n.id);
    const view = viewRef.current;
    if (!view) return;
    const t = zoomTransform(view);
    const opening = !open.has(n.id);
    // Where the box is on screen now; with the summary sheet up, keep it in the visible top part.
    const sx = (n.x + n.w / 2) * t.k + t.x;
    let sy = (n.y + n.h / 2) * t.k + t.y;
    if (sheet) sy = Math.min(sy, view.clientHeight * 0.26);
    if (n.childCount > 0) {
      anchor.current = { id: n.id, sx, sy, reveal: opening };
      setOpen(prev => {
        const next = new Set(prev);
        if (opening) next.add(n.id); else next.delete(n.id);
        return next;
      });
    } else if (sheet) {
      apply({ k: t.k, x: sx - (n.x + n.w / 2) * t.k, y: sy - (n.y + n.h / 2) * t.k });
    }
  };

  const expandable = useMemo(() => expandableIds(data), [data]);
  const openAll = () => { pendingFit.current = 0.2; setOpen(new Set(expandable)); };
  const closeAll = () => { pendingFit.current = narrow ? 0.85 : 0.6; setOpen(new Set(['root'])); };

  const zoomBy = (factor: number) => {
    const view = viewRef.current, z = zoomRef.current;
    if (view && z) select(view).call(z.scaleBy, factor, [view.clientWidth / 2, view.clientHeight / 2]);
  };

  const links = layout.links.map(l => {
    const a = byId.get(l.from)!, b = byId.get(l.to)!;
    return { id: l.id, d: link({ s: [a.x + a.w, a.y + a.h / 2], t: [b.x, b.y + b.h / 2] }) ?? '', status: b.status };
  });

  const leaves = data.parts.flatMap(p => p.leaves);
  const totals = countStatuses(leaves);

  const close = () => select_(null);

  const label = (n: GraphNode) => {
    if (n.kind === 'root') return `The question: ${data.root.title}`;
    const found = findNode(data, n.id);
    if (!found || found.kind === 'root') return n.id;
    return found.kind === 'part' ? `Part ${found.part.number}: ${found.part.title}` : found.leaf.title;
  };

  const cardInner = (n: GraphNode) => {
    const found = findNode(data, n.id);
    const badge = n.childCount > 0 ? <span className="ig-badge" aria-hidden="true">{n.open ? '−' : '+'} {childLabel(n.kind, n.childCount)}</span> : null;
    if (!found || found.kind === 'root') {
      return <>
        <span className="tree-node-kicker">The question</span>
        <strong className="ig-title">{data.root.title}</strong>
        {badge}
      </>;
    }
    if (found.kind === 'part') {
      return <>
        <span className="ig-head"><TreeIcon status={found.part.status}/><span className="tree-node-kicker">Part {found.part.number} · {found.part.dates}</span></span>
        <strong className="ig-title">{found.part.title}</strong>
        {badge}
      </>;
    }
    return <>
      <span className="ig-head"><TreeIcon status={found.leaf.status}/><span className="tree-node-kicker">{found.leaf.id}</span></span>
      <strong className="ig-title">{found.leaf.title}</strong>
      {badge}
    </>;
  };

  return <div className={`ig ${narrow ? 'ig-narrow' : 'ig-medium'} ${sheet ? 'ig-sheeted' : 'ig-wide'}`} ref={frameRef}>
    <ul className="tree-legend" aria-label="Legend">
      {statusOrder.map(s => <li key={s} className={`tree-${s}`}><TreeIcon status={s}/><span>{statusInfo[s].name}<small>{totals[s]} tests</small></span></li>)}
    </ul>
    <div className="ig-controls" role="toolbar" aria-label="Graph controls">
      <button type="button" className="button" onClick={openAll}>Open all</button>
      <button type="button" className="button" onClick={closeAll}>Close all</button>
      <button type="button" className="button" onClick={() => fit(0.15)}><Maximize2 size={15}/> Fit to screen</button>
      <span className="ig-zoom">
        <button type="button" className="button" aria-label="Zoom in" onClick={() => zoomBy(1.3)}><Plus size={16}/></button>
        <button type="button" className="button" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.3)}><Minus size={16}/></button>
      </span>
    </div>
    <p className="ig-hint">Select a box: it opens what comes next on the right and shows its summary. Select it again to fold it. Drag to move, scroll or pinch to zoom.</p>
    <div className="ig-body">
      <div className="ig-canvas">
        <div className="ig-view" ref={viewRef} role="application" aria-label="Idea tree graph">
          <div className="ig-stage" ref={stageRef}>
            <svg className="ig-links" width={Math.max(1, layout.bounds.maxX + 4)} height={1} aria-hidden="true">
              {links.map(l => <path key={l.id} d={l.d} className={`ig-link ${l.status ? `tree-${l.status}` : ''}`}/>)}
            </svg>
            {layout.nodes.map(n => <button
              key={n.id}
              type="button"
              className={`ig-card ig-${n.kind} ${n.status ? `tree-${n.status}` : ''}`}
              style={{ left: n.x, top: n.y, width: n.w, height: n.h }}
              aria-current={selectedId === n.id}
              aria-expanded={n.childCount > 0 ? n.open : undefined}
              aria-label={label(n)}
              onClick={() => onNode(n)}
            >{cardInner(n)}</button>)}
          </div>
        </div>
        {sheet && node && <div className="ig-sheet">
          <button type="button" className="ig-sheet-close" aria-label="Close summary" onClick={close}><X size={18}/></button>
          <TreeDetail node={node} data={data} pageIds={pageIds} testLink={testLink} onSelect={select_}/>
        </div>}
      </div>
      {!sheet && <aside className="ig-side" aria-label="Summary of the selected box">
        <TreeDetail node={node} data={data} pageIds={pageIds} testLink={testLink} onSelect={select_}/>
      </aside>}
    </div>
  </div>;
}
