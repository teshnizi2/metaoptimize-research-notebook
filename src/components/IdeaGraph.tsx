import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
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

const PAD = 20;
const PANEL_MAX = 380;

export function IdeaGraph({ data = ideaTree, pageIds, testLink, initialNode = null, onSelectChange, header }: {
  data?: IdeaTreeData;
  pageIds: ReadonlySet<string>;
  testLink: TestLink;
  initialNode?: string | null;
  onSelectChange?: (id: string | null) => void;
  /** Shown at the left of the toolbar row (the page title and view switch). */
  header?: ReactNode;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<ZoomBehavior<HTMLDivElement, unknown> | null>(null);
  const anchor = useRef<{ id: string; sx: number; sy: number; reveal: boolean } | null>(null);
  const first = useRef(true);
  const pending = useRef<number | 'start' | null>(null);
  const frameWidth = useWidth(frameRef);
  const narrow = frameWidth > 0 && frameWidth < 700;
  const sizes = narrow ? narrowSizes : wideSizes;

  const [selectedId, setSelectedId] = useState<string | null>(initialNode && findNode(data, initialNode) ? initialNode : null);
  const [open, setOpen] = useState<Set<string>>(() => new Set(['root', ...(initialNode && findNode(data, initialNode) ? ancestorsOf(data, initialNode) : [])]));
  const layout = useMemo(() => layoutGraph(data, open, sizes), [data, open, sizes]);
  const byId = useMemo(() => new Map(layout.nodes.map(n => [n.id, n])), [layout]);
  const node: TreeNode | null = findNode(data, selectedId);

  /** Width of the canvas that is not covered by the summary panel (the panel sits on the right when the graph is wide). */
  const visibleWidth = (view: HTMLElement, panelOpen: boolean) => view.clientWidth - (panelOpen && !narrow ? Math.min(PANEL_MAX, view.clientWidth * 0.46) + 10 : 0);

  const apply = useCallback((v: View) => {
    const view = viewRef.current, z = zoomRef.current;
    if (!view || !z) return;
    select(view).call(z.transform, zoomIdentity.translate(v.x, v.y).scale(v.k));
  }, []);

  // Pan and zoom. Drag moves; the wheel scrolls (so the text stays readable); Ctrl/Cmd + wheel and two-finger pinch zoom.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const z = zoom<HTMLDivElement, unknown>().scaleExtent([MIN_K, MAX_K])
      .filter(e => (e.type === 'wheel' ? e.ctrlKey || e.metaKey : !e.button && !e.ctrlKey))
      .on('zoom', e => {
        if (stageRef.current) stageRef.current.style.transform = `translate(${e.transform.x}px, ${e.transform.y}px) scale(${e.transform.k})`;
      });
    zoomRef.current = z;
    select(view).call(z).on('dblclick.zoom', null);
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : 1;
      select(view).call(z.translateBy, (-e.deltaX * unit) / zoomTransform(view).k, (-e.deltaY * unit) / zoomTransform(view).k);
    };
    view.addEventListener('wheel', onWheel, { passive: false });
    return () => { select(view).on('.zoom', null); view.removeEventListener('wheel', onWheel); };
  }, []);

  /** Readable start: full-size cards, the question and the first parts in view, the rest a drag or scroll away. */
  const startView = useCallback((view: HTMLElement): View => {
    const col = layout.nodes.filter(n => n.depth <= 1);
    const minY = Math.min(...col.map(n => n.y)), maxY = Math.max(...col.map(n => n.y + n.h)), maxX = Math.max(...col.map(n => n.x + n.w));
    const root = byId.get('root')!;
    const k = Math.min(1, (view.clientWidth - 2 * PAD) / maxX);
    const vh = view.clientHeight;
    // Everything fits: centre it. Otherwise centre on the question, never starting lower than the first part.
    const y = (maxY - minY) * k + 2 * PAD <= vh ? (vh - (maxY - minY) * k) / 2 - minY * k : Math.min(vh / 2 - (root.y + root.h / 2) * k, PAD - minY * k);
    return { k, x: PAD, y };
  }, [layout, byId]);

  const fit = useCallback(() => {
    const view = viewRef.current;
    if (view) apply(fitView(layout.bounds, view.clientWidth, view.clientHeight, MIN_K, 1.1));
  }, [apply, layout]);

  // First paint: a readable start, or the selected box when a shared link names one.
  useLayoutEffect(() => {
    if (!frameWidth || !first.current) return;
    first.current = false;
    const view = viewRef.current;
    if (!view) return;
    const target = selectedId ? byId.get(selectedId) : null;
    if (target) {
      const k = narrow ? Math.min(1, (view.clientWidth - 2 * PAD) / (sizes.part.w + sizes.gap + sizes.leaf.w)) : 1;
      const cx = narrow ? view.clientWidth / 2 : Math.min(visibleWidth(view, true) / 2, view.clientWidth / 2);
      apply({ k, x: cx - (target.x + target.w / 2) * k, y: view.clientHeight * (narrow ? 0.24 : 0.5) - (target.y + target.h / 2) * k });
    } else {
      apply(startView(view));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameWidth, narrow, layout, byId, selectedId, apply, startView]);

  // After a box opens or folds, keep it where it was on screen, and bring its new children into view at the same size.
  useLayoutEffect(() => {
    const a = anchor.current, view = viewRef.current;
    if (!view) return;
    if (pending.current !== null) {
      apply(pending.current === 'start' ? startView(view) : fitView(layout.bounds, view.clientWidth, view.clientHeight, pending.current, 1.1));
      pending.current = null;
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
        const vw = visibleWidth(view, selectedId !== null), vh = view.clientHeight;
        const right = Math.max(...kids.map(c => c.x + c.w)) * t.k + x, left = n.x * t.k + x;
        if (right > vw - PAD) x -= Math.min(right - (vw - PAD), Math.max(0, left - PAD));
        const top = Math.min(...kids.map(c => c.y)) * t.k + y, bottom = Math.max(...kids.map(c => c.y + c.h)) * t.k + y;
        if (!narrow && bottom - top <= vh - 2 * PAD) {
          if (top < PAD) y += PAD - top;
          else if (bottom > vh - PAD) y -= bottom - (vh - PAD);
        }
      }
    }
    apply({ x, y, k: t.k });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, byId, apply, narrow]);

  const select_ = (id: string | null) => { setSelectedId(id); onSelectChange?.(id); };

  const onNode = (n: GraphNode) => {
    select_(n.id);
    const view = viewRef.current;
    if (!view) return;
    const t = zoomTransform(view);
    const opening = !open.has(n.id);
    // Where the box is on screen now. With the bottom sheet up keep it in the visible top part; with the side panel up keep it left of the panel.
    let sx = (n.x + n.w / 2) * t.k + t.x;
    let sy = (n.y + n.h / 2) * t.k + t.y;
    if (narrow) sy = Math.min(sy, view.clientHeight * 0.26);
    else sx = Math.min(sx, visibleWidth(view, true) - PAD - (n.w / 2) * t.k);
    if (n.childCount > 0) {
      anchor.current = { id: n.id, sx, sy, reveal: opening };
      setOpen(prev => {
        const next = new Set(prev);
        if (opening) next.add(n.id); else next.delete(n.id);
        return next;
      });
    } else {
      apply({ k: t.k, x: sx - (n.x + n.w / 2) * t.k, y: sy - (n.y + n.h / 2) * t.k });
    }
  };

  const expandable = useMemo(() => expandableIds(data), [data]);
  const openAll = () => { pending.current = 0.2; setOpen(new Set(expandable)); };
  const closeAll = () => { pending.current = 'start'; setOpen(new Set(['root'])); };

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

  const panel = node && <>
    <button type="button" className="ig-panel-close" aria-label="Close summary" onClick={close}><X size={18}/></button>
    <TreeDetail node={node} data={data} pageIds={pageIds} testLink={testLink} onSelect={select_}/>
  </>;

  return <div className={`ig ${narrow ? 'ig-narrow' : 'ig-wide'}`} ref={frameRef}>
    <div className="ig-bar">
      <div className="ig-bar-head">{header}</div>
      <div className="ig-controls" role="toolbar" aria-label="Graph controls">
        <button type="button" className="button" onClick={openAll}>Open all</button>
        <button type="button" className="button" onClick={closeAll}>Close all</button>
        <button type="button" className="button" onClick={fit}><Maximize2 size={15}/> Fit to screen</button>
        <button type="button" className="button ig-icon" aria-label="Zoom in" onClick={() => zoomBy(1.3)}><Plus size={16}/></button>
        <button type="button" className="button ig-icon" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.3)}><Minus size={16}/></button>
      </div>
    </div>
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
      <ul className="ig-legend" aria-label="Legend">
        {statusOrder.map(s => <li key={s} className={`tree-${s}`}><TreeIcon status={s}/><span>{statusInfo[s].short}<small>{totals[s]}</small></span></li>)}
      </ul>
      <p className="ig-hint">Select a box to open what comes next and read its summary; select it again to fold it. Drag or scroll to move. Ctrl or ⌘ + scroll, or pinch, to zoom.</p>
      {node && <aside className={narrow ? 'ig-sheet' : 'ig-panel'} aria-label="Summary of the selected box">{panel}</aside>}
    </div>
  </div>;
}
