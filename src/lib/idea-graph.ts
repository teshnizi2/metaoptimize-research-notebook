import { hierarchy, tree } from 'd3-hierarchy';
import type { IdeaTreeData, TreeLeaf, TreeStatus } from './idea-tree';

export type GraphKind = 'root' | 'part' | 'leaf';

export interface GraphNode {
  id: string;
  kind: GraphKind;
  depth: number;
  /** Top-left corner and size of the card, in graph units. */
  x: number;
  y: number;
  w: number;
  h: number;
  status: TreeStatus | null;
  /** Direct children, whether or not they are shown. */
  childCount: number;
  open: boolean;
}

export interface GraphLink { id: string; from: string; to: string }

export interface GraphLayout {
  nodes: GraphNode[];
  links: GraphLink[];
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
}

export interface GraphSizes {
  root: { w: number; h: number };
  part: { w: number; h: number };
  leaf: { w: number; h: number };
  gap: number;
  pitch: number;
}

export const wideSizes: GraphSizes = { root: { w: 250, h: 170 }, part: { w: 270, h: 98 }, leaf: { w: 240, h: 84 }, gap: 72, pitch: 110 };
export const narrowSizes: GraphSizes = { root: { w: 156, h: 156 }, part: { w: 196, h: 98 }, leaf: { w: 200, h: 84 }, gap: 40, pitch: 110 };

/** Children of a node in the full tree: root -> parts -> tests, where a test with a `parent` hangs under that test. */
export function buildChildren(data: IdeaTreeData): Map<string, string[]> {
  const kids = new Map<string, string[]>();
  const add = (parent: string, id: string) => kids.set(parent, [...(kids.get(parent) ?? []), id]);
  for (const part of data.parts) {
    add('root', part.id);
    for (const leaf of part.leaves) add(leaf.parent ?? part.id, leaf.id);
  }
  return kids;
}

/** Id of the parent of every node. */
export function buildParents(data: IdeaTreeData): Map<string, string> {
  const parents = new Map<string, string>();
  for (const [parent, ids] of buildChildren(data)) for (const id of ids) parents.set(id, parent);
  return parents;
}

/** Ids of all nodes that have children, i.e. everything "Open all" should open. */
export function expandableIds(data: IdeaTreeData): string[] {
  return [...buildChildren(data).keys()];
}

/** Ids that must be open to show `id`. */
export function ancestorsOf(data: IdeaTreeData, id: string): string[] {
  const parents = buildParents(data);
  const out: string[] = [];
  for (let p = parents.get(id); p; p = parents.get(p)) out.push(p);
  return out;
}

export function layoutGraph(data: IdeaTreeData, open: ReadonlySet<string>, sizes: GraphSizes): GraphLayout {
  const kids = buildChildren(data);
  const statusOf = new Map<string, TreeStatus>();
  for (const part of data.parts) {
    statusOf.set(part.id, part.status);
    for (const leaf of part.leaves as TreeLeaf[]) statusOf.set(leaf.id, leaf.status);
  }
  const partIds = new Set(data.parts.map(p => p.id));
  const kindOf = (id: string): GraphKind => (id === 'root' ? 'root' : partIds.has(id) ? 'part' : 'leaf');

  const root = hierarchy<string>('root', id => (open.has(id) ? kids.get(id) ?? [] : []).length ? kids.get(id) : undefined);
  tree<string>().nodeSize([sizes.pitch, 1]).separation(() => 1)(root);

  // Column offsets: a column is as wide as its kind of card; test columns repeat for deeper levels.
  const widths = (depth: number) => (depth === 0 ? sizes.root.w : depth === 1 ? sizes.part.w : sizes.leaf.w);
  const colX: number[] = [0];
  const maxDepth = Math.max(...root.descendants().map(d => d.depth));
  for (let d = 1; d <= maxDepth; d += 1) colX[d] = colX[d - 1] + widths(d - 1) + sizes.gap;

  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  let minY = Infinity, maxY = -Infinity, maxX = 0;
  for (const d of root.descendants()) {
    const kind = kindOf(d.data);
    const { w, h } = sizes[kind];
    const y = (d.x ?? 0) - h / 2;
    const x = colX[d.depth];
    minY = Math.min(minY, y); maxY = Math.max(maxY, y + h); maxX = Math.max(maxX, x + w);
    nodes.push({ id: d.data, kind, depth: d.depth, x, y, w, h, status: statusOf.get(d.data) ?? null, childCount: kids.get(d.data)?.length ?? 0, open: open.has(d.data) });
    if (d.parent) links.push({ id: `${d.parent.data}>${d.data}`, from: d.parent.data, to: d.data });
  }
  return { nodes, links, bounds: { minX: 0, minY, maxX, maxY } };
}

export interface View { x: number; y: number; k: number }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Pan and zoom that show `box` inside a viewport; a box that is too big at `minK` is anchored at its top-left instead. */
export function fitView(box: { minX: number; minY: number; maxX: number; maxY: number }, vw: number, vh: number, minK: number, maxK: number, pad = 20): View {
  const bw = box.maxX - box.minX, bh = box.maxY - box.minY;
  const k = clamp(Math.min((vw - 2 * pad) / bw, (vh - 2 * pad) / bh), minK, maxK);
  const x = bw * k <= vw - 2 * pad ? (vw - bw * k) / 2 - box.minX * k : pad - box.minX * k;
  const y = bh * k <= vh - 2 * pad ? (vh - bh * k) / 2 - box.minY * k : pad - box.minY * k;
  return { x, y, k };
}

/** Short label for the badge on a card that has children. */
export function childLabel(kind: GraphKind, count: number): string {
  if (kind === 'root') return `${count} parts`;
  if (kind === 'part') return `${count} test${count === 1 ? '' : 's'}`;
  return `${count} follow-up${count === 1 ? '' : 's'}`;
}
