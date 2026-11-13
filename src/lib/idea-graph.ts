import { hierarchy, tree } from 'd3-hierarchy';
import type { IdeaTreeData, TreeLeaf, TreeStatus } from './idea-tree';
import { attachedIdeas, ideaBranches, ideaParent, type FutureIdea } from './future-ideas';

/** `idea` is a future idea (a test or part's follow-up not run yet); `branch` is a new branch of ideas off the question. Both are drawn dashed and carry no verdict. */
export type GraphKind = 'root' | 'part' | 'leaf' | 'idea' | 'branch';

/** The card size a kind of box uses: an idea is as big as a test, a new branch as big as a part. */
export const sizeKind = (kind: GraphKind): 'root' | 'part' | 'leaf' => (kind === 'idea' ? 'leaf' : kind === 'branch' ? 'part' : kind);

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
  /** How many of those children are future ideas (or, for the question, new branches). */
  ideaCount: number;
  open: boolean;
  /** Radial view only: where the card's centre is, as an angle (radians, clockwise from twelve o'clock) and a distance from the question. */
  angle?: number;
  radius?: number;
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

export const wideSizes: GraphSizes = { root: { w: 260, h: 190 }, part: { w: 280, h: 104 }, leaf: { w: 250, h: 104 }, gap: 72, pitch: 114 };
// On a phone the question, the gap and the part column together fit one 375 px screen at full size.
export const narrowSizes: GraphSizes = { root: { w: 140, h: 210 }, part: { w: 176, h: 104 }, leaf: { w: 200, h: 104 }, gap: 28, pitch: 114 };

/** Children of a node in the full tree: root -> parts -> tests, where a test with a `parent` hangs under that test. */
export function buildChildren(data: IdeaTreeData, ideas: readonly FutureIdea[] = []): Map<string, string[]> {
  const kids = new Map<string, string[]>();
  const add = (parent: string, id: string) => kids.set(parent, [...(kids.get(parent) ?? []), id]);
  for (const part of data.parts) {
    add('root', part.id);
    for (const leaf of part.leaves) add(leaf.parent ?? part.id, leaf.id);
  }
  // Future ideas come after the tests of the box they grow from; new branches after the parts.
  const shown = attachedIdeas(data, ideas);
  for (const b of ideaBranches(shown)) add('root', b.id);
  for (const idea of shown) add(ideaParent(idea), idea.id);
  return kids;
}

/** Ids of the future-idea boxes and new-branch boxes of a tree. */
export function ideaBoxIds(data: IdeaTreeData, ideas: readonly FutureIdea[]): Set<string> {
  const shown = attachedIdeas(data, ideas);
  return new Set([...shown.map(i => i.id), ...ideaBranches(shown).map(b => b.id)]);
}

/** Id of the parent of every node. */
export function buildParents(data: IdeaTreeData, ideas: readonly FutureIdea[] = []): Map<string, string> {
  const parents = new Map<string, string>();
  for (const [parent, ids] of buildChildren(data, ideas)) for (const id of ids) parents.set(id, parent);
  return parents;
}

/** Ids of all nodes that have children, i.e. everything "Open all" should open. */
export function expandableIds(data: IdeaTreeData, ideas: readonly FutureIdea[] = []): string[] {
  return [...buildChildren(data, ideas).keys()];
}

/** Ids that must be open to show `id`. */
export function ancestorsOf(data: IdeaTreeData, id: string, ideas: readonly FutureIdea[] = []): string[] {
  const parents = buildParents(data, ideas);
  const out: string[] = [];
  for (let p = parents.get(id); p; p = parents.get(p)) out.push(p);
  return out;
}

export function layoutGraph(data: IdeaTreeData, open: ReadonlySet<string>, sizes: GraphSizes, ideas: readonly FutureIdea[] = []): GraphLayout {
  const kids = buildChildren(data, ideas);
  const futureIds = ideaBoxIds(data, ideas);
  const branchIds = new Set(ideaBranches(attachedIdeas(data, ideas)).map(b => b.id));
  const statusOf = new Map<string, TreeStatus>();
  for (const part of data.parts) {
    statusOf.set(part.id, part.status);
    for (const leaf of part.leaves as TreeLeaf[]) statusOf.set(leaf.id, leaf.status);
  }
  const partIds = new Set(data.parts.map(p => p.id));
  const kindOf = (id: string): GraphKind => (id === 'root' ? 'root' : partIds.has(id) ? 'part' : branchIds.has(id) ? 'branch' : futureIds.has(id) ? 'idea' : 'leaf');

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
    const { w, h } = sizes[sizeKind(kind)];
    const y = (d.x ?? 0) - h / 2;
    const x = colX[d.depth];
    minY = Math.min(minY, y); maxY = Math.max(maxY, y + h); maxX = Math.max(maxX, x + w);
    nodes.push({ id: d.data, kind, depth: d.depth, x, y, w, h, status: statusOf.get(d.data) ?? null, childCount: kids.get(d.data)?.length ?? 0, ideaCount: (kids.get(d.data) ?? []).filter(c => futureIds.has(c)).length, open: open.has(d.data) });
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
export function childLabel(kind: GraphKind, count: number, future = 0): string {
  const s = (n: number) => (n === 1 ? '' : 's');
  if (kind === 'branch') return `${count} idea${s(count)}`;
  const main = count - future;
  const parts: string[] = [];
  if (main > 0) parts.push(kind === 'root' ? `${main} parts` : kind === 'part' ? `${main} test${s(main)}` : `${main} follow-up${s(main)}`);
  if (future > 0) parts.push(kind === 'root' ? `${future} new branch${future === 1 ? '' : 'es'}` : `💡 ${future}`);
  return parts.join(' · ');
}

export interface RadialSizes {
  root: { w: number; h: number };
  part: { w: number; h: number };
  leaf: { w: number; h: number };
  /** Distance from the question to the ring of parts. */
  r1: number;
  /** Distance from the part ring to the first ring of tests. */
  start: number;
  /** Arc length one test takes along its ring. */
  slot: number;
  /** Tests of one level take turns on this many radii, which packs them closer. */
  lanes: number;
  /** Distance between those radii. */
  lane: number;
}

// lanes * slot is at least a card's diagonal and lane is wider than a card, so neighbours on a ring never touch.
export const wideRadial: RadialSizes = { root: { w: 260, h: 180 }, part: { w: 190, h: 100 }, leaf: { w: 176, h: 62 }, r1: 400, start: 230, slot: 48, lanes: 4, lane: 190 };
export const narrowRadial: RadialSizes = { root: { w: 190, h: 190 }, part: { w: 150, h: 96 }, leaf: { w: 150, h: 58 }, r1: 330, start: 200, slot: 42, lanes: 4, lane: 162 };

/**
 * Radial tree: the question in the centre, the parts on a ring around it, each part's tests further out in the part's own
 * direction, follow-up tests further still. The place of every box is worked out with everything open, so opening or folding
 * a box never moves another one; only the open boxes are returned.
 */
export function radialLayout(data: IdeaTreeData, open: ReadonlySet<string>, s: RadialSizes, ideas: readonly FutureIdea[] = []): GraphLayout {
  const kids = buildChildren(data, ideas);
  const futureIds = ideaBoxIds(data, ideas);
  const branchIds = new Set(ideaBranches(attachedIdeas(data, ideas)).map(b => b.id));
  const statusOf = new Map<string, TreeStatus>();
  for (const part of data.parts) {
    statusOf.set(part.id, part.status);
    for (const leaf of part.leaves as TreeLeaf[]) statusOf.set(leaf.id, leaf.status);
  }
  const partIds = new Set(data.parts.map(p => p.id));
  const kindOf = (id: string): GraphKind => (id === 'root' ? 'root' : partIds.has(id) ? 'part' : branchIds.has(id) ? 'branch' : futureIds.has(id) ? 'idea' : 'leaf');

  const bigSlot = Math.hypot(s.leaf.w, s.leaf.h) + 12;
  const partSlot = Math.hypot(s.part.w, s.part.h) + 10;
  // The ring of parts is at least large enough for its parts to sit side by side.
  const r1 = Math.max(s.r1, Math.ceil(((kids.get('root')?.length ?? data.parts.length) * partSlot * 1.08) / (2 * Math.PI)));
  const ring = (depth: number, k: number) => (depth === 0 ? 0 : depth === 1 ? r1 : depth === 2 ? r1 + s.start * k : r1 + s.start * k + s.lanes * s.lane + (depth - 3) * s.lane);
  const arc = (depth: number) => (depth === 1 ? partSlot : depth === 2 ? s.slot : bigSlot);

  const lay = (k: number) => {
    const root = hierarchy<string>('root', id => kids.get(id));
    tree<string>().nodeSize([1, 1]).separation((a, b) => (arc(a.depth) * (a.parent === b.parent ? 1 : 1.6)) / ring(a.depth, k))(root);
    const xs = root.descendants().map(d => d.x ?? 0);
    const min = Math.min(...xs);
    return { root, min, total: Math.max(...xs) - min + Math.max(arc(1) / ring(1, k), (arc(2) * 1.6) / ring(2, k)) };
  };
  // Smallest distance between the part ring and the test rings at which everything fits round the circle.
  let lo = 1, hi = 12;
  if (lay(1).total > 2 * Math.PI) {
    for (let i = 0; i < 30; i += 1) { const mid = (lo + hi) / 2; if (lay(mid).total > 2 * Math.PI) lo = mid; else hi = mid; }
  } else hi = 1;
  const { root, min, total } = lay(hi);
  const turn = (2 * Math.PI) / total;

  const lane = new Map<string, number>();
  const byDepth = new Map<number, typeof root[]>();
  for (const d of root.descendants()) byDepth.set(d.depth, [...(byDepth.get(d.depth) ?? []), d]);
  (byDepth.get(2) ?? []).sort((a, b) => (a.x ?? 0) - (b.x ?? 0)).forEach((d, i) => lane.set(d.data, i % s.lanes));

  const shown = new Set<string>(['root']);
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  root.each(d => {
    if (d.parent && !(shown.has(d.parent.data) && open.has(d.parent.data))) return;
    shown.add(d.data);
    const kind = kindOf(d.data);
    const { w, h } = s[sizeKind(kind)];
    const angle = ((d.x ?? 0) - min) * turn;
    const radius = ring(d.depth, hi) + (d.depth === 2 ? (lane.get(d.data) ?? 0) * s.lane : 0);
    const cx = radius * Math.sin(angle), cy = -radius * Math.cos(angle);
    minX = Math.min(minX, cx - w / 2); maxX = Math.max(maxX, cx + w / 2); minY = Math.min(minY, cy - h / 2); maxY = Math.max(maxY, cy + h / 2);
    nodes.push({ id: d.data, kind, depth: d.depth, x: cx - w / 2, y: cy - h / 2, w, h, status: statusOf.get(d.data) ?? null, childCount: kids.get(d.data)?.length ?? 0, ideaCount: (kids.get(d.data) ?? []).filter(c => futureIds.has(c)).length, open: open.has(d.data), angle, radius });
    if (d.parent) links.push({ id: `${d.parent.data}>${d.data}`, from: d.parent.data, to: d.data });
  });
  return { nodes, links, bounds: { minX, minY, maxX, maxY } };
}
