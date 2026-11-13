/** The Idea tree page keeps its state in the address: `view` (radial, tree = left to right, list, changed), `node` and `keys`.
 *  Choosing a box must change `node` and nothing else, so a click never throws you into another view. */
export type TreeView = 'radial' | 'tree' | 'list';
export type PageView = TreeView | 'changed';

export const PAGE_VIEWS: PageView[] = ['radial', 'tree', 'list', 'changed'];

/** The view named in the address, or the default: radial, and left to right on a phone. */
export function viewOf(params: URLSearchParams, phone: boolean): PageView {
  const asked = params.get('view');
  return (PAGE_VIEWS as string[]).includes(asked ?? '') ? (asked as PageView) : phone ? 'tree' : 'radial';
}

/** The same address with only `node` changed (removed when `id` is null). */
export function withNode(params: URLSearchParams, id: string | null): URLSearchParams {
  const next = new URLSearchParams(params);
  if (id) next.set('node', id); else next.delete('node');
  return next;
}

/** The same address with another view; a view change drops the one-off `keys` start. */
export function withView(params: URLSearchParams, view: PageView): URLSearchParams {
  const next = new URLSearchParams(params);
  next.set('view', view);
  next.delete('keys');
  return next;
}

/** The address of a test in the tree for a link from another page: the default view opens (no `view`). A link from a tree view passes that view. */
export function treeAddress(id: string, view?: PageView): string {
  const q = new URLSearchParams();
  if (view) q.set('view', view);
  q.set('node', id);
  return `/idea-tree?${q.toString()}`;
}
