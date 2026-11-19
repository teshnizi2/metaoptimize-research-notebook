/**
 * The site's structure in one table: the sidebar groups, their entries, the sub-tabs inside an entry,
 * the breadcrumb and the page titles all read from here. To add a page, add its route to `sections`
 * and put it in a group (as an entry, or as a tab of an entry). See docs/MAINTENANCE.md.
 */

export type GroupId = 'start' | 'results' | 'progress' | 'record';

/** Top-level sections; `detail` sections also own a `/:id` route. */
export const sections = [
  { to: '/', label: 'Overview', detail: false, group: 'start' },
  { to: '/learn', label: 'Learn', detail: false, group: 'start' },
  { to: '/achievements', label: 'What we achieved', detail: false, group: 'results' },
  { to: '/main-idea', label: 'The main idea', detail: false, group: 'results' },
  { to: '/better', label: 'Making it better', detail: false, group: 'results' },
  { to: '/weekly', label: 'Weekly record', detail: true, group: 'progress' },
  { to: '/activity', label: 'Research log', detail: false, group: 'progress' },
  { to: '/idea-tree', label: 'Idea tree', detail: false, group: 'record' },
  { to: '/experiments', label: 'Experiments', detail: true, group: 'record' },
  { to: '/warnings', label: 'Warnings & limits', detail: false, group: 'record' },
  { to: '/evidence', label: 'Evidence', detail: false, group: 'record' },
  { to: '/figures', label: 'Figures & tables', detail: true, group: 'record' },
  { to: '/runs', label: 'Run ledger', detail: true, group: 'record' },
  { to: '/code', label: 'Source code', detail: true, group: 'record' },
] as const satisfies readonly { to: string; label: string; detail: boolean; group: GroupId }[];

export interface NavTab { to: string; label: string }
export interface NavEntry { to: string; label: string; icon: string; tabs?: NavTab[] }
export interface NavGroup { id: GroupId; label: string; blurb: string; entries: NavEntry[] }

/** The sidebar: 9 entries in 4 groups. An entry with `tabs` stays highlighted on every tab's route. */
export const groups: NavGroup[] = [
  { id: 'start', label: 'Start', blurb: 'This page, and Learn: the guide, the glossary, the videos and a study path in order.', entries: [
    { to: '/', label: 'Overview', icon: 'overview' },
    { to: '/learn', label: 'Learn', icon: 'learn' },
  ] },
  { id: 'results', label: 'Results', blurb: 'What the project can stand behind, why the main idea lost, and each round of making it better.', entries: [
    { to: '/achievements', label: 'What we achieved', icon: 'achievements' },
    { to: '/main-idea', label: 'The main idea', icon: 'main-idea' },
    { to: '/better', label: 'Making it better', icon: 'better' },
  ] },
  { id: 'progress', label: 'Progress', blurb: 'One page per week, and the dated research log from before the weekly pages.', entries: [
    { to: '/weekly', label: 'Weekly record', icon: 'weekly', tabs: [
      { to: '/weekly', label: 'Weeks' },
      { to: '/activity', label: 'Research log' },
    ] },
  ] },
  { id: 'record', label: 'Research record', blurb: 'The map of every test, every experiment with its warnings, and the evidence behind them: figures, tables, runs and code.', entries: [
    { to: '/idea-tree', label: 'Idea tree', icon: 'idea-tree' },
    { to: '/experiments', label: 'Experiments', icon: 'experiments', tabs: [
      { to: '/experiments', label: 'Questions' },
      { to: '/warnings', label: 'Warnings & limits' },
    ] },
    { to: '/figures', label: 'Evidence', icon: 'evidence', tabs: [
      { to: '/figures', label: 'Figures & tables' },
      { to: '/runs', label: 'Run ledger' },
      { to: '/code', label: 'Source code' },
    ] },
  ] },
];

export const NOT_FOUND_LABEL = 'Page not found';

/** The section a path belongs to, matched on whole route segments; undefined for unknown routes. */
function sectionOf(pathname: string) {
  const path = pathname.replace(/\/+$/, '') || '/';
  const [, first = '', second, ...rest] = path.split('/');
  const section = sections.find(s => s.to === `/${first}`);
  if (!section) return undefined;
  if (second === undefined) return section;
  return section.detail && second !== '' && rest.length === 0 ? section : undefined;
}

/** The breadcrumb and title label for a path, matched on whole route segments. */
export function sectionLabel(pathname: string): string {
  return sectionOf(pathname)?.label ?? NOT_FOUND_LABEL;
}

/** The sidebar group a path belongs to (its name is the page's eyebrow and the breadcrumb's first part). */
export function groupOf(pathname: string): NavGroup | undefined {
  const section = sectionOf(pathname);
  return section && groups.find(g => g.id === section.group);
}

/** The sidebar entry to highlight for a path: the entry itself, or the entry whose tabs hold the path's section. */
export function navEntryOf(pathname: string): NavEntry | undefined {
  const section = sectionOf(pathname);
  if (!section) return undefined;
  const to = section.to === '/evidence' ? '/figures' : section.to;
  for (const g of groups) for (const e of g.entries) {
    if (e.to === to || e.tabs?.some(t => t.to === to)) return e;
  }
  return undefined;
}

/** The sub-tabs to show on a path: only on a tab's own index page, not on its detail pages. */
export function sectionTabs(pathname: string): { entry: NavEntry; tabs: NavTab[] } | undefined {
  const path = pathname.replace(/\/+$/, '') || '/';
  for (const g of groups) for (const e of g.entries) {
    if (e.tabs?.some(t => t.to === path)) return { entry: e, tabs: e.tabs };
  }
  return undefined;
}
