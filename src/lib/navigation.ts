/** Top-level sections; `detail` sections also own a `/:id` route. */
export const sections = [
  { to: '/', label: 'Overview', detail: false },
  { to: '/experiments', label: 'Experiments', detail: true },
  { to: '/figures', label: 'Figures & tables', detail: true },
  { to: '/code', label: 'Source code', detail: true },
  { to: '/runs', label: 'Run ledger', detail: true },
  { to: '/warnings', label: 'Warnings & limits', detail: false },
  { to: '/activity', label: 'Research log', detail: false },
] as const;

export const NOT_FOUND_LABEL = 'Page not found';

/** The breadcrumb and title label for a path, matched on whole route segments. */
export function sectionLabel(pathname: string): string {
  const path = pathname.replace(/\/+$/, '') || '/';
  const [, first = '', second, ...rest] = path.split('/');
  const section = sections.find(s => s.to === `/${first}`);
  if (!section) return NOT_FOUND_LABEL;
  if (second === undefined) return section.label;
  return section.detail && second !== '' && rest.length === 0 ? section.label : NOT_FOUND_LABEL;
}
