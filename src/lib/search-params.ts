/**
 * Apply filter changes to a URL query. Typing in a search box updates `q` on every
 * keystroke; those edits replace the current history entry so the browser Back button
 * leaves the page instead of stepping back one letter at a time. Any other filter
 * change pushes a new entry.
 */
export function applySearchParamChanges(current: URLSearchParams, changes: Record<string, string>) {
  const params = new URLSearchParams(current);
  for (const [key, value] of Object.entries(changes)) value ? params.set(key, value) : params.delete(key);
  const keys = Object.keys(changes);
  return { params, replace: keys.length > 0 && keys.every(key => key === 'q') };
}
