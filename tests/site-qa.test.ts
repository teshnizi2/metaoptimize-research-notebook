// Regression tests for defects found in the site QA pass of 2026-09-16.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const css = read('../src/styles.css');
/** Declarations of every rule whose selector list contains `selector` exactly. */
function declarations(selector: string): string {
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)];
  return rules.filter(([, selectors]) => selectors.split(',').map(s => s.trim()).includes(selector)).map(([, , body]) => body).join(';');
}

// ---- Back button: typing in a search box must not add one history entry per keystroke ----
test('search-box edits replace the history entry; other filter changes push one', async () => {
  const { applySearchParamChanges } = await import('../src/lib/search-params.ts');
  const typed = applySearchParamChanges(new URLSearchParams('area=Mechanism&q=vg'), { q: 'vgg' });
  assert.equal(typed.params.toString(), 'area=Mechanism&q=vgg');
  assert.equal(typed.replace, true);
  const cleared = applySearchParamChanges(new URLSearchParams('q=vgg&kind=research'), { q: '' });
  assert.equal(cleared.params.toString(), 'kind=research');
  assert.equal(cleared.replace, true);
  const filtered = applySearchParamChanges(new URLSearchParams('q=vgg'), { outcome: 'fail', kind: '' });
  assert.equal(filtered.params.toString(), 'q=vgg&outcome=fail');
  assert.equal(filtered.replace, false);
  assert.equal(applySearchParamChanges(new URLSearchParams(), { q: 'x', area: 'A' }).replace, false);
  const input = new URLSearchParams('q=a');
  applySearchParamChanges(input, { q: 'ab' });
  assert.equal(input.toString(), 'q=a', 'the current params are not mutated');
});

test('the experiment, figure and code search boxes use the history-replacing update', () => {
  for (const page of ['Experiments', 'Figures', 'Code']) {
    const source = read(`../src/pages/${page}.tsx`);
    assert.match(source, /applySearchParamChanges\(/, page);
    assert.match(source, /setParams\([^;]*\{\s*replace\s*[:}]/, page);
  }
});

// ---- Breadcrumb and document title for unknown routes ----
test('section labels match whole route segments and unknown routes are not called Overview', async () => {
  const { sectionLabel } = await import('../src/lib/navigation.ts');
  assert.equal(sectionLabel('/'), 'Overview');
  assert.equal(sectionLabel('/experiments'), 'Experiments');
  assert.equal(sectionLabel('/experiments/'), 'Experiments');
  assert.equal(sectionLabel('/experiments/CVK2'), 'Experiments');
  assert.equal(sectionLabel('/code/src-0bddfee4bf28ee84'), 'Source code');
  assert.equal(sectionLabel('/runs/run-4683077'), 'Run ledger');
  assert.equal(sectionLabel('/warnings'), 'Warnings & limits');
  assert.equal(sectionLabel('/activity'), 'Research log');
  for (const path of ['/nonexistent/deep/route', '/codex', '/runsheet', '/experiments/CVK2/extra', '/activity/phase-01', '/warnings/x'])
    assert.equal(sectionLabel(path), 'Page not found', path);
});

// ---- Screen readers must be told which register filter is active ----
test('register filter selection is exposed for every filter button', async () => {
  const { registerFilterSelection } = await import('../src/lib/research.ts');
  assert.deepEqual(registerFilterSelection({ outcome: '', kind: '' }), { all: true, research: false, outcomes: { success: false, fail: false, mixed: false, unresolved: false }, methodChecks: false });
  assert.deepEqual(registerFilterSelection({ outcome: 'fail', kind: '' }).outcomes, { success: false, fail: true, mixed: false, unresolved: false });
  assert.equal(registerFilterSelection({ outcome: '', kind: 'research' }).research, true);
  assert.equal(registerFilterSelection({ outcome: 'fail', kind: 'research' }).research, false);
  assert.equal(registerFilterSelection({ outcome: '', kind: 'method-check' }).methodChecks, true);
  const source = read('../src/pages/Experiments.tsx');
  const filters = source.slice(source.indexOf('className="outcome-filters"'), source.indexOf('className="filter-bar"'));
  const buttons = filters.match(/<button\b[^>]*>/g) || [];
  assert.equal(buttons.length, 5, 'All, Research questions, the outcome map, Method checks, Corrected');
  for (const button of buttons) assert.match(button, /aria-pressed=/, button);
  const figures = read('../src/pages/Figures.tsx');
  for (const button of figures.slice(figures.indexOf('className="detail-tabs"'), figures.indexOf('className="filter-bar"')).match(/<button\b[^>]*>/g) || [])
    assert.match(button, /aria-pressed=/, button);
});

// ---- Layout defects ----
test('ledger search fields lay the icon beside a bordered input', () => {
  const field = declarations('.filter-search');
  assert.match(field, /display:flex/);
  assert.match(field, /align-items:center/);
  assert.match(field, /border:1px solid/);
  assert.match(declarations('.filter-search input'), /border:0/);
  assert.match(declarations('.filter-search:focus-within'), /outline:/);
});

test('warning-card status pills keep their own height instead of stretching to the card', () => {
  assert.match(declarations('.warning-card>.status'), /align-self:flex-start/);
});

test('run metric cards do not restyle the execution-state pill as a block label', () => {
  assert.equal(declarations('.metric-card span'), '', 'a descendant span rule also hits the pill and its dot');
  assert.match(declarations('.metric-card>span:not(.status)'), /display:block/);
});

test('global search and run pages render editorial experiment copy', () => {
  const layout = read('../src/components/Layout.tsx');
  const ledger = read('../src/pages/Ledger.tsx');
  assert.match(layout, /experimentQuestion\(e\)/);
  assert.doesNotMatch(layout, /\{e\.goal\}/);
  assert.match(ledger, /experimentQuestion\(e\)/);
  assert.match(ledger, /experimentResult\(e\)/);
  assert.doesNotMatch(ledger, /\{e\.(?:goal|result)\}/);
});

test('research log dates and entry types are separated', () => {
  assert.match(declarations('.activity-date'), /display:flex/);
  assert.match(declarations('.activity-date'), /flex-direction:column/);
});

// ---- Typing speed: URL-bound search boxes dropped keystrokes ----
// React Router applies location changes inside startTransition by default. A search
// input whose value comes from the URL then snaps back to its previous value until
// the transition commits, so a key pressed in that window is appended to the stale
// text: typing "mt014" quickly on /warnings produced ?q=4. Location updates must be
// synchronous for these controlled inputs.
test('the router applies location updates synchronously so search inputs keep every keystroke', () => {
  const main = read('../src/main.tsx');
  assert.match(main, /<BrowserRouter\s+useTransitions=\{false\}\s*>/);
});
