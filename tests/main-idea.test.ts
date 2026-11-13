import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { findNode, ideaTree } from '../src/lib/idea-tree.ts';
import { barCharts, lineCharts, mainIdeaTests, reasons } from '../src/lib/main-idea.ts';

const root = new URL('../', import.meta.url);

function leafText(id: string): string {
  const node = findNode(ideaTree, id);
  assert.equal(node?.kind, 'leaf', `${id} is a test`);
  if (node?.kind !== 'leaf') return '';
  return `${node.leaf.title} ${node.leaf.result} ${node.leaf.summary}`;
}

function readCsv(path: string): Record<string, string>[] {
  const lines = readFileSync(new URL(path, root), 'utf8').trim().split('\n');
  const split = (line: string) => { const out: string[] = []; let cur = ''; let q = false; for (const ch of line) { if (ch === '"') q = !q; else if (ch === ',' && !q) { out.push(cur); cur = ''; } else cur += ch; } out.push(cur); return out; };
  const head = split(lines[0]);
  return lines.slice(1).map(l => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}

test('main idea bar charts: every number appears in the summary of the test it cites', () => {
  for (const chart of barCharts) {
    for (const row of chart.rows) {
      assert.ok(leafText(row.test).includes(row.quote), `${chart.id}: "${row.quote}" in ${row.test}`);
      // The number on screen, the bar length and the quoted text all agree.
      const shown = Number(row.display.replace('−', '-').replace('+', ''));
      assert.equal(Math.abs(shown), Math.abs(row.value), `${chart.id}: ${row.display}`);
      assert.ok(row.quote.includes(String(Math.abs(row.value)).replace(/^(\d)$/, '$1.0')) || row.quote.includes(String(Math.abs(row.value))), `${chart.id}: ${row.quote} holds ${row.value}`);
    }
    if (chart.reference) assert.ok(leafText(chart.reference.test).includes(chart.reference.quote), `${chart.id} reference`);
    for (const t of chart.tests) assert.ok(findNode(ideaTree, t), t);
  }
});

test('main idea line charts: every point equals the recorded table value to two decimals', () => {
  for (const chart of lineCharts) {
    const table = readCsv(chart.csv);
    const found = chart.csvRows.map(spec => {
      const hit = table.filter(r => Object.entries(spec.key).every(([k, v]) => r[k] === v));
      assert.equal(hit.length, 1, `${chart.id}: one row for ${JSON.stringify(spec.key)}`);
      return Math.round(Number(hit[0][spec.column]) * 100) / 100;
    });
    const shown = chart.id === 'dial'
      ? [...chart.series[0].points, ...chart.series[1].points, chart.references[0].value, chart.references[1].value]
      : chart.series[0].points;
    assert.deepEqual(shown, found, chart.id);
    for (const s of chart.series) assert.equal(s.points.length, chart.x.length, `${chart.id} ${s.label}`);
  }
});

test('main idea: the dial chart backs the claim that every point trails plain per-layer', () => {
  const dial = lineCharts.find(c => c.id === 'dial')!;
  dial.series.forEach((s, i) => { for (const p of s.points) assert.ok(p < dial.references[i].value); });
  // B3.5 reports the best partial pull trailing by 0.253 and 0.819 (r = 0.03, 0.1, 0.3 only).
  const gap = (i: number) => dial.references[i].value - Math.max(...dial.series[i].points.slice(1, 4));
  assert.ok(Math.abs(gap(1) - 0.25) < 0.01 && Math.abs(gap(0) - 0.82) < 0.01);
});

test('main idea: reasons cite tests and parts that exist, and the cards cover the six tests of Part 3', () => {
  assert.equal(reasons.length, 4);
  for (const r of reasons) { assert.ok(r.evidence.length > 0, r.title); for (const e of r.evidence) assert.ok(findNode(ideaTree, e.id), e.id); }
  for (const id of ['B3.1', 'B3.2', 'B3.3', 'B3.4', 'B3.5', 'B3.6', 'B1.2']) assert.ok(mainIdeaTests.includes(id), id);
  assert.ok(ideaTree.parts.every(p => p.id !== 'B11'), 'no Part 11 on the public site');
});
