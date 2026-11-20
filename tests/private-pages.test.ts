import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { learnFileType } from '../api/learn-file.ts';
import { sectionLabel, NOT_FOUND_LABEL } from '../src/lib/navigation.ts';
import {
  axisFor, BETTER_URL, blocks, CHART_NAME, chartUrl, fmt, inline, parseBetter, parseWeek, parseWeekIndex, pct, readItem, stamp,
  WEEKLY_INDEX_URL, weekUrl, type IntervalChartData,
} from '../src/lib/private-pages.ts';

/** Made-up placeholders only: the real text, numbers and charts are unpublished research and never go in this repository. */
const better = {
  title: 'Placeholder page', updated: 'Placeholder date', intro: ['Placeholder intro.', 7, ''],
  words: [{ term: 'Term', meaning: 'Meaning' }, { term: 'No meaning' }],
  rounds: [
    {
      id: 'r1', name: 'Placeholder round', dates: 'Day 1', tried: ['Placeholder step', null], why: '',
      chart: { title: 'Placeholder chart', axis: 'Axis', band: { from: -0.5, label: 'Band' }, items: [
        { label: 'A', value: 0.2, lo: -0.1, hi: 0.5, tone: 'hit' }, { label: 'B', value: -0.3, lo: -1, hi: 0.4 },
        { label: 'C', value: -0.9, lo: -1.4, hi: -0.4 }, { label: 'Bad', value: 'x', lo: 0, hi: 1 }, { value: 1, lo: 0, hi: 2 },
      ] },
      table: { caption: 'Placeholder table', columns: ['Col 1', 'Col 2'], rows: [['a', 'b'], ['only one'], 'junk'] },
      verdicts: [{ tone: 'miss', text: 'Placeholder **verdict**' }, { tone: 'odd', text: 'Unknown tone' }, { text: '' }],
      misses: ['Placeholder miss'], reading: 'Placeholder reading',
    },
    { id: '', name: 'No id' }, 'junk', null,
  ],
  standing: { headline: '', bullets: ['One'], next: ['Next'], note: 'Note' },
};

test('the "Making it better" file is read defensively', () => {
  const b = parseBetter(better)!;
  assert.equal(b.title, 'Placeholder page');
  assert.deepEqual(b.intro, ['Placeholder intro.', '7']);
  assert.deepEqual(b.words, [{ term: 'Term', meaning: 'Meaning' }]);
  assert.equal(b.rounds.length, 1);
  const r = b.rounds[0];
  assert.deepEqual(r.tried, ['Placeholder step']);
  assert.deepEqual(r.chart!.items.map(i => i.label), ['A', 'B', 'C']);
  assert.equal(r.chart!.items[1].tone, 'plain');
  assert.deepEqual(r.table!.rows, [['a', 'b'], ['only one', '']]);
  assert.deepEqual(r.verdicts.map(v => v.tone), ['miss', 'plain']);
  assert.equal(b.standing!.headline, 'Where things stand now');
});

test('the optional figures section is read defensively and old files are unchanged', () => {
  assert.equal(parseBetter(better)!.figures, null);
  const withFigures = parseBetter({
    ...better,
    figures: { headline: 'Placeholder pictures', items: [
      { title: 'One', text: 'Placeholder text', image: 'a-chart-1', caption: 'Placeholder caption' },
      { title: 'Bad name', image: 'a/b.png' }, { title: '', image: 'ok' }, { image: 'ok' }, 'junk',
    ] },
  })!;
  assert.deepEqual(withFigures.figures, { headline: 'Placeholder pictures', items: [{ title: 'One', text: 'Placeholder text', image: 'a-chart-1', caption: 'Placeholder caption' }] });
  assert.equal(parseBetter({ ...better, figures: { items: [] } })!.figures, null);
  assert.equal(parseBetter({ ...better, figures: 'x' })!.figures, null);
});

test('nothing usable gives null, never an error', () => {
  for (const bad of [null, undefined, 'text', 3, [], {}, { rounds: [] }, { rounds: [{ id: 'x' }] }]) assert.equal(parseBetter(bad), null);
  for (const bad of [null, 'x', [], {}, { id: 'not-a-date', title: 'T' }, { id: '2099-01-02' }]) assert.equal(parseWeek(bad), null);
  assert.deepEqual(parseWeekIndex(null), []);
  assert.deepEqual(parseWeekIndex({ weeks: 'x' }), []);
});

const week = {
  id: '2099-01-02', title: 'Placeholder week', range: 'Range', oneline: 'One line', updated: 'Day 3 10:00 (week still open)', open: true,
  summary: ['First', 'Second', ''], achievements: ['Placeholder result', 7, ''],
  questions: [{ id: 'q1', title: 'Placeholder question', said: [{ when: 'Day 1', text: 'Placeholder words' }, { when: 'x', text: '' }], answer: 'Placeholder answer', tables: [{ columns: ['A'], rows: [['1']] }, { columns: [], rows: [] }], charts: [{ name: 'a-chart', caption: 'Cap' }, { name: '../../secret', caption: 'Bad' }, { name: 'A Chart' }],
    details: 'Placeholder detail', detailTables: [{ columns: ['B'], rows: [['2']] }, 'x'], detailCharts: [{ name: 'b-chart', caption: 'More' }, { name: 'b.png' }] }, { title: '' }, 5],
  tasks: [{ title: 'Placeholder task', question: 'Q?', why: 'Why', decisions: '- one\n- two', outcome: 'Done' }],
};

test('a week is read defensively, with chart names limited to safe file names', () => {
  const w = parseWeek(week)!;
  assert.equal(w.open, true);
  assert.deepEqual(w.summary, ['First', 'Second']);
  assert.equal(w.questions.length, 1);
  const q = w.questions[0];
  assert.deepEqual(q.said, [{ when: 'Day 1', text: 'Placeholder words' }]);
  assert.equal(q.tables.length, 1);
  assert.deepEqual(q.charts, [{ name: 'a-chart', caption: 'Cap' }]);
  assert.equal(q.details, 'Placeholder detail');
  assert.equal(q.detailTables.length, 1);
  assert.deepEqual(q.detailCharts, [{ name: 'b-chart', caption: 'More' }]);
  assert.deepEqual(w.achievements, ['Placeholder result', '7']);
  assert.equal(w.tasks[0].id, 't1');
  assert.deepEqual([w.tasks[0].details, w.tasks[0].detailTables, w.tasks[0].detailCharts], ['', [], []]);
  assert.equal(stamp(w.updated), 'Day 3 10:00');
  assert.equal(stamp('Day 3 10:00'), 'Day 3 10:00');
  assert.ok(CHART_NAME.test('w3-t55-finals') && !CHART_NAME.test('../x') && !CHART_NAME.test('a.png') && !CHART_NAME.test('A'));
});

test('the index lists the newest week first and drops entries without a date id', () => {
  const idx = parseWeekIndex({ weeks: [{ id: '2099-01-02', title: 'B', questions: 2, tasks: 3, achievements: 4 }, { id: '2099-01-09', title: 'C', open: true }, { id: 'x', title: 'No' }, { id: '2098-12-26', title: 'A' }] });
  assert.deepEqual(idx.map(w => [w.id, w.questions, w.tasks, w.achievements, w.open]), [['2099-01-09', 0, 0, 0, true], ['2099-01-02', 2, 3, 4, false], ['2098-12-26', 0, 0, 0, false]]);
  assert.deepEqual(parseWeekIndex([{ id: '2099-01-02', title: 'Bare list' }]).map(w => w.id), ['2099-01-02']);
});

test('small markdown becomes paragraphs, lists, bold and code, and never raw HTML', () => {
  assert.deepEqual(blocks('One\n\n- a\n- b\nTwo'), [{ kind: 'p', text: 'One' }, { kind: 'ul', items: ['a', 'b'] }, { kind: 'p', text: 'Two' }]);
  assert.deepEqual(inline('a **b** `c` d'), [{ kind: 'text', text: 'a ' }, { kind: 'bold', text: 'b' }, { kind: 'text', text: ' ' }, { kind: 'code', text: 'c' }, { kind: 'text', text: ' d' }]);
  assert.deepEqual(inline('<img src=x onerror=alert(1)>'), [{ kind: 'text', text: '<img src=x onerror=alert(1)>' }]);
  assert.deepEqual(inline('**'), [{ kind: 'text', text: '**' }]);
});

test('the interval chart: numbers with a real minus sign, a shared axis and the reading of each row', () => {
  assert.equal(fmt(0.571), '+0.57');
  assert.equal(fmt(-0.33), '−0.33');
  assert.equal(fmt(0), '0.00');
  assert.equal(fmt(-0), '−0.00');
  const chart = parseBetter(better)!.rounds[0].chart as IntervalChartData;
  const axis = axisFor(chart);
  assert.ok(axis.min <= -1.4 && axis.max >= 0.5 && axis.min < axis.max);
  assert.ok(axis.ticks.includes(0) && axis.ticks[0] === axis.min && axis.ticks.at(-1) === axis.max);
  assert.equal(Math.round(pct(axis.min, axis)), 0);
  assert.equal(Math.round(pct(axis.max, axis)), 100);
  assert.deepEqual(chart.items.map(i => readItem(i, chart)), ['within', 'mean-only', 'misses']);
  const zero: IntervalChartData = { ...chart, band: null, zero: true };
  assert.deepEqual(zero.items.map(i => readItem(i, zero)), ['includes-zero', 'includes-zero', 'below']);
  assert.equal(readItem(chart.items[0], { ...chart, band: null, zero: false }), null);
});

test('the private route serves the new files and nothing near them', () => {
  assert.match(learnFileType('better/better.json') ?? '', /json/);
  assert.match(learnFileType('weekly/index.json') ?? '', /json/);
  assert.match(learnFileType('weekly/2099-01-02.json') ?? '', /json/);
  assert.equal(learnFileType('weekly/charts/w3-t55-finals.png'), 'image/png');
  for (const bad of [
    'better.json', 'better/other.json', 'better/better.json/x', 'better/../video/B01.mp4', 'weekly/other.json', 'weekly/2099-1-2.json',
    'weekly/2099-01-02.json.bak', 'weekly/charts/x.PNG', 'weekly/charts/x.png/y', 'weekly/charts/../index.json', 'weekly/charts/a_b.png', 'weekly/charts/.png',
    'weekly/charts/x.svg', 'weekly/charts/x.html', 'weekly//index.json',
  ]) assert.equal(learnFileType(bad), null, bad);
  assert.equal(BETTER_URL, '/learn-files/better/better.json');
  assert.equal(WEEKLY_INDEX_URL, '/learn-files/weekly/index.json');
  assert.equal(weekUrl('2099-01-02'), '/learn-files/weekly/2099-01-02.json');
  assert.equal(chartUrl('a-chart'), '/learn-files/weekly/charts/a-chart.png');
});

test('the two sections have labels, and a week is a detail route', () => {
  assert.equal(sectionLabel('/better'), 'Making it better');
  assert.equal(sectionLabel('/weekly'), 'Weekly record');
  assert.equal(sectionLabel('/weekly/2099-01-02'), 'Weekly record');
  assert.equal(sectionLabel('/weekly/2099-01-02/x'), NOT_FOUND_LABEL);
  assert.equal(sectionLabel('/better/x'), NOT_FOUND_LABEL);
});

const root = path.resolve(import.meta.dirname, '..');
function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === '.git') continue;
    const file = path.join(dir, name);
    if (statSync(file).isDirectory()) walk(file, out); else out.push(file);
  }
  return out;
}

test('the real weekly text, numbers and charts are not in the repository', () => {
  const files = ['content', 'public', 'src', 'api', 'scripts', 'docs'].flatMap(d => { try { return walk(path.join(root, d)); } catch { return []; } });
  assert.deepEqual(files.filter(f => /(?:^|[\\/])(?:\d{4}-\d{2}-\d{2}|better|weekly|w\d-[a-z0-9-]+|fig-t\d+[a-z0-9-]*)\.(?:json|png|md)$/i.test(f)), [], 'a private content file must not be committed');
});

test('both pages show an empty state when the private content cannot be fetched', () => {
  const read = (f: string) => readFileSync(path.join(root, f), 'utf8');
  assert.match(read('src/pages/Better.tsx'), /!data \? <PrivateEmpty/);
  assert.match(read('src/pages/Weekly.tsx'), /weeks\.length === 0 \? <PrivateEmpty/);
  assert.match(read('src/pages/Weekly.tsx'), /!week \? \(valid/);
  assert.doesNotMatch(read('src/components/NotebookParts.tsx'), /dangerouslySetInnerHTML/);
  assert.doesNotMatch(read('src/pages/Weekly.tsx') + read('src/pages/Better.tsx'), /dangerouslySetInnerHTML/);
});

// The hosted build leaves out scripts/*.py (see .vercelignore), so this check runs where the script exists and says so where it does not.
const exportScript = path.join(root, 'scripts/weekly-export.py');
test('the weekly export reads a data file, keeps the index current and leaves out money, limits and file paths', { skip: existsSync(exportScript) ? false : 'scripts/weekly-export.py is not in this copy (excluded from the hosted build)' }, () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'weekly-export-'));
  const src = path.join(dir, '2099-01-02.md');
  writeFileSync(src, [
    'title: Placeholder week', 'range: Day 1 to Day 8.', 'oneline: Placeholder line.', 'updated: Day 8 09:00 (week closed)',
    'summary:', '- First thing.', '- Second thing. The plan cost 3% of the usage limit.', '- Third thing.',
    'achievements:', '- Placeholder result.', '- Paid $5 for it.', '',
    '## Q | Placeholder question', 'said: Day 1 | Placeholder words', 'answer: A real answer. The notes are in ~/secret/notes.md. Another real sentence.',
    'table: Placeholder table', '| Name | Value |', '|---|---|', '| keep | 1 |', '| price | $5 |',
    'details: Placeholder detail. The budget was 10 hours.', 'dtable: Placeholder detail table', '| Name | Value |', '|---|---|', '| keep | 2 |',
    'dchart: w3-spend | Spend chart', 'dchart: ok-detail | Fine detail chart', '',
    '## T | Placeholder task', 'question: Q?', 'outcome: - Kept line', '- The budget was 10 hours', '- The mother said yes', 'chart: w3-gpu | Cap chart', 'chart: ok-chart | Fine chart',
  ].join('\n'));
  const out = path.join(dir, 'out');
  execFileSync('python3', ['-I', path.join(root, 'scripts/weekly-export.py'), src, out]);
  const w = JSON.parse(readFileSync(path.join(out, '2099-01-02.json'), 'utf8'));
  assert.equal(w.id, '2099-01-02');
  assert.deepEqual(w.summary, ['First thing.', 'Second thing.', 'Third thing.']);
  assert.equal(w.questions[0].answer, 'A real answer. Another real sentence.');
  assert.deepEqual(w.questions[0].tables[0].rows, [['keep', '1']]);
  assert.deepEqual(w.achievements, ['Placeholder result.']);
  assert.equal(w.questions[0].details, 'Placeholder detail.');
  assert.deepEqual(w.questions[0].detailTables[0].rows, [['keep', '2']]);
  assert.deepEqual(w.questions[0].detailCharts.map((c: { name: string }) => c.name), ['ok-detail']);
  assert.equal(w.tasks[0].outcome, '- Kept line\n- The lead chat said yes');
  assert.deepEqual(w.tasks[0].charts.map((c: { name: string }) => c.name), ['ok-chart']);
  assert.doesNotMatch(JSON.stringify(w), /usage limit|~\/|\$5|budget|mother/i);
  const idx = JSON.parse(readFileSync(path.join(out, 'index.json'), 'utf8'));
  assert.deepEqual(idx.weeks.map((x: { id: string; questions: number; tasks: number; achievements: number }) => [x.id, x.questions, x.tasks, x.achievements]), [['2099-01-02', 1, 1, 1]]);
  assert.ok(parseWeek(w) && parseWeekIndex(idx).length === 1, 'the site reads what the script writes');
});
