import test from 'node:test';
import assert from 'node:assert/strict';
import { ageLabel, artifactDateKey, formatDateFact, parseArtifactDateCatalog, safeDateHref } from '../src/lib/artifact-dates.ts';
import type { DateFact } from '../src/lib/artifact-dates.ts';

const day = (at: string): DateFact => ({ at, precision: 'day', basis: 'Recorded calendar date' });
const second = (at: string): DateFact => ({ at, precision: 'second', basis: 'Recorded scheduler timestamp' });
const catalog = () => ({ schemaVersion: 1, snapshotId: 'snapshot-A', snapshotSha256: 'a'.repeat(64), runsSha256: 'b'.repeat(64), publishedAt: '2026-09-15', exportedAt: '2026-09-15T13:11:13.784665+00:00', recordedAt: '2026-09-15T14:00:00Z', records: {
  'figure:page-22': { kind: 'figure', id: 'page-22', contentSha256: 'c'.repeat(64), experimentIds: ['CVK2'], firstRecorded: day('2026-09-15') },
} });

test('day precision stays a calendar day and offset-qualified timestamps display in UTC', () => {
  assert.equal(formatDateFact(day('2026-09-14')), '14 Sept 2026');
  assert.equal(formatDateFact(second('2026-09-15T00:30:00+02:00')), '14 Sept 2026');
  assert.equal(formatDateFact(second('2026-09-15T00:30:00.123456789+02:00')), '14 Sept 2026');
  assert.equal(formatDateFact(second('2026-09-15T00:30:00+14:00')), '14 Sept 2026');
  assert.equal(formatDateFact(day('2024-02-29')), '29 Feb 2024');
});

test('unknown, partial and impossible dates are never normalized into invented dates', () => {
  for (const fact of [undefined, null, day('2026-02-29'), day('2026-02-30'), day('2026-09'), day('2026'), day('2026-09-15T00:00:00Z'), second('2026-09-15T00:00:00'), second('2026-02-30T12:00:00Z'), second('2026-09-15T24:00:00Z'), second('2026-09-15T00:00:00+14:01'), second('2026-09-15T00:00:00-15:00')]) {
    assert.equal(formatDateFact(fact), 'Not recorded');
    assert.equal(ageLabel(fact, '2026-09-15T12:00:00Z'), null);
  }
});

test('age follows UTC calendar boundaries and unknown dates never appear recent', () => {
  assert.equal(ageLabel(day('2026-09-15'), '2026-09-15T12:00:00Z'), 'Today');
  assert.equal(ageLabel(second('2026-09-15T00:30:00+02:00'), '2026-09-15T12:00:00Z'), '1 day ago');
  assert.equal(ageLabel(day('2026-08-15'), '2026-09-15T12:00:00Z'), '31 days ago');
  assert.equal(ageLabel(day('2026-09-16'), '2026-09-15T12:00:00Z'), 'Date is in the future');
  assert.equal(ageLabel(day('2026-09-15'), 'invalid clock'), null);
});

test('date provenance links stay within the notebook or the two known public repositories', () => {
  const github = 'https://github.com/teshnizi2/hierarchical-metaoptimize/commit/627d69ffd624d768178719b4c5b52b0e3b0e9ed5';
  const credentials = new URL(github);
  credentials.username = 'fixture-user'; credentials.password = 'fixture-password';
  assert.equal(safeDateHref(github), github);
  assert.equal(safeDateHref('/runs/run-5004252'), '/runs/run-5004252');
  for (const href of [undefined, '', '//example.org/x', '/\\example.org/x', 'javascript:alert(1)', 'https://example.org', 'https://github.com.evil.test/teshnizi2/hierarchical-metaoptimize', 'https://github.com/other/repo', credentials.href, '/data/../secret']) {
    assert.equal(safeDateHref(href), null);
  }
});

test('catalog parsing keeps unknown creation distinct from the recorded and exported dates', () => {
  const result = parseArtifactDateCatalog(catalog(), 'snapshot-A');
  assert.equal(result.records[artifactDateKey('figure', 'page-22')].created, undefined);
  assert.equal(result.records['figure:page-22'].firstRecorded?.at, '2026-09-15');
  assert.equal(result.publishedAt, '2026-09-15');
});

test('wrong snapshots, malformed records, bad dates and misleading coverage are rejected', () => {
  assert.throws(() => parseArtifactDateCatalog(catalog(), 'snapshot-B'), /snapshot/i);
  for (const mutate of [
    (c: any) => { c.schemaVersion = 2; },
    (c: any) => { c.records = {}; },
    (c: any) => { c.records['figure:page-22'].kind = 'run'; },
    (c: any) => { c.records['figure:page-22'].firstRecorded.at = '2026-02-30'; },
    (c: any) => { c.records['figure:page-22'].contentSha256 = 'unknown'; },
    (c: any) => { c.records['figure:page-22'].firstRecorded.href = 'javascript:alert(1)'; },
    (c: any) => { c.records['figure:page-22'].runWindow = { started: day('2026-09-10'), finished: day('2026-09-14'), datedRuns: 28, totalRuns: 27 }; },
    (c: any) => { c.records['figure:page-22'].runWindow = { started: day('2026-09-15'), finished: day('2026-09-14'), datedRuns: 27, totalRuns: 27 }; },
  ]) {
    const bad = catalog(); mutate(bad);
    assert.throws(() => parseArtifactDateCatalog(bad, 'snapshot-A'));
  }
});
