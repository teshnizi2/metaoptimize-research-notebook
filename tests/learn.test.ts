import assert from 'node:assert/strict';
import test from 'node:test';
import { LEARN_FILES, formatLength, leftOutVideos, learnVideos, mapLinks, pdfUrl, studyPath, subtitleUrl, videoUrl, videosOf } from '../src/lib/learn.ts';
import { srtToVtt } from '../src/lib/srt-to-vtt.ts';

const everyString = (): string[] => [
  ...learnVideos.flatMap(v => [v.id, v.title, v.learn]),
  ...studyPath.flatMap(s => s.kind === 'guide' ? [s.title, s.youtube?.url ?? '', s.youtube?.label ?? ''] : [s.videoId]),
  ...mapLinks.flatMap(l => [l.to, l.label, l.detail]),
];

test('the beginner series is B01 to B16 and the first series is 01 to 07, each once', () => {
  assert.deepEqual(videosOf('beginner').map(v => v.id), Array.from({ length: 16 }, (_, i) => `B${String(i + 1).padStart(2, '0')}`));
  assert.deepEqual(videosOf('first').map(v => v.id), Array.from({ length: 7 }, (_, i) => `0${i + 1}`));
  assert.equal(new Set(learnVideos.map(v => `${v.series}${v.id}`)).size, learnVideos.length);
});

test('the videos about the review paper and the status videos built on it are left out', () => {
  const ids = new Set(learnVideos.map(v => `${v.series}:${v.id}`));
  for (const gone of ['beginner:B17', 'beginner:B18', 'first:08', 'first:09']) assert.ok(!ids.has(gone), gone);
  assert.equal(leftOutVideos.length, 4);
  const text = everyString().join('\n');
  assert.doesNotMatch(text, /PaperFactory|review paper|co-author|supervisor|professor/i);
});

test('every video has a title, a one-line summary and a length', () => {
  for (const v of learnVideos) {
    assert.ok(v.title.length > 3, v.id);
    assert.ok(v.learn.length > 20 && !v.learn.includes('\n'), v.id);
    assert.ok(v.seconds > 60 && v.seconds < 600, v.id);
  }
});

test('the study path keeps the study-list order and links only to videos on the site', () => {
  const videoSteps = studyPath.flatMap(s => s.kind === 'video' ? [s.videoId] : []);
  assert.deepEqual(videoSteps, videosOf('beginner').map(v => v.id));
  assert.equal(studyPath.length, 26);
  assert.deepEqual(studyPath.slice(0, 5).map(s => s.kind), ['guide', 'guide', 'guide', 'guide', 'video']);
  const pages = studyPath.flatMap(s => s.kind === 'guide' ? [s.page] : []);
  assert.deepEqual(pages, [...pages].sort((a, b) => a - b));
  for (const s of studyPath) if (s.kind === 'guide' && s.youtube) assert.match(s.youtube.url, /^https:\/\/www\.youtube\.com\/watch\?v=[\w-]{11}$/);
});

test('no local file paths reach the page', () => {
  assert.doesNotMatch(everyString().join('\n'), /file:|\/Users\/|iCloud|\.mp4|\.srt|\.pdf|Desktop/i);
});

test('file addresses come from one setting', () => {
  assert.equal(LEARN_FILES.available, false, 'stays off until the files are really served');
  assert.equal(videoUrl('B01'), `${LEARN_FILES.base}/video/B01.mp4`);
  assert.equal(subtitleUrl('01'), `${LEARN_FILES.base}/subs/01.vtt`);
  assert.equal(pdfUrl('guide', 5), `${LEARN_FILES.base}/pdf/guide.pdf#page=5`);
  assert.equal(pdfUrl('glossary'), `${LEARN_FILES.base}/pdf/glossary.pdf`);
});

test('lengths read as minutes and seconds', () => {
  assert.equal(formatLength(302), '5:02');
  assert.equal(formatLength(196), '3:16');
});

test('srt becomes WebVTT', () => {
  const vtt = srtToVtt('﻿1\r\n00:00:01,500 --> 00:00:03,000\r\nHello, world\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,250\r\nBye\r\n');
  assert.equal(vtt, 'WEBVTT\n\n1\n00:00:01.500 --> 00:00:03.000\nHello, world\n\n2\n00:00:03.000 --> 00:00:04.250\nBye\n');
});
