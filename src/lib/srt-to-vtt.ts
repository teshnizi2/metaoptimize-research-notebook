/** Converts SubRip subtitles to WebVTT: a header, and a dot instead of a comma in each timestamp. */
export function srtToVtt(srt: string): string {
  const body = srt.replace(/^﻿/, '').replace(/\r\n?/g, '\n').trim()
    .replace(/(\d{2}:\d{2}:\d{2}),(\d{3})/g, '$1.$2');
  return `WEBVTT\n\n${body}\n`;
}
