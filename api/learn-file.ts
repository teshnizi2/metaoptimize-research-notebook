import { get } from '@vercel/blob';

/** Serves the Learn page's videos, subtitles and PDFs from the private Blob store.
 *  The deployment is behind the Vercel sign-in, so this route is too; the files are never public.
 *  /learn-files/<path> is rewritten here by vercel.json. Only the paths below are served. */
const ID = '(?:B(?:0[1-9]|1[0-6])|0[1-7])';
const ALLOWED: { pattern: RegExp; type: string; cache?: string }[] = [
  { pattern: new RegExp(`^video/${ID}\\.mp4$`), type: 'video/mp4' },
  { pattern: new RegExp(`^subs/${ID}\\.vtt$`), type: 'text/vtt; charset=utf-8' },
  { pattern: /^pdf\/(?:guide|glossary)\.pdf$/, type: 'application/pdf' },
  // The Idea tree's future ideas (unpublished research): one JSON file, never cached for long, never public.
  { pattern: /^ideas\/ideas\.json$/, type: 'application/json; charset=utf-8', cache: 'private, no-cache' },
];

export function learnFileType(path: string): string | null {
  return ALLOWED.find(a => a.pattern.test(path))?.type ?? null;
}

function cacheFor(path: string): string {
  return ALLOWED.find(a => a.pattern.test(path))?.cache ?? 'private, max-age=3600';
}

export async function GET(request: Request): Promise<Response> {
  const path = new URL(request.url).searchParams.get('p') ?? '';
  const type = learnFileType(path);
  if (!type) return new Response('Not found', { status: 404 });
  const range = request.headers.get('range');
  const blob = await get(path, { access: 'private', headers: range ? { Range: range } : undefined });
  if (!blob || !blob.stream) return new Response('Not found', { status: 404 });
  const headers = new Headers({
    'Content-Type': type,
    'Accept-Ranges': 'bytes',
    'Cache-Control': cacheFor(path),
    'Content-Disposition': 'inline',
    'X-Content-Type-Options': 'nosniff',
  });
  for (const name of ['content-length', 'content-range', 'etag']) {
    const value = blob.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(blob.stream, { status: headers.has('content-range') ? 206 : 200, headers });
}
