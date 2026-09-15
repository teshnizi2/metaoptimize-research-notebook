import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useResearch } from './data';
import { artifactDateKey, parseArtifactDateCatalog } from './lib/artifact-dates';
import type { ArtifactDateCatalog, ArtifactKind } from './lib/artifact-dates';

interface DateState { catalog: ArtifactDateCatalog | null; loading: boolean; error: string }
const Context = createContext<DateState>({ catalog: null, loading: false, error: 'Date metadata is unavailable.' });

export function ArtifactDatesProvider({ children }: { children: ReactNode }) {
  const { meta } = useResearch();
  const [state, setState] = useState<DateState>({ catalog: null, loading: true, error: '' });
  useEffect(() => {
    const controller = new AbortController();
    setState({ catalog: null, loading: true, error: '' });
    fetch('/data/artifact-dates.json', { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Date metadata could not be loaded.'); return response.json(); })
      .then(value => {
        if (controller.signal.aborted) return;
        const catalog = parseArtifactDateCatalog(value, meta.snapshotId);
        if (catalog.publishedAt !== meta.asOf || Date.parse(catalog.exportedAt) !== Date.parse(meta.generatedAt)) throw new Error('Date metadata is out of sync with this snapshot.');
        setState({ catalog, loading: false, error: '' });
      })
      .catch(error => { if (!controller.signal.aborted) setState({ catalog: null, loading: false, error: error instanceof Error ? error.message : 'Date metadata is unavailable.' }); });
    return () => controller.abort();
  }, [meta.snapshotId, meta.asOf, meta.generatedAt]);
  return <Context.Provider value={state}>{children}</Context.Provider>;
}

export function useArtifactDates(kind: ArtifactKind, id: string) {
  const state = useContext(Context);
  const record = state.catalog?.records[artifactDateKey(kind, id)] ?? null;
  return { ...state, record, error: state.error || (!state.loading && !record ? 'Date not recorded.' : '') };
}
