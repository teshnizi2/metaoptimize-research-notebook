# General Research Homepage Implementation Plan

> Use the existing authorized subagent workflow. Root owns aggregation, tests and release; the UI worker owns the homepage and its styles. Preserve other agents' changes.

**Goal:** Replace the homepage's single-experiment emphasis with a general visual review of all research.

**Architecture:** Derive overall and per-area outcome counts from the unchanged experiment register. Reuse dates, outcome colors, navigation and detailed experiment plots. Link compact, genuinely dated notebook updates to the existing log.

**Tech Stack:** Existing React/TypeScript, Node test runner and Vite; GitHub/Vercel publication.

## Tasks

- [x] Root: write failing tests in `tests/overview.test.ts` for `areaOutcomeCounts(experiments, areas)` and `recentNotebookUpdates(activity, limit = 4)` from `src/lib/overview.ts`. Area summaries have `{label, total, counts}`; counts use all five existing Outcome keys. Test real grouping, stale supplied counts, unlisted/empty areas, genuine note/correction/warning selection, valid UTC dates, ordering, limits and immutable inputs.
- [x] Root: implement the two minimal pure helpers; run `npx tsx --test tests/overview.test.ts` and confirm green.
- [x] UI worker: update `src/pages/Overview.tsx` and its scoped CSS in `src/styles.css`. Remove the hard-coded recent experiment table and CVK2 feature. Render all-research counts, area/outcome links and dated updates from the helper. Retain header/downloads/date components. Keep the CVK2 experiment component unchanged.
- [x] Root: run JavaScript tests, date catalog checks, production build and browser checks. Inspect all five outcome labels/counts, the area sum, area/outcome filters, event links, desktop/mobile overflow and CVK2 detail access. Confirm scientific assets and date catalogs have no diff.
- [x] Independent reviewer: assess the integrated data/UI semantics and exact changes; resolve concrete defects.
- [x] Root: append a concise journal note, sync it, regenerate and verify the portable source package. Integration and production verification follow the release gate below; the receipt is stored outside Git to avoid a verification-only redeployment.

Release gate: push connected main, wait for the actual deployment, verify live overview content and protected downloads, and report the corrected homepage concisely.

Helper verification: all five tests failed before implementation and pass afterward. Independent helper/spec review passed; the full register totals are 27 Goal met, 30 Goal missed, 20 Mixed, 11 Open and 23 Corrected.

Integrated verification: all 49 JavaScript tests and the production build passed. Independent review validated all 59 summary/filter URLs and four then-current update links. Desktop 1280px and mobile 390px show labeled counts with no document overflow; the mobile data table scrolls within its own region. Science assets, dates and the specific CVK2 chart remain unchanged.

Portable package: PASS, including clean install, all 49 tests and production build; 3,461 packaged files. Actual Baseline comparisons → Goal met link opens the correct four experiment rows. The new homepage log link opens its matching dated entry.
