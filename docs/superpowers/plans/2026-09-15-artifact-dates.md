# Artifact Dates Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` for the independent implementation tasks below. Preserve others' edits; root integrates and publishes.

**Goal:** Show trustworthy artifact and research dates throughout the notebook without inventing historical creation times.

**Architecture:** Versioned, hash-bound date metadata is validated before each build and loaded alongside the research snapshot. Reusable date rows render explicit date meanings and age; existing research bytes remain unchanged.

**Tech Stack:** React, TypeScript, Node scripts, Python maintainer importer, existing Git/Vercel deployment.

## 1. Evidence and date catalog (campaign_evidence)

Files: `scripts/export_artifact_dates.py`, `scripts/sync-artifact-dates.mjs`, `tests/test_artifact_dates.py`, `tests/artifact-dates-sync.test.ts`, `content/artifact-dates.json`, `public/data/artifact-dates.json`.

- [x] Extract scheduler and exact-path Git dates into the private workspace evidence fixture, with public-safe provenance and actual hashes.
- [x] Write failing tests for strict facts, catalog coverage and content-hash mismatches, stable refreshes, and changed-record dating.
- [x] Implement one Node inventory function for artifact bytes and canonical entity JSON; expose `--inventory` for the importer.
- [x] Import the verified fixture into the agreed catalog; reuse prior facts only for matching evidence and keep unchanged record dates stable.
- [x] Validate and synchronize the catalog. Reject snapshot/run hash mismatches, unknown IDs, missing required items, bad facts and unsafe provenance links. Test `node --import tsx --test tests/artifact-dates-sync.test.ts` and Python unit cases.

## 2. Browser date semantics and loading (root)

Files: `src/lib/artifact-dates.ts`, `src/artifact-dates.tsx`, `src/main.tsx`, `tests/artifact-dates.test.ts`.

- [x] Define the spec's types and write failing tests for invalid dates, missing timezone, leap days, UTC boundaries, unknown/future ages and safe links.
- [x] Implement strict parsing, UTC formatting, and labeled relative ages. Never parse incomplete dates as precise dates.
- [x] Add `ArtifactDatesProvider` and `useArtifactDates(kind, id)`, returning `{record, catalog, loading, error}`. Check schema and current snapshot identity; date errors do not erase the research page.
- [x] Wrap existing routes in the provider. Keep all current research values and outcomes untouched.

## 3. Compact date UI (isolation_evidence)

Files: `src/components/ArtifactDates.tsx`, `common.tsx`, `CutChart.tsx`, `Layout.tsx`, `src/pages/{Figures,Experiments,Code,Ledger,Overview}.tsx`, `src/styles.css`.

- [x] Use the agreed provider/types. Compact variant shows the appropriate recorded date and age; detail variant shows all available meanings, including unknown original creation.
- [x] Add the component at every spec placement. Preserve routes, search, pagination, source provenance, colors and comparison/results content.
- [x] Verify typecheck, then browser layouts at desktop and mobile widths. Check dates in the inline CVK2 chart and CSV dialog, not only figure cards.

## 4. Maintenance, review and release (root + reviewer)

Files: `package.json`, `scripts/package_notebook.py`, README/maintenance guides, journal, portable ZIP. Review by verdict_design_check.

- [x] Add date validation/sync before the existing build, and document verified refresh commands and date meanings.
- [x] Include the public catalog, its content source and Node validator in the portable source package. Keep private evidence/importer inputs out.
- [x] Run meaningful date tests plus existing tests and required data checks; confirm scientific assets' hashes did not change.
- [x] Independently review spec compliance, edge cases and security. Resolve findings before release.
- [x] Append a dated maintainer note, synchronize both catalogs, verify portable install/test/build, and review the exact changes.
Release gate: commit the feature, integrate to `main`, and push the connected repository. Verify the actual deployed metadata, visible dates and original access protection. The release receipt is kept outside this repository so verifying a deployment does not itself trigger another deployment.

## Verification notes

- Browser layouts: desktop 1280/1664px and mobile 390px. Figure detail and CSV preview screenshots inspected; date text remains 14px, no document overflow, neutral dates and scientific verdict colors stay separate. The CVK2 chart shows First recorded 15 September and Last linked run 14 September.
- All 44 JavaScript tests passed. The original 39 Python checks had one pre-existing public-GitHub-username false positive; the narrow approved-prefix regression and actual public-text scan both pass after correction. All other 38 checks passed. The importer additionally passed its final 10-test run after the provenance wording correction. All 11 packaging unit tests passed.
- Both catalog copies validate 3,642 records. No-op refresh preserves every byte. Exact science/image/table/source/log bytes remain unchanged from the starting main revision.
- Known producer dates are available for 70 CSVs; unknown historical creation remains explicit. The reviewer’s known-creation wording contradiction was corrected without changing any date or hash.

- Final independent review: PASS, including the corrected date bases and the narrowly scoped public-repository test exception. Portable install/test/build and deterministic ZIP verification: PASS (3,459 packaged files).
