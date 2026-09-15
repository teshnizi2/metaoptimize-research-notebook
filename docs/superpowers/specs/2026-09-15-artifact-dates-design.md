# Dates on research artifacts

The user wants visible creation dates so readers can judge when figures and other evidence may be old. Keep the existing compact research-group interface, public code repositories, and protected website.

## Date meanings

- Show actual scheduler submission, start and finish dates for runs. The retained accounting receipt establishes the historical offset; retain offset-qualified timestamps and display UTC consistently.
- Show original source files' first Git record at their own path and most recent content change at the pinned revision. A copied file does not inherit another path's earlier date. These are recorded dates, not a claim about original creation.
- Original figure/PDF and most table creation dates are unknown. Show **First recorded** from the notebook's Git history and **Exported** from the hash-matched export receipt. The 70 portfolio CSVs with verified producer timestamps and original/public hash chains may show **Created**. Never replace unknown creation with publication, deployment, filesystem, or phase-window dates.
- Linked-run windows use actual recorded start and finish dates, with coverage counts. They describe linked runs, not an asserted individual figure measurement date.
- Display relative age alongside the dated fact it describes. Age never changes scientific success/failure or automatically makes an old experiment invalid.
- Unavailable, mismatched, or malformed date metadata is visibly unavailable. Unknown dates never become today's date.

## Durable records

A separate `content/artifact-dates.json` catalog is version controlled and copied to `public/data/artifact-dates.json` by a Node maintenance/build validator. Scientific JSON, numerical tables, original logs, and image bytes remain unchanged. The catalog covers figures, tables, source files, experiments, runs, warnings, the CVK2 inline chart model, PDF, and evidence bundle.

Each item is tied to its current file bytes or canonical JSON record with SHA-256. A maintainer refresh preserves all unchanged facts; changed records retain their first recorded date and receive a clearly labeled new notebook-record date. Existing historical source/scheduler facts are retained only while their verified evidence still matches. Build validation blocks missing or mismatched records, preventing stale dates from silently shipping. Source ZIP contents include the catalog and validator; the ZIP cannot hash itself inside its own date catalog.

## Public contract

Keys are `kind:id`, with kinds `figure`, `table`, `source`, `experiment`, `run`, `warning`, `chart`, and `download`. A date fact is `{at, precision: "day"|"second", basis, href?}`. ISO timestamps require an explicit timezone; day values are strict calendar dates. `href`, if supplied, is a verified public provenance link.

A record contains `kind`, `id`, `contentSha256`, `experimentIds`, and optional facts named `created`, `firstRecorded`, `updated`, `exported`, `submitted`, `started`, and `finished`. `created` remains absent when unknown. `firstRecorded` and `updated` refer to the documented scope in `basis`. An optional `runWindow` contains `started`, `finished`, `datedRuns`, and `totalRuns`.

The catalog contains `schemaVersion: 1`, `snapshotId`, `snapshotSha256`, `runsSha256`, `publishedAt`, `exportedAt`, `recordedAt`, and `records`. `recordedAt` dates catalog maintenance and is never displayed as artifact creation. JSON content hashes use recursively sorted object keys and ordinary JSON arrays; the Node inventory generator is the single canonical hashing implementation.

## Interface

One wrapping, readable date component serves cards and detail pages. Compact rows prioritize the relevant date (run finish/start, source update, or artifact first record). Detail rows show the available facts, original creation unknown where appropriate, and linked-run range. Bold values, muted labels, and neutral colors keep dates distinct from verdicts. Date metadata errors use an explicit amber message.

Place dates on shared figure cards, figure detail, table lists/previews, experiment list/detail, source lists/details, run list/detail, warning rows, the inline CVK2 plot, and the report/evidence download area. Keep snapshot publication separately labeled. Existing filters, links and ordering remain intact.

## Verification

Test calendar validation, timezone boundaries, unknown/future dates, relative age, and provenance URLs. Test catalog coverage, hashes, unchanged-date preservation, changed-record dating, scheduler coverage, stale-metadata rejection, and rebuild stability. Verify desktop/mobile readability, date distinctions, linked sources, preserved private access, and an automatic GitHub deployment. No new scientific jobs are needed.

## Chosen approach

Use verified metadata plus a small durable catalog. Filesystem dates would record local copies; browser-only dates would reset on publication. A separate database is unnecessary for this static notebook. The existing clean, dedicated portal checkout is isolated on `feat/artifact-dates`; no additional working copy is needed.
