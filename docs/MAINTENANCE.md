# Maintaining the research notebook

The website publishes a versioned research snapshot. Imported evidence is stored in `public/data/research.json` and `public/data/runs.json`; maintainer notes are stored in `content/journal.json`. The browser reads these published files. It does not write research history or contact ALICE.

Run the commands below from the website project directory. Use Node.js 22.12 or later and install the locked dependencies with `npm ci`. Evidence import also requires Python 3, Pillow, and pypdf; the exporter can use the configured Codex bundled Python when those libraries are absent from the current interpreter.

## Append a note, correction, or warning

Use the experiment IDs shown in the experiment register. The current snapshot contains 148 IDs (130 research questions and 18 method checks). The command validates against the actual imported register, including `CVK2`, rather than a hard-coded count.

```sh
npm run log -- --type note --title "Interpretation note" --text "Describe the observation and its evidence here." --experiments CVK2
npm run log -- --type correction --title "Correction to an earlier note" --text "Reference the earlier entry ID, explain the correction, and cite the supporting source." --experiments MT098,CVK2
npm run log -- --type warning --title "Scope limitation" --text "Describe the limitation that qualifies this experiment." --experiments CVK2
```

These are syntax examples; replace the example content with a real update before executing. The command generates a unique entry ID and the actual UTC recording time. Optional `--id J-UNIQUE-ID` and `--date 2026-09-15T09:00:00.000Z` let a maintainer retain a documented historical ID/date. Use an explicit historical date only when supported by the record. Valid dates are `YYYY-MM-DD` or UTC ISO timestamps ending in `Z`.

Entries require a nonempty title, nonempty text, and at least one existing experiment ID. Only `note`, `correction`, and `warning` are accepted. Unknown IDs, duplicate entry IDs, duplicate linked experiment IDs, invalid dates, unsupported options, and corrupt stored JSON cause a nonzero exit without modifying the journal. Obvious private paths, credentials, contact addresses, IP addresses, and infrastructure account assignments are also rejected before append or publication; entries are never silently rewritten. These patterns support maintainer review and do not establish that arbitrary prose is safe to publish. An ID must begin with a letter and contain only letters, numbers, underscores, or hyphens, with at most 80 characters. Titles allow 160 characters and text allows 20,000.

`npm run log` saves immediately to the local file, independently of the browser. Each write preserves previous entries, uses an exclusive writer lock, and atomically replaces the complete JSON file. `npm run build` validates the journal, requires already published entries to remain an unchanged prefix, and copies it to `public/data/journal.json`. It also validates and synchronizes the artifact date catalog before compiling the website. An initial `[]` journal is valid. Commit both journal files to retain the durable research history in version control.

Append a new correction instead of editing or deleting an earlier entry. A journal note does not revise an imported experiment outcome, registered score, or numerical table. Those changes require verified evidence and a new import. Journal warnings appear in both **Warnings & limits** and **Research log**, linked to their experiments. They carry a journal label, with no invented resolution state.

For a note-only update:

```sh
node scripts/sync-journal.mjs
python3 scripts/package_notebook.py --verify
npm run build
npm run preview -- --port 4173
```

Review `/activity` and any linked experiment pages, then follow the deployment steps below. The package command creates the source download and verifies `npm ci`, `npm test`, and `npm run build` in an isolated extraction. The final local build then includes the refreshed source download. Recreate the source ZIP after the last journal, interface, or evidence change; otherwise the download can lag behind the website.

## Import a verified evidence update

Prepare the verified campaign workspace first. The import expects its `outputs/tables/complete_experiment_register.csv`, `complete_run_inventory.csv`, `complete_figure_index.csv`, `research_timeline.csv`, supporting evidence JSON files under `work/`, the completed PDF, and supporting tables/assets. It imports the established evidence schema; it does not infer new scientific verdicts from raw jobs.

Set the two paths for the machine performing the import. Also set `RESEARCH_REVISION` and `ARCHIVE_REVISION` to verified full 40-character commit SHAs available on GitHub. The research revision identifies the original code tree; the archive revision must already contain the exact exported `public/source` bytes. For new evidence, first commit and push those sanitized source snapshots on a protected nonproduction branch. Then generate the mappings, package and test the complete update before publishing `main`. Never claim that a changed local file is identical to the pinned original. `PUBLICATION_DATE` is the date of this publication snapshot; measurement windows and historical phase dates remain in the source evidence.

```sh
RESEARCH_WORKSPACE="/path/to/verified-campaign-workspace"
RESEARCH_REPO="/path/to/hierarchical-metaoptimize"
PUBLICATION_DATE="$(date -u +%F)"

python3 scripts/export_sources.py --repo "$RESEARCH_REPO" --workspace "$RESEARCH_WORKSPACE" --public ./public --audit "$RESEARCH_WORKSPACE/work/portal_source_audit.json"
python3 scripts/export_research.py --workspace "$RESEARCH_WORKSPACE" --public-dir ./public --as-of "$PUBLICATION_DATE"
python3 scripts/export_research.py --public-dir ./public --check
npx tsx scripts/export_github_links.ts --research-repo "$RESEARCH_REPO" --research-revision "$RESEARCH_REVISION" --archive-revision "$ARCHIVE_REVISION"
npm run dates:refresh -- --evidence "$RESEARCH_WORKSPACE/work/artifact-date-evidence.json"
npm run dates:check
npm run test:data
node scripts/sync-journal.mjs
python3 scripts/package_notebook.py --verify
npm run build
```

The source import must run first because evidence export validates its source catalog. The source repository is read only. The exporters replace generated public snapshots/assets and produce audit receipts in the campaign workspace. Review `work/portal_source_audit.json` and `work/portal_data_audit.json`, including coverage, references, hashes, privacy checks, and explicit missing evidence. Inspect the version-control diff before publishing.

Record the evidence update with a separate `npm run log` entry linked to the affected experiments, then repeat journal synchronization, source packaging, and the final build after appending it. Import regenerates the base publication history, while `content/journal.json` preserves maintainer history. Do not remove old experiment IDs that existing journal entries reference; a removed ID will block synchronization until the historical reference is restored in the evidence index.

The current validation checks the established 148-record, 2,863-run snapshot (130 research questions, 18 method checks, 10 areas) and its registered CVK2 result. If a later campaign changes the evidence schema or coverage, update the importer and its tests deliberately from verified source evidence before publishing. Do not bypass failed validation to make an import pass.

## Outcome model and the partition audit

`scripts/register_model.py` is the single, reviewable place where register rows become published outcomes. Both exporters import it.

- Research questions carry one of four outcomes: `success` (Goal met), `fail` (Goal missed), `mixed` (Mixed) or `unresolved` (Open). `corrected` is a separate badge with a `correction` record (note and source); it never replaces the outcome.
- `CORRECTION_REMAP` maps each of the 23 rows the register exported with outcome `correction`: eight research questions get a real outcome, and fifteen process checks become `method-check` records with no research outcome. An unmapped `correction` row stops the export.
- `PARTITION_AUDIT` imports MASTER-TABLE section 10 (lines 175-211) from the pinned campaign commit `d69b23a`, whose file hash is checked. Each row has an explicit rule, outcome, batches, optional correction note and one-line reason. New IDs are `MT<line>` in that commit (MT175-MT211); existing IDs are never renumbered.
- MASTER-TABLE source anchors are resolved by row content. The original IDs came from an uncommitted MASTER-TABLE snapshot that was one line longer from line 20 on, so anchoring by line number alone would point most records at their neighbouring row.

Run the source export before the research export (the research export requires a source link for every record). `export_research.py --audit PATH` writes its receipt outside the campaign workspace when that workspace must stay read only. The Python data tests accept `NOTEBOOK_WORKSPACE`, `NOTEBOOK_RESEARCH_REPO` and `NOTEBOOK_DATA_AUDIT` when the portal is not checked out inside the campaign workspace.

## Review and deploy

Start `npm run preview -- --port 4173`, then inspect:

- `/runs`: search and combine model, dataset, batch, execution-state, and experiment filters; advance beyond the first 50 rows and reset filters.
- `/runs/run-<jobId>`: use a real ID from the ledger. Check full parameters, accuracy window, completed epochs, experiment links, and explicit missing-log messages. CVK runs with a published `logHref` should show and download their raw log.
- `/warnings?experiment=CVK2` and `/activity`: test filters, linked experiments, dated entries, and empty results.
- A direct reload of an experiment/run route, the PDF, a source text download, a CSV, and the evidence bundle.

Deployment is a maintainer action after review. Use the existing Vercel project association and configured account; do not create a replacement project accidentally. Check `.vercel/project.json` locally if present and confirm the destination in the Vercel CLI before publication. Keep account credentials and local association files out of public assets and version control.

```sh
npm run deploy
```

The local `npm run deploy` command runs the JavaScript tests, validates and synchronizes the journal and date catalog, regenerates and verifies the source ZIP, builds production assets, then calls `vercel --prod --archive=tgz`. This order keeps the downloadable source aligned with the publication and uploads the thousands of assets as an archive. The packaging command writes `public/assets/notebook-source.zip` plus the source deliverable in the campaign outputs and records hashes, privacy coverage, and portability checks in `work/notebook_source_package_report.json`. Vercel's remote build runs `npm test` followed by `npm run build`; the Python packager is local tooling and is excluded from the hosted deployment. Run the Python data checks above first whenever imported evidence changes. Commit the reviewed source and generated data using the project's normal version-control workflow. Once Vercel reports a successful deployment, open its reported production URL and verify the snapshot ID, a new journal entry, direct route reloads, and downloads. Build success alone does not verify a deployment.

## Metric and provenance boundaries

- Run execution state describes inventory completion or supersession. Scientific success/failure belongs to the linked experiment. A completed job can have a collapsed trajectory or a failed scientific objective.
- `Run.epochs` is `epochs_done`, the number of completed training epochs. `parameters.epochs_requested` holds the requested budget.
- Displayed run accuracy is `plateau5`, the mean test accuracy over the final five completed epochs, only when `window_ok=1`. Missing/invalid windows are not replaced by final or best accuracy. The legacy `plateau` field uses a different historical window and remains explicitly separate in source parameters.
- An experiment/run association identifies the documented batch family. It does not assert that every run participated in every reported contrast.
- All 2,863 runs in this snapshot have a sanitized raw log linked by `logHref`, with original/public SHA-256 pairs. If a future snapshot omits an archived log, retain its explicit archived-but-unpublished availability label rather than implying that no log exists.
- Source downloads can be redacted research records. Their original and public hashes distinguish the two copies. A related current source file is not automatically the exact historical executable; use explicit frozen-source verification when available.
- Activity dates identify documented phase windows, completion events, publication snapshots, or the maintainer's recording time. They are not inferred per-run launch times or a live cluster feed.

## Recover a failed journal operation

Read the command's error before retrying. Validation failures leave the previous journal intact. Restore malformed JSON or a changed published prefix from version control, then append a correction. If `.journal.lock` remains after an interrupted process, inspect its recorded PID and verify that no journal or build-time sync writer is running before removing that specific stale lock. Never clear another active writer's lock.

The two JSON files and version control provide the persistence boundary. There is no browser-local storage, collaborative editor, or backend database. The hosted notebook uses native Vercel access protection. A saved local entry becomes visible to readers only after a successful build and publication.

## Research group access and GitHub

The hosted notebook and downloads served by Vercel require site access. Keep **Vercel Authentication → All Deployments** enabled in the project settings, including after redeployment.

Both GitHub repositories are public and readable without invitations. The notebook repository includes published source and evidence; site login does not make those files private. Never place access links, passwords, tokens or professor contact details in the journal, code archive or repository.

The existing Vercel project is connected to the notebook GitHub repository with `main` as its production branch. After a verified update, synchronize the journal, regenerate the source ZIP, test, build, commit the reviewed changes and push `main`; Vercel starts the build automatically. Confirm the new deployment is ready and still denies anonymous HTML, data, source, log and download requests before sharing it. The CLI deployment command remains available for an authorized manual release. Do not disable protection to make a browser check or automated verification pass.

The research group uses one revocable Vercel shareable link attached to the stable production alias. It is stored separately from this repository. Both professors can use the same link without a Vercel account. Anyone holding it can read the notebook; distribute it only within the group. The link remains valid until revoked, and its continued access must be checked after a production update. Access protection stays enabled for every deployment.

## Artifact dates

The UI reads `public/data/artifact-dates.json`; its maintained source is `content/artifact-dates.json`. The build validates complete artifact coverage, date precision, links, snapshot hashes and current evidence bytes before synchronizing the reader copy. `npm run dates:check` verifies both copies without changing them.

- **Created** requires a dated producer record for the exact original artifact. Unknown creation remains explicitly unrecorded.
- **First recorded** and **Updated** identify the documented file or notebook-record scope. Git dates do not prove original creation. Copied protocols use their own path history.
- **Exported** dates the public export of the bytes; it does not date model training.
- **Submitted**, **Started** and **Finished** come from retained scheduler accounting. Offset-qualified timestamps display in UTC.
- **Linked run window** gives the start/finish range and dated coverage for runs linked through the experiment register. It is not an asserted timestamp for every figure measurement.
- Relative age uses the reader's current clock. Age alone does not mark a scientific result failed or automatically obsolete. ZIP member dates are fixed packaging metadata, not research dates.

A website rebuild or journal note does not refresh unchanged artifact dates. Keep both date-catalog files together with their evidence. If validation detects stale hashes or missing records, supply a verified metadata update; never silence the error by changing dates to today's date.

For the maintained repository, after updating the scientific export and verifying new date evidence, refresh the catalog from the local evidence fixture:

```sh
npm run dates:refresh
npm run dates:check
```

The default fixture is `../artifact-date-evidence.json` relative to the portal. Supply `npm run dates:refresh -- --evidence /path/to/verified-artifact-date-evidence.json` on another machine. The importer records the time a changed notebook record was saved; it preserves unchanged dates and only retains historical facts when their source evidence matches. Review the catalog diff, then synchronize the journal, package, test, build and publish. Keep the raw evidence fixture outside public assets and version control.
