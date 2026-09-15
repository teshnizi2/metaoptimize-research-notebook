# Maintaining the research notebook

The website publishes a versioned research snapshot. Imported evidence is stored in `public/data/research.json` and `public/data/runs.json`; maintainer notes are stored in `content/journal.json`. The browser reads these published files. It does not write research history or contact ALICE.

Run the commands below from the website project directory. Use Node.js 22.12 or later and install the locked dependencies with `npm ci`. Evidence import also requires Python 3, Pillow, and pypdf; the exporter can use the configured Codex bundled Python when those libraries are absent from the current interpreter.

## Append a note, correction, or warning

Use the experiment IDs shown in the experiment register. The current snapshot contains 111 IDs. The command validates against the actual imported register, including `CVK2`, rather than a hard-coded count.

```sh
npm run log -- --type note --title "Interpretation note" --text "Describe the observation and its evidence here." --experiments CVK2
npm run log -- --type correction --title "Correction to an earlier note" --text "Reference the earlier entry ID, explain the correction, and cite the supporting source." --experiments MT098,CVK2
npm run log -- --type warning --title "Scope limitation" --text "Describe the limitation that qualifies this experiment." --experiments CVK2
```

These are syntax examples; replace the example content with a real update before executing. The command generates a unique entry ID and the actual UTC recording time. Optional `--id J-UNIQUE-ID` and `--date 2026-09-15T09:00:00.000Z` let a maintainer retain a documented historical ID/date. Use an explicit historical date only when supported by the record. Valid dates are `YYYY-MM-DD` or UTC ISO timestamps ending in `Z`.

Entries require a nonempty title, nonempty text, and at least one existing experiment ID. Only `note`, `correction`, and `warning` are accepted. Unknown IDs, duplicate entry IDs, duplicate linked experiment IDs, invalid dates, unsupported options, and corrupt stored JSON cause a nonzero exit without modifying the journal. Obvious private paths, credentials, contact addresses, IP addresses, and infrastructure account assignments are also rejected before append or publication; entries are never silently rewritten. These patterns support maintainer review and do not establish that arbitrary prose is safe to publish. An ID must begin with a letter and contain only letters, numbers, underscores, or hyphens, with at most 80 characters. Titles allow 160 characters and text allows 20,000.

`npm run log` saves immediately to the local file, independently of the browser. Each write preserves previous entries, uses an exclusive writer lock, and atomically replaces the complete JSON file. `npm run build` validates the journal, requires already published entries to remain an unchanged prefix, and copies it to `public/data/journal.json` before compiling the website. An initial `[]` journal is valid. Commit both journal files to retain the durable research history in version control.

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

Set the two paths for the machine performing the import. `PUBLICATION_DATE` is the date of this publication snapshot; measurement windows and historical phase dates remain in the source evidence.

```sh
RESEARCH_WORKSPACE="/path/to/verified-campaign-workspace"
RESEARCH_REPO="/path/to/hierarchical-metaoptimize"
PUBLICATION_DATE="$(date -u +%F)"

python3 scripts/export_sources.py --repo "$RESEARCH_REPO" --workspace "$RESEARCH_WORKSPACE" --public ./public --audit "$RESEARCH_WORKSPACE/work/portal_source_audit.json"
python3 scripts/export_research.py --workspace "$RESEARCH_WORKSPACE" --public-dir ./public --as-of "$PUBLICATION_DATE"
python3 scripts/export_research.py --public-dir ./public --check
npm run test:data
node scripts/sync-journal.mjs
python3 scripts/package_notebook.py --verify
npm run build
```

The source import must run first because evidence export validates its source catalog. The source repository is read only. The exporters replace generated public snapshots/assets and produce audit receipts in the campaign workspace. Review `work/portal_source_audit.json` and `work/portal_data_audit.json`, including coverage, references, hashes, privacy checks, and explicit missing evidence. Inspect the version-control diff before publishing.

Record the evidence update with a separate `npm run log` entry linked to the affected experiments, then repeat journal synchronization, source packaging, and the final build after appending it. Import regenerates the base publication history, while `content/journal.json` preserves maintainer history. Do not remove old experiment IDs that existing journal entries reference; a removed ID will block synchronization until the historical reference is restored in the evidence index.

The current validation checks the established 111-experiment, 2,863-run snapshot and its registered CVK2 result. If a later campaign changes the evidence schema or coverage, update the importer and its tests deliberately from verified source evidence before publishing. Do not bypass failed validation to make an import pass.

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

The local `npm run deploy` command runs the JavaScript tests, synchronizes the journal, regenerates and verifies the source ZIP, builds production assets, then calls `vercel --prod --archive=tgz`. This order keeps the downloadable source aligned with the publication and uploads the thousands of assets as an archive. The packaging command writes `public/assets/notebook-source.zip` plus the source deliverable in the campaign outputs and records hashes, privacy coverage, and portability checks in `work/notebook_source_package_report.json`. Vercel's remote build uses `npm run build`; the Python packager is local tooling and is excluded from the hosted deployment. Run the Python data checks above first whenever imported evidence changes. Commit the reviewed source and generated data using the project's normal version-control workflow. Once Vercel reports a successful deployment, open its reported production URL and verify the snapshot ID, a new journal entry, direct route reloads, and downloads. Build success alone does not verify a deployment.

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

The two JSON files and version control provide the persistence boundary. There is no browser-local storage, collaborative editor, authentication service, or backend database. A saved local entry becomes visible to readers only after a successful build and publication.
