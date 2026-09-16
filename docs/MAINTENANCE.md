# Maintaining the research notebook

The website publishes a versioned research snapshot. Imported evidence is stored in `public/data/research.json` and `public/data/runs.json`; maintainer notes are stored in `content/journal.json`. The browser reads these published files. It does not write research history or contact ALICE.

Run the commands below from the website project directory. Use Node.js 22.12 or later and install the locked dependencies with `npm ci`. Evidence import also requires Python 3, Pillow, and pypdf; the exporter can use the configured Codex bundled Python when those libraries are absent from the current interpreter.

## Append a note, correction, or warning

Use the experiment IDs shown in the experiment register. The current snapshot contains 160 IDs (142 research questions and 18 method checks). The command validates against the actual imported register, including `CVK2`, rather than a hard-coded count.

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

The current validation checks the established 160-record, 3,049-run snapshot (142 research questions, 18 method checks, 10 areas; `EXPECTED_STATS` in `export_research.py`) and its registered CVK2 result. If a later campaign changes the evidence schema or coverage, update the importer and its tests deliberately from verified source evidence before publishing. Do not bypass failed validation to make an import pass.

## Outcome model and the partition audit

`scripts/register_model.py` is the single, reviewable place where register rows become published outcomes. Both exporters import it.

- Research questions carry one of four outcomes: `success` (Goal met), `fail` (Goal missed), `mixed` (Mixed) or `unresolved` (Open). `corrected` is a separate badge with a `correction` record (note and source); it never replaces the outcome.
- `CORRECTION_REMAP` maps each of the 23 rows the register exported with outcome `correction`: eight research questions get a real outcome, and fifteen process checks become `method-check` records with no research outcome. An unmapped `correction` row stops the export.
- `PARTITION_AUDIT` imports MASTER-TABLE section 10 (lines 175-211) from the pinned campaign commit `d69b23a`, whose file hash is checked. Each row has an explicit rule, outcome, batches, optional correction note and one-line reason. New IDs are `MT<line>` in that commit (MT175-MT211); existing IDs are never renumbered.
- `APPENDED_ROWS` imports the rows appended after the audit (MASTER-TABLE lines 212-217, CORRECTIONS 217-226) from the pinned campaign commit `64e4f47`, whose file hash is checked; lines 1-211 must match the partition-audit commit except the header lines 3 and 5. Each row has an explicit rule, research area, batches, figure page and one-line reason that starts with the row's first registered verdict token: MT212 cgn1, MT213 cpl1, MT214 cvh1, MT215 cuc1 (Baseline comparisons), MT216 cgn2, MT217 cpl2. The import itself does not rewrite earlier records these rows bear on; `APPENDED_BEARS_ON` lists them (MT019, MT020, MT162, MT163, MT166, MT213), and changes to earlier records go through `ROW_AMENDMENTS` below. Their 93 runs (cgn1 6, cpl1 15, cvh1 12, cuc1 30, cgn2 15, cpl2 15) link through their batches once the run inventory includes them; see the run-inventory note below.
- `ROW_AMENDMENTS` applies rows the campaign amended in place, pinned to campaign commit `6e33fd8` (CORRECTIONS 229), whose MASTER-TABLE hash is checked; it may differ from `64e4f47` only on header line 3 and rows 19, 162, 163, 166 and 213, and no line may move. Each entry names the record (row 19 is MT019, 162 is MT163, 163 is MT164, 166 is MT020 and 213 is MT213; the older IDs came from a MASTER-TABLE snapshot one line longer, and the cau1 row keeps its original ID), its outcome before and after, and a one-line reason. An outcome may change only where the row's verdict column changed: MT019 moved from Open to Goal met (row 19: OPEN -> CONFIRMED, RESCOPED 229) and gains the `cuc1` batch; the other four keep their outcome and gain the amendment note in their scope. A drifted outcome stops the export. The previous outcome and wording stay on the record as an `amendment` warning; a result moved by later data is not a corrected claim, so the Corrected badge is not set. The ID fixture `tests/fixtures/register-ids-2026-09-16.json` records the MT019 change explicitly as `outcomeChange`.
- `LANDED_ROWS` imports the `cvt1` landing (MASTER-TABLE line 218, CORRECTIONS 230) from the pinned campaign commit `82867bb`, whose file hash is checked; it may differ from `6e33fd8` only on header line 3 and row 19's verdict-cell formatting fix (the superseded token OPEN moved into a bold `[SUPERSEDED: OPEN]` bracket; every other row-19 cell and the verdict wording from "Moved at" on must be unchanged), and no line may move. That fix changes no record's text; it shows only in the published MASTER-TABLE source copy. MT218 is Mixed under `VERDICT_RULES`: the returned branch `STEP-SIZE-NEEDED-VOTE-SUFFICES` answers both halves of the question, but its registered account missed INJECT's 8-30 band (30.14). `LANDED_INTERVENTIONS` names its vote-weight arms from the campaign's hash-pinned `results/CORPUS-EXCLUSIONS.tsv`: the record gains an intervention warning, and each of the 9 MUTE / DOSE / INJECT runs carries `intervention`, `interventionWitness` and `interventionNote` parameters, because the run inventory gives those rows the cell key of the plain `k01` / HEAD arm. The export stops if the list and the inventory disagree. The extended run inventory appends the 15 `cvt1` corpus rows after the 93 earlier ones (2,971 rows).
- `CGN3_ROWS` imports the `cgn3` landing (MASTER-TABLE line 219, CORRECTIONS 231) from the pinned campaign commit `e3a43da`, whose file hash is checked; it may differ from `82867bb` only on header line 3 and rows 213 and 218, and no line may move. MT219 is Goal met under `VERDICT_RULES`: the registered branch `RESCUE-SURVIVES` answers the question, and every arm and RHO land in the tested FREEZE-HOLDS account's registered bands; the level settling before the pin and `CEIL-BELOW` stay in its reason and scope. Its 12 runs are 430-epoch runs. `CGN3_AMENDMENTS` applies the two in-place amendments of the same commit, each checked to change one column and keep the superseded wording: MT218's scope takes row 218's amended scale cell verbatim (the pooling claim bracketed as not true when written), and MT213 gains an `Amended at CORRECTIONS 231` note that cvt1 has landed, with the pending-cvt1 clause of its CORRECTIONS 229 note retired. Neither moves an outcome; each adds a `warning-<id>-amendment-231` record. Because the live corpus was re-aggregated at that ingest, the extended run inventory keeps the published 2,971 rows byte for byte and appends the 12 `cgn3` rows (2,983 rows).
- `CVT23_ROWS` imports the `cvt3` and `cvt2` landings (MASTER-TABLE lines 220-221, CORRECTIONS 235 and 236) from the pinned campaign commit `9c5d72a`, whose file hash is checked; it may differ from `e3a43da` only on header lines 3 and 5 and rows 213 and 216, and no line may move. Line 5 (the bottom-line paragraph, amended at CORRECTIONS 234 and 236) feeds no record and appears only in the published MASTER-TABLE source copy. MT220 (`OWN-STEP-NECESSARY`) and MT221 (`GRADED` + `TOP-ATTENUATED`) are Mixed under `VERDICT_RULES`, as MT218: each branch answers its question, but a registered expectation was defied (cvt3's own account misses its MUTECTL band by 0.18 pp and `P_COAL` comes from the control falling below `k01`; for cvt2 no registered account predicted the pair of words and the stated `TOP-FLAT` prior missed by 0.26 pp). Each reason quotes all FINAL tokens. `CVT23_AMENDMENTS` applies the two row amendments of the same pin, each checked to be one contiguous bracketed edit of one column that keeps any replaced wording: MT216's result takes row 216's amended cell verbatim (CORRECTIONS 234: 'ISO above `kL`' held at 100 epochs only), and MT213 gains an `Amended at CORRECTIONS 236` note ('whoever casts it' corrected to the one caster that was run). Neither moves an outcome; each adds a `warning-<id>-amendment-<number>` record. `CVT23_INTERVENTIONS` names the 24 intervened runs (cvt3 MUTE50 / MUTEDOWN / MUTECTL, cvt2 K13-K2000) from `results/CORPUS-EXCLUSIONS.tsv` at `9c5d72a`, whose hash is checked and whose cvt1 pin must be a byte prefix; they carry the same run parameters as cvt1's. The extended run inventory keeps the published 2,983 rows byte for byte and appends the 36 new corpus rows (3,019 rows).
- `CVT45_ROWS` imports the `cvt4` and `cvt5` landings (MASTER-TABLE lines 222-223, CORRECTIONS 240 and 241) from the pinned campaign commit `643264c`, whose file hash is checked; it may differ from `9c5d72a` only on header lines 3 and 5 (line 5 again feeds no record), and no line may move. No existing row was amended, so no earlier record changes. MT222 (`OWN-STEP-MAGNITUDE`) and MT223 (`K-DEPENDENT-EQUILIBRIUM` + `RESCUE-SURVIVES` + `K33-HOLDS-PINNED`) are Goal met under `VERDICT_RULES`, as MT219: each registered branch answers its question and the tested account is in band on every registered band (cvt4's 9, cvt5's 8), with the bounds of CORRECTIONS 240.4 and 241.4 kept in the reason and scope. Neither corrects an earlier claim, so neither carries the Corrected badge. `CVT45_INTERVENTIONS` names the 18 intervened runs (cvt4's `BETA_HOLD` step-size holds HOLDLOW / HOLDSHARED / HOLDHIGH / HOLDHEAD, cvt5's `VOTE_W` arms K13 / K33) from `results/CORPUS-EXCLUSIONS.tsv` at `643264c`, whose hash is checked; each pinned copy of the list must be a byte prefix of the next. The run note names the intervention kind (step-size hold or vote weight). cvt5's runs are 300-epoch runs. The extended run inventory keeps the published 3,019 rows byte for byte and appends the 30 new corpus rows (3,049 rows). `jobs/run_cifar_cvt4.sh` exists only on the cluster, so MT222 carries a source-unavailable warning rather than a substitute link.
- `ADDED_PHASES` adds research phases the campaign's `research_timeline.csv` does not have, in the same format (currently phase-09, 15-16 Sep 2026, linked to MT212-MT223; cvt4 and cvt5 were launched on 16 Sep and landed at 22:10 UTC the same day). The timeline CSV itself is published unchanged. The timeline CSV itself is published unchanged.
- The published `assets/tables/complete_experiment_register.csv` is regenerated from this register: one row per record, with `kind`, the four-way `outcome` and `outcome_label`, and `corrected` / `correction_note` as separate columns. The campaign's 111-row export is an input, not the published table.
- `outcomes_by_area.csv` and `goal_outcomes.csv` are regenerated from the register too: outcome counts by area (four outcomes, method checks and Corrected in separate columns; the retired `correction` column is gone), and the report's page goal table with each page's current linked records and their outcome counts.
- MASTER-TABLE cells are Markdown. `clean()` in `register_model.py` removes bold, emphasis and code marks and turns backslash escapes into the literal character (`S\*` becomes `S*`, `\|` becomes `|`), so no escape reaches the published text; `tests/markdown-escapes.test.ts` fails if one does.
- MASTER-TABLE source anchors are resolved by row content. The original IDs came from an uncommitted MASTER-TABLE snapshot that was one line longer from line 20 on, so anchoring by line number alone would point most records at their neighbouring row.

Run the source export before the research export (the research export requires a source link for every record). `export_research.py --audit PATH` writes its receipt outside the campaign workspace when that workspace must stay read only. `--run-inventory PATH` publishes an extended copy of `complete_run_inventory.csv` kept outside that workspace: the campaign inventory unchanged, followed by the corpus rows (`results/all_runs.csv`) of landed batches, with raw logs found in the archived `runs/` and `runs_alice2/` `.out` files and sanitized by the same exporter. The Python data tests accept `NOTEBOOK_WORKSPACE`, `NOTEBOOK_RESEARCH_REPO`, `NOTEBOOK_DATA_AUDIT` and `NOTEBOOK_RUN_INVENTORY` when the portal is not checked out inside the campaign workspace. Most of them read only the published snapshot; the tests that also read one of those four inputs (`tests/external_inputs.py`) run when its variable is set or its default path exists, and are skipped with a reason naming the variable otherwise, so `npm run test:data` in a plain clone reports skips instead of errors. A set variable always runs its tests, so a wrong path fails instead of skipping. Set all four before an evidence import, and check that the run reports no skips.

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
- All 3,049 runs in this snapshot have a sanitized raw log linked by `logHref`, with original/public SHA-256 pairs. If a future snapshot omits an archived log, retain its explicit archived-but-unpublished availability label rather than implying that no log exists.
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
