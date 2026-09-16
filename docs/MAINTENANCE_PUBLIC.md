# Maintaining a portable research notebook

This package contains a published research snapshot and the application that displays it. The reader interface is read only. Maintainer entries persist in `content/journal.json`, and the production build validates and copies them to `public/data/journal.json`.

## Install, test, and preview

Use Node.js 22.12 or later. From the extracted project folder:

```sh
npm ci
npm test
npm run build
npm run preview -- --port 4173
```

The locked dependency install requires npm registry access. The application then serves included fonts and evidence locally. It does not need a cluster login, private file path, backend, or deployment account. `npm run dev` starts the development server.

## Append a versioned maintainer entry

Replace the example text below with a real, public update. Link IDs from the current experiment register:

```sh
npm run log -- --type note --title "Interpretation note" --text "Describe the observation and its evidence." --experiments CVK2
```

Supported types are `note`, `correction`, and `warning`. The command generates a unique ID and the actual UTC recording time. Optional `--id J-UNIQUE-ID` and `--date YYYY-MM-DD` retain a documented ID or historical date; UTC ISO timestamps are also accepted. A title, text, and at least one valid experiment ID are required. Multiple experiment IDs use commas. Titles allow 160 characters and text allows 20,000.

The command rejects invalid or duplicate IDs, empty text, unsupported types, invalid dates, and obvious private infrastructure or credential content. It does not silently redact an entry. Review wording before retrying, and keep private operational details outside the public record. These checks support maintainer review; pattern matching cannot establish that arbitrary prose is safe to publish.

Each successful command persists a new entry without changing earlier entries. Writes use an exclusive lock and atomic file replacement. Published entries must remain an unchanged prefix when building; correct an error by appending a new entry that references the earlier ID. A journal correction does not itself revise an imported scientific outcome, registered score, or numerical table.

```sh
npm test
npm run build
npm run preview -- --port 4173
```

Inspect `/activity`, the linked experiments, and `/warnings` if the entry is a warning. Save both journal files in version control with the application source and published data. `node scripts/sync-journal.mjs` validates and synchronizes the journal without compiling. The browser never stores or writes journal history.

If a write was interrupted, inspect the recorded process ID in `content/.journal.lock` and confirm no writer is active before removing that specific stale lock. Validation failures leave prior history intact. Restore damaged or rewritten history from version control and append a correction.

## Preserve or replace the evidence snapshot

Interface changes can use the included verified JSON, figures, tables, source text, and logs directly. Keep their content and hashes intact. The local ingestion/export scripts, including the GitHub mapping generator, are excluded because they rely on a separate verified research workspace. Obtain a complete verified export from the maintainer rather than guessing GitHub revisions.

For a later research update, obtain a complete, verified public export from the research maintainer. Preserve stable historical experiment IDs, replace the corresponding public data/assets/source set together, including a matching verified `data/github-links.json`, retain journal history, and record the update in a new journal entry. Review the supplied provenance and validation receipts before building. Removing an experiment referenced by an existing journal entry blocks synchronization until the historical reference is restored.

`npm test` checks application behavior and the journal workflow. It does not reproduce model training or replace scientific validation of a new export. `npm run build` validates journal links and compiles the website; it does not rederive registered scores.

## Publish the static website

After testing, deploy the generated `dist` folder with your chosen static hosting provider. The included `vercel.json` configures a Vite build, output directory, security headers, and route fallbacks for Vercel. Use your own authorized project and account; no deployment association or credentials are supplied.

For an existing authorized Vercel project with its CLI configured, run:

```sh
npm test
npm run build
vercel --prod --archive=tgz
```

Confirm the destination before publication. Keep local deployment associations, credentials, environment files, dependency directories, and build output out of the versioned source. If the hosted site should retain the source-download button, place the original downloaded ZIP at `public/assets/notebook-source.zip` before the production build. It was excluded from its own archive to avoid recursion.

After the host reports success, verify the reported production URL: check the snapshot ID, the new journal entry, direct reloads of experiment and run pages, filters, CSV/PDF/source downloads, and one available raw log. A successful local build alone does not verify a deployment.

## Interpret the data correctly

- A run's execution state describes inventory completion or supersession. Scientific outcomes belong to experiments; completed jobs can have failed research goals or collapsed trajectories.
- Completed epochs and requested epochs are different fields. Accuracy is the valid final-five-epoch mean, with no final/best fallback. Historical plateau values use a separate window.
- A run link identifies the documented batch family and does not assert that the run belongs to every reported contrast.
- All 2,956 runs in this snapshot have sanitized raw logs linked by `logHref`, with original/public SHA-256 pairs. If a future snapshot omits an archived log, preserve its explicit archived-but-unpublished availability distinction.
- Research code records contain both original and public hashes. Redactions can change bytes while retaining scientific content and line references. Related current code is not automatically exact historical runtime provenance.
- History records documented phases, completions, imports, or maintainer recording times. It does not infer per-run launch times or show live cluster status.

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

The portable package includes the catalog and Node validator. Its private-evidence importer is deliberately omitted. For a new scientific snapshot, obtain the matching verified date catalog from the maintainer together with the replacement evidence. You can edit the interface or append journal notes without changing historical artifact dates.
