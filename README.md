# MetaOptimize Research Notebook

A working notebook for the researcher and two professors: experiment questions, visual findings, numerical tables, related research code, execution parameters, warnings, and dated history.

The snapshot currently covers 111 experiment questions, 2,863 recorded runs, 54 report figures, and the completed CVK2 cut-location test. Coverage and publication metadata come from `public/data/research.json`. Job completion is separate from scientific success, and each conclusion retains its comparison and scope.

The homepage summarizes the whole register through overall outcomes, an area-by-area table and dated notebook updates. Outcome counts link to the matching experiments; detailed plots belong to each experiment page. These counts cover different research goals and audits and are not a method win rate or a record of professor reviews.

## Repositories and access

- [Notebook website and versioned log](https://github.com/teshnizi2/metaoptimize-research-notebook)
- [Original research code](https://github.com/teshnizi2/hierarchical-metaoptimize)
- [Hosted notebook](https://metaoptimize-research-notebook.vercel.app)

Both GitHub repositories are public and readable without invitations. The notebook repository includes published source and evidence, so site login does not make those files private. The code viewer retains verified original-revision links and explicitly labeled archived copies.

Vercel is connected to the notebook repository, with production built from `main`. Native Vercel Authentication protects every deployment and its downloads. This project setting protects the hosted routes, not copies published on GitHub. Keep it enabled when publishing updates.

## Dates and age

Figures, tables, code, experiments, runs and downloads show their recorded dates. **Created** is used only when a producer record establishes creation; otherwise the notebook labels **First recorded** or **Exported** explicitly. Code dates come from the pinned file history, and run dates come from recorded scheduler accounting. Open **Date provenance** for the basis of a date. Timestamp displays use UTC.

Age describes the labeled artifact or record; it does not change a scientific verdict. Unchanged records keep their dates across website rebuilds. The versioned catalog is in `content/artifact-dates.json`, with a validated reader copy in `public/data/artifact-dates.json`. ZIP entry timestamps are fixed for reproducibility; use the catalog for research dates.

## Run the portable website

Extract the source ZIP into a new folder and open a terminal in that folder. Install Node.js 22.12 or later, then run:

```sh
npm ci
npm test
npm run build
npm run preview -- --port 4173
```

Open the address printed by the preview server. Use `npm run dev` while changing the interface. The locked npm install needs registry access; after installation, the site serves its evidence, fonts, code, and logs from local files. No private directories, cluster connection, database, or account credentials are required to build and browse it.

The downloadable package includes application source, configuration, locked dependencies, JavaScript tests, versioned journal content, and published data/assets/source text. Local ingestion tools and private infrastructure configuration are deliberately omitted. The portable copy removes npm scripts that require those omitted tools or a preconfigured deployment account.

The archive cannot contain itself. To retain the website's source-download link when rehosting, copy the original downloaded ZIP into `public/assets/notebook-source.zip` before building. All other included downloads remain available directly.

## Maintain the research record

See [the maintenance guide](docs/MAINTENANCE.md) for journal updates, validation, preview checks, and publishing the static `dist` folder. In the portable bundle this guide describes the portable workflow. A maintainer can append a note, correction, or warning with `npm run log`; the website itself stays read only. Saved entries become visible after a build and publication.

Preserve the verified evidence files when changing presentation. Adding new research results requires a separately verified evidence export; browser edits and journal notes do not change registered scientific outcomes.

## Provenance and completeness

- `MANIFEST.json` in the downloadable archive records each payload file's SHA-256 hash and byte size. It excludes itself from its own file list; the separately reported ZIP hash identifies the complete archive.
- `public/data/source-index.json` records `originalSha256` for the unredacted source file used for this export and `publicSha256` for its published copy. These differ when private infrastructure text was redacted. Public copies preserve scientific content and source line references, but a redacted file is not necessarily a byte-identical executable reproduction.
- Source associations identify related records, not automatically the exact executable used in a historical run. Use explicit frozen-source verification where it is recorded.
- All 2,863 runs in this snapshot have an included sanitized raw log, linked by `logHref`, with original/public SHA-256 pairs. The package report records the exact published-log count. If a future snapshot has an archived-but-unpublished log, its ledger entry must retain that explicit availability distinction; an unpublished log is different from a nonexistent log.
- Displayed run accuracy is the mean over the final five completed epochs (`plateau5`) only when its window is valid. Final accuracy, best accuracy, and the legacy plateau metric remain separate source fields.
- The website records a publication snapshot. It does not report live cluster state or infer launch dates from publication dates.

Research evidence remains subject to its recorded caveats and original attribution. Included source associations and hashes should accompany any reuse or scientific claim.

The research group uses one revocable Vercel shareable link attached to the stable production alias. It is stored separately from this repository. Both professors can use the same link without a Vercel account. Anyone holding it can read the notebook; distribute it only within the group. The link remains valid until revoked, and its continued access must be checked after a production update. Access protection stays enabled for every deployment.
