# MetaOptimize Research Notebook

## Authorized outcome
Publish the completed research report and the corresponding code on Vercel. Make experiments, results, charts, tables, warnings, run records and dated research history easy to find and cross-reference. Preserve the research repository and original report. Use large readable text, bold goals, and semantic outcome colors.

## Architecture and alternatives
Use a versioned static React + Vite notebook generated from the verified research exports, with stable experiment IDs and relative links. A PDF-only landing page would hide the required provenance. A database-backed collaborative editor adds authentication and editing semantics; the user has been asked whether collaborators need write access. Build the common browsing surface now and use a durable, append-only file-backed update log for maintainer updates unless the user requests collaborative editing.

## Public interface
Editorial laboratory dashboard, warm off-white canvas, dark green accents, Manrope typography and readable monospace code. Persistent navigation: Overview, Experiments, Figures, Code, Run ledger, Warnings, Activity. Global search and per-page filters. Every record gets a shareable URL. Experiment detail tabs show results, figures, data, code, runs, warnings/history. Scope limitations remain visible. Scientific outcomes are separate from execution status. No claims of live cluster monitoring.

## Data integrity
111 documented questions/audits, 2863 recorded runs (includes cancelled history),54 report pages. CVK2 has27 completed runs and an unresolved registered peak set16/19/22. Bind charts and code to experiment IDs via explicit evidence mappings. Mark generic shared code and unavailable source associations honestly. Preserve original source hashes and label redacted public copies. Remove private machine paths, account names, emails and credentials from public assets. Publish only research materials relevant to this notebook.

## Maintenance
Version the website and its exported data in a dedicated project. Build/import commands validate references and append an activity event for changed snapshots. A maintainer CLI records notes and warnings against valid experiment IDs; rebuild/redeploy publishes them to everyone. Existing events are retained. Public visitors browse by default pending the optional edit-access answer.

## Verification
Test search/status filtering, ID resolution, persistent log validation, and complete data/code/figure references before implementation. Build with type checking. Verify security/redaction and all public file links. Test desktop/mobile browser flows and direct route reloads. Deploy only the reviewed build to Vercel production and verify anonymous access to the URL, deep links and downloads.
