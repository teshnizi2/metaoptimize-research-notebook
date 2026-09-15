# Research Notebook Implementation Plan

> **For agentic workers:** Use Superpowers test-driven development for behavioral changes. Root integrates independent source-catalog and research-export tasks, then reviews specification and code quality.

**Goal:** Publish a complete, connected MetaOptimize research notebook on Vercel.

**Architecture:** Static React/Vite application with generated, typed JSON and immutable source assets. Versioned append-only activity log supports maintenance without browser-local data loss.

**Tech Stack:** React, TypeScript, Vite, Lucide icons, Manrope fonts, CSS, Python export tooling, Node tests, Vercel.

- [x] Inspect verified inputs, source repository, Vercel access and user authorization.
- [x] Define schema and public interface; public read-only access with maintainer-published log updates.
- [x] Source catalog: test valid code mappings/redaction, export source files and references without touching the research repo.
- [x] Research export: test all111 records,54figures and2863runs, code/data links and honest warnings; export JSON and public assets.
- [x] UI domain: write failing tests for multi-field search, combined filters and related-record lookup; implement small pure helpers.
- [x] Build routes: overview, experiment index/detail, figures, code viewer, run ledger, warnings and activity with empty/error states and stable URLs.
- [x] Maintenance: test and implement append-only validated research log updates, repeatable import, build and deploy commands.
- [x] Review spec compliance, numeric scope, privacy, cross-links and implementation quality; fix concrete issues.
- [x] Verify desktop/mobile browser workflows, production build, all internal references and download integrity.
- [x] Deploy to Vercel, verify anonymous access and direct links, package website source as an output deliverable.

Review notes: mixed Markdown heading depth could over-associate sources; fixed and independently tested. Journal warnings now reach experiment details and global search. Source labels distinguish the notebook base from original/public per-file hashes. Final export expands raw-log publication to every recorded run. Desktop1280px/mobile390px browsing, code-line search, job navigation, warning filtering, and CSV preview checked.

Final evidence checks: 291 source files, 2,863 original/public log hash pairs, 296,636 unchanged epoch accuracy lines, 549,709 unchanged numeric CSV cells, and provenance links for all 111 records. Generic private-storage prefixes blocked at source import and package boundaries. One unavailable source explicitly retained as a limitation (MT111/bin/agree2.py). 24 JavaScript, 12 source plus 3 focused privacy, 16 distinct data export, and 9 packaging tests passed. The portable package passed a clean install, test, and production build; the final package differs only in its .vercelignore upload exclusion, verified by complete manifest comparison.

Publication complete: https://metaoptimize-research-notebook.vercel.app (deployment dpl_DBBDdy6YDZcha6aKL73XPVgKJzNa; application commit e58287a). All 31 anonymous endpoint checks passed, including direct routes, exact-byte evidence downloads, security headers, and absent-file 404 responses. Fresh production browser navigation verified result → code, job → log → experiment, source line links, warning search, and research history; no console errors or warnings. Public source archive SHA-256: 3fd74d052a61b5d7d6b302192904702af0aaf0bfb3d500ec2f06115166bd9851. Deployment receipt: outputs/MetaOptimize_Website_Deployment.json in the publication workspace.
