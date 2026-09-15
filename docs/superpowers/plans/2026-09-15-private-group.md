# Private Group Notebook Implementation Plan

> For agentic workers: root integrates bounded parallel UI, GitHub-source and access-control work. Preserve the study repository and others' edits. Use test-first practice for behavioral changes, then specification and quality review.

**Goal:** A simple, protected notebook for the research group, connected to its private GitHub repositories.

**Architecture:** Retain React/Vite and immutable evidence; use verified GitHub source mappings and native Vercel protection where feasible. Existing repository/project identities remain unchanged.

**Tech stack:** TypeScript/React/Vite, existing Node tests and Python exporter, GitHub and Vercel.

- [x] Verify both repositories are private and the original scientific repository remains untouched.
- [x] Connect the existing Vercel project to the website GitHub repository.
- [x] Confirm native authentication protects every deployed URL and all evidence downloads: 20 anonymous checks passed for the primary alias and original unique deployment.
- [x] Resolve three-reader access without unnecessary accounts or new paid features.
- [x] Simplify Layout/Overview/styles while preserving readable charts, tables and navigation.
- [x] Test then implement immutable GitHub mappings in Code page/helper/catalog; use explicit archived-copy fallback.
- [x] Review spec coverage and source/protection correctness; fix concrete issues.
- [x] Refresh maintenance instructions and portable source after all edits.
- [x] Run appropriate tests/build, inspect desktop/mobile, deploy through connected GitHub, verify authorized access and anonymous denial.

Root owns deployment settings, documentation and packaging. Source worker owns Code.tsx and GitHub mapping artifacts. UI worker owns Layout.tsx, Overview.tsx and styles.css. Security worker initially performs a read-only access-control feasibility review, then reviews the chosen boundary.

Pre-release checks:32 JavaScript tests pass; clean source-package installation, tests and build pass. Package SHA-256:17aa203f53bd7516214d37e79082a32dc1a010c45443578220867ee80ff068aa (3,451members). Existing package privacy guards pass, and a separate known-credential exclusion check passed across13,155tracked/archive payloads. Native protection, group-session downloads, source mapping and UI SPEC/security reviews passed. Root also inspected the compact results view and original-scorer GitHub action; browser reports no console errors or warnings.

Release verification PASS: GitHub push d0665a3245c7ccb95b3ffa97a42a7ae86683aa6f created production deployment dpl_FAEFbyyrn1XnEESKHvnAKaXBZVxQ. All33 authorized content checks and50 access-denial checks passed across production, branch/project aliases, the new unique deployment and the old unique deployment. The original private group link persisted on the same alias while its deployment ID changed; the token is removed from the browser URL after cookie establishment. Secrets remain outside Git and source archives. Root inspected the live compact layout; no console warnings/errors or horizontal overflow. The detailed receipt is stored in the publication workspace outputs/MetaOptimize_Private_Website_Verification.json.
