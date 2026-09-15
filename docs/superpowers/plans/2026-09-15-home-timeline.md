# Homepage research timeline

Restore the eight-row chronological research table requested by the user, alongside the general outcome and area reviews. Use the existing documented phase windows, tests, observations and next questions from the research snapshot. Preserve the original CSV and all scientific assets and dates.

Rows show readable bold tests and observations, labeled current outcome counts for their linked questions, short essential scope notes, and links to the records and phase log. Do not give a mixed group of research goals one invented phase verdict. Current verdicts are not necessarily the historical interpretation of the observation. Keep the overlapping windows and disclose the 31 unique linked records and 80 records without phase links; do not imply that these other records lack dated run logs.

Add a phase filter to the existing experiment register so each row and the records without a phase link can be inspected directly. Derive membership from explicit activity links, deduplicate within a phase, and preserve membership across overlapping phases. Incomplete phase metadata remains visible without invented window dates.

- [x] Test timeline parsing, chronological ordering, overlap/coverage, original CSV agreement, phase filtering and missing metadata.
- [x] Implement the table and connected register filter, preserving the general homepage and experiment-specific charts.
- [x] Verify desktop/mobile rendering, links, filter behavior, source/date integrity and scientific cautions.
- [x] Add a dated notebook note, build and verify the portable source package.
- [ ] Integrate, push to connected GitHub main, verify Vercel content and private access.

Implementation is isolated from unrelated untracked duplicates in the main checkout. Only owned files will be staged. A read-only evidence review confirms MT098's failed prediction bridge, the separate isolation batches and unmet dynamics gates, and CVK2's unresolved prediction despite completed runs.

Local verification: 55 JavaScript tests pass, including six timeline tests and 20 malformed-field cases; independent review confirms all 48 phase/outcome combinations and no remaining findings. Production build succeeds. Mobile phase filter is full width, the document stays within a 390px viewport, and principal timeline cells use 16px text. Original research/run data, CSV, PDF, evidence bundle and date catalog are byte-identical. Portable package privacy/determinism and clean install/test/build pass (3,464 files). Live release verification is recorded outside the repository.
