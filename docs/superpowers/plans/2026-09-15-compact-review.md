# Compact review implementation

- [x] Update date-order regression tests, demonstrate failure, then sort the phase list newest first with missing dates last.
- [x] Replace the home timeline with a compact linked scrolling list and Show all/Show less controls; simplify surrounding home sections.
- [x] Move expanded phase information into the existing log destination (independent worker owns the detail component and ActivityPage wiring).
- [x] Review and verify desktop/mobile interaction, accessible navigation and unchanged science/date assets.
- [ ] Append a dated note, verify portable packaging, integrate and publish through GitHub; save the live verification receipt outside the repository.

Root owns the homepage, timeline helper, main CSS and tests. The worker owns ResearchPhaseDetail and the narrow ActivityPage integration. Preserve unrelated duplicate files in the main checkout.

Integrated checks: 56 JavaScript tests pass, including seven timeline tests; production build passes. Independent review issues for CSS imports in Node tests and stable phase links were reproduced and fixed. Desktop scrolling reaches all older rows; Show all reveals all eight and Show less resets the list to the newest work. Main evidence/date files are unchanged. Release receipts are stored outside the repository.

Portable source verification passes with 3,466 files: privacy scan, deterministic archive hashes, clean npm install, all 56 tests, and production build. The 390px mobile document stays within the viewport; only the detailed evidence table scrolls horizontally. All eight phases remain reachable in the home scrolling list.
