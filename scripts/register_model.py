#!/usr/bin/env python3
"""Explicit outcome model for the experiment register, shared by both exporters.

Two operator-approved decisions (2026-09-15) are encoded here, as data, so the
mapping is reviewable in one place and never applied ad hoc:

1. "Corrected" is a badge, not an outcome. Every research question carries one
   of four outcomes (success / fail / mixed / unresolved, shown as Goal met /
   Goal missed / Mixed / Open) plus a separate ``corrected`` flag with its
   correction history. The 23 register rows exported with outcome "correction"
   are re-mapped by ``CORRECTION_REMAP``: eight research questions get a real
   outcome, fifteen process checks become ``method-check`` records that are
   listed on their own and excluded from research outcome counts.

2. The count-matched partition audit (docs/MASTER-TABLE.md section 10, added at
   campaign commit d69b23a, CORRECTIONS 211) is imported row by row from the
   pinned commit. ``PARTITION_AUDIT`` maps each row's verdict to an outcome
   with a one-line reason, using ``VERDICT_RULES`` -- the rule families the
   existing register already follows (majority mapping of its 111 rows).

3. Rows appended to MASTER-TABLE after the partition audit (lines 212-217 at
   campaign commit 64e4f47, CORRECTIONS 217-226) are imported the same way by
   ``APPENDED_ROWS``: one explicit rule, research area, batches, figure page and
   one-line reason per row, with the verdict rules of decision 2. The pinned
   file must keep lines 1-211 identical to the partition-audit commit apart from
   its header counts (lines 3 and 5), so no earlier record can drift.

4. Rows the campaign later amends in place (CORRECTIONS 229 at commit 6e33fd8:
   MASTER-TABLE rows 19, 162, 163, 166 and 213) are applied by ``ROW_AMENDMENTS``:
   the pinned file may differ from the appended-rows commit only on those lines
   and header line 3, each record states its outcome before and after with a
   one-line reason, and an outcome may move only where the row's verdict column
   moved (row 19, MT019: Open -> Goal met).

5. The row appended at the cvt1 landing (MASTER-TABLE line 218 at campaign commit
   82867bb, CORRECTIONS 230) is imported by ``LANDED_ROWS`` in the same way. The pinned
   file may differ from the amendment commit only on header line 3 and row 19's
   verdict-cell formatting. Its intervened arms (MUTE, DOSE, INJECT) are named from the
   campaign's hash-pinned ``results/CORPUS-EXCLUSIONS.tsv`` by ``LANDED_INTERVENTIONS``.

6. The row appended at the cgn3 landing (MASTER-TABLE line 219 at campaign commit
   e3a43da, CORRECTIONS 231) is imported by ``CGN3_ROWS``. The pinned file may differ from
   the cvt1 pin only on header line 3 and rows 213 and 218, which CORRECTIONS 231 amended in
   place with the superseded wording kept; ``CGN3_AMENDMENTS`` carries those amendments into
   MT213 and MT218 without moving an outcome.

7. The rows appended at the cvt3 and cvt2 landings (MASTER-TABLE lines 220-221 at campaign commit
   9c5d72a, CORRECTIONS 235-236) are imported by ``CVT23_ROWS``. The pinned file may differ from the
   cgn3 pin only on header lines 3 and 5 and rows 213 and 216, which CORRECTIONS 234 and 236 amended
   in place with the superseded wording kept; ``CVT23_AMENDMENTS`` carries those amendments into MT213
   and MT216 without moving an outcome. Their intervened arms are named from the same exclusion list,
   which may only have grown by appended rows since the cvt1 pin.

8. The rows appended at the cvt4 and cvt5 landings (MASTER-TABLE lines 222-223 at campaign commit
   643264c, CORRECTIONS 240-241) are imported by ``CVT45_ROWS``. The pinned file may differ from the
   cvt3/cvt2 pin only on header lines 3 and 5; no existing row was amended. Their intervened arms (cvt4's
   step-size holds, cvt5's vote weights) are named from the same exclusion list, grown by appended rows only.

9. CORRECTIONS 244 (campaign commit 0ade9cc) inserted bracketed wording amendments into rows 222 and 223 (and line 5,
   which feeds no record). ``C244_AMENDMENTS`` carries them into MT222 and MT223 without moving an outcome; the pinned
   file may differ from the cvt4/cvt5 pin only on those three lines, and only by the inserted brackets.

10. The rows appended at the cvt6 and cvt7 landings (MASTER-TABLE lines 224-225 at campaign commit dae2a49, CORRECTIONS
    246-247) are imported by ``CVT67_ROWS``. The pinned file may differ from the CORRECTIONS 244 pin only on header line 3
    and on line 5, by inserted brackets; no existing row was amended. Their intervened arms (cvt6's step-size and
    complement holds, some runs carrying both; cvt7's group holds) are named from the same exclusion list. The landing's
    in-place corrections of its own registrations (CORRECTIONS 242, 243.4) are checked as bracket insertions and shown on
    MT224 / MT225 as documented limitations.

11. The rows appended at the cvt8 and cvt9 landings (MASTER-TABLE lines 226-227 at campaign commit ba01f54, CORRECTIONS
    252-253) are imported by ``CVT89_ROWS``. The pinned file may differ from the cvt6/cvt7 pin only on header line 3 and on
    line 5, by inserted brackets; no existing row was amended. Their intervened arms (cvt8's group holds, three of them with
    the rest group forced too; cvt9's step-size and complement holds, two of them cut to an update window, so three holds per
    run) are named from the same exclusion list. The landing's one in-place correction of cvt9's registration (CORRECTIONS
    249.3) is checked as a bracket insertion and shown on MT227 as a documented limitation. The CVT89 pin also covers the
    landing's final audit (campaign commit 3cf4201, CORRECTIONS 253.15): MASTER-TABLE may differ from ba01f54 only by row
    227's one in-place wording correction (checked by ``check_in_place_correction``), CORRECTIONS only by the inserted 253.15
    section and its note. ``CVT89_AMENDMENTS`` carries the corrected cell into MT227's scope without moving an outcome.

12. The four MUST-tier landings (MASTER-TABLE lines 228-231 at campaign commit 40d29cf, CORRECTIONS 264-266) are imported
    by ``MUST_ROWS``. The pinned file may differ from the cvt8/cvt9 final-audit pin on header line 3 alone -- line 5, the
    bottom-line paragraph, was deliberately not amended this cycle -- and no line may move; no existing row was amended,
    so no earlier record changes (``MUST_BEARS_ON`` records the relationship). ``docs/CORRECTIONS.md`` is append-only from
    that pin: the ingest commit 77c6de9 added entries 254-263 and the landing replaced its closing "Next free number"
    trailer with entries 264-266. Only ``cmo1`` owes exclusion rows, and they are of a NEW kind: its M9 and W0 arms deviate
    from the standard cell in a base-optimiser CLI FLAG alone, which no patch announces and no CSV column carries, so the
    campaign lists them under the ARGS-value witness kinds of CORRECTIONS 263 (``ARGS_KINDS``). ``MUST_ARGS_DEVIATIONS``
    names those 18 runs; ``args_deviation()`` reads such a row and ``args_witness_line()`` checks it against the run's own
    ``ARGS:`` line under argparse last-wins semantics. ``cst1``, ``cct1`` and ``cmg1`` own no exclusion row at all.

13. The last four MUST-tier landings (MASTER-TABLE lines 232-235 at campaign commit 2972d48, CORRECTIONS 270-273) are
    imported by ``MECH4_ROWS``. The pinned file may differ from the MUST-tier pin on header line 3 and on row 229, which
    CORRECTIONS 273 amended in place, and no line may move; line 5 was again deliberately left unamended. ``docs/CORRECTIONS.md``
    is append-only across the ingest 66a19fb (corpus 3,181 -> 3,253, +72 runs, 0 changed, 45 exclusion rows; entries 267-269)
    and this landing (entries 270-273). All four batches owe exclusion rows, under the existing ``GROUP_HOLD`` and ``VOTE_W``
    kinds and the two kinds CORRECTIONS 269 added, ``DECAY_MASK`` and ``SHADOW_VOTE``; six cwd2 rows carry three kinds at once.
    ``MECH4_AMENDMENTS`` carries the row-229 amendment into MT229 and MOVES its outcome from Open to Mixed, because the frozen
    successor registered at CORRECTIONS 268 reaches a branch whose registered clauses split -- the isolation rescue transfers
    to a second meta step size, the vote-dominance nomination misses its bar by 41 records of 1,500. The Corrected badge is
    NOT set: an outcome moved by later data is not a corrected earlier claim.

Record text is plain text: ``clean()`` removes Markdown bold, emphasis and code marks from every MASTER-TABLE cell and
from the text fields of the campaign's register export (``BASE_TEXT_FIELDS``), which copies the cells with their marks.

IDs: existing IDs are never renumbered. New MASTER-TABLE rows are keyed
``MT<line>`` on their line in the pinned commit (MT175-MT211, then MT212-MT217, then MT218, then MT219, then MT220-MT221,
then MT222-MT223, then MT224-MT225, then MT226-MT227, then MT228-MT231, then MT232-MT235);
no ID from the original register is at or above MT167. Source anchors into
MASTER-TABLE are resolved by row content, never by line number alone, because
the site's IDs were assigned from an uncommitted MASTER-TABLE snapshot that is
one line longer from line 20 on.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

OUTCOMES = ("success", "fail", "mixed", "unresolved")
OUTCOME_LABELS = {"success": "Goal met", "fail": "Goal missed", "mixed": "Mixed", "unresolved": "Open"}
KINDS = ("research", "method-check")
REGISTER_EXPORT = "complete_experiment_register.csv exported 2026-09-15 (snapshot 2026-09-15-d42635e5a8d68f4b)"

# ---------------------------------------------------------------------------
# 1. The 23 rows whose register outcome was "correction" (approved split).
# ---------------------------------------------------------------------------
# id -> (kind, outcome or None for method checks, one-line reason)
CORRECTION_REMAP = {
    # Research questions: a real outcome for the stated goal, plus the badge.
    "MT014": ("research", "success", "The goal was met: the tuned cosine baseline peaks near 3e-3. The earlier 1e-3 reading is the corrected claim."),
    "MT060": ("research", "mixed", "Nodewise beats layerwise but weightwise loses. The 17.2 pp cost claim compared a coarser partition and was corrected."),
    "MT051": ("research", "fail", "The threshold-width robustness readings did not hold: three verdicts reversed and were withdrawn."),
    "MT063": ("research", "fail", "Neither the pooling inversion nor the scalar crossover survives the corrected stratification and metric."),
    "MT071": ("research", "fail", "No tunable shrinkage strength was found: the per-step operator saturates, and the pooling reading was withdrawn."),
    "MT074": ("research", "fail", "r was not shown to be a pooling dial. Closest call in this mapping: g4m also refuted the rescaling explanation, so Mixed is defensible."),
    "MT086": ("research", "fail", "The claimed excess sign agreement does not hold: 53.1% belongs to 62 groups, whose independence null is 55.07%."),
    "MT154": ("research", "fail", "conv2 privilege fails; the rival account needs renaming and its substantive ambiguity remains."),
    # Process checks: errors caught in the campaign's own methods.
    "MT108": ("method-check", None, "Method check: the legacy plateau column averaged different windows across budgets."),
    "MT109": ("method-check", None, "Method check: truncated runs had entered the response surface."),
    "MT110": ("method-check", None, "Method check: the weightwise arm was dropped on a budget and metric mix-up."),
    "MT111": ("method-check", None, "Method check: derived probe fields encoded assumptions the raw data contradicted."),
    "MT112": ("method-check", None, "Method check: probes were tabulated while jobs were still running."),
    "MT113": ("method-check", None, "Method check: guard occupancy was measured at tensor rather than coordinate resolution."),
    "MT115": ("method-check", None, "Method check: a pooled gate statistic hid one pinned seed."),
    "MT116": ("method-check", None, "Method check: argmin summaries were read past their own validity flags."),
    "MT117": ("method-check", None, "Method check: submit-script headers and queue trims did not match the intended configuration."),
    "MT118": ("method-check", None, "Method check: the FairShare submission floor did not address the active queue limit."),
    "MT119": ("method-check", None, "Method check: a stale operational list of unfinished runs was reconciled."),
    "MT122": ("method-check", None, "Method check: the validity-gate prose named the wrong guard."),
    "MT123": ("method-check", None, "Method check: a restricted citation search manufactured an orphan-run backlog."),
    "MT124": ("method-check", None, "Method check: published headline contrasts were audited for unmatched axes."),
    "MT158": ("method-check", None, "Method check: the census of the CIFAR-100 ms=1e-4 stratum was stale."),
}

# ---------------------------------------------------------------------------
# 2. MASTER-TABLE section 10: the count-matched partition audit.
# ---------------------------------------------------------------------------
PARTITION_AUDIT_COMMIT = "d69b23abcc14547e68017dba91d47a40af6f345e"
MASTER_TABLE = "docs/MASTER-TABLE.md"
CORRECTIONS_DOC = "docs/CORRECTIONS.md"
MASTER_TABLE_SHA256 = "e6f158ea1020f020c1753efe62355014b7148804c3c1def0a9bfeea14b380d88"
SECTION_HEADING_LINE = 169
FIRST_ROW, LAST_ROW = 175, 211
SECTION, AREA = 10, "Count-matched partition audit"
NEW_ID_FLOOR = 167  # Existing IDs stop at MT166; everything from MT167 up is reserved for appended rows.
PHASE_ID = "phase-03"  # research_timeline.csv "24 Aug-3 Sep / Partition tests"; CORRECTIONS 104-134 were committed 2026-08-24..09-03.
FIGURE_IDS = ["page-8"]  # The report's count-matched page. The PDF predates section 10 and has no per-row page.
WHY = "Count-matched partition audit (MASTER-TABLE section 10): does the partition matter once the group count is held fixed?"

# Verdict rule families, taken from how the existing register maps verdict
# families to outcomes (CONFIRMED -> success 19/34, REFUTED/DEAD -> fail 15/30,
# OPEN/UNRESOLVED -> unresolved 9/15; split verdicts -> mixed).
VERDICT_RULES = {
    "met": "Goal met: every registered clause answers the question in its registered direction",
    "missed": "Goal missed: the row's registered hypothesis or claim is refuted, dead or dropped",
    "mixed": "Mixed: the registered clauses split between met and missed, or a registered expectation was defied",
    "open": "Open: no registered verdict, or the primary is void, undecided, blocked, unidentifiable or withdrawn before data",
    "method": "Method check: the row audits the campaign's own instrument, design or analysis, not MetaOptimize",
}

# line -> (rule, batches, corrected note or None, one-line reason)
PARTITION_AUDIT = {
    175: ("met", [], None, "Registered headline claim: uniform beats aligned in 20 of 20 count-matched cells. A synthesis; runs link on the per-batch rows."),
    176: ("met", ["tw0"], None, "T1 CONFIRMS and T4 DECIDED: the weightwise deficit is not a clamp artefact."),
    177: ("met", ["at1"], None, "A1 REPRODUCES and A2 CONFIRMS the out-of-sample sign-agreement prediction."),
    178: ("mixed", ["ck1"], None, "K1 REPRODUCES; K2 REFUTES its own registered prediction that the partition does not matter at fixed m. Judgement call: the row's question was answered yes."),
    179: ("met", ["ck1"], None, "K3 CONFIRMS a monotone fine-end slope with no knee."),
    180: ("met", ["cx2"], None, "X1 and X2 CONFIRM; the within-batch rise is a tie, which the scope keeps."),
    181: ("met", ["mm1"], None, "M1 CONFIRMS the first measured count-matched D; M2 REPRODUCES."),
    182: ("mixed", ["pp1"], None, "P1 REPLICATES the size-distribution effect; P2 is a bounded NULL on alignment, not a refutation."),
    183: ("open", ["bn1"], None, "The primary tail test T1 is UNDECIDED (0.005 pp short of SURVIVES); T2 and T3 are secondary."),
    184: ("open", ["ar1"], "A1 SURVIVES and A2 COLLAPSES were later removed from the primaries: the corrected occupancy instrument (CORRECTIONS 117.1) finds ar1 floor-bound.", "A3 VOID: 0 of 12 arms box-free, and the cell was removed from the primaries."),
    185: ("missed", ["cc1"], None, "C1 MIXED, so direction C IS DROPPED: the field does not predict accuracy at matched count (as for MT098)."),
    186: ("met", ["cc1"], None, "C2 REPLICATES D at fresh seeds and C3 COLLAPSES G inside the same batch, as registered."),
    187: ("open", [], "The registered singleton law gap(f) was WITHDRAWN before any run (CORRECTIONS 111.2).", "WITHDRAWN before data, and the cross-architecture test is UNTESTABLE; the question stays unanswered."),
    188: ("mixed", ["fa1"], None, "F2 SURVIVES, but F3 defied its registered expectation (the box changed the optimiser) and G_w is underpowered."),
    189: ("method", [], "The fixed-m dose-response design was WITHDRAWN AND DELETED before a GPU-hour (CORRECTIONS 114.1).", "Audits whether a planned design could measure its target; killed before running."),
    190: ("met", ["g3m"], None, "H1 GENERALISES, H2 NULL REPLICATES and H3 MECHANISM CONFIRMED on ResNet-34."),
    191: ("met", ["gc1"], "\"C100 gives the LARGEST D in the corpus\" was WITHDRAWN at CORRECTIONS 117.11.", "CLOSE-CONFIRMED on CIFAR-100 (the scorer's band() applied to the re-derived numbers)."),
    192: ("method", [], "The box instrument was reading the wrong array (CORRECTIONS 117); ar1 left the primaries and fa1 carries a ceiling disclosure.", "Audits the occupancy instrument every D reading rests on."),
    193: ("method", [], "The imported count slope was DELETED (CORRECTIONS 117.10).", "Audits an analysis correction method: a count slope imported across batches."),
    194: ("open", ["gn1"], None, "NO TRANSFER VERDICT IS ISSUED (commensurability bar failed) and THIS IS NOT A NULL."),
    195: ("missed", [], None, "REFUTED: size-1 groups do not cause the non-monotonicity (CORRECTIONS 119.0)."),
    196: ("open", [], "gf1 was WITHDRAWN because its equivalence branch could never fire (CORRECTIONS 119.9).", "gf1 WITHDRAWN and gf2 BLOCKED on PATCH_NODEFLOOR; the dose question is open."),
    197: ("met", ["rl3"], None, "RULE 11 CLOSED ON R18/CIFAR-10: D is unchanged at matched optima (scope R18/C10 only)."),
    198: ("open", ["aw1"], "The cycle-91 headline \"D TRANSFERS TO AdamW\" was WITHDRAWN at CORRECTIONS 123.1.", "UNRESOLVED at n=6 by the registered scorer."),
    199: ("open", ["gm2"], None, "NO REGISTERED VERDICT: no scorer was registered before the runs, so the favourable result carries no registered outcome."),
    200: ("met", ["hz3"], "\"THE GAP GROWS WITH BUDGET\" was WITHDRAWN at CORRECTIONS 123.2; the trend was repaired by hz3q.", "MECHANISM SURVIVES THE HORIZON: the level survives three times the budget."),
    201: ("open", ["nl1"], None, "NO REGISTERED VERDICT: CORRECTIONS 123.4 registered an analysis rule, not a scorer."),
    202: ("open", ["r50"], None, "NO REGISTERED VERDICT for the ResNet-50 cell."),
    203: ("open", ["ml2"], None, "The meta axis was VOIDED (every ARGS line repeats --alg-meta); the batch enters only as a Lion replicate."),
    204: ("open", [], None, "NOT IDENTIFIABLE FROM THIS CORPUS: the design has rank 3."),
    205: ("missed", [], "\"THE LEVEL SLOPE IS DEAD\" was later amended to M7 NOT SEPARABLE once the gn1-GroupNorm cell was removed (CORRECTIONS 126.1).", "The predictive framing is DEAD: nothing measured predicts D out of sample better than its mean."),
    206: ("open", ["sm3"], None, "VOID AS DESIGNED: every ARGS line repeats --alg-meta; the M9 candidate stayed untested."),
    207: ("met", ["bm2"], None, "REPLICATES at both base-optimiser levels (Q <= 3.841)."),
    208: ("missed", ["sm4"], None, "H1 REFUTED: D >= +0.55 at the second-moment corner, so M9 (a second moment shrinks D) is refuted."),
    209: ("open", [], "\"Identified moderator\" was WITHDRAWN to \"candidate moderator\" at CORRECTIONS 128.1.", "Candidate moderator only: the base optimiser is not separated from its submission batch."),
    210: ("mixed", ["rp1"], None, "T1 NULL replicates the alignment null, but T1b is underpowered and an unexpected run-seed effect is unresolved."),
    211: ("missed", ["hz3q"], "\"D does not grow ... we cannot resolve whether it decays\" was WITHDRAWN at CORRECTIONS 134.", "NOT FLAT -- D DECLINES WITH BUDGET: the registered flatness hypothesis fails, although D stays positive."),
}
RULE_OUTCOME = {"met": "success", "missed": "fail", "mixed": "mixed", "open": "unresolved", "method": None}

# ---------------------------------------------------------------------------
# 3. Rows appended after the partition audit (CORRECTIONS 217-226).
# ---------------------------------------------------------------------------
APPENDED_COMMIT = "64e4f47caaf0fa6c9834e49ee54cd12ff45d8bc5"
APPENDED_MASTER_TABLE_SHA256 = "4cfaa96ee4fc563baeb620afea0e08c91c5a4979d12c39f9e45025ad278b41d2"
APPENDED_FIRST_ROW, APPENDED_LAST_ROW = 212, 217
APPENDED_HEADER_LINES = {3, 5}  # Run/GPU-hour header and bottom line, amended in place by each landing.
AREAS = {1: "Baseline comparisons", 9: "Mechanism and isolation", 10: AREA}  # 10: the appended cgw1 row (MT240), CORRECTIONS 296
# research_timeline.csv ends at 14 Sep. The notebook adds this documented phase:
# the rows were launched 2026-09-15 (CORRECTIONS 216, commit 918c0aa) and landed
# 2026-09-16 (CORRECTIONS 217-226, commits 2d09417-64e4f47).
ADDED_PHASES = [{
    "id": "phase-09", "date": "2026-09-15", "period": "15-17 Sep", "title": "Off BatchNorm, off residuals, long horizons",
    "test": "GroupNorm and residual-free ResNet-18 isolation; VGG rescue at 328 epochs; unaugmented CIFAR-100 baseline",
    "observed_result": "Gap and carrier set transfer to GroupNorm; one BN scale rescues without residuals; VGG rescue holds at 328 epochs; unaugmented deficit +12.18 pp",
    "next_question": "Identity versus magnitude remains unresolved on every network",
    # MT212-MT217 (CORRECTIONS 217-226), MT218, the cvt1 landing of 16 Sep (CORRECTIONS 227 launch, 230 landing),
    # MT219, the cgn3 landing of 16 Sep (CORRECTIONS 228 launch, 231 landing), and MT220-MT221, the cvt3 and
    # cvt2 landings of 16 Sep (CORRECTIONS 232 / 233 registration and launch, 235 / 236 landing), and MT222-MT223,
    # the cvt4 and cvt5 batches registered and launched on 16 Sep (CORRECTIONS 237 / 238) and landed at campaign
    # commit 643264c (CORRECTIONS 240 / 241; 2026-09-17 00:10 +0200, 16 Sep 22:10 UTC).
    # MT224-MT225, the cvt6 and cvt7 batches registered and launched on 16-17 Sep (CORRECTIONS 242 / 243; 16 Sep 23:19 UTC)
    # and landed at campaign commit dae2a49 (CORRECTIONS 246 / 247; 2026-09-17 03:54 +0200, 17 Sep 01:54 UTC).
    # MT226-MT227, the cvt8 and cvt9 batches registered on 17 Sep (CORRECTIONS 249 / 248; campaign commits 0ad379d and 5c9934d,
    # 03:21-03:26 UTC), launched the same morning (249.12-13 / 248.12-13) and landed at campaign commit ba01f54 (CORRECTIONS
    # 252 / 253; 2026-09-17 08:50 +0200, 17 Sep 06:50 UTC). The phase still ends on 17 Sep.
    "experimentIds": [f"MT{line}" for line in range(APPENDED_FIRST_ROW, 227 + 1)],
    "source": "docs/CORRECTIONS.md 216-226 at 64e4f47",
}, {
    # The MUST-have tier of docs/LIMITS-PREP.md section 5.1, approved by the operator and registered as a block on
    # 17 Sep (CORRECTIONS 255 cmo1, 256 cst1, 257 cct1, 259 cmg1), launched the same day and landed on 18 Sep: the
    # batches were scored at campaign commits 6d09d1f and 8e341e4, ingested once at 77c6de9 (corpus 3,127 -> 3,181)
    # and written up at 40d29cf (CORRECTIONS 264-266). A separate phase from phase-09: these four ask what the carrier
    # account NEEDS -- another base-optimiser configuration, another meta step size, a non-collapsing dataset, the
    # merge direction -- rather than which network or horizon it survives.
    "id": "phase-10", "date": "2026-09-17", "period": "17-18 Sep", "title": "What the carrier account needs",
    "test": "One-factor-at-a-time base-optimiser legs; a second meta step size; the one non-collapsing cell; merging the carriers instead of isolating them",
    "observed_result": "The collapse survives momentum 0.9 but not weight decay 0; CIFAR-10 does not collapse and its carriers never dominate; the carrier merge costs ~4 pp, inside the 5 pp bar; the second meta step size is blocked by a scorer tolerance defect",
    "next_question": "Which weight-decay route acts, and is carrier separation ever necessary",
    # MUST_FIRST_ROW / MUST_LAST_ROW are 228 and 231; they are defined further down the file, so the four IDs are written out.
    "experimentIds": ["MT228", "MT229", "MT230", "MT231"],
    "source": "docs/CORRECTIONS.md 264-266 at 40d29cf",
}, {
    # The rest of the MUST-have tier of docs/LIMITS-PREP.md section 5.1 (R1 cvt10, S1b cwd1, N1 csv1, N4 cwd2), registered
    # and launched on 18 Sep (CORRECTIONS 258, 260, 261, 262; campaign commits 8e341e4-635132f, 18:12-18:45 UTC), scored the
    # same night, ingested once at 66a19fb (corpus 3,181 -> 3,253) and written up at 2972d48 (CORRECTIONS 270-273). A
    # separate phase from phase-10: these four ask WHICH TENSORS carry the collapse and THROUGH WHICH ROUTE, and whether the
    # carrier's step size is necessary -- a mechanism decomposition, not another thing the carrier account needs. The same
    # landing amended row 229 in place (MT229), carrying CORRECTIONS 268's frozen successor into the record.
    "id": "phase-11", "date": "2026-09-18", "period": "18-19 Sep", "title": "Which tensors, and through which route",
    "test": "One carrier held against three at PlainNet's dose; the coupled weight decay masked on the normalisation scales, and on one carrier scale; one carrier's applied step size separated from the vote it casts",
    "observed_result": "Any one of the three ResNet carriers held alone stalls the run, provided the others still vote; removing the coupled weight decay from the 20 BatchNorm scales removes the collapse entirely, and removing it from one PlainNet scale removes the held-step damage too; both the applied step and the vote carry part of that damage",
    "next_question": "By what route the coupled decay acts, which the instrument cannot watch on the arms that stall",
    # MECH4_FIRST_ROW / MECH4_LAST_ROW are 232 and 235; they are defined further down the file, so the four IDs are written out.
    "experimentIds": ["MT232", "MT233", "MT234", "MT235"],
    "source": "docs/CORRECTIONS.md 270-273 at 2972d48",
}, {
    # cwd3, the one batch of the cycle: the ResNet carrier-only decay mask that CORRECTIONS 271 and 274 named as the gap
    # and that the campaign's own write-up listed as its top-ranked open question. Registered, proved and launched on
    # 19 Sep (CORRECTIONS 275, campaign commits 3c22eda-1bbc6d1), scored the same afternoon (3a530b9), ingested once at
    # 064dff6 (corpus 3,253 -> 3,268) and written up at 97eb049 (CORRECTIONS 278). A separate phase from phase-11: that
    # phase asked which tensor SET and which route; this one asks whether the three nominated carriers' OWN decay is
    # enough, against two count-matched non-carrier sets at the same depth.
    "id": "phase-12", "date": "2026-09-19", "period": "19 Sep", "title": "The carriers' own decay",
    "test": "Coupled weight decay removed from the three nominated ResNet carriers alone, against a count-matched non-carrier triple, a class-pure non-carrier pair and the network-wide 20-scale mask, every arm scalar",
    "observed_result": "The three carriers alone remove the collapse as fully as all twenty scales do, while both matched non-carrier sets stay on the floor; the residual left to the other seventeen is a bound, not a measurement, and its sign is horizon-dependent",
    "next_question": "Whether it is those three tensors or any three normalisation scales of that width at that depth, which needs a two-carrier arm nobody has run",
    # MECH5_FIRST_ROW / MECH5_LAST_ROW are both 236; they are defined further down the file, so the ID is written out.
    "experimentIds": ["MT236"],
    "source": "docs/CORRECTIONS.md 278 at 97eb049",
}, {
    # cwd4, the batch that decomposes cwd3's three-carrier mask: registered, proved and launched on 20 Sep (CORRECTIONS
    # 280, campaign commits 5f16f40-edf3bfa), scored the same afternoon (5b5b372), ingested at 8b9fbd2 (corpus 3,268 ->
    # 3,289) and written up at 26baf4a (CORRECTIONS 283). A separate phase from phase-12: that phase asked whether the
    # three nominated carriers' own decay is enough; this one asks whether the rescue is a matter of WHICH tensors or of
    # HOW MANY, with a class-pure matched pair at count two and a dose ladder of three single-carrier arms.
    "id": "phase-13", "date": "2026-09-20", "period": "20 Sep", "title": "Which carriers, or how many",
    "test": "The coupled weight decay removed from the carrier PAIR against a class-pure, count-, width-, depth- and numel-matched carrier-free pair, and from each of the three carriers alone, every arm scalar and the three-carrier mask re-run in batch as the recovery reference",
    "observed_result": "At matched count two the carrier pair recovers and the matched non-carrier pair stays on the floor, and two of the three single carriers reach recovery alone, so the effect is not a function of count alone; but the third single lands far outside its registered band, the nearer of the two recovering singles clears its bar by 0.40 SE, and the term-magnitude and position-class rivals are both sharper than before",
    "next_question": "Whether the ordering among the singles is identity, Kim's position class or term magnitude, none of which this network can separate",
    # MECH6_FIRST_ROW / MECH6_STEP_ROW are both 237; they are defined further down the file, so the ID is written out.
    "experimentIds": ["MT237"],
    "source": "docs/CORRECTIONS.md 283 at 26baf4a",
}, {
    # cwd5, the coupled weight-decay ladder the area chair's corner-case charge made the gating experiment: registered,
    # proved and launched on 20 Sep (CORRECTIONS 281, campaign commits 1d2a8b7-7ff4685), scored the same evening
    # (bfda844), ingested at 91fcd57 (corpus 3,289 -> 3,316) and written up at 8554afa (CORRECTIONS 285), 20 Sep 22:41
    # UTC. A separate phase from phase-13: that phase asks WHICH tensors inside the mechanism, this one asks whether the
    # mechanism exists at all away from the campaign's own weight decay -- a scope question, not a decomposition.
    "id": "phase-14", "date": "2026-09-20", "period": "20-21 Sep", "title": "Does the collapse exist at normal weight decays",
    "test": "Four rungs of the base weight decay -- 0.1, 1e-2, 1e-3 and the standard CIFAR 5e-4 -- with BOTH grains run in batch at every rung, plus a carrier companion at the second rung",
    "observed_result": "The collapse is present at the campaign's coupled 0.1 and absent at all three lower rungs, where the scalar arm is if anything slightly ahead of its own layerwise arm; four rungs bracket the change between 1e-2 and 0.1 and locate nothing inside that decade, and the carrier companion is unreadable because its rung did not collapse",
    "next_question": "Where inside the unrun decade the configuration breaks, and whether decoupled weight decay behaves the same way",
    # MECH6_LAST_ROW is 238; it is defined further down the file, so the ID is written out.
    "experimentIds": ["MT238"],
    "source": "docs/CORRECTIONS.md 285 at 8554afa",
}, {
    # caw2, the ICML plan's Step 1 (the standard-recipe cell), redesigned after caw1 was withdrawn before launch: registered,
    # proved and submitted on 22 Sep (CORRECTIONS 290, campaign commits 87be0fa-b2eeed5), scored the same morning (aa80a93),
    # ingested at 477a853 (corpus 3,316 -> 3,383, with cgw1) and written up at 8a99001 (CORRECTIONS 295). A separate phase
    # from phase-14: that phase asked whether the collapse exists away from the campaign's weight decay on the SGDm base;
    # this one asks whether it exists at the standard AdamW + Adam recipe, with each confound in its own cell.
    "id": "phase-15", "date": "2026-09-22", "period": "22 Sep", "title": "Does the collapse reach the standard recipe",
    "test": "Nine arms on the mechanism cell: the SGDm + Lion control, the meta swap alone, the base swap alone, the standard AdamW + Adam recipe, and that recipe at ten times the weight decay as a dose arm, with the realised per-step shrink measured on every arm",
    "observed_result": "The standard recipe does not collapse at either decay, the meta swap alone still collapses and the base swap alone does not; but the dose arm reached only about a seventeenth of the control's realised shrink, so whether the recipe would survive the control's dose is undecided, and the base swap cannot be separated from the lower dose it produces",
    "next_question": "Whether the standard recipe collapses when its realised shrink is held at the control's level",
    # MECH7_FIRST_ROW is 239; it is defined further down the file, so the ID is written out.
    "experimentIds": ["MT239"],
    "source": "docs/CORRECTIONS.md 295 at 8a99001",
}, {
    # cgw1, the ICML plan's G3 (the count-matched audit's core cell at standard weight decay): registered on 22 Sep
    # (CORRECTIONS 291, cfc12ff), submitted the same night (38e96af), scored at e9d3884, ingested at 477a853 with caw2 and
    # written up at 8a99001 (CORRECTIONS 296). A separate phase from phase-15: it asks whether the partition audit's own
    # headline, not the mechanism, depends on the weight decay.
    "id": "phase-16", "date": "2026-09-22", "period": "22 Sep", "title": "Does the partition audit survive standard weight decay",
    "test": "The audit's most load-bearing cell verbatim except the base weight decay, at 0.1, 1e-2 and the standard value 5e-4, with the uniform and aligned partitions at matched count and plain scalar and layerwise run in batch at every rung",
    "observed_result": "The uniform-over-aligned sign is reproduced at 0.1 and holds at 1e-2, and at 5e-4 it lies between the registered bars; at 5e-4 plain scalar is above both partitions by 2.6 to 2.8 points, but the harness's 5e-4 is far less decayed than a standard recipe, so no reading settles whether the audit effect depends on non-standard decay",
    "next_question": "Whether the scalar-over-partition reading holds on the audit's CIFAR-100 cell and survives re-tuning at 5e-4",
    # MECH7_LAST_ROW is 240; it is defined further down the file, so the ID is written out.
    "experimentIds": ["MT240"],
    "source": "docs/CORRECTIONS.md 296 at 8a99001",
}]

# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Figure pages: the report PDF predates these rows, so each links the page of the
# record it extends (ResNet isolation page 19, VGG rescue page 21, baseline page 23).
APPENDED_ROWS = {
    212: ("met", 9, ["cgn1"], ["page-19"], None, "GAP-REPLICATES and FULL-SIZE: the scalar-to-layerwise gap exists on GroupNorm ResNet-18 (a 100-epoch snapshot)."),
    213: ("mixed", 9, ["cpl1"], ["page-19"], None, "HEAD-CARRIES-PLAIN: ISO rescues against its twin, but ONE stays at the floor and 189.2's literal forms missed their registered branches."),
    214: ("met", 9, ["cvh1"], ["page-21"], None, "RESCUE-SURVIVES: rho 1.0001 at 328 epochs, and all six registered point predictions landed in band."),
    215: ("met", 1, ["cuc1"], ["page-23"], None, "DEFICIT-HOLDS with an interior ladder: tuned SGD beats the best method cell by +12.18 pp without augmentation on CIFAR-100."),
    216: ("met", 9, ["cgn2"], ["page-19"], None, "IDENTITY-TRANSFERS-GN: the three carriers rescue on GroupNorm, the matched set does not, and 50 alone rescues."),
    217: ("met", 9, ["cpl2"], ["page-19"], None, "HEAD-CARRIES-ALONE-PLAIN: layer4.1.bn2.weight alone rescues, its twin does not, and D_PAIR stays inside the frozen 5.0 pp bar."),
}
# Earlier records these rows bear on. The export does not rewrite earlier records;
# they are listed here so the relationship stays reviewable.
APPENDED_BEARS_ON = {
    215: ["MT019", "MT020"],  # closes row 19's two remaining counts (CIFAR-100, meta-side tuning)
    214: ["MT166"],           # retires cvi1's HORIZON-100-ONLY bound for the ISO arm
    216: ["MT162", "MT163"],  # the BatchNorm isolation and identity legs, repeated on GroupNorm
    217: ["MT213"],           # measures cpl1's decomposition by subtraction
}


# ---------------------------------------------------------------------------
# 4. In-place amendments to existing MASTER-TABLE rows (CORRECTIONS 229).
# ---------------------------------------------------------------------------
# The bookkeeping pass at campaign commit 6e33fd8 edited six MASTER-TABLE lines in
# place (header line 3 and rows 19, 162, 163, 166, 213) and moved no line. Each
# amended row that has a record here is listed with the record's outcome before
# and after, a one-line reason, and the note shown with the record. Only row 19's
# verdict column changed (OPEN -> CONFIRMED, RESCOPED 229), so MT019 is the only
# outcome change; the other four keep their outcome and gain the amendment note.
# An outcome moved by later data is not a corrected earlier claim, so the
# Corrected badge is not set; the previous wording is kept in a warning instead.
AMENDMENT_COMMIT = "6e33fd8de088cae04d0ecf7805f7fd86894aeba7"
AMENDMENT_MASTER_TABLE_SHA256 = "c2fe5c47a79daf3d29e231d32aca1968510e838add6775b1da6ed4de1e17bbf2"
AMENDMENT_SOURCE = "docs/CORRECTIONS.md [CORRECTIONS 229]"
AMENDMENT_MARK = "[AMENDED at cycle 152, CORRECTIONS 229"
AMENDED_LINES = {3, 19, 162, 163, 166, 213}
# id -> (MASTER-TABLE line, outcome before, outcome after, verdict token the amended
#        verdict cell must start with (None: the verdict cell must be unchanged),
#        batches added, one-line reason, replacement result/reason/scope or None)
ROW_AMENDMENTS = {
    "MT019": (19, "unresolved", "success", "CONFIRMED, RESCOPED 229", ["cuc1"],
              "CONFIRMED, RESCOPED 229: cau1 (CIFAR-10, +3.6173 pp) and cuc1 (CIFAR-100, +12.1780 pp against a 5-cell meta grid) closed the two counts that kept the fairness audit Open.",
              {"result": "Every registered fairness flaw is closed: budget and search budget (CORRECTIONS 30/44) and the parent's unaugmented setup on both datasets, where tuned SGD+momentum+cosine still beats the method by +3.6173 pp (CIFAR-10, cau1) and +12.1780 pp (CIFAR-100, cuc1).",
               "reason": "Goal met: the deficit is not an artefact of the three audited unfairnesses (MASTER-TABLE row 19 moved OPEN -> CONFIRMED, RESCOPED at CORRECTIONS 229).",
               "scope": "Limits that travel with the verdict: one baseline family (SGD+momentum+cosine); the paper's alpha0 1e-6 is untested on CIFAR-100 (on CIFAR-10 the paper's config m6 is 15.1160 pp behind); one granularity per batch, so meta-side tuning is bounded, not exhausted (cuc1's best cell sits at the ms-HIGH grid edge, +0.28 SE above the centre). No method-side counterweight, ResNet18 only, 100 epochs only, and the AUC reading reverses at the best baseline rung on both datasets."}),
    "MT163": (162, "success", "success", None, [],
              "Outcome unchanged (IDENTITY-OPERATIVE holds). 'ResNet-only' is rescoped to this batch's DEPTH control: the nominated set rescuing where a matched non-carrier set does not has since been measured on VGG (cvi1), GroupNorm (cgn2) and PlainNet (cpl2). The tensor list does not transfer, and identity vs magnitude is still not separated on any network.",
              None),
    "MT164": (163, "success", "success", None, [],
              "Outcome unchanged (GAP-REPLICATES). 'No isolation arm has run on VGG' and 'bn8.weight is a hypothesis' are overtaken: cvi1 isolated bn8.weight against its twin (+31.3033 pp) and cvh1 held the rescue to 328 epochs. The gap and the isolation stay separate claims for the tensor list and for identity vs magnitude.",
              None),
    "MT020": (166, "success", "success", None, [],
              "Outcome unchanged (DEFICIT-HOLDS on CIFAR-10). 'CIFAR-100 not licensed' and 'the FAIR row stays open' are answered by cuc1 (MT215, +12.1780 pp) and MT019's move to Goal met. This batch's own method side is still two alpha0 cells at one meta-stepsize.",
              None),
    "MT213": (213, "mixed", "mixed", None, [],
              "Outcome unchanged (Mixed). 'By subtraction' and 'never isolated alone' are overtaken by cpl2 (MT217): layer4.1.bn2.weight alone rescues (+52.1367 pp over its twin). 'Alone' holds only against the bar (ISO minus HEAD is +3.5940 pp inside the 5.0 pp bar), HEAD's rescue is a delay, and identity vs magnitude is still open (cvt1 has not landed).",
              None),
}


# ---------------------------------------------------------------------------
# 5. Rows appended after the amendments (CORRECTIONS 230).
# ---------------------------------------------------------------------------
# Campaign commit 82867bb (cycle 152, CORRECTIONS 230) appended the cvt1 landing as
# line 218, recounted header line 3, and fixed a formatting slip in row 19's verdict
# cell (the superseded token OPEN moved into a bold [SUPERSEDED: ...] bracket; no
# verdict or wording change). Nothing else may differ from the amendment pin, and
# no line may move. The row is imported with the verdict rules of decision 2.
LANDED_COMMIT = "82867bb0880956a0563b2a5a5bc17fce67f87f1a"
LANDED_MASTER_TABLE_SHA256 = "030b36ce3bc89d5902e50024cc85f36ab2a2898a9bc35b0af83a05f4398e30b7"
LANDED_FIRST_ROW, LANDED_LAST_ROW = 218, 218
LANDED_EDITED_LINES = {3, 19}
LANDED_ROW19_TOKEN, LANDED_ROW19_BRACKET = "CONFIRMED, RESCOPED 229", "[SUPERSEDED: OPEN]"
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Mixed, not Goal met: the returned branch answers both halves of the question, but its
# registered account (STEP-AND-VOTE, 227.5) missed one registered band -- INJECT 30.14
# against 8-30 -- which is "a registered expectation was defied" (VERDICT_RULES["mixed"]).
LANDED_ROWS = {
    218: ("mixed", 9, ["cvt1"], ["page-19"], None, "STEP-SIZE-NEEDED-VOTE-SUFFICES: muting layer4.1.bn2.weight's vote while it keeps the shared step size does not rescue (MUTE at k01), and a carrier-sized vote cast by its bn1 twin collapses HEAD's complement (P_INJECT +34.65 pp); but INJECT is a partial collapse (INJECT-PARTIAL) and misses its registered 8-30 band by 0.14 pp."),
}
LANDED_BEARS_ON = {218: ["MT213", "MT217"]}  # cpl1 / cpl2: the PlainNet HEAD rescue whose mechanism cvt1 intervenes on
# The intervened arms. results/all_runs.csv has no column for the vote-weight patch, so
# MUTE and DOSE rows carry k01's cell key and INJECT rows HEAD's; the campaign lists them in
# results/CORPUS-EXCLUSIONS.tsv (hash-pinned below). Wording follows CORRECTIONS 230.
INTERVENTIONS_TSV = "results/CORPUS-EXCLUSIONS.tsv"
INTERVENTIONS_TSV_SHA256 = "31105e32a9c9c7566863330a989ef986f047ae5fa7083fb61c2b0e1af1284eb5"
LANDED_INTERVENTIONS = {
    "MT218": {
        "batch": "cvt1", "arms": {"MUTE": 3, "DOSE": 3, "INJECT": 3},
        "title": "MUTE, DOSE and INJECT are vote-weight interventions, not plain arms",
        "note": ("MUTE, DOSE and INJECT ran PATCH_VOTEWEIGHT (VOTE_W=<tensor>:<w> multiplies one tensor's term inside the "
                 "unnormalised shared meta-gradient sum, before the sign). MUTE = scalar grouping with layer4.1.bn2.weight's "
                 "term x0 (it keeps the shared step size and stops voting); DOSE = the same x0.1; INJECT = HEAD's grouping "
                 "with the twin layer4.1.bn1.weight x691 (fixed K, sign kept) inside the complement. VOTE_W rides only the "
                 "run's own VOTE_W: witness line, so the run inventory writes MUTE and DOSE rows with k01's cell key "
                 "(granularity scalar) and INJECT rows with HEAD's cell key: seed for seed they differ from those arms only in "
                 "run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain scalar / plain HEAD "
                 "measurements. The 9 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them before pooling runs by "
                 "cell. The k01 and HEAD arms print VOTE_W: off and are ordinary measurements of their cells."),
    },
}


def landed_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt1 landing; only the header, row 19's formatting and one appended row may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{LANDED_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != LANDED_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {LANDED_COMMIT[:12]} does not match the pinned landed-row bytes")
    lines = raw.decode("utf-8").splitlines()
    before = amended_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != LANDED_LAST_ROW or len(before) != LANDED_FIRST_ROW - 1 or changed != LANDED_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {LANDED_COMMIT[:7]} moved a line or edited lines other than {sorted(LANDED_EDITED_LINES)}: {sorted(changed)}")
    old, new = table_cells(before[18]), table_cells(lines[18])
    moved_at = "Moved at cycle 152"
    if (len(new) != 7 or [c for i, c in enumerate(old) if i != 4] != [c for i, c in enumerate(new) if i != 4]
            or not clean(new[4]).startswith(LANDED_ROW19_TOKEN) or LANDED_ROW19_BRACKET not in clean(new[4])
            or old[4][old[4].index(moved_at):] != new[4][new[4].index(moved_at):]):
        raise ValueError("MASTER-TABLE row 19 changed beyond its verdict-cell formatting fix")
    return lines


def intervened_runs(repo: Path) -> dict[str, dict]:
    """job_id -> the campaign's exclusion row for runs whose CSV cell key hides a harness intervention."""
    raw, previous = b"", None
    # Each pin must be a byte prefix of the next: the list may only grow by appended rows.
    for commit, expected in [(LANDED_COMMIT, INTERVENTIONS_TSV_SHA256), (CVT23_COMMIT, CVT23_INTERVENTIONS_TSV_SHA256), (CVT45_COMMIT, CVT45_INTERVENTIONS_TSV_SHA256),
                             (CVT67_COMMIT, CVT67_INTERVENTIONS_TSV_SHA256), (CVT89_COMMIT, CVT89_INTERVENTIONS_TSV_SHA256),
                             (MUST_COMMIT, MUST_INTERVENTIONS_TSV_SHA256), (MECH4_COMMIT, MECH4_INTERVENTIONS_TSV_SHA256),
                             (MECH5_COMMIT, MECH5_INTERVENTIONS_TSV_SHA256),
                             (MECH6_CWD4_INGEST_COMMIT, MECH6_CWD4_INTERVENTIONS_TSV_SHA256),
                             (MECH6_INGEST_COMMIT, MECH6_INTERVENTIONS_TSV_SHA256),
                             (MECH7_INGEST_COMMIT, MECH7_INTERVENTIONS_TSV_SHA256)]:
        pinned = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{INTERVENTIONS_TSV}"])
        if hashlib.sha256(pinned).hexdigest() != expected:
            raise ValueError(f"{INTERVENTIONS_TSV} at {commit[:12]} does not match the pinned bytes")
        if not pinned.startswith(raw):
            raise ValueError(f"{INTERVENTIONS_TSV} changed rows listed at {previous[:7]}; it may only grow by appended rows")
        raw, previous = pinned, commit
    body = [line for line in raw.decode("utf-8").splitlines() if line and not line.startswith("#")]
    rows = list(csv.DictReader(body, delimiter="\t"))
    runs = {}
    for eid, spec in {**LANDED_INTERVENTIONS, **CVT23_INTERVENTIONS, **CVT45_INTERVENTIONS, **CVT67_INTERVENTIONS, **CVT89_INTERVENTIONS,
                      **MECH4_INTERVENTIONS, **MECH5_INTERVENTIONS, **MECH6_INTERVENTIONS}.items():
        for row in listed_rows(rows, eid, spec):
            if is_args_deviation(row["witness"]):
                raise ValueError(f"{INTERVENTIONS_TSV} lists {row['run']} as a patch intervention with an ARGS-value witness")
            intervention_kinds(row["intervention"])  # every listed hold must be a known kind
            runs[row["job_id"]] = {**row, "experimentId": eid, "argsDeviation": False}
    # CORRECTIONS 263's ARGS-value rows: no patch ran, so the row is read by its flag and checked against the run's own
    # ARGS line rather than a "<PATCH>: on" line (CORRECTIONS 255, cmo1's M9 and W0 arms).
    for eid, spec in {**MUST_ARGS_DEVIATIONS, **MECH6_ARGS_DEVIATIONS, **MECH7_ARGS_DEVIATIONS}.items():
        for row in listed_rows(rows, eid, spec):
            if not is_args_deviation(row["witness"]):
                raise ValueError(f"{INTERVENTIONS_TSV} lists {row['run']} as an ARGS-value deviation without an ARGS-value witness")
            kind, _flag, value, _standard = args_deviation(row["intervention"], row["witness"])  # the flag, its value and the witness must agree
            extra = args_extra_args(row["intervention"], row["witness"])
            if extra:
                # CORRECTIONS 294's TWO-ARGS row: listed ONCE by one ARGS kind, every deviating kind registered per
                # (batch, arm). An unregistered run deviating on two ARGS kinds stops the export, as it FAILs on the
                # campaign side.
                registered = MECH7_MULTI_ARGS.get((row["batch"], row["arm"]))
                if registered is None:
                    raise ValueError(f"{INTERVENTIONS_TSV} lists {row['run']} as an unregistered TWO-ARGS row")
                listed = {kind: value, **{k: v for k, _f, v in extra}}
                if set(listed) != set(registered) or any(not args_equal(listed[k], registered[k]) for k in registered):
                    raise ValueError(f"{INTERVENTIONS_TSV} lists {row['run']} with ARGS axes {listed} against the registered {registered}")
            runs[row["job_id"]] = {**row, "experimentId": eid, "argsDeviation": True}
    if len(runs) != len(rows):
        raise ValueError(f"{INTERVENTIONS_TSV} lists runs with no registered intervention record")
    return runs


def listed_rows(rows: list[dict], eid: str, spec: dict) -> list[dict]:
    """The exclusion rows of one record's batch, with the registered arm counts checked."""
    mine = [row for row in rows if row["batch"] == spec["batch"]]
    arms = {arm: sum(row["arm"] == arm for row in mine) for arm in spec["arms"]}
    if arms != spec["arms"] or len(mine) != sum(spec["arms"].values()):
        raise ValueError(f"{INTERVENTIONS_TSV} does not list the registered intervened arms of {eid}: {arms}")
    return mine


# The exclusion list's intervention column is free text: one PATCH=value per hold, separated by spaces (CORRECTIONS 245:
# cvt6's forced arms carry BETA_HOLD and COMP_HOLD at once; CORRECTIONS 251: cvt8's forced arms GROUP_HOLD and REST_HOLD,
# cvt9's EARLY / LATE BETA_HOLD, COMP_HOLD and WINDOW_HOLD). Each kind is named on the run page; an unknown patch stops
# the export rather than being guessed. REST_HOLD forces the rest group (the complement of a GROUP_HOLD group) onto a
# replay; WINDOW_HOLD cuts the run's BETA_HOLD trajectory to an update window <n0>:<n1|end>, the floor outside.
# CORRECTIONS 269 added the last two, and cwd2's HIGHWD0 / LOWWD0 are the first rows to carry three kinds at once:
# DECAY_MASK sets the BASE optimiser's coupled weight decay to 0 on a named tensor set (<spec>, in the weight update and in
# the meta trace); SHADOW_VOTE separates one tensor's APPLIED step size from the term it casts into the shared
# meta-gradient sum (<vote>:<applied>:<name>).
INTERVENTION_KINDS = {"VOTE_W": "vote-weight", "BETA_HOLD": "step-size hold", "COMP_HOLD": "complement step-size hold",
                      "GROUP_HOLD": "group step-size hold", "REST_HOLD": "rest-group step-size hold", "WINDOW_HOLD": "update-window hold",
                      "DECAY_MASK": "coupled weight-decay mask", "SHADOW_VOTE": "shadow-vote"}


def intervention_kinds(intervention: str) -> list[tuple[str, str]]:
    """[(patch, value), ...] for the exclusion list's intervention cell, in the order written."""
    parts = intervention.split(" ") if intervention else []
    kinds = []
    for part in parts:
        patch, sep, value = part.partition("=")
        if not sep or not value or patch not in INTERVENTION_KINDS:
            raise ValueError(f"Unknown intervention in {INTERVENTIONS_TSV}: {intervention!r}")
        kinds.append((patch, value))
    if not kinds or len({patch for patch, _ in kinds}) != len(kinds):
        raise ValueError(f"Unreadable intervention in {INTERVENTIONS_TSV}: {intervention!r}")
    return kinds


def intervention_opening(looks_like: str) -> str:
    """The run note's opening clause. cvt10's ISOSPLIT is the one listed arm with NO plain twin at its cell key, and the
    exclusion list says so in the looks_like column instead of naming an arm (CORRECTIONS 258.9, 270.6 F4); "Not a plain
    no free arm at this cell key (...) measurement" is not a sentence, so that row gets its own wording."""
    if not looks_like.startswith("no free arm"):
        return f"Not a plain {looks_like} measurement"
    key = looks_like.split("(", 1)[1].rsplit(")", 1)[0] if "(" in looks_like else looks_like
    return f"Not a plain measurement of its cell key ({key}), which no free arm anywhere in the corpus shares"


def intervention_phrase(intervention: str) -> str:
    """'step-size hold intervention'; 'a and b interventions' for a two-kind run; 'a, b and c interventions' for three."""
    names = [INTERVENTION_KINDS[patch] for patch, _ in intervention_kinds(intervention)]
    return f"{names[0]} intervention" if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] + " interventions"


def additional_witness(lines: list[str], patch: str, value: str) -> str:
    """The run log's one '<patch>: on' line, checked against the listed value.

    Each kind has its own value grammar, and a value written for another kind is refused rather than guessed: floor,
    tri:<P> or rec:<id> for the step-size holds, <n0>:<n1|end> for WINDOW_HOLD, <spec> for DECAY_MASK, and
    <vote>:<applied>:<name> for SHADOW_VOTE.

    The exclusion list carries the witness of a run's first hold only; every further hold is witnessed by the run's own log."""
    found = [line for line in lines if line.startswith(f"{patch}: on ")]
    if len(found) != 1:
        raise ValueError(f"A run log must print exactly one '{patch}: on' line; found {len(found)}")
    mode = value.rsplit(":", 2)
    window = re.fullmatch(r"(\d+):(\d+|end)", value)
    if patch == "DECAY_MASK":
        # The mask names its tensor set, which the log echoes as spec=<value> beside the count it resolved to.
        if not value or ":" in value or " " in value:
            raise ValueError(f"Unreadable {patch} value: {value!r}")
        expected = f" spec={value} "
    elif patch == "SHADOW_VOTE":
        vote, applied, name = (mode + ["", "", ""])[:3]
        if len(mode) != 3 or not vote or not applied or not name:
            raise ValueError(f"Unreadable {patch} value: {value!r}")
        # The named tensor appears in the log's items= list, between its index and its element count.
        if f":{name}:" not in found[0]:
            raise ValueError(f"The run log's {patch} line does not match the listed {value!r}")
        expected = f" vote={vote} applied={applied} "
    elif patch == "WINDOW_HOLD" or window:
        if patch != "WINDOW_HOLD" or not window:
            raise ValueError(f"Unreadable {patch} value: {value!r}")
        expected = f" n0={window[1]} n1={window[2]} "
    elif value.endswith(":floor") or value == "floor":
        expected = " mode=floor"
    elif len(mode) >= 2 and mode[-2] == "tri":
        expected = f" mode=tri P={mode[-1]} "
    elif len(mode) >= 2 and mode[-2] == "rec":
        expected = f" mode=rec id={mode[-1]} "
    else:
        raise ValueError(f"Unreadable {patch} value: {value!r}")
    if expected not in found[0] + " ":
        raise ValueError(f"The run log's {patch} line does not match the listed {value!r}")
    return found[0]


# The exclusion list's witness column normally quotes the run's own "<PATCH>: on ..." line. CORRECTIONS 263 added a SECOND
# kind of row, because cmo1 (CORRECTIONS 255) changes the BASE optimiser through CLI FLAGS rather than a patched tree: its
# M9* arms run `--momentum-param-base 0.9` and its W0* arms `--weight-decay-base 0`, everything else at the standard cell.
# No patch announces such a deviation and no column of results/all_runs.csv carries it, so there is no "<KIND>: on" line to
# read. The row's intervention cell is the flag as it was written on the command line ("--momentum-param-base 0.9"), its
# witness cell is "<KIND>: <flag>=<value>", and the evidence is the run's OWN `ARGS:` line -- the one prefix every run
# prints -- read with argparse last-wins semantics, exactly as analysis/argsline_guard.py reads it under STANDING RULE 20.
# An unknown kind, a witness that does not match its own intervention cell, or a listed value that is the standard value
# stops the export rather than being guessed. kind -> (flag, standard value, plain-English name of the factor).
ARGS_KINDS = {"ARGS_MOMENTUM_BASE": ("momentum-param-base", "0.99", "base momentum"),
              "ARGS_WD_BASE": ("weight-decay-base", "0.1", "base weight decay")}
ARGS_LINE_PREFIX = "ARGS:"


def args_equal(left: str, right: str) -> bool:
    """Numeric comparison where both sides parse as numbers, so 0.9 and 0.90 are one value; otherwise exact."""
    try:
        return float(left) == float(right)
    except ValueError:
        return left == right


def is_args_deviation(witness: str) -> bool:
    """True for an exclusion row whose witness names an ARGS-value kind rather than a patch's ON line."""
    return witness.split(":", 1)[0] in ARGS_KINDS


# CORRECTIONS 284's TWO-AXIS row: a run that BOTH deviates on an ARGS value AND prints an ON "<KIND>:" line, which one
# witness column cannot describe. The campaign writes both axes into the intervention cell, the ARGS flag first and each
# patch clause after it, separated by " + "; the witness column carries the ARGS witness and the patch's own ON line is
# read back from the run's own log. A patch spec may itself contain "+" (DECAY_MASK=a+b+c), so the axes split on the
# spaced separator alone. cwd5's CARW2 is the corpus's first such row.
ARGS_AXIS_SEPARATOR = " + "


def args_axes(intervention: str) -> tuple[str, list[str]]:
    """(the ARGS flag clause, [further patch clauses]) for a one- or two-axis ARGS-value exclusion row."""
    parts = [part.strip() for part in intervention.split(ARGS_AXIS_SEPARATOR)]
    if not parts[0] or any(not part for part in parts):
        raise ValueError(f"Unreadable two-axis intervention in {INTERVENTIONS_TSV}: {intervention!r}")
    return parts[0], parts[1:]


# CORRECTIONS 294's TWO-ARGS row: a run that deviates on TWO ARGS kinds at once (caw2's dose arms XS / XL: the AdamW base's
# own beta1 0.9 AND the dose weight decay 1.0). Its intervention cell names every ARGS flag as written on the command line,
# ARGS axes before any patch axis, and it is listed ONCE by the witness of any one of them; MECH7_MULTI_ARGS registers the
# (batch, arm)s that may do this, with every deviating kind's value.
def args_clause(clause: str) -> tuple[str, str, str] | None:
    """(kind, flag, value) for an ARGS axis "--<flag> <value>", None for a patch axis; an unregistered flag stops the export."""
    if not clause.startswith("--"):
        return None
    flag, sep, value = clause[2:].partition(" ")
    if not sep or not value or " " in value:
        raise ValueError(f"Unreadable ARGS axis in {INTERVENTIONS_TSV}: {clause!r}")
    kind = next((name for name, (expected, _standard, _label) in ARGS_KINDS.items() if expected == flag), None)
    if kind is None:
        raise ValueError(f"The intervention axis {clause!r} is not a registered ARGS-value flag")
    return kind, flag, value


def args_clauses(intervention: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """([(kind, flag, value) for every ARGS axis], [every patch axis]) of an ARGS-value exclusion row."""
    flag_clause, rest = args_axes(intervention)
    args, patches = [], []
    for part in [flag_clause, *rest]:
        parsed = args_clause(part)
        if parsed is None:
            patches.append(part)
        elif patches:
            raise ValueError(f"An ARGS axis must come before every patch axis: {intervention!r}")
        else:
            args.append(parsed)
    if len({flag for _kind, flag, _value in args}) != len(args):
        raise ValueError(f"The intervention cell names one ARGS flag twice: {intervention!r}")
    return args, patches


def args_extra_kinds(intervention: str) -> list[tuple[str, str]]:
    """[(patch, value), ...] for the PATCH axes of a two-axis ARGS-value row, in the order written; [] for one axis."""
    _args, patches = args_clauses(intervention)
    return intervention_kinds(" ".join(patches)) if patches else []


def args_extra_args(intervention: str, witness: str) -> list[tuple[str, str, str]]:
    """[(kind, flag, value), ...] for the ARGS axes a TWO-ARGS row carries beyond its witness's kind; [] otherwise."""
    kind = witness.partition(": ")[0]
    return [axis for axis in args_clauses(intervention)[0] if axis[0] != kind]


def args_deviation(intervention: str, witness: str) -> tuple[str, str, str, str]:
    """(kind, flag, value, standard value) for an ARGS-value exclusion row, or a ValueError."""
    kind, sep, rest = witness.partition(": ")
    if not sep or kind not in ARGS_KINDS:
        raise ValueError(f"Unknown ARGS-value witness in {INTERVENTIONS_TSV}: {witness!r}")
    flag, equals, value = rest.partition("=")
    expected, standard, _name = ARGS_KINDS[kind]
    if not equals or not value or flag != expected or " " in rest:
        raise ValueError(f"Unreadable ARGS-value witness in {INTERVENTIONS_TSV}: {witness!r}")
    flag_clause, extra = args_axes(intervention)
    if not flag_clause.startswith("--"):
        raise ValueError(f"The {kind} intervention cell must open with the flag as written on the command line: {intervention!r}")
    args, _patches = args_clauses(intervention)
    if len(args) == 1 and flag_clause != f"--{flag} {value}":
        raise ValueError(f"The {kind} intervention cell must open with the flag as written on the command line: {intervention!r}")
    if (kind, flag, value) not in args:
        raise ValueError(f"The {kind} intervention cell does not carry the witness's flag --{flag} {value}: {intervention!r}")
    if extra:
        args_extra_kinds(intervention)  # every further patch axis must be a known patch kind with a value
    if args_equal(value, standard):
        raise ValueError(f"{kind} lists the standard value {standard} as a deviation")
    for other, other_flag, other_value in args:
        if other != kind and args_equal(other_value, ARGS_KINDS[other][1]):
            raise ValueError(f"A further ARGS axis lists the standard value --{other_flag} {other_value}: {intervention!r}")
    return kind, flag, value, standard


def args_phrase(kind: str) -> str:
    """'base momentum flag --momentum-param-base', for the run note and the record's warning."""
    flag, _standard, name = ARGS_KINDS[kind]
    return f"{name} flag --{flag}"


def args_effective(payload: str) -> dict[str, str]:
    """flag -> value for an ARGS payload, the LAST occurrence winning, as argparse and analysis/argsline_guard.py read it.

    argsline_guard is a REGISTERED campaign file and is not imported (RULE 16); its tokenizer and last-wins rule are
    re-typed here, as analysis/corpus_exclusions.py re-types them on the campaign side (CORRECTIONS 263)."""
    try:
        tokens = shlex.split(payload)
    except ValueError:
        tokens = payload.split()
    effective, index = {}, 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        flag, equals, inline = token.partition("=")
        if equals:
            effective[flag], index = inline, index + 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            effective[flag], index = tokens[index + 1], index + 2
        else:
            effective[flag], index = "", index + 1
    return effective


def args_witness_line(lines: list[str], flag: str, value: str, standard: str) -> str:
    """The run's own ARGS line, checked to carry <flag> at the listed value; the returned witness quotes that flag only.

    The raw ARGS line also carries the run's private save directory, so the witness published beside the run is the
    checked flag alone. The sanitized raw log is published in full, so a reader can read the whole line there."""
    found = [line for line in lines if line.startswith(ARGS_LINE_PREFIX)]
    if len(found) != 1:
        raise ValueError(f"A run log must print exactly one '{ARGS_LINE_PREFIX}' line; found {len(found)}")
    listed = args_effective(found[0][len(ARGS_LINE_PREFIX):]).get(f"--{flag}")
    if listed is None or not args_equal(listed, value):
        raise ValueError(f"The run log's ARGS line carries --{flag} {listed!r}, not the listed {value!r}")
    if args_equal(listed, standard):
        raise ValueError(f"The run log's ARGS line carries the standard --{flag} {standard}, so the run does not deviate")
    return f"{ARGS_LINE_PREFIX} --{flag} {listed}"


# ---------------------------------------------------------------------------
# 6. The cgn3 landing (CORRECTIONS 231): one appended row, two rows amended in place.
# ---------------------------------------------------------------------------
# Campaign commit e3a43da (cycle 152, CORRECTIONS 231) appended the cgn3 landing as
# line 219, recounted header line 3, and applied the verifier's two fixes in place,
# superseded wording kept in brackets: row 213 (cpl1, MT213) gains an UPDATE bracket
# saying cvt1 has landed, and row 218 (cvt1, MT218) marks "every cell-pooling reader
# drops them" as not true when written. Nothing else may differ from the landed pin,
# and no line may move. Neither amendment touches a verdict column, so no outcome moves.
CGN3_COMMIT = "e3a43dacf1ef91f2fe3e7bd6d9c5efb775b6ced9"
CGN3_MASTER_TABLE_SHA256 = "22faae0797b22d17fbb6426872f02d0959e175fda1ec17503a09a79f69b63eb1"
CGN3_FIRST_ROW, CGN3_LAST_ROW = 219, 219
CGN3_EDITED_LINES = {3, 213, 218}
CGN3_SOURCE = "docs/CORRECTIONS.md [CORRECTIONS 231]"
CGN3_MARK = "CORRECTIONS 231"
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Goal met: the registered branch RESCUE-SURVIVES answers the question, the tested
# FREEZE-HOLDS account (228.5) is IN on every arm and on RHO (k01 14.84 in 14-22, kL 66.39
# in 55-72, ISO 63.36 in 57-72, ONE 56.62 in 42-68, RHO 1.2235 in 0.90-1.40), the pin stamp
# separates it from NEVER-PINS, and the scorer passes 87 gates with 0 failures. The three
# bounds of 231.4 (the level settled before the pin; CEIL-BELOW; DIFFERS-FROM-CGN2) are
# descriptive limits and conditional stamps, not registered bands the result missed.
CGN3_ROWS = {
    219: ("met", 9, ["cgn3"], ["page-19"], None, "RESCUE-SURVIVES: the GroupNorm isolation rescue holds to 430 epochs (RHO 1.2235) after its complement's step size pins, every arm lands in the tested FREEZE-HOLDS account's registered band, and ONE survives with a rising share of the kL gap (0.68 -> 0.81). Bounded: the level settled before the pin, and ISO ends 3.02 pp below kL (CEIL-BELOW)."),
}
CGN3_BEARS_ON = {219: ["MT216"]}  # cgn2: the 100-epoch GroupNorm isolation this horizon test extends (its ISO-above-kL was a transient)
# id -> amendment. "cell" is the MASTER-TABLE column the campaign amended (0-based).
# scope "cell": the record's scope takes the amended cell verbatim, superseded clause and all.
# scope "note": the record keeps its text and gains "Amended at CORRECTIONS 231: <reason>";
#   "retire" names wording of the notebook's own CORRECTIONS 229 note that the row now
#   supersedes, removed from the scope and from the 229 amendment record.
CGN3_AMENDMENTS = {
    "MT213": {"line": 213, "outcome": "mixed", "cell": 5, "scope": "note", "retire": (" (cvt1 has not landed).", "."),
              "reason": ("Outcome unchanged (Mixed). cvt1 landed at CORRECTIONS 230 as MT218 (STEP-SIZE-NEEDED-VOTE-SUFFICES): silencing "
                         "layer4.1.bn2.weight's term in the shared meta-gradient sum while it keeps the shared step size does not rescue "
                         "(MUTE 10.9860 vs k01 11.9493; the DOWN vote is re-carried by 47, 44, 41 and 38), isolating it does (HEAD 64.7940), "
                         "and its bn1 twin's term x691 inside HEAD's complement re-pins that complement at epoch 37.6 and drops the arm to "
                         "30.1400 (0.344 of the gap kept). Its rescue needs its own step size; magnitude is still not separated from "
                         "identity in level.")},
    "MT218": {"line": 218, "outcome": "mixed", "cell": 2, "scope": "cell", "retire": None,
              "reason": ("Outcome unchanged (Mixed). The row said every cell-pooling reader drops the 9 MUTE / DOSE / INJECT rows; that was "
                         "not true when written. No file in analysis/ imports corpus_exclusions.py, and cPL2_plainnet_head_score.py "
                         "--selftest pools them (SIGMA_PLAIN 8.6778, df 24, against 0.4107, df 15, with the rows dropped; a NOTE that "
                         "feeds no bar). The clause stays in the row as superseded; future noise-floor or corpus-sigma derivations must "
                         "drop the rows with corpus_exclusions.filter_rows first.")},
}


def cgn3_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cgn3 landing; only the header, rows 213 and 218 and one appended row may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CGN3_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CGN3_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CGN3_COMMIT[:12]} does not match the pinned cgn3-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = landed_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != CGN3_LAST_ROW or len(before) != CGN3_FIRST_ROW - 1 or changed != CGN3_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {CGN3_COMMIT[:7]} moved a line or edited lines other than {sorted(CGN3_EDITED_LINES)}: {sorted(changed)}")
    for amendment in CGN3_AMENDMENTS.values():
        line, cell = amendment["line"], amendment["cell"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if len(new) != 7 or [c for i, c in enumerate(old) if i != cell] != [c for i, c in enumerate(new) if i != cell]:
            raise ValueError(f"MASTER-TABLE row {line} changed outside its amended column {cell}")
        # The superseded wording is kept: dropping the SUPERSEDED label and the new bracket gives back the old cell.
        kept = norm(re.sub(r"\[SUPERSEDED[^:\]]*:\s*", "", new[cell]))
        if CGN3_MARK not in new[cell] or not kept.startswith(norm(old[cell])):
            raise ValueError(f"MASTER-TABLE row {line}: the CORRECTIONS 231 amendment must be bracketed and keep the superseded wording")
    return lines


def cgn3_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE line 219 (cgn3)."""
    return appended_rows(lines, CGN3_ROWS, CGN3_FIRST_ROW, CGN3_LAST_ROW, CGN3_COMMIT)


def apply_cgn3_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply CORRECTIONS 231's in-place amendments of rows 213 and 218 to their records."""
    lines, before = cgn3_master_table(repo), landed_master_table(repo)
    if {a["line"] for a in CGN3_AMENDMENTS.values()} != CGN3_EDITED_LINES - {3}:
        raise ValueError("Every row amended at CORRECTIONS 231 needs exactly one record amendment")
    by_id = {row["id"]: row for row in rows}
    missing = set(CGN3_AMENDMENTS) - by_id.keys()
    if missing:
        raise ValueError(f"CORRECTIONS 231 amendments name absent records: {sorted(missing)}")
    amended = []
    for row in rows:
        spec = CGN3_AMENDMENTS.get(row["id"])
        if not spec:
            amended.append(row)
            continue
        line = spec["line"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        question = re.sub(r"^\[[^\]]*\]\s*", "", clean(new[0]))
        if norm(row["original_question"])[:60] not in norm(question):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != spec["outcome"]:
            raise ValueError(f"{row['id']} outcome drifted before its CORRECTIONS 231 amendment: {row['outcome']} (expected {spec['outcome']})")
        row = dict(row)
        if spec["scope"] == "cell":
            if row["scope"].count(clean(old[spec["cell"]])) != 1:
                raise ValueError(f"{row['id']}: the amended MASTER-TABLE cell is not in the record's scope exactly once")
            row["scope"] = row["scope"].replace(clean(old[spec["cell"]]), clean(new[spec["cell"]]))
        else:
            retired, replacement = spec["retire"]
            if row["scope"].count(retired) != 1 or not row.get("amendment"):
                raise ValueError(f"{row['id']}: the CORRECTIONS 229 wording to retire is not in the record exactly once")
            row["scope"] = row["scope"].replace(retired, replacement) + f" Amended at CORRECTIONS 231: {spec['reason']}"
            earlier = json.loads(row["amendment"])
            earlier["reason"] = earlier["reason"].replace(retired, replacement)
            row["amendment"] = json.dumps(earlier, ensure_ascii=False)
        sources = json.loads(row.get("sources") or "[]")
        # Anchors that pinned the row's text before CORRECTIONS 231 now point at the amended row.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        row["sources"] = json.dumps([*sources, CGN3_SOURCE], ensure_ascii=False)
        row["later_amendment"] = json.dumps({"line": line, "commit": CGN3_COMMIT, "previousOutcome": spec["outcome"], "outcome": spec["outcome"],
                                             "reason": spec["reason"], "source": f"{MASTER_TABLE} line {line} at {CGN3_COMMIT[:7]}; CORRECTIONS 231"}, ensure_ascii=False)
        amended.append(row)
    return amended


# ---------------------------------------------------------------------------
# 7. The cvt3 and cvt2 landings (CORRECTIONS 234-236): two appended rows, two rows amended in place.
# ---------------------------------------------------------------------------
# Campaign commit 1ef1ba9 (CORRECTIONS 234) amended header line 5 and rescoped row 216 (cgn2, MT216) in
# place; commit 9c5d72a (cycle 152, CORRECTIONS 235 + 236) appended the cvt3 and cvt2 landings as lines
# 220 and 221, recounted header line 3, amended line 5 again, and corrected a wording slip in row 213
# (cpl1, MT213) in place, superseded wording kept verbatim. Nothing else may differ from the cgn3 pin, and
# no line may move. Line 5 (the bottom-line paragraph) feeds no record; it is published only in the
# MASTER-TABLE source copy. Neither row amendment touches a verdict column, so no outcome moves.
CVT23_COMMIT = "9c5d72ab54dca57f94f143d23bfe722804b6aa6e"
CVT23_MASTER_TABLE_SHA256 = "1943409bd0692900fabf41037a9f8654b8de27cad1fcfe6b5fdca54b2b07bc85"
CVT23_FIRST_ROW, CVT23_LAST_ROW = 220, 221
CVT23_EDITED_LINES = {3, 5, 213, 216}
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Both Mixed, as MT218: the returned branch answers the question, but a registered expectation was defied.
# cvt3 (235.3): the returned branch's own account, OWN-STEP-NECESSARY (232.4), hits 5 of 6 -- MUTECTL 7.82
# misses its 8-16 band by 0.18 pp -- and the PRIMARY P_COAL is positive only because that control fell
# below k01. cvt2 (236.3): no registered account's outcome is GRADED | TOP-ATTENUATED, the prior stated at
# 233.5 (TOP-FLAT) missed by 0.26 pp, and the sign-saturation derivation failed at K33.
CVT23_ROWS = {
    220: ("mixed", 9, ["cvt3"], ["page-19"], None, "OWN-STEP-NECESSARY: silencing layer4.1.bn2.weight and the whole 20-tensor DOWN coalition, with every tensor on one shared step size, leaves the scalar arm at k01 (MUTEDOWN 11.33 vs k01 11.57), which strengthens MT218's 'the rescue needs 50 on its own step size'; but the account misses its registered MUTECTL band by 0.18 pp, P_COAL (+3.51 pp) comes from the control falling below k01 rather than from the coalition (MUTEDOWN - MUTE50 +0.03 pp), and the DOWN vote was re-carried by the next tier, so 'no silenced set rescues' is not licensed."),
    221: ("mixed", 9, ["cvt2"], ["page-19"], None, "GRADED + TOP-ATTENUATED: the twin's injected vote collapses HEAD's complement by degree in K (share of the HEAD gap kept 0.772 at K13 down to 0.256 at K2000), so cvt1's 0.344 residue is not a fixed floor; but no registered account predicted this pair of words, the stated TOP-FLAT prior missed by 0.26 pp, both words sit on near bars (K33 0.32 pp and TOP 0.26 pp past the 5.0 pp bar), and K13 is still rising at 100 epochs, so its level may be a delay."),
}
CVT23_BEARS_ON = {220: ["MT218"], 221: ["MT218"]}  # cvt1: MT220 re-tests its MUTE at the coalition level, MT221 doses its INJECT
# id -> amendment of CORRECTIONS 234 or 236. "cell" is the amended MASTER-TABLE column (0-based); "number" the
# CORRECTIONS entry that made it. The amended cell must differ from the cgn3 pin in one contiguous span that
# carries the entry's mark and keeps any removed wording verbatim.
# field "result": the record's result takes the amended cell verbatim (the rescope bracket and all).
# field "note": the record keeps its text and gains "Amended at CORRECTIONS <number>: <reason>".
CVT23_AMENDMENTS = {
    "MT216": {"line": 216, "outcome": "success", "cell": 3, "number": 234, "field": "result",
              "reason": ("Outcome unchanged (Goal met). Rescoped at CORRECTIONS 234: 'ISO sits +1.81 pp above kL' and the ISO-TRACKS-KL "
                         "stamp held at 100 epochs only; both are kept verbatim in the result. cgn3 (MT219) ran this cell to 430 epochs: kL "
                         "kept climbing, passes ISO for good at epoch ~174 and ends 3.02 pp above it (CEIL-BELOW), while the rescue itself "
                         "survives (RESCUE-SURVIVES, RHO 1.2235). The ordering against layerwise does not survive; the identity transfer does.")},
    "MT213": {"line": 213, "outcome": "mixed", "cell": 5, "number": 236, "field": "note",
              "reason": ("Outcome unchanged (Mixed). Row 213's CORRECTIONS 231 update said a carrier-sized vote collapses a complement "
                         "'whoever casts it'. That claimed more than was tested and was corrected in place: one other caster "
                         "(layer4.1.bn1.weight) was run, into one complement (HEAD's), at one K in cvt1 (MT218) and at five K in cvt2 "
                         "(MT221); whether any other tensor's term does the same is untested. The superseded wording stays in the row.")},
}
CVT23_INTERVENTIONS_TSV_SHA256 = "aeb82070350bfd6dd230877ea98bea30f9e15537a30789b3a7e9a603bc22461f"
CVT23_INTERVENTIONS = {
    "MT220": {
        "batch": "cvt3", "arms": {"MUTE50": 3, "MUTEDOWN": 3, "MUTECTL": 3},
        "title": "MUTE50, MUTEDOWN and MUTECTL are vote-weight interventions, not plain arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT23_COMMIT[:7]}; CORRECTIONS 232, 235 and 236",
        "note": ("MUTE50, MUTEDOWN and MUTECTL ran cvt1's unchanged PATCH_VOTEWEIGHT tree (VOTE_W=<tensor>:<w>, one or more "
                 "tensors separated by /, multiplies each listed tensor's term inside the unnormalised shared meta-gradient sum, "
                 "before the sign), all on the scalar grouping. MUTE50 = layer4.1.bn2.weight's term x0 (cvt1's MUTE, replicated); "
                 "MUTEDOWN = 50 plus RULE C's 20-tensor DOWN coalition (15 BN scales and 5 convs) x0; MUTECTL = 50 plus RULE K's "
                 "count- and class-matched non-DOWN set (15 BN biases of the same modules and 5 smaller convs) x0, not mass-matched. "
                 "VOTE_W rides only the run's own VOTE_W: witness line, so the run inventory writes all 9 rows with k01's cell key "
                 "(granularity scalar): seed for seed they differ from k01 only in run, job_id, node, wallclock_min and the accuracy "
                 "columns. They are NOT plain scalar measurements. The 9 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them "
                 "before pooling runs by cell. The k01 and HEAD arms print VOTE_W: off and are ordinary measurements of their cells."),
    },
    "MT221": {
        "batch": "cvt2", "arms": {"K13": 3, "K33": 3, "K152": 3, "K691": 3, "K2000": 3},
        "title": "K13, K33, K152, K691 and K2000 are vote-weight interventions, not plain HEAD arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT23_COMMIT[:7]}; CORRECTIONS 233 and 236",
        "note": ("The five ladder arms ran cvt1's unchanged PATCH_VOTEWEIGHT tree on HEAD's grouping ({50} [52,1]) with the twin "
                 "layer4.1.bn1.weight's term multiplied by K = 13, 33, 152, 691 or 2000 (fixed K, sign kept) inside the complement's "
                 "unnormalised shared meta-gradient sum, before the sign; K691 is cvt1's INJECT byte for byte. VOTE_W rides only the "
                 "run's own VOTE_W: witness line, so the run inventory writes all 15 rows with HEAD's cell key: seed for seed they "
                 "differ from HEAD only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain HEAD "
                 "measurements. The 15 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them before pooling runs by cell. The "
                 "k01 and HEAD arms print VOTE_W: off and are ordinary measurements of their cells."),
    },
}


def cvt23_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt3 + cvt2 landing; only header lines 3 and 5, rows 213 and 216 and two appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT23_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CVT23_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CVT23_COMMIT[:12]} does not match the pinned cvt3/cvt2-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cgn3_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != CVT23_LAST_ROW or len(before) != CVT23_FIRST_ROW - 1 or changed != CVT23_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {CVT23_COMMIT[:7]} moved a line or edited lines other than {sorted(CVT23_EDITED_LINES)}: {sorted(changed)}")
    for amendment in CVT23_AMENDMENTS.values():
        line, cell = amendment["line"], amendment["cell"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if len(new) != 7 or [c for i, c in enumerate(old) if i != cell] != [c for i, c in enumerate(new) if i != cell]:
            raise ValueError(f"MASTER-TABLE row {line} changed outside its amended column {cell}")
        removed, added = changed_span(old[cell], new[cell])
        # One contiguous edit that carries the entry's mark and keeps whatever it replaced, verbatim.
        if f"CORRECTIONS {amendment['number']}" not in added or norm(removed) not in norm(added):
            raise ValueError(f"MASTER-TABLE row {line}: the CORRECTIONS {amendment['number']} amendment must be bracketed and keep the superseded wording")
    return lines


def changed_span(old: str, new: str) -> tuple[str, str]:
    """The one contiguous span in which two strings differ: (removed from old, added in new)."""
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(old), len(new)) - prefix and old[-1 - suffix] == new[-1 - suffix]:
        suffix += 1
    return old[prefix:len(old) - suffix], new[prefix:len(new) - suffix]


def cvt23_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 220 (cvt3) and 221 (cvt2) and attach their intervention notes."""
    return with_interventions(appended_rows(lines, CVT23_ROWS, CVT23_FIRST_ROW, CVT23_LAST_ROW, CVT23_COMMIT), CVT23_INTERVENTIONS)


def apply_cvt23_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply CORRECTIONS 234's and 236's in-place amendments of rows 216 and 213 to their records."""
    lines, before = cvt23_master_table(repo), cgn3_master_table(repo)
    if {a["line"] for a in CVT23_AMENDMENTS.values()} != CVT23_EDITED_LINES - {3, 5}:
        raise ValueError("Every row amended at CORRECTIONS 234 or 236 needs exactly one record amendment")
    missing = set(CVT23_AMENDMENTS) - {row["id"] for row in rows}
    if missing:
        raise ValueError(f"CORRECTIONS 234/236 amendments name absent records: {sorted(missing)}")
    amended = []
    for row in rows:
        spec = CVT23_AMENDMENTS.get(row["id"])
        if not spec:
            amended.append(row)
            continue
        line, number = spec["line"], spec["number"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        question = re.sub(r"^\[[^\]]*\]\s*", "", clean(new[0]))
        if norm(row["original_question"])[:60] not in norm(question):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != spec["outcome"]:
            raise ValueError(f"{row['id']} outcome drifted before its CORRECTIONS {number} amendment: {row['outcome']} (expected {spec['outcome']})")
        row = dict(row)
        if spec["field"] == "result":
            if row["result"] != clean(old[spec["cell"]]):
                raise ValueError(f"{row['id']}: the record's result is not the MASTER-TABLE cell CORRECTIONS {number} amended")
            row["result"] = clean(new[spec["cell"]])
        else:
            row["scope"] = f"{row['scope']} Amended at CORRECTIONS {number}: {spec['reason']}"
        sources = json.loads(row.get("sources") or "[]")
        # Anchors that pinned the row's text at the cgn3 landing now point at the amended row.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        row["sources"] = json.dumps([*sources, f"docs/CORRECTIONS.md [CORRECTIONS {number}]"], ensure_ascii=False)
        further = json.loads(row.get("further_amendments") or "[]")
        further.append({"number": number, "line": line, "commit": CVT23_COMMIT, "previousOutcome": spec["outcome"], "outcome": spec["outcome"],
                        "reason": spec["reason"], "source": f"{MASTER_TABLE} line {line} at {CVT23_COMMIT[:7]}; CORRECTIONS {number}"})
        row["further_amendments"] = json.dumps(further, ensure_ascii=False)
        amended.append(row)
    return amended


# ---------------------------------------------------------------------------
# 8. The cvt4 and cvt5 landings (CORRECTIONS 240-241): two appended rows, no row amended.
# ---------------------------------------------------------------------------
# Campaign commit 643264c (cycle 152, CORRECTIONS 240 + 241) appended the cvt4 and cvt5 landings as lines 222
# and 223, recounted header line 3 and amended the bottom-line paragraph (line 5) in place, superseded clauses
# kept. Nothing else may differ from the cvt3/cvt2 pin, and no line may move. Line 5 feeds no record; it is
# published only in the MASTER-TABLE source copy. No existing row was amended, so no earlier record changes.
CVT45_COMMIT = "643264c8e4056710b29b437cab34c74f3f81c37b"
CVT45_MASTER_TABLE_SHA256 = "3c1363e3be150813efdfa74662c2d7e17e3ef3c99c3a0ff6b1e2ba477ddd77b6"
CVT45_FIRST_ROW, CVT45_LAST_ROW = 222, 223
CVT45_EDITED_LINES = {3, 5}
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Both Goal met, as MT219 (cgn3): the registered branch answers the question, the tested account is in band on
# every registered band, the positive control reproduces and G-BITE passes on every run. The bounds of 240.4 and
# 241.4 are descriptive limits and stamps, not registered bands the result missed, and neither row corrects an
# earlier claim, so no Corrected badge.
# cvt4 (240.3): OWN-STEP-MAGNITUDE (237.5, the brief's account) hits all 9 bands -- HOLDLOW 5.73 pp and HOLDHEAD
# 4.93 pp inside AT-HEAD, HOLDSHARED 2.30 pp and HOLDHIGH 2.57 pp inside AT-K01 -- and the next-closest account
# (OWN-STEP-GRADED) misses HOLDHIGH by 8.87 pp.
# cvt5 (241.3): the tested EQUILIBRIUM + FREEZE-HOLDS account (238.5) hits all 8 bands (4 arms x 2 horizons),
# every other account misses one by >= 6.92 pp, no branch bar is near (closest DROP13, 4.58 pp) and the
# registration stated no prior. The scorer's printed RESCUE-SURVIVES licence mentions an exact clamp "at ~96-100"
# that this batch did not show (241.7(3)); no gate or token reads it, so it stays a limit in the scope.
CVT45_ROWS = {
    222: ("met", 9, ["cvt4"], ["page-19"], None, "OWN-STEP-MAGNITUDE: with layer4.1.bn2.weight's vote out of the complement's sum in every held arm, its own step size held large (HOLDHIGH, MUTE's measured trajectory replayed) stalls the arm at k01 (11.13 vs k01 11.69) and held at the floor from init keeps HEAD's level (HOLDLOW 65.46 vs HEAD 64.73); P_HIGH +53.61 pp = +94.55 SE, the registered account hits all 9 of its bands, and the replayed HEAD trajectory reproduces HEAD (P_CTL +0.07 pp). Bounded: HOLDHIGH's unforced complement collapsed too (pinned at epoch 40.4, HEAD's at 97.8-99.6), so a direct stall is not separated from one through the complement, and there is no time gate or dose curve."),
    223: ("met", 9, ["cvt5"], ["page-19"], None, "K-DEPENDENT-EQUILIBRIUM + RESCUE-SURVIVES + K33-HOLDS-PINNED: at 300 epochs K13 holds 15.95 pp above K33 (+23.04 SE) and moved +0.42 pp from its own 100-epoch level, so cvt2's GRADED (MT221) is not a delay; HEAD's PlainNet rescue survives its complement's magnitude pin (RHO 1.0012, level frozen for 236 post-pin epochs); K33 stays pinned; the tested EQUILIBRIUM + FREEZE-HOLDS account hits all 8 registered bands and no branch bar is near. Bounded: the level-holds and frozen-at-pin stamps sit only 0.96-1.34 pp inside READ_BAR, K13's step size hovers near the floor rather than freezing, and its level settled (epoch 105) before its pin (~160)."),
}
CVT45_BEARS_ON = {222: ["MT218", "MT220"], 223: ["MT221", "MT218"]}  # cvt4 varies the step size cvt1 / cvt3 left shared; cvt5 runs cvt2's K13 / K33 and HEAD to 300 epochs
CVT45_INTERVENTIONS_TSV_SHA256 = "f8455571b0f1e1ce246aedb89a0dbfda50b485bfea28496cc29f29524e7c96dd"
CVT45_INTERVENTIONS = {
    "MT222": {
        "batch": "cvt4", "arms": {"HOLDLOW": 3, "HOLDSHARED": 3, "HOLDHIGH": 3, "HOLDHEAD": 3},
        "title": "HOLDLOW, HOLDSHARED, HOLDHIGH and HOLDHEAD are step-size hold interventions, not plain HEAD arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT45_COMMIT[:7]}; CORRECTIONS 237, 239 and 240",
        "note": ("The four held arms ran cvt1's tree plus the opt-in PATCH_BETAHOLD (BETA_HOLD=<tensor>:<mode>), on HEAD's grouping "
                 "({50} [52,1]). The patch overwrites the held group's beta (layer4.1.bn2.weight's log step size) after every meta "
                 "update and its clamp, and at init; the held group's momentum is dead state. HOLDLOW = the -15 floor from init; "
                 "HOLDSHARED = the complement's live beta (cvt1's MUTE in exact arithmetic); HOLDHIGH = an exogenous max-rate replay "
                 "of MUTE's measured shared trajectory (tri:9428: peak -4.39 at epoch 18.9, clamp at 40.1); HOLDHEAD = a replay of "
                 "HEAD's own median trajectory for tensor 50 (tri:5041, the hold's positive control). BETA_HOLD rides only the run's "
                 "own BETA_HOLD: witness line, so the run inventory writes all 12 rows with HEAD's cell key: seed for seed they differ "
                 "from HEAD only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain HEAD measurements. "
                 "The 12 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them before pooling runs by cell. The k01 and HEAD "
                 "arms print BETA_HOLD: off and VOTE_W: off and are ordinary measurements of their cells."),
    },
    "MT223": {
        "batch": "cvt5", "arms": {"K13": 3, "K33": 3},
        "title": "K13 and K33 are vote-weight interventions, not plain HEAD arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT45_COMMIT[:7]}; CORRECTIONS 238 and 241",
        "note": ("K13 and K33 ran cvt1's unchanged PATCH_VOTEWEIGHT tree on HEAD's grouping ({50} [52,1]) with the twin "
                 "layer4.1.bn1.weight's term multiplied by K = 13 or 33 (fixed K, sign kept) inside the complement's unnormalised "
                 "shared meta-gradient sum, before the sign: cvt2's K13 and K33 strings byte for byte, run to 300 epochs. VOTE_W rides "
                 "only the run's own VOTE_W: witness line, so the run inventory writes all 6 rows with HEAD's cell key: seed for seed "
                 "they differ from HEAD only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain HEAD "
                 "measurements. The 6 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them before pooling runs by cell. The "
                 "k01 and HEAD arms print VOTE_W: off and are ordinary 300-epoch measurements of their cells, kept out of 100-epoch "
                 "cells by epochs_requested."),
    },
}


def cvt45_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt4 + cvt5 landing; only header lines 3 and 5 and two appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT45_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CVT45_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CVT45_COMMIT[:12]} does not match the pinned cvt4/cvt5-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cvt23_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != CVT45_LAST_ROW or len(before) != CVT45_FIRST_ROW - 1 or changed != CVT45_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {CVT45_COMMIT[:7]} moved a line or edited lines other than {sorted(CVT45_EDITED_LINES)}: {sorted(changed)}")
    return lines


def cvt45_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 222 (cvt4) and 223 (cvt5) and attach their intervention notes."""
    return with_interventions(appended_rows(lines, CVT45_ROWS, CVT45_FIRST_ROW, CVT45_LAST_ROW, CVT45_COMMIT), CVT45_INTERVENTIONS)


# ---------------------------------------------------------------------------
# 9. CORRECTIONS 244: bracketed wording amendments inserted into rows 222 and 223 (and line 5), no outcome moved.
# ---------------------------------------------------------------------------
# Campaign commit 0ade9cc (cycle 152, CORRECTIONS 244, the fix track) applied the wording fixes the cvt4 and cvt5
# verifiers had asked for and the landing had not carried. Each is a pure insertion of one
# "**[AMENDED at cycle 152, CORRECTIONS 244: ...]**" bracket: two in row 222's so-what cell (sufficiency at this cell;
# P_COUP a floor location), one leading row 223's so-what cell (the bound first; the equilibrium is the level's), and
# one on line 5, which feeds no record. Nothing else may differ from the cvt4/cvt5 pin, no line may move, and removing
# the inserted brackets must give the pinned cell back byte for byte. Both records keep their outcome (Goal met) and
# gain a warning-<id>-amendment-244 record; the scope takes the amended cell, so the earlier wording stays in it.
C244_COMMIT = "0ade9ccfd887fdd421f402084227396b83747e4b"
C244_MASTER_TABLE_SHA256 = "0aa32e5d943788ecf449f58680540637e6f9130045436bd4a6f468f423821e59"
C244_EDITED_LINES = {5, 222, 223}
C244_TAG = "CORRECTIONS 244"
C244_AMENDMENTS = {
    "MT222": {"line": 222, "outcome": "success", "cell": 5, "number": 244,
              "reason": ("Outcome unchanged (Goal met). Amended at CORRECTIONS 244, the cvt4 verifier's wording fixes 2 and 3, inserted in "
                         "brackets with the earlier wording kept: 'the rescue needs 50 on a SMALL step size' is read as sufficiency at this "
                         "cell -- the one large trajectory tested (MUTE's, replayed) is sufficient to stall, and 50 at the floor or on HEAD's "
                         "own trajectory is sufficient to keep the rescue; there is no dose curve and no time window. P_COUP ~ 0 is a reading "
                         "between two arms at the same floor location, not a measured absence of coupling; COUPLING's predicted HOLDHIGH band "
                         "56-72 was missed by 44.87 pp.")},
    "MT223": {"line": 223, "outcome": "success", "cell": 5, "number": 244,
              "reason": ("Outcome unchanged (Goal met). Amended at CORRECTIONS 244, the cvt5 verifier's wording fixes 4 and 5, which reached "
                         "the landing cut off: the row now leads with its bound -- no branch bar within 4.5 pp, soft stamps (0.96-1.34 pp "
                         "inside READ_BAR 1.384), and K13's 'pinned' a hover of the median r over the last quarter, not a clamp -- and reads "
                         "K-DEPENDENT-EQUILIBRIUM as an equilibrium of the level only: K13's TEST settled from ~epoch 105, ~55 epochs before "
                         "its complement first reached r <= 2.")},
}


def strip_inserted_brackets(text: str, tag: str) -> str:
    """``text`` without the "[... <tag> ...]" brackets inserted into it (bold marks and one adjacent space included)."""
    pattern = re.compile(r"(?:\*\*)?\[[^\[\]]*?" + re.escape(tag) + r"[^\[\]]*\](?:\*\*)?")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        start, end = match.span()
        if end < len(text) and text[end] == " ":
            end += 1
        elif start > 0 and text[start - 1] == " ":
            start -= 1
        text = text[:start] + text[end:]


def c244_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at CORRECTIONS 244; only lines 5, 222 and 223 may differ from the cvt4/cvt5 pin, by inserted brackets."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{C244_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != C244_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {C244_COMMIT[:12]} does not match the pinned CORRECTIONS 244 bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cvt45_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != len(before) or changed != C244_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {C244_COMMIT[:7]} moved a line or edited lines other than {sorted(C244_EDITED_LINES)}: {sorted(changed)}")
    for n in changed:
        if strip_inserted_brackets(lines[n - 1], C244_TAG) != before[n - 1]:
            raise ValueError(f"MASTER-TABLE line {n} at {C244_COMMIT[:7]} changed more than inserted {C244_TAG} brackets")
    return lines


def apply_c244_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply CORRECTIONS 244's bracketed amendments of rows 222 and 223 to their records."""
    lines, before = c244_master_table(repo), cvt45_master_table(repo)
    if {a["line"] for a in C244_AMENDMENTS.values()} != C244_EDITED_LINES - {5}:
        raise ValueError("Every row amended at CORRECTIONS 244 needs exactly one record amendment")
    missing = set(C244_AMENDMENTS) - {row["id"] for row in rows}
    if missing:
        raise ValueError(f"CORRECTIONS 244 amendments name absent records: {sorted(missing)}")
    amended = []
    for row in rows:
        spec = C244_AMENDMENTS.get(row["id"])
        if not spec:
            amended.append(row)
            continue
        line, number, cell = spec["line"], spec["number"], spec["cell"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if len(old) != 7 or len(new) != 7 or [i for i in range(7) if old[i] != new[i]] != [cell]:
            raise ValueError(f"MASTER-TABLE line {line} at {C244_COMMIT[:7]} must change exactly cell {cell}")
        if row["master_table_line"] != str(line):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != spec["outcome"]:
            raise ValueError(f"{row['id']} outcome drifted before its CORRECTIONS {number} amendment: {row['outcome']} (expected {spec['outcome']})")
        row = dict(row)
        if row["scope"].count(clean(old[cell])) != 1:
            raise ValueError(f"{row['id']}: the amended MASTER-TABLE cell is not in the record's scope exactly once")
        row["scope"] = row["scope"].replace(clean(old[cell]), clean(new[cell]))
        sources = json.loads(row.get("sources") or "[]")
        # Anchors that pinned the row's text at the cvt4/cvt5 landing now point at the amended row.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        row["sources"] = json.dumps([*sources, f"docs/CORRECTIONS.md [CORRECTIONS {number}]"], ensure_ascii=False)
        further = json.loads(row.get("further_amendments") or "[]")
        further.append({"number": number, "line": line, "commit": C244_COMMIT, "previousOutcome": spec["outcome"], "outcome": spec["outcome"],
                        "reason": spec["reason"], "source": f"{MASTER_TABLE} line {line} at {C244_COMMIT[:7]}; CORRECTIONS {number}"})
        row["further_amendments"] = json.dumps(further, ensure_ascii=False)
        amended.append(row)
    return amended


# ---------------------------------------------------------------------------
# 10. The cvt6 and cvt7 landings (CORRECTIONS 246-247): two appended rows, no row amended; registrations corrected in place.
# ---------------------------------------------------------------------------
# Campaign commit dae2a49 (cycle 152, CORRECTIONS 246 + 247) appended the cvt6 and cvt7 landings as lines 224 and 225,
# recounted header line 3 and amended the bottom-line paragraph (line 5) with four inserted brackets, superseded clauses kept.
# Nothing else may differ from the CORRECTIONS 244 pin and no line may move; line 5 must give the pinned line back byte for
# byte once the CORRECTIONS 246 and 247 brackets are removed. Line 5 feeds no record. No existing row was amended, so no
# earlier record changes (MT222's bound, which line 5 now calls partly resolved, is left as its row still reads).
CVT67_COMMIT = "dae2a49675a950e9848543435526a3ba864b145a"
CVT67_MASTER_TABLE_SHA256 = "d33f4dfe929e9576282914641645bce376aa73fa8658e4431e57c49ca01b9f22"
CVT67_FIRST_ROW, CVT67_LAST_ROW = 224, 225
CVT67_EDITED_LINES = {3, 5}
CVT67_LINE5_TAGS = ("CORRECTIONS 246", "CORRECTIONS 247")
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# cvt6 is Mixed, as MT218 and MT221: the registered branch GRADED answers the 2x2 question (240.4(1) PARTLY RESOLVED), the
# complement-hold positive control reproduces and G-BITE passes 21/21, but no registered account fits every band (246.3):
# GRADED's own account hits 3 of 5 (HIGHHEADPATH 20-50 missed by 8.51 pp, P_50 +6...+52 by 1.53 pp), DIRECT and
# EITHER-SUFFICES 3 of 5 too, so the observed pattern is a GRADED sub-case no account predicted -- "a registered expectation
# was defied". Not Goal missed: nothing registered was refuted as a hypothesis and no control failed.
# cvt7 is Goal met, as MT219 and MT222: the registered branch TRANSFERS-GRADED answers the question ("partly transfers"), its
# account hits every registered band (247.3: HOLDHIGH 50.23 in 34-56, P_HIGH 19.81 in +6...+44), the positive control
# HOLDISO reproduces, G-BITE passes 15/15, no bar is near and no numeric prior was registered (243.5). Its bounds -- dose
# and held set confounded with network, the free complement not ISO-like -- are descriptive limits (247.4), not missed
# bands, and stay in the reason and scope, as MT222's direct-vs-complement bound did.
# Neither row corrects an earlier published claim, so neither carries the Corrected badge.
CVT67_ROWS = {
    224: ("mixed", 9, ["cvt6"], ["page-19"], None, "GRADED: at cvt1's cell on PlainNet18_c100, HIGHHEADPATH (50 on MUTE's replayed trajectory, the complement forced open-loop onto HEAD's recorded path) sits at the k01 location (11.49 vs k01 12.07) and LOWMUTEPATH (50 at the floor, the complement forced onto MUTE's collapse schedule) sits between at 41.56; P_50 +53.53 pp = +94.40 SE and P_C +23.47 pp = +41.38 SE, the complement-hold control keeps HEAD's level (P_CTLC -0.12 pp), cvt4's HOLDLOW and HOLDHIGH replicate, and G-BITE passes 21/21. But no registered account fits every band: GRADED's own account hits 3 of 5 (HIGHHEADPATH misses 20-50 by 8.51 pp, P_50 misses by 1.53 pp), as do DIRECT and EITHER-SUFFICES, so the pattern is a GRADED sub-case no account predicted. Bounded: the forced arms are open-loop, there is one shape per factor with no dose curve or time gate, readings at k01 are locations, and no single-route sentence or necessity is licensed."),
    225: ("met", 9, ["cvt7"], ["page-19"], None, "TRANSFERS-GRADED: on ResNet18_c100 at ciso1's cell, holding ISO's three-carrier group on k01's measured trajectory (HOLDHIGH) gives 50.23, between k01 22.95 and ISO 70.04 (P_HIGH +19.81 pp = +34.94 SE), while the floor (HOLDLOW 70.30) and ISO's own replayed trajectory (HOLDISO 70.06, the positive control) keep ISO's level; the TRANSFERS-GRADED account hits every registered band, G-BITE passes 15/15, no bar is near and no numeric prior was registered, so PlainNet's small-step-size sentence partly transfers. Bounded: dose and held set are confounded with network (replay peak -5.21 vs PlainNet's -4.39; three carriers held together vs idx 50 alone), the free complement did not stay ISO-like (pinned at 52.8-54.0 vs ISO's 92.6-99.4), so direct vs via-complement is not separated, and larger trajectories were not tested."),
}
CVT67_BEARS_ON = {224: ["MT222", "MT218"], 225: ["MT222"]}  # cvt6 splits the route cvt4's HOLDHIGH left confounded; cvt7 carries cvt4's sentence to ResNet
CVT67_INTERVENTIONS_TSV_SHA256 = "d0b955cbfce5c485c48c23d53feadd1fee581f379a221596fa65cf99c7eefe93"
CVT67_INTERVENTIONS = {
    "MT224": {
        "batch": "cvt6", "arms": {"HOLDLOW": 3, "HOLDHIGH": 3, "HIGHHEADPATH": 3, "LOWMUTEPATH": 3, "LOWHEADPATH": 3},
        "title": "HOLDLOW, HOLDHIGH, HIGHHEADPATH, LOWMUTEPATH and LOWHEADPATH are step-size hold interventions, not plain HEAD arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT67_COMMIT[:7]}; CORRECTIONS 242, 245 and 246",
        # All 21 runs (all seven arms) ran harness_cvt6 through jobs/run_cifar_cvt6.sh (CORRECTIONS 242.12 guard 6); COMP_HOLD was set on
        # the 9 forced runs only, and HOLDLOW / HOLDHIGH printed 'COMP_HOLD: off' as k01 and HEAD did (results/cvt6_rule20_full.txt: off x12).
        "note": ("All seven arms ran harness_cvt6 (cvt4's tree plus the opt-in PATCH_COMPHOLD) through jobs/run_cifar_cvt6.sh, on HEAD's "
                 "grouping ({50} [52,1]). COMP_HOLD (=tri:<P> | rec:<id>) was set only on the three forced arms, HIGHHEADPATH, LOWMUTEPATH "
                 "and LOWHEADPATH; HOLDLOW and HOLDHIGH ran with COMP_HOLD off, as k01 and HEAD did. PATCH_BETAHOLD holds tensor 50's beta (layer4.1.bn2.weight's log step size); PATCH_COMPHOLD overwrites "
                 "the complement's beta after that hold. HOLDLOW = 50 at the -15 floor with the complement free (Lion); HOLDHIGH = 50 on "
                 "MUTE's replayed trajectory (tri:9428) with the complement free (cvt4's two arms, replicated); HIGHHEADPATH = 50 on "
                 "tri:9428 with the complement forced onto HEADPATH (the knot-by-knot median of 12 landed HEAD complements, replay file "
                 "74be71fa); LOWMUTEPATH = 50 at the floor with the complement forced onto MUTEPATH (tri:9428); LOWHEADPATH = 50 at the "
                 "floor with the complement on HEADPATH (the complement hold's positive control). HIGHHEADPATH, LOWMUTEPATH and "
                 "LOWHEADPATH hold both groups, so they are open-loop: both Lion momentum entries are dead state. The holds ride only "
                 "the run's own BETA_HOLD: and COMP_HOLD: witness lines, so the run inventory writes all 15 rows with HEAD's cell key: "
                 "seed for seed they differ from HEAD only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT "
                 "plain HEAD measurements. The 15 rows are listed in results/CORPUS-EXCLUSIONS.tsv by their BETA_HOLD witness; the 9 "
                 "forced rows' COMP_HOLD line is checked through MULTI_KIND (CORRECTIONS 245) and shown on each run as its additional "
                 "witness. Drop them before pooling runs by cell. The k01 and HEAD arms print VOTE_W: off, BETA_HOLD: off and "
                 "COMP_HOLD: off and are ordinary measurements of their cells."),
    },
    "MT225": {
        "batch": "cvt7", "arms": {"HOLDLOW": 3, "HOLDHIGH": 3, "HOLDISO": 3},
        "title": "HOLDLOW, HOLDHIGH and HOLDISO are group step-size hold interventions, not plain ISO arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT67_COMMIT[:7]}; CORRECTIONS 243 and 247",
        "note": ("The three held arms ran a tree built from the live ResNet harness plus PATCH_VOTEWEIGHT, PATCH_BETAHOLD and the "
                 "opt-in PATCH_GROUPHOLD (GROUP_HOLD=<n1>+<n2>+<n3>:floor|tri:<P>), on ISO's grouping [59,3], which isolates the three "
                 "carriers layer4.0.bn2.weight, layer4.0.shortcut.1.weight and layer4.1.bn2.weight together in one group with their "
                 "vote out of the complement's sum. The patch holds that group's beta (its log step size), named by its complete "
                 "membership, from init; the complement stays on Lion. HOLDLOW = the group at the -15 floor; HOLDHIGH = an exogenous "
                 "max-rate replay of k01's measured shared trajectory at this cell (tri:8609, peak -5.21); HOLDISO = a replay of ISO's "
                 "own median carrier trajectory (tri:5153, the hold's positive control). GROUP_HOLD rides only the run's own "
                 "GROUP_HOLD: witness line, so the run inventory writes all 9 rows with ISO's cell key: seed for seed they differ "
                 "from ISO only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain ISO measurements. "
                 "The 9 rows are listed in results/CORPUS-EXCLUSIONS.tsv; drop them before pooling runs by cell. With them in, the "
                 "ResNet18_c100 noise floor depends on the filter (0.688153 filtered against 2.208672 unfiltered, CORRECTIONS 247.11). "
                 "The k01 and ISO arms print VOTE_W: off, BETA_HOLD: off and GROUP_HOLD: off and are ordinary measurements of their cells."),
    },
}
# The landing also corrected its own registrations in place (docs/CORRECTIONS.md, brackets inserted, superseded wording
# kept): five brackets in CORRECTIONS 242 and one in 242.14 (246.9), one in 243.4 (247.12). They are registration text,
# not a record's published claim, so they carry no Corrected badge; each record shows them as a documented limitation.
# The pinned file at dae2a49 must equal the one at the ingest commit 45eac02 on every earlier line once the tagged
# brackets are removed from exactly these lines, plus the two appended entries.
CVT67_PRE_CORRECTIONS_COMMIT = "45eac02ad5afc87ad66c203a575034b2aaa02558"
CVT67_PRE_CORRECTIONS_SHA256 = "903cca25b6449b8ec82ce64c7a669d2de796ee9880ed7de45bc4d3614569b13e"
CVT67_CORRECTIONS_SHA256 = "caac42cf2f56edd30d9a043f03a15b47c6308614bda6167bfbb110f3515024db"
CVT67_CORRECTED_LINES = {32557: "CORRECTIONS 246", 32588: "CORRECTIONS 246", 32589: "CORRECTIONS 246", 32592: "CORRECTIONS 246",
                         32715: "CORRECTIONS 246", 32775: "CORRECTIONS 246", 32834: "CORRECTIONS 247"}
CVT67_REGISTRATION_CORRECTIONS = {
    "MT224": {"number": 246, "entry": "246.9",
              "reason": ("CORRECTIONS 246.9 corrected cvt6's registration (CORRECTIONS 242) in place, with brackets and the superseded "
                         "wording kept. RULE H's replay path (HEADPATH) is the median of 12 of the 15 landed 100-epoch HEAD runs at this "
                         "cell, not of every one: cpl2 HEAD s75-77 were omitted, and the 15-run median differs from the replay by at most "
                         "0.1601 log units (mean 0.0100), re-derived from the raw probe records; corrected in four places. 242.14's 'the "
                         "first run COMPLETED': by sacct end times 8 runs had ended at that pass. 242.3(2)'s 'epoch 18.2': update 9,036 = "
                         "epoch 18.07. The registration stays valid, because the replay file is what was registered, witnessed and bitten; "
                         "no number, gate, token or outcome moves.")},
    "MT225": {"number": 247, "entry": "247.12",
              "reason": ("CORRECTIONS 247.12 corrected a cross-reference in cvt7's registration (CORRECTIONS 243.4) in place, with a "
                         "bracket: the after-launch hashes are recorded in 243.13, not 243.11. Cosmetic; no number, gate, token or "
                         "outcome moves.")},
}


def cvt67_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt6 + cvt7 landing; only header line 3, bracket insertions on line 5 and two appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT67_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CVT67_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CVT67_COMMIT[:12]} does not match the pinned cvt6/cvt7-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = c244_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != CVT67_LAST_ROW or len(before) != CVT67_FIRST_ROW - 1 or changed != CVT67_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {CVT67_COMMIT[:7]} moved a line or edited lines other than {sorted(CVT67_EDITED_LINES)}: {sorted(changed)}")
    line5 = lines[4]
    for tag in reversed(CVT67_LINE5_TAGS):
        line5 = strip_inserted_brackets(line5, tag)
    if line5 != before[4]:
        raise ValueError(f"MASTER-TABLE line 5 at {CVT67_COMMIT[:7]} changed more than inserted {' / '.join(CVT67_LINE5_TAGS)} brackets")
    return lines


def cvt67_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the landing and at the ingest; earlier entries may differ only by the listed inserted brackets."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT67_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != CVT67_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT67_COMMIT[:12]} does not match the pinned bytes")
    previous = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT67_PRE_CORRECTIONS_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(previous).hexdigest() != CVT67_PRE_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT67_PRE_CORRECTIONS_COMMIT[:12]} does not match the pinned bytes")
    lines, before = raw.decode("utf-8").splitlines(), previous.decode("utf-8").splitlines()
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) <= len(before) or changed != set(CVT67_CORRECTED_LINES):
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT67_COMMIT[:7]} edited lines other than the listed corrections: {sorted(changed ^ set(CVT67_CORRECTED_LINES))}")
    for n, tag in CVT67_CORRECTED_LINES.items():
        if tag not in lines[n - 1] or strip_inserted_brackets(lines[n - 1], tag) != before[n - 1]:
            raise ValueError(f"{CORRECTIONS_DOC} line {n} at {CVT67_COMMIT[:7]} changed more than an inserted {tag} bracket")
    if not any(line.startswith("## 246. ") for line in lines[len(before):]) or not any(line.startswith("## 247. ") for line in lines[len(before):]):
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT67_COMMIT[:7]} does not append entries 246 and 247")
    return lines, before


def cvt67_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 224 (cvt6) and 225 (cvt7), attach their intervention notes and registration corrections."""
    rows = with_interventions(appended_rows(lines, CVT67_ROWS, CVT67_FIRST_ROW, CVT67_LAST_ROW, CVT67_COMMIT), CVT67_INTERVENTIONS)
    for row in rows:
        spec = CVT67_REGISTRATION_CORRECTIONS[row["id"]]
        entries = sorted({line for line, tag in CVT67_CORRECTED_LINES.items() if tag == f"CORRECTIONS {spec['number']}"})
        row["registration_corrections"] = json.dumps([{"number": spec["number"], "reason": spec["reason"], "lines": entries,
                                                       "source": f"{CORRECTIONS_DOC} at {CVT67_COMMIT[:7]}; CORRECTIONS {spec['entry']}"}], ensure_ascii=False)
    return rows


# ---------------------------------------------------------------------------
# 11. The cvt8 and cvt9 landings (CORRECTIONS 252-253): two appended rows, no row amended; one registration corrected in place.
# ---------------------------------------------------------------------------
# Campaign commit ba01f54 (cycle 153, CORRECTIONS 252 + 253) appended the cvt8 and cvt9 landings as lines 226 and 227,
# recounted header line 3 and amended the bottom-line paragraph (line 5) with inserted brackets, superseded clauses kept.
# Nothing else may differ from the cvt6/cvt7 pin and no line may move; line 5 must give the pinned line back byte for byte
# once the CORRECTIONS 252 and 253 brackets are removed. Line 5 feeds no record. No existing row was amended, so no earlier
# record changes: MT225's dose confound, which 252 resolves toward dose, and MT224's GRADED reading, which 253 replicates,
# are left as their rows still read (CVT89_BEARS_ON records the relationship).
CVT89_COMMIT = "ba01f54ae6a69c8ad98103993b8d47a48c3be281"
CVT89_MASTER_TABLE_SHA256 = "eca9b99ce2f5c5502320ac29421e1ec89d71bb58524b8bc285b801ce43713885"
CVT89_FIRST_ROW, CVT89_LAST_ROW = 226, 227
CVT89_EDITED_LINES = {3, 5}
CVT89_LINE5_TAGS = ("CORRECTIONS 252", "CORRECTIONS 253")
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# Both are Mixed, as MT218, MT221 and MT224: each registered branch answers its question, the positive control reproduces
# and G-BITE passes 21/21, but a registered expectation was defied -- no registered account fits every band.
# cvt8 (252.3): of the four DOSE-family accounts, three hit 6 of 7 (DOSE x DIRECT misses HIGHISOPATH by 2.19 pp, DOSE x
# PARTIAL-VIA misses BIGISOPATH by 10.58, DOSE-DIRECT-ONLY-AT-BIG misses HIGHISOPATH by 3.81) and DOSE x VIA hits 5 (misses
# HIGHISOPATH by 3.81 and BIGISOPATH by 42.58), and both big-dose arms returned registered
# BELOW-K01 tokens (every seed under its same-seed k01 run), so DOSE-FULL names a location and BIGROUTE-DIRECT compares two
# arms below k01, where a recovery under 5 pp cannot be seen (252.8 item 3: not evidence that the free complement plays no
# role). The scorer's two RULE 16 text defects (the BIGROUTE-DIRECT licence's 'the partial loss'; the broken-REST_HOLD
# null coded at NO-DOSE levels) reach no gate, branch or stamp and stay in the reason and scope.
# cvt9 (253.3): DOSE-GRADED's account misses the MIDDOSE 20-50 band by 5.86 pp, DOSE-THRESHOLD misses RESDOSE by 5.84,
# WINDOW-GRADED misses EARLY by 1.63 and BOTH-WINDOWS misses EARLY and LATE; the tokens come from the registered bars, so the
# band misses are descriptive, but the pattern is one no account predicted. Its FINAL is one 2.92 pp move from
# REPLICATE-FAILED, which stays in the reason.
# Not Goal missed for either: no registered hypothesis was refuted and no control failed. Not Goal met: see above.
# Neither row corrects an earlier published claim, so neither carries the Corrected badge.
CVT89_ROWS = {
    226: ("mixed", 9, ["cvt8"], ["page-19"], None, "DOSE-FULL+ROUTE-PARTIAL+BIGROUTE-DIRECT: on ResNet18_c100 at ciso1's cell, ISO's three-carrier group on PlainNet's dose (HOLDBIG, tri:9428, complement free) puts the run at the k01 location (19.38 vs k01 23.06) while k01's own dose (HOLDHIGH 49.63) replicates cvt7's partial loss (P_DOSE +30.25 pp = +53.85 SE); forcing ISO's complement path recovers part of the loss at k01's dose (HIGHISOPATH 58.19, P_ROUTE +8.56 pp = +15.23 SE); at PlainNet's dose, with the complement forced, the carriers alone reproduce HOLDBIG's stall (BIGISOPATH 19.42, P_ROUTE_BIG +0.05 pp), but both arms sit below k01, so a recovery under 5 pp could not be seen there and this is not evidence that the free complement's collapse plays no role in HOLDBIG; the complement-hold control keeps ISO's level (LOWISOPATH 70.47, P_CTLC -0.21 pp) and G-BITE passes 21/21. But no registered account fits every band: of the four DOSE-family accounts, three hit 6 of 7 (DOSE x DIRECT misses HIGHISOPATH by 2.19 pp) and DOSE x VIA hits 5, both big-dose arms sit below k01 on every seed (HOLDBIG 3.68 pp = -6.54 SE, BIGISOPATH 3.63 pp under), and the scorer carries two RULE 16 text defects that reach no gate (the BIGROUTE-DIRECT licence says 'the partial loss'; the broken-REST_HOLD null is coded at NO-DOSE levels, so only G-BITE separates it). Bounded: the nearest bar is 3.56 pp under ROUTE-PARTIAL, the forced complement is open-loop, three carriers are held where PlainNet held one, and there are two doses with no time gate."),
    227: ("mixed", 9, ["cvt9"], ["page-19"], None, "DOSE-GRADED + WINDOW-GRADED: on PlainNet18_c100 at cvt1's cell, with the complement forced onto HEADPATH in every held arm, idx 50's level falls through two BETWEEN states over the three rungs tried (MIDDOSE tri:7235 55.86, RESDOSE tri:8609 21.84) to the k01 location at tri:9428 (HIGHHEADPATH 11.21 vs k01 12.13), and either half of the large trajectory alone gives a large partial loss (EARLY 18.37, LATE 27.71; P_WIN = LATE - EARLY +9.34 pp = +15.31 SE, same sign on every seed); the control LOWHEADPATH (65.21) replicates cvt6 and G-BITE passes 21/21. But no registered account fits every band (DOSE-GRADED's misses the MIDDOSE 20-50 band by 5.86 pp, WINDOW-GRADED's misses EARLY by 1.63 pp; the tokens come from the registered bars and the band misses are descriptive), so the pattern is one no account predicted. Bounded: one 2.92 pp move turns the whole FINAL into REPLICATE-FAILED, dose is the triangle family with no threshold or functional form, the one window pair is open-loop, cut at the peak and located only to updates 9403-9501, and no necessity or single-window sentence is licensed."),
}
CVT89_BEARS_ON = {226: ["MT225", "MT224"], 227: ["MT224", "MT222"]}  # cvt8 resolves cvt7's dose confound and transplants cvt6's HIGHHEADPATH; cvt9 doses and windows cvt6's HIGHHEADPATH (replicating 240's OWN-STEP-MAGNITUDE)
CVT89_INTERVENTIONS_TSV_SHA256 = "ff9533348f9f2f0db04c66e7f6cbe8f191cd05bc59cb1c7dc359976db7cbe191"
CVT89_INTERVENTIONS = {
    "MT226": {
        "batch": "cvt8", "arms": {"HOLDHIGH": 3, "HOLDBIG": 3, "HIGHISOPATH": 3, "BIGISOPATH": 3, "LOWISOPATH": 3},
        "title": "HOLDHIGH, HOLDBIG, HIGHISOPATH, BIGISOPATH and LOWISOPATH are group step-size hold interventions, not plain ISO arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT89_COMMIT[:7]}; CORRECTIONS 248, 251 and 252",
        # All 21 runs ran harness_cvt8 through jobs/run_cifar_cvt8.sh (CORRECTIONS 248 guard 6); results/cvt8_rule20_full.txt:
        # GROUP_HOLD off x6 (k01, ISO), REST_HOLD off x12 (k01, ISO, HOLDHIGH, HOLDBIG), REST_HOLD on x9.
        "note": ("All seven arms ran harness_cvt8 (cvt7's tree plus the opt-in PATCH_RESTHOLD) through jobs/run_cifar_cvt8.sh; every arm "
                 "but k01 used ISO's grouping [59,3], which isolates the three carriers layer4.0.bn2.weight, layer4.0.shortcut.1.weight "
                 "and layer4.1.bn2.weight together in one group with their vote out of the complement's sum. PATCH_GROUPHOLD "
                 "(GROUP_HOLD=<n1>+<n2>+<n3>:floor|tri:<P>) holds that group's beta (its log step size) from init; PATCH_RESTHOLD "
                 "(REST_HOLD=rec:<id>) then forces the 59-tensor complement's beta onto a replay. HOLDHIGH = the group on k01's "
                 "measured trajectory (tri:8609, peak -5.21) with the complement free (cvt7's arm, replicated); HOLDBIG = the group "
                 "on tri:9428 (peak -4.39, PlainNet's schedule bitwise) with the complement free; HIGHISOPATH = tri:8609 with the "
                 "complement forced onto ISOPATH (the record-by-record median complement of 12 eligible ISO runs, replay file "
                 "08ab25f3); BIGISOPATH = tri:9428 with the complement on ISOPATH; LOWISOPATH = the group at the -15 floor with the "
                 "complement on ISOPATH (the complement hold's positive control). REST_HOLD was set only on HIGHISOPATH, BIGISOPATH "
                 "and LOWISOPATH; HOLDHIGH and HOLDBIG ran with REST_HOLD off, as k01 and ISO did. The three forced arms hold both "
                 "groups, so they are open-loop: both Lion momentum entries are dead state. The holds ride only the run's own "
                 "GROUP_HOLD: and REST_HOLD: witness lines, so the run inventory writes all 15 rows with ISO's cell key: seed for "
                 "seed they differ from ISO only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT plain "
                 "ISO measurements. The 15 rows are listed in results/CORPUS-EXCLUSIONS.tsv by their GROUP_HOLD witness; the 9 "
                 "forced rows' REST_HOLD line is checked through MULTI_KIND (CORRECTIONS 251) and shown on each run as its "
                 "additional witness. Drop them before pooling runs by cell. The k01 and ISO arms print VOTE_W: off, BETA_HOLD: off, "
                 "GROUP_HOLD: off and REST_HOLD: off and are ordinary measurements of their cells."),
    },
    "MT227": {
        "batch": "cvt9", "arms": {"LOWHEADPATH": 3, "HIGHHEADPATH": 3, "MIDDOSE": 3, "RESDOSE": 3, "EARLY": 3, "LATE": 3},
        "title": "LOWHEADPATH, HIGHHEADPATH, MIDDOSE, RESDOSE, EARLY and LATE are step-size hold interventions, not plain HEAD arms",
        "source": f"{INTERVENTIONS_TSV} at {CVT89_COMMIT[:7]}; CORRECTIONS 249, 251 and 253",
        # All 21 runs ran harness_cvt9 through jobs/run_cifar_cvt9.sh (CORRECTIONS 249); results/cvt9_rule20_full.txt:
        # BETA_HOLD off x3 and COMP_HOLD off x3 (k01), COMP_HOLD on x18, WINDOW_HOLD on x6 (EARLY, LATE), off x15.
        "note": ("All seven arms ran harness_cvt9 (cvt6's tree plus the opt-in PATCH_WINDOWHOLD) through jobs/run_cifar_cvt9.sh; the six "
                 "held arms used HEAD's grouping ({50} [52,1]). PATCH_BETAHOLD holds tensor 50's beta (layer4.1.bn2.weight's log step "
                 "size); PATCH_COMPHOLD (COMP_HOLD=rec:cvt6_headpath) forces the 52-tensor complement's beta onto HEADPATH (the "
                 "knot-by-knot median of 12 landed HEAD complements, replay file 74be71fa) in every held arm; PATCH_WINDOWHOLD "
                 "(WINDOW_HOLD=<n0>:<n1|end>) cuts 50's tri hold to an update window, the floor outside. LOWHEADPATH = 50 at the -15 "
                 "floor (the control, cvt6's arm replicated); HIGHHEADPATH = 50 on tri:9428 (peak -4.39, cvt6's arm replicated); "
                 "MIDDOSE = tri:7235 (peak -6.58); RESDOSE = tri:8609 (peak -5.21, cvt7's ResNet replay); EARLY = tri:9428 for updates "
                 "n < 9429, then the floor; LATE = the floor, then tri:9428 from update 9429. WINDOW_HOLD was set only on EARLY and "
                 "LATE. All six hold both groups, so they are open-loop: both Lion momentum entries are dead state. The holds ride "
                 "only the run's own BETA_HOLD:, COMP_HOLD: and WINDOW_HOLD: witness lines, so the run inventory writes all 18 rows "
                 "with HEAD's cell key: seed for seed they differ from HEAD only in run, job_id, node, wallclock_min and the accuracy "
                 "columns. They are NOT plain HEAD measurements. The 18 rows are listed in results/CORPUS-EXCLUSIONS.tsv by their "
                 "BETA_HOLD witness; their COMP_HOLD lines, and EARLY and LATE's WINDOW_HOLD lines, are checked through MULTI_KIND "
                 "(CORRECTIONS 251) and shown on each run as additional witnesses. The probe records locate the window cut only to "
                 "updates 9403-9501; the exact update 9429 rests on the witness line and a CPU test (CORRECTIONS 251). Drop them "
                 "before pooling runs by cell. The batch has no HEAD arm; its k01 arm prints VOTE_W: off, BETA_HOLD: off, COMP_HOLD: "
                 "off and WINDOW_HOLD: off and is an ordinary measurement of its cell."),
    },
}
# The landing also corrected cvt9's registration in place (docs/CORRECTIONS.md, one bracket inserted, superseded figure
# kept): 249.3's RULE E integral (253.9). It is registration text, not a record's published claim, so it carries no
# Corrected badge; MT227 shows it as a documented limitation. cvt8's landing corrected no registration text. The pinned
# file at ba01f54 must equal the one at the ingest commit 1cb52f7 on every earlier line once the tagged bracket is removed
# from exactly this line, plus the two appended entries; and every line of the cvt6/cvt7 pin (entries up to 247) is unchanged.
CVT89_PRE_CORRECTIONS_COMMIT = "1cb52f74f469932a6fdb6e12d488d51a711f8b1d"
CVT89_PRE_CORRECTIONS_SHA256 = "1c6f8038e1433bf069e0ce6a55aebd4e18f654f4b0466c1e634660cf5fe775d0"
CVT89_CORRECTIONS_SHA256 = "f782dcb762e9d08d71df717272498f4583060bf0291119986760faef40c0a45b"
CVT89_CORRECTED_LINES = {33610: "CORRECTIONS 253"}
CVT89_REGISTRATION_CORRECTIONS = {
    "MT227": {"number": 253, "entry": "253.9",
              "reason": ("CORRECTIONS 253.9 corrected cvt9's registration (CORRECTIONS 249.3, RULE E) in place, with a bracket and the "
                         "superseded figure kept: LATE's integrated (r - 1) is 40,647,952.4 = 4.06480e7 at 6 significant figures, not the "
                         "truncated 4.06479e7 (the cvt9 verifier's fix 4). The ratio 1.00103 is correct and no gate, bar or schedule "
                         "reads the figure, so no number, gate, token or outcome moves.")},
}


def cvt89_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt8 + cvt9 landing; only header line 3, bracket insertions on line 5 and two appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT89_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CVT89_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CVT89_COMMIT[:12]} does not match the pinned cvt8/cvt9-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cvt67_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != CVT89_LAST_ROW or len(before) != CVT89_FIRST_ROW - 1 or changed != CVT89_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {CVT89_COMMIT[:7]} moved a line or edited lines other than {sorted(CVT89_EDITED_LINES)}: {sorted(changed)}")
    line5 = lines[4]
    for tag in reversed(CVT89_LINE5_TAGS):
        line5 = strip_inserted_brackets(line5, tag)
    if line5 != before[4]:
        raise ValueError(f"MASTER-TABLE line 5 at {CVT89_COMMIT[:7]} changed more than inserted {' / '.join(CVT89_LINE5_TAGS)} brackets")
    return lines


def cvt89_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the landing and at the ingest; earlier entries may differ only by the listed inserted bracket."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT89_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != CVT89_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_COMMIT[:12]} does not match the pinned bytes")
    previous = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT89_PRE_CORRECTIONS_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(previous).hexdigest() != CVT89_PRE_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_PRE_CORRECTIONS_COMMIT[:12]} does not match the pinned bytes")
    lines, before = raw.decode("utf-8").splitlines(), previous.decode("utf-8").splitlines()
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) <= len(before) or changed != set(CVT89_CORRECTED_LINES):
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_COMMIT[:7]} edited lines other than the listed corrections: {sorted(changed ^ set(CVT89_CORRECTED_LINES))}")
    for n, tag in CVT89_CORRECTED_LINES.items():
        if tag not in lines[n - 1] or strip_inserted_brackets(lines[n - 1], tag) != before[n - 1]:
            raise ValueError(f"{CORRECTIONS_DOC} line {n} at {CVT89_COMMIT[:7]} changed more than an inserted {tag} bracket")
    if not any(line.startswith("## 252. ") for line in lines[len(before):]) or not any(line.startswith("## 253. ") for line in lines[len(before):]):
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_COMMIT[:7]} does not append entries 252 and 253")
    earlier, _ = cvt67_corrections(repo)
    if lines[:len(earlier)] != earlier:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_COMMIT[:7]} changed an entry of the cvt6/cvt7 pin {CVT67_COMMIT[:7]}")
    return lines, before


def cvt89_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 226 (cvt8) and 227 (cvt9), attach their intervention notes and registration correction."""
    rows = with_interventions(appended_rows(lines, CVT89_ROWS, CVT89_FIRST_ROW, CVT89_LAST_ROW, CVT89_COMMIT), CVT89_INTERVENTIONS)
    if not set(CVT89_REGISTRATION_CORRECTIONS) <= {row["id"] for row in rows}:
        raise ValueError("CVT89 registration corrections name a record that is not a cvt8/cvt9 row")
    for row in rows:
        spec = CVT89_REGISTRATION_CORRECTIONS.get(row["id"])
        if not spec:
            continue
        entries = sorted({line for line, tag in CVT89_CORRECTED_LINES.items() if tag == f"CORRECTIONS {spec['number']}"})
        row["registration_corrections"] = json.dumps([{"number": spec["number"], "reason": spec["reason"], "lines": entries,
                                                       "source": f"{CORRECTIONS_DOC} at {CVT89_COMMIT[:7]}; CORRECTIONS {spec['entry']}"}], ensure_ascii=False)
    return rows


# ---------------------------------------------------------------------------
# 11b. The final audit of the cvt8 / cvt9 landing (CORRECTIONS 253.15): row 227 corrected in place, wording only.
# ---------------------------------------------------------------------------
# Campaign commit 6d09d1f appended CORRECTIONS 253.15: row 227's bound (1) said "DOSE-GRADED changes on one arm only if
# RESDOSE rises ~39 pp", but the registered decide() of analysis/cVT9_dosewindow_score.py returns DOSE-NONMONOTONE whenever
# RESDOSE - MIDDOSE > 5, so MIDDOSE falling by more than 39.02 pp flips the word too. Campaign commit 3cf4201 then corrected
# row 227's so-what cell in place -- the two-route sentence, followed by one "**[CORRECTED IN PLACE at cycle 153,
# CORRECTIONS 253.15: ... SUPERSEDED wording, kept verbatim: '<old words>']**" bracket -- and added a one-line note to
# 253.15. The CVT89 pin covers that commit too. Its MASTER-TABLE may differ from the landing pin (ba01f54) only in row
# 227's so-what cell, and only by that one correction: removing the bracket and putting the old words back must give the
# pinned cell back byte for byte, and the bracket must quote the old words verbatim. Its CORRECTIONS may differ from the
# landing pin only by the 253.15 section, note included, inserted before the closing "Next free number" line. No number,
# gate, token or outcome moves: MT227 stays Mixed without a Corrected badge, its scope takes the corrected cell (so the old
# words stay in it, bracketed) and it gains a warning-MT227-amendment-253 record. The rows, the exclusion list and the
# registration correction stay pinned to the landing commit.
CVT89_AUDIT_COMMIT = "3cf42011d04339a08631d9b17c413a0121152edb"
CVT89_AUDIT_MASTER_TABLE_SHA256 = "577c77f756dc1c4bcfb41edd462f3c766f441d67ebf206a8a0baefb57b80c0fe"
CVT89_AUDIT_CORRECTIONS_SHA256 = "69399df3a856636c36a1fee2823fe59dbd5163b07dcfd9f64e69014c4418fec2"
CVT89_AUDIT_SECTION = "### 253.15 "
CVT89_AUDIT_NOTE = "*Note, added after this entry: MASTER-TABLE row 227 has since been amended in place."
CVT89_AMENDMENTS = {
    "MT227": {"line": 227, "outcome": "mixed", "cell": 5, "number": 253, "entry": "253.15", "tag": "CORRECTIONS 253.15",
              "superseded": "and DOSE-GRADED changes on one arm only if RESDOSE rises ~39 pp",
              "amended": "and DOSE-GRADED changes on one arm only if RESDOSE rises, or MIDDOSE falls, by ~39 pp (DOSE-NONMONOTONE)",
              "reason": ("Outcome unchanged (Mixed). Amended at CORRECTIONS 253.15, the landing's final audit: row 227's bound said "
                         "DOSE-GRADED changes on one arm only if RESDOSE rises ~39 pp, but the registered decide() returns "
                         "DOSE-NONMONOTONE whenever RESDOSE - MIDDOSE > 5, which at the landed means (difference -34.02) happens if "
                         "RESDOSE rises or MIDDOSE falls by more than 39.02 pp (a MIDDOSE at 16.84 is still BETWEEN). The row now "
                         "names both routes and keeps the superseded wording in its bracket. No number, gate, token or outcome moves.")},
}


def check_in_place_correction(before: str, after: str, spec: dict) -> None:
    """``after`` is ``before`` with exactly one in-place wording correction in cell ``spec["cell"]``, or a ValueError.

    The correction replaces ``spec["superseded"]`` by ``spec["amended"]`` once and inserts one bracket carrying
    ``spec["tag"]`` that quotes the superseded words verbatim; nothing else in the row may change."""
    tag, cell = spec["tag"], spec["cell"]
    old, new = table_cells(before), table_cells(after)
    if len(old) != 7 or len(new) != 7 or [i for i in range(7) if old[i] != new[i]] != [cell]:
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the {tag} correction must change exactly cell {cell}")
    brackets = re.findall(r"\[[^\[\]]*?" + re.escape(tag) + r"[^\[\]]*\]", new[cell])
    if len(brackets) != 1 or f"SUPERSEDED wording, kept verbatim: '{spec['superseded']}']" not in brackets[0]:
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the {tag} correction needs one bracket keeping the superseded wording verbatim")
    stripped = strip_inserted_brackets(new[cell], tag)
    if stripped.count(spec["amended"]) != 1 or stripped.replace(spec["amended"], spec["superseded"], 1) != old[cell]:
        raise ValueError(f"MASTER-TABLE line {spec['line']} changed more than the {tag} correction")


def cvt89_audit_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the final audit (CORRECTIONS 253.15); only row 227's one in-place correction may differ from ba01f54."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT89_AUDIT_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != CVT89_AUDIT_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {CVT89_AUDIT_COMMIT[:12]} does not match the pinned final-audit bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cvt89_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != len(before) or changed != {spec["line"] for spec in CVT89_AMENDMENTS.values()}:
        raise ValueError(f"MASTER-TABLE at {CVT89_AUDIT_COMMIT[:7]} moved a line or edited lines other than row 227: {sorted(changed)}")
    for spec in CVT89_AMENDMENTS.values():
        check_in_place_correction(before[spec["line"] - 1], lines[spec["line"] - 1], spec)
    return lines


def cvt89_audit_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the final audit and at the landing pin; only the 253.15 section and its note may be added."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{CVT89_AUDIT_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != CVT89_AUDIT_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_AUDIT_COMMIT[:12]} does not match the pinned final-audit bytes")
    lines = raw.decode("utf-8").splitlines()
    before, _ = cvt89_corrections(repo)
    # The landing closes with "Next free number: **254**."; the 253.15 section is inserted just above that line.
    inserted = lines[len(before) - 1:-1]
    if (len(lines) <= len(before) or lines[:len(before) - 1] != before[:-1] or lines[-1] != before[-1]
            or not before[-1].startswith("Next free number: ") or not inserted[0].startswith(CVT89_AUDIT_SECTION)
            or any(line.startswith("#") for line in inserted[1:]) or sum(line.startswith(CVT89_AUDIT_NOTE) for line in inserted) != 1):
        raise ValueError(f"{CORRECTIONS_DOC} at {CVT89_AUDIT_COMMIT[:7]} changed more than the inserted 253.15 section and its note")
    return lines, before


def apply_cvt89_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply the final audit's in-place correction of row 227 (CORRECTIONS 253.15) to MT227's scope."""
    lines, before = cvt89_audit_master_table(repo), cvt89_master_table(repo)
    cvt89_audit_corrections(repo)
    missing = set(CVT89_AMENDMENTS) - {row["id"] for row in rows}
    if missing:
        raise ValueError(f"CORRECTIONS 253.15 amendments name absent records: {sorted(missing)}")
    amended = []
    for row in rows:
        spec = CVT89_AMENDMENTS.get(row["id"])
        if not spec:
            amended.append(row)
            continue
        line, cell, entry = spec["line"], spec["cell"], spec["entry"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if row["master_table_line"] != str(line):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != spec["outcome"]:
            raise ValueError(f"{row['id']} outcome drifted before its CORRECTIONS {entry} amendment: {row['outcome']} (expected {spec['outcome']})")
        row = dict(row)
        if row["scope"].count(clean(old[cell])) != 1:
            raise ValueError(f"{row['id']}: the corrected MASTER-TABLE cell is not in the record's scope exactly once")
        row["scope"] = row["scope"].replace(clean(old[cell]), clean(new[cell]))
        sources = json.loads(row.get("sources") or "[]")
        # The anchor that pinned the row's text at the landing now points at the corrected row.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        row["sources"] = json.dumps([*sources, f"docs/CORRECTIONS.md [CORRECTIONS {entry}]"], ensure_ascii=False)
        further = json.loads(row.get("further_amendments") or "[]")
        further.append({"number": spec["number"], "line": line, "commit": CVT89_AUDIT_COMMIT, "previousOutcome": spec["outcome"], "outcome": spec["outcome"],
                        "reason": spec["reason"], "source": f"{MASTER_TABLE} line {line} at {CVT89_AUDIT_COMMIT[:7]}; CORRECTIONS {entry}"})
        row["further_amendments"] = json.dumps(further, ensure_ascii=False)
        amended.append(row)
    return amended


# ---------------------------------------------------------------------------
# 12. The four MUST-tier landings (CORRECTIONS 264-266): four appended rows, no row amended.
# ---------------------------------------------------------------------------
# Campaign commit 40d29cf (cycle 154, CORRECTIONS 264 + 265 + 266) appended the cmo1, cst1, cct1 and cmg1 landings as
# MASTER-TABLE lines 228-231 and recounted header line 3. Nothing else may differ from the cvt8/cvt9 final-audit pin and
# no line may move. Line 5, the bottom-line paragraph, was DELIBERATELY left unamended this cycle (CORRECTIONS 266.12:
# cmo1's W0k01 71.5600 is still 6.03 pp below that cell's tuned SGD+momentum+cosine peak 77.5927, so GAP_in stands), so
# unlike every landing since cvt3 the pin edits header line 3 alone. No existing row was amended, so no earlier record
# changes; MUST_BEARS_ON records which earlier records these four bear on.
MUST_COMMIT = "40d29cf908c788dffc868d926ca069d89fdea1a6"
MUST_MASTER_TABLE_SHA256 = "3bae7bd5bcdaad6c3bc9213944971f4171f29cdf3e34cf90463b1e24319696b1"
MUST_FIRST_ROW, MUST_LAST_ROW = 228, 231
MUST_EDITED_LINES = {3}
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# MT228 (cmo1) is Mixed, as MT218, MT221, MT224, MT226 and MT227: the momentum leg answers both halves of the question in
# its registered direction, but the weight-decay leg defies a registered expectation -- at base weight decay 0 the scalar
# arm is the BEST arm in the batch, so there is no gap for the isolation to close and the registered ISO reading is
# stamped W0-ISO-AT-REF and NOT read (264.3). Half the question is therefore unanswerable at that cell, and the batch
# additionally ran on two GPU models that were neither registered nor gated (264.4(4)).
# MT229 (cst1) is Open: the registered scorer STOPPED at G-DECOMP on all nine runs and exited 1, so the batch reaches no
# branch and owns no licence sentence -- VERDICT_RULES["open"], "the primary is void, undecided, blocked". Its levels and
# contrasts are labelled DESCRIPTIVE ONLY by the row itself and are not a result. The blocking gate is a scorer tolerance
# DEFECT, reported under RULE 16 and deliberately NOT fixed (265.3); a frozen successor must be registered before its
# output is read.
# MT230 (cct1) is Goal met, as MT219, MT222, MT223 and MT225: both registered words are returned (RATIO 0.9641 and
# DOM_C 0.0000), the scorer exits 0 with every gate passing, no registered band is missed and neither bar is near
# (DOM_C 0 of 1,500 against a 150-record bar; k01 could fall 5.817 pp and still read NOT-COLLAPSED). Its bounds --
# association not causation, the co-varying dataset and head width, three seeds, one cell -- are scope, not a missed band.
# MT231 (cmg1) is Mixed: the merge question is answered and the single licensed sentence is granted, but no registered
# account's LEVELS are reproduced (A3 predicts MCAR 68.4166 against the observed 65.4335; A1 and A2 predict 59.998), so
# ACCOUNT-A3-MERGE-HARMLESS is a BRANCH match only, and the verdict turns on a near bar with 0.8850 pp (1.84 SE) of margin.
# None of the four corrects an earlier published claim, so none carries the Corrected badge: the two refuted sentences
# (264.6 W1's epoch-duplicate claim, 266.6's N-pin clause) were claims of the landing reports themselves, corrected inside
# the same entries before any of this reached the register.
MUST_ROWS = {
    228: ("mixed", 9, ["cmo1"], ["page-19"], None, "M9:COLLAPSE-PERSISTS/ISO-RESCUES+W0:COLLAPSE-IS-CONFIG/ISO-UNREADABLE: the momentum leg answers both halves of the question in its registered direction -- at SGDm base momentum 0.9 the scalar collapse PERSISTS (M9k01 24.5833 against its own layerwise M9kL 69.3913, gap 44.8080 pp = +80.56 SE, ratio 0.3543 against the anchor's 0.3300) and isolating ciso1's three carriers STILL RESCUES (M9ISO 70.9047, clearing the one-sided M9ISO >= M9kL - 5 bar by 6.5133 pp and in fact +1.5133 pp above M9kL) -- and the anchor reproduces at fresh seeds (G_A +46.2620, R_A +47.6340). But the weight-decay leg defies a registered expectation: at base weight decay 0 the scalar arm is the BEST arm in the batch (W0k01 71.5600, +48.7713 pp above the anchor k01 and +3.3900 pp above its own layerwise W0kL), so there is no gap for the isolation to close and the registered ISO reading is stamped W0-ISO-AT-REF and NOT read, leaving the rescue half of the question unanswerable at that cell. One value per factor and one factor at a time, with no CTL arm at either new configuration and the interaction not run; the batch also ran on two GPU models that were neither registered nor gated (10 RTX 2080 Ti, 17 NVIDIA L4), which leaves both half-words untouched because every deciding contrast is hardware-matched seed for seed, but makes DI_M9 a fully confounded comparison."),
    229: ("open", 9, ["cst1"], ["page-19"], None, "UNRESOLVED-DECOMPOSITION: the registered scorer STOPPED at G-DECOMP on all nine runs and exited 1, so the batch reaches NO branch, owns NO licence sentence and NO number in its row is a result; its levels and contrasts are labelled DESCRIPTIVE ONLY. The blocking gate is a scorer TOLERANCE DEFECT, reported under RULE 16 and deliberately NOT fixed: an ABSOLUTE 1e-3 bar is applied to a quotient whose float32 quantisation residual is bounded by 0.5*ulp(beta)/ms = 1.589e-03 at ms 3e-4, and the posted beta equals the registered update BIT FOR BIT on 6,876 of 6,876 unclamped cst1 coordinates (92,788 of 92,788 once the verifier widens it across cst1 and cct1), worst residual 1.358e-03, so the harness applied exactly the registered update and the bar simply cannot be met at this meta step (the corrected forward bound: safe for ms >= 4.77e-04, crossover 4.7684e-04). Every other gate PASSES and G-SPEC excludes the BROKEN-SPEC null, but a FROZEN SUCCESSOR must be REGISTERED BEFORE its output is read, and the blocked reading is bar-sensitive -- DOM_C misses its frozen 0.50 bar by 41 records of 1,500 and TOP3_C by SIX."),
    230: ("met", 9, ["cct1"], ["page-19"], None, "NOT-COLLAPSED+CARRIERS-DO-NOT-DOMINATE: both registered words are returned on CIFAR-10 -- the scalar arm does not collapse in batch (RATIO = k01 / kL = 0.9641 >= 0.90, GAP-IN-BATCH +3.2607 pp = +22.22 SE) and the three carriers fix the shared sign on 0 of k01's 1,500 records (DOM_C 0.0000 <= 0.10) against 0.8069 at the collapsing CIFAR-100 cell -- the scorer exits 0 with every gate passing, no registered band is missed and neither bar is near (DOM_C 0 of 1,500 against a 150-record bar, and k01 could fall 5.817 pp and still read NOT-COLLAPSED). The carriers ARE still the largest single terms (modal top-3 {50,53,59} on 1,041 of 1,500, tensor 59 the argmax on 1,243, BN scales a median 0.545 of the summed abs(L)) but never OUT-WEIGH the other 59, SHARE_C median 0.4228 < 0.5. Its limits -- two cells only, dataset and head width co-varying, the CIFAR-100 side a registration calibration on earlier runs, three seeds -- are scope that the licence states as ASSOCIATION, not registered bands the result missed."),
    231: ("mixed", 9, ["cmg1"], ["page-19"], None, "NO-MERGE-HARMS: the merge question is answered and the single licensed sentence is granted -- DELTA_ID = MCTL - MCAR = +3.0225 pp = +6.27 SE with all four same-seed pairs positive, D_CAR = KLS - MCAR = +4.1150 pp stays INSIDE the frozen 5.0 pp margin, KLS lands in the corpus layerwise band, and G-PARTITION's non-vacuity is re-proved on the real records by 24 cross-reads with 0 pass, which excludes the BROKEN-PARTITION null. But a registered expectation was defied: NO account's LEVELS are reproduced (A3 predicts MCAR 68.4166 against the observed 65.4335, and A1 / A2 predict 59.998), so ACCOUNT-A3-MERGE-HARMLESS is a BRANCH match only and is descriptive. The verdict also turns on a NEAR BAR -- D_CAR is 0.8850 pp (1.84 SE) short of the margin while being +8.54 SE from zero and below the bar in all four seeds, so the carrier merge measurably costs ~4 pp -- DELTA_ID is sub-margin, so the harm may be called neither carrier-specific nor absent, TRAIN-AGREES is disclosed as VACUOUS, and the control triple is not role-matched."),
}
# Earlier records these rows bear on. The import does not rewrite them; the relationship is listed so it stays reviewable.
MUST_BEARS_ON = {
    228: ["MT162", "MT155"],  # ciso1's collapse and rescue, re-run at two base-optimiser settings; W0k01 71.5600 lands level with cdn1's method arm but 6.03 pp under that cell's tuned baseline, so MT155 does not move
    229: ["MT162", "MT163"],  # ciso1's isolation and cdep1's count-matched control, re-asked at a second meta step size; the gate blocks the reading
    230: ["MT162"],           # ciso1's carrier dominance, asked at the one non-collapsing cell the campaign can reach
    231: ["MT163", "MT162"],  # cdep1's matched triple and ciso1's carriers, merged instead of isolated: the necessity direction
}
# The 18 deviating arms. results/all_runs.csv has no column for a base-optimiser CLI flag, so cmo1's M9* and W0* rows
# carry the plain 0.99 / 0.1 cell key; the campaign lists them in the same hash-pinned exclusion file under the ARGS-value
# witness kinds added at CORRECTIONS 263. cst1, cct1 and cmg1 own no exclusion row: none of their runs prints an
# intervention line and their arms are separated by columns the corpus already carries (266.1, 265.1).
MUST_INTERVENTIONS_TSV_SHA256 = "76c910b8f598a4a72a4158391720220a46f1bc70273fa5ecca903cfc58b4edfd"
MUST_ARGS_DEVIATIONS = {
    "MT228": {
        "batch": "cmo1", "arms": {"M9k01": 3, "M9kL": 3, "M9ISO": 3, "W0k01": 3, "W0kL": 3, "W0ISO": 3},
        "title": "cmo1's M9 and W0 arms differ from the standard cell in a base-optimiser CLI flag, not in any CSV column",
        "source": f"{INTERVENTIONS_TSV} at {MUST_COMMIT[:7]}; CORRECTIONS 255, 263 and 264",
        "note": ("All nine cmo1 arms ran the LIVE harness with NO patch: the six deviating arms change the BASE optimiser through "
                 "command-line flags alone. M9k01, M9kL and M9ISO ran --momentum-param-base 0.9 with weight decay held at the "
                 "standard 0.1; W0k01, W0kL and W0ISO ran --weight-decay-base 0 with momentum held at the standard 0.99; the "
                 "anchor arms k01, kL and ISO ran both flags at the standard 0.99 / 0.1. ONE value per factor and ONE factor at a "
                 "time: the interaction (0.9 AND 0) was not run. Neither flag is a column of results/all_runs.csv, so the run "
                 "inventory writes all 18 rows with the plain 0.99 / 0.1 cell key: seed for seed they differ from their anchor arm "
                 "only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT measurements of the standard "
                 "cell. There is no PATCH ON line to read here, because no patch ran: the witness is the run's OWN ARGS: line, the "
                 "one prefix every run prints, read with argparse last-wins semantics, and each row is listed in "
                 "results/CORPUS-EXCLUSIONS.tsv under the ARGS-value witness kind ARGS_MOMENTUM_BASE or ARGS_WD_BASE added at "
                 "CORRECTIONS 263. Drop them before pooling runs by cell. The batch's own anchor arms are ordinary measurements of "
                 "their cells, and cst1, cct1 and cmg1 own no exclusion row at all."),
    },
}
# docs/CORRECTIONS.md is append-only from the cvt8/cvt9 final audit to this landing: the ingest commit 77c6de9 (corpus
# 3,127 -> 3,181, the 18 ARGS-value exclusion rows) added entries 254-263, and the landing replaced that file's closing
# "Next free number" trailer with entries 264, 265 and 266 and a new trailer. No earlier line moved or changed, so no
# registration text was corrected in place this cycle and no record gains a registration warning.
MUST_INGEST_COMMIT = "77c6de90ff6dcc5444f1c864bafc6dbdfd6b2476"
MUST_INGEST_CORRECTIONS_SHA256 = "0219271152528978d4eccba038b40e9b9987828934dd2629b8a8742bb70c8778"
MUST_CORRECTIONS_SHA256 = "bb2702e1ddc5d7b7c8ccd3c6f12a21645ca2b9a59743f68e99d61e4c4f8f8454"
MUST_ENTRIES = (264, 265, 266)
MUST_TRAILER = "Next free number: "


def must_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the four MUST-tier landings; only header line 3 and four appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MUST_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != MUST_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {MUST_COMMIT[:12]} does not match the pinned MUST-tier-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = cvt89_audit_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != MUST_LAST_ROW or len(before) != MUST_FIRST_ROW - 1 or changed != MUST_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {MUST_COMMIT[:7]} moved a line or edited lines other than {sorted(MUST_EDITED_LINES)}: {sorted(changed)}")
    return lines


def must_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the landing and at the ingest; the ingest's closing trailer is the only line that may go."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MUST_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != MUST_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_COMMIT[:12]} does not match the pinned bytes")
    previous = subprocess.check_output(["git", "-C", str(repo), "show", f"{MUST_INGEST_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(previous).hexdigest() != MUST_INGEST_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_INGEST_COMMIT[:12]} does not match the pinned bytes")
    lines, before = raw.decode("utf-8").splitlines(), previous.decode("utf-8").splitlines()
    if len(lines) <= len(before) or not before[-1].startswith(MUST_TRAILER) or not lines[-1].startswith(MUST_TRAILER):
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_COMMIT[:7]} does not append entries below the ingest's closing trailer")
    if lines[:len(before) - 1] != before[:-1]:
        changed = [n for n in range(1, len(before)) if lines[n - 1] != before[n - 1]]
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_COMMIT[:7]} changed an earlier line: {changed[:8]}")
    appended = [line for line in lines[len(before) - 1:] if line.startswith("## ")]
    if [line.split(".")[0] for line in appended] != [f"## {number}" for number in MUST_ENTRIES]:
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_COMMIT[:7]} does not append exactly entries {MUST_ENTRIES}: {appended[:4]}")
    earlier, _ = cvt89_audit_corrections(repo)
    if lines[:len(earlier)] != earlier:
        raise ValueError(f"{CORRECTIONS_DOC} at {MUST_COMMIT[:7]} changed an entry of the cvt8/cvt9 audit pin {CVT89_AUDIT_COMMIT[:7]}")
    return lines, before


def must_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 228-231 (cmo1, cst1, cct1, cmg1) and attach cmo1's ARGS-value deviation note."""
    rows = appended_rows(lines, MUST_ROWS, MUST_FIRST_ROW, MUST_LAST_ROW, MUST_COMMIT)
    for row in rows:
        spec = MUST_ARGS_DEVIATIONS.get(row["id"])
        if spec:
            row["args_deviations"] = json.dumps({"title": spec["title"], "note": spec["note"], "batch": spec["batch"],
                                                 "arms": spec["arms"], "source": spec["source"]}, ensure_ascii=False)
    return rows


# ---------------------------------------------------------------------------
# 13. The cvt10, cwd1, csv1 and cwd2 landings (CORRECTIONS 270-273): four appended rows, row 229 amended in place.
# ---------------------------------------------------------------------------
# Campaign commit 2972d48 (cycle 155, CORRECTIONS 270 + 271 + 272 + 273) appended the last four MUST-tier landings as
# MASTER-TABLE lines 232-235, recounted header line 3 and amended row 229 (cst1, MT229) in place, carrying the supersession
# CORRECTIONS 268.11 left owed. Nothing else may differ from the MUST-tier pin and no line may move. Line 5, the bottom-line
# paragraph, was again deliberately left unamended (273.10: no verdict of this cycle moves the denominator result).
# docs/CORRECTIONS.md is append-only across the two commits: the ingest 66a19fb (corpus 3,181 -> 3,253, +72 runs, 0 changed,
# 45 exclusion rows) replaced the MUST-tier landing's closing "Next free number" trailer with entries 267, 268 and 269, and
# this landing replaced that trailer with entries 270-273.
MECH4_COMMIT = "2972d4858bf1f1cfcb81d8a7f2e1f87231f71312"
MECH4_MASTER_TABLE_SHA256 = "b6128197755137e265cda5388733054c42aeab369ed015a1bfaf179203d2ac4e"
MECH4_FIRST_ROW, MECH4_LAST_ROW = 232, 235
MECH4_EDITED_LINES = {3, 229}
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# All four are Goal met under VERDICT_RULES, as MT219, MT222, MT223, MT225 and MT230: each returns its registered branch in
# the registered direction, each scorer exits 0 with every gate passing, each positive control reproduces, G-BITE excludes
# the batch's own broken-patch null on records, and NO registered account misses a band and NO control fails -- the test the
# MT218-MT231 precedent applies before Goal met is granted. What each row bounds (sufficiency only; one dose, one tensor set
# or one tensor; one cell; one horizon; three or four seeds; heterogeneous ungated GPU hardware on cwd1 and cwd2; the
# counterfactual shadow vote on csv1) is SCOPE that the reason and the record's scope carry, not a band the result missed,
# exactly as MT230's association-not-causation bounds are. None of the four corrects an earlier published notebook claim, so
# none carries the Corrected badge: the wording each landing fixed was the wording of its own report, fixed inside the same
# entry before any of it reached the register (270.6, 271.6, 272.6, 273.6), and the RULE 16 defects each reports are
# reported and NOT fixed.
MECH4_ROWS = {
    232: ("met", 9, ["cvt10"], ["page-19"], None, "ONE-SUFFICES+ONE50-STALLS+ONE59-STALLS+ONE53-STALLS+SPLIT-NO-EFFECT: on ResNet18_c100 at ciso1's cell, each of the three carriers isolated alone in its own step-size group and held on PlainNet's measured dose (tri:9428, from init) is SUFFICIENT to stall while its free twin rescues -- ONE50 64.7967 -> ONE50BIG 21.1340 (P_ONE50 +43.6627 pp = +78.50 SE), ONE59 67.7053 -> ONE59BIG 21.0973 (P_ONE59 +46.6080 pp = +83.80 SE), ONE53 57.5033 -> ONE53BIG 22.2893 (P_ONE53 +35.2140 pp = +63.31 SE) -- which CLOSES 252.8(1)'s held-set confound at this cell and carries 240 / 246's PlainNet sentence to ResNet for each carrier singly; but the same one-tensor hold does nothing once 53 and 59 leave the shared vote (ISOSPLIT 67.0893, P_SPLIT +3.0767 pp only), so the one-tensor stall needs the remaining carriers voting in the shared step size. The anchor reproduces (k01 22.7540, ISO 70.1660, D_ISO +47.4120 pp = +85.24 SE), cvt8's DOSE-FULL replicates in batch at fresh seeds (HOLDBIG3 18.7233, 6.0307 pp inside the REPLICATE-FAILED bar), the registered scorer exits 0 with every gate passing, G-BITE passes 30/30 and excludes the BROKEN-SINGLETON-HOLD null on records before any level is read, and no registered band is missed. Bounded: every reading is SUFFICIENCY and never necessity; all four stalled arms sit BELOW k01 rather than at it, so ...-AT-K01 is a LOCATION that H-FLOOR predicts equally; the tightest word is SPLIT-NO-EFFECT, which clears ISO - 5 by 1.9233 pp = 3.46 SE while the three -STALLS words clear k01 + 2 by 2.4647-3.6567 pp; ISOSPLIT has no free [59,1,2] control and no plain twin anywhere in the corpus; one dose, one network, one cell, 100 epochs, three seeds, no per-run GPU-hardware census; and one RULE 16 defect is reported and NOT fixed -- a host-dependent DESCRIPTIVE rounding cell that no gate, bar, level, contrast, branch word or stamp reads."),
    233: ("met", 9, ["cwd1"], ["page-19"], None, "COLLAPSE-VANISHES: on ResNet18_c100 at ciso1's cell, masking the base optimiser's coupled weight decay on the 20 BatchNorm scales -- in the weight update AND in the meta trace -- removes the scalar collapse entirely: k01 22.9513 -> k01NWD 70.7760 (P_NWD +47.8247 pp = +85.99 SE), and the masked scalar arm sits ABOVE its own masked layerwise reference (kLNWD 69.3420, G_NWD -1.4340 pp = -2.58 SE, inside the MATCH bar 5.0), with RATIO 1.0207 so the collapse is not reduced but GONE. The registered scorer exits 0 with every gate passing, G-BITE passes 9/9 with the mask record audit k=20 on exactly the six masked runs and k=0 on exactly the three unmasked ones, excluding the BROKEN-MASK null that predicts COLLAPSE-PERSISTS exactly; a live-model check confirms the mask is the 20 BatchNorm2d scales by module type at the 20 registered indices; and no bar is near, VANISHES clearing by 6.4340 pp = 11.57 SE with the branch word unchanged under every single seed and every leave-one-seed-out. This is the decomposition cmo1's W0 half could not do, localising the precondition to 4,800 of 11,220,132 parameters. Bounded: the mask is ONE tensor set and BOTH routes at once, so the batch answers which tensors and NOT which route, and cannot say which of the 20 scales matters; the reference arm is also masked, so G_NWD is masked-vs-masked; there is no ISO arm; coupled decay only, one WD value, one network, one cell, 100 epochs, three seeds; 7 runs ran on an NVIDIA L4 and 2 on an RTX 2080 Ti, unregistered and ungated; the collapsing arm carries no weight-norm readout at all, and where the readout exists the scales grow rather than shrink; and two RULE 16 defects are reported and NOT fixed."),
    234: ("met", 9, ["csv1"], ["page-19"], None, "BOTH-ROUTES: on PlainNet18_c100 at cvt1's cell, separating idx 50's APPLIED step size from the term it casts into the shared meta-gradient sum splits the damage between the two routes -- pinning the applied step at the clamp floor from init while the vote is rebuilt at the shared step size leaves the run BETWEEN (SHADOWLOW 49.8790, P_APPLIED +37.7610 pp = +72.58 SE), and removing that shadow term recovers HEAD's level (NAIVELOW 65.2420 against HEAD 64.4293, P_VOTE +15.3630 pp = +31.89 SE of the 52.3113 pp HEAD - k01 gap). The registered vote-share premise is MET (share(k01) 0.5271 >= 0.30 and share(SHADOWLOW) 0.3592 >= half of it, SHADOW-VOTE-DOMINANT), the identity control INERT is inside its registered +/-2.0 bar (P_INERT +0.2440 pp), MUTE reproduces 230 at the k01 location, the registered scorer exits 0 with every gate passing, no bar is near, and G-BITE passes 18/18 with a working negative control -- 47 counterfactual readings on this batch's own records, 0 passes, including the BROKEN-SHADOW null whose levels are exactly APPLIED-STEP-NECESSARY's. This gives the campaign the carrier NECESSITY statement L2 was missing and closes the RULE 20 half CORRECTIONS 267.2 left owed. Bounded: only ONE small dose was tested -- the clamp floor, from init -- so the honest sentence is that an applied step ABOVE THE CLAMP FLOOR on idx 50 is necessary for the FULL stall, not that a LARGE one is, which is a RULE 16 defect reported against the registered licence string itself and NOT fixed; the shadow vote is COUNTERFACTUAL, so nothing here speaks about unmodified MetaOptimize's own vote; INERT has n = 1 and the batch is unbalanced; MUTE sits 0.8153 pp below k01, a LOCATION; INERT is disclosed as NOT bitwise k01 over the landed 100-epoch run although 262.3 proved that on-path over 300 real steps, so the identity gate passes at its registered tolerance and the word bitwise belongs to the proof job; and a second RULE 16 defect, seven host-dependent DESCRIPTIVE display ties, is reported and NOT fixed."),
    235: ("met", 9, ["cwd2"], ["page-19"], None, "WD-ROUTE + SCALAR-NEEDS-CARRIER-WD: on PlainNet18_c100 at cvt6 / cvt9's cell, masking coupled weight decay on layer4.1.bn2.weight alone -- 512 parameters of 11,046,308 -- removes the damage of the externally held large step on that scale (HIGHHEADPATH 10.7127 -> HIGHWD0 65.6260, P_WD +54.9133 pp = +98.73 SE against the damage to explain R_HIGH +54.2880, F_WD 1.012, the remainder P_LEFT -0.6253 pp inside the NULL bar) and removes the scalar collapse itself (k01 12.0673 -> k01WD0 65.7500, P_SC +53.6827 pp = +96.52 SE). In the three held arms every applied step size is exogenous, so the mask can change only tensor 50's weight update and the ROUTE reading is clean; the two anchors and the control replicate cvt6 and cvt9 between batch, REPLICATE-FAILED sits 3.3547 pp (6.03 SE) away while the two words that are read clear their bar by 5.6253 and 5.7493 pp, the registered scorer exits 0 with every gate passing, and G-BITE passes 15/15 with 15/15 cross-reads refused, excluding the BROKEN-MASK null whose levels are exactly STEP-ROUTE + SCALAR-WITHOUT-CARRIER-WD. 240's OWN-STEP-MAGNITUDE, 246's HIGHHEADPATH-AT-K01 and 253's DOSE-GRADED are therefore REINTERPRETED: the own large step acts through the shrinkage it multiplies. Bounded: heterogeneous GPU hardware, unregistered and ungated, with the arm-centred device effect measured at A100 +0.0460 / L4 -0.0375 / RTX 2080 Ti -0.0727 pp against the 54.9 pp the verdict turns on and seed confounded with device; the mechanism is NOT watched where it is claimed to act, because the patch records weight norms only on masked arms, so both stalling arms carry none and where it is measured there is no filter collapse at all, which leaves Zhou et al. arXiv:2001.11216 neither confirmed nor excluded; HIGHHEADPATH sits 1.3547 pp below k01, a LOCATION; k01WD0 changes both routes at once; the complement is forced onto a replay and the held arms are open loop; one tensor, one cell, one network, 100 epochs, three seeds, decoupled weight decay untested; and two RULE 16 defects are reported and NOT fixed."),
}
# Earlier records these rows bear on. The import does not rewrite them; the relationship is listed so it stays reviewable.
MECH4_BEARS_ON = {
    232: ["MT226", "MT225"],  # cvt8's DOSE-FULL, replicated in batch, and cvt7's TRANSFERS-GRADED: the held set, not the dose, was the open half
    233: ["MT228", "MT162"],  # cmo1's W0 leg, decomposed by tensor set, on ciso1's collapse cell
    234: ["MT218", "MT224"],  # cvt1's MUTE and cvt6's GRADED: the same carrier, with the applied step separated from the vote
    235: ["MT224", "MT227"],  # cvt6's HIGHHEADPATH and cvt9's DOSE-GRADED, reinterpreted as acting through the coupled decay
}
# The 45 intervened rows of this ingest. cvt10's five held arms use the EXISTING GROUP_HOLD kind (258.7); cwd1, cwd2 and
# csv1 use the two kinds CORRECTIONS 269 added to analysis/corpus_exclusions.py -- DECAY_MASK (the base optimiser's coupled
# weight decay set to 0 on a named tensor set, in the weight update and in the meta trace) and SHADOW_VOTE (idx 50's applied
# step size separated from the term it casts into the shared sum). Nine cwd2 rows are multi-kind and six of those carry
# three kinds at once; csv1's MUTE rows keep the existing VOTE_W kind. The file is byte-identical at the ingest 66a19fb,
# where the rows were appended, and at this landing, which did not touch it.
MECH4_INTERVENTIONS_TSV_SHA256 = "7da3901cc1378ef10441f2055b03b53b8cb1a1fda2e32ee099117e6fe0ed572a"
MECH4_INTERVENTIONS = {
    "MT232": {
        "batch": "cvt10", "arms": {"HOLDBIG3": 3, "ONE50BIG": 3, "ONE59BIG": 3, "ONE53BIG": 3, "ISOSPLIT": 3},
        "title": "cvt10's five held arms are group step-size holds, not plain arms",
        "source": f"{INTERVENTIONS_TSV} at {MECH4_COMMIT[:7]}; CORRECTIONS 258 and 270",
        "note": ("HOLDBIG3, ONE50BIG, ONE59BIG, ONE53BIG and ISOSPLIT ran PATCH_GROUPHOLD (GROUP_HOLD=<names>:tri:9428 pins "
                 "one step-size group's beta to cvt4 / cvt6 / cvt8's measured triangular trajectory from init -- PlainNet's "
                 "measured dose). HOLDBIG3 holds all three carriers as one group of three, as cvt8's HOLDBIG did; ONE50BIG, "
                 "ONE59BIG and ONE53BIG hold a group of ONE carrier, which is a new place in the patch's registered grammar; "
                 "ISOSPLIT holds group 1 of a three-group partition, 50 held with {53,59} free in their own group. Each held "
                 "arm's spec is byte-identical to its free twin's, so the run inventory writes the held rows with the free "
                 "arm's cell key: seed for seed they differ from it only in run, job_id, node, wallclock_min and the accuracy "
                 "columns. They are NOT plain measurements of those cells. ISOSPLIT is the one arm in the corpus with NO "
                 "plain twin at its cell key, and its exclusion row says so. The 15 rows are listed in "
                 "results/CORPUS-EXCLUSIONS.tsv under the existing GROUP_HOLD kind -- this batch added no new kind; drop them "
                 "before pooling runs by cell. The five free arms print GROUP_HOLD: off and are ordinary measurements."),
    },
    "MT233": {
        "batch": "cwd1", "arms": {"k01NWD": 3, "kLNWD": 3},
        "title": "cwd1's two masked arms run with the coupled weight decay switched off on the 20 BatchNorm scales",
        "source": f"{INTERVENTIONS_TSV} at {MECH4_COMMIT[:7]}; CORRECTIONS 260, 269 and 271",
        "note": ("k01NWD and kLNWD ran PATCH_DECAYMASK (DECAY_MASK=normscale), which sets the BASE optimiser's coupled weight "
                 "decay to 0 on a named tensor set -- in the weight update AND in the meta trace h <- gamma(1 - wd*a)h - delta. "
                 "normscale is the 20 one-dimensional *.weight tensors, which on the live model are exactly the 20 "
                 "BatchNorm2d scales by module type, 4,800 of 11,220,132 parameters, the three carriers 50 / 53 / 59 among "
                 "them; the conv and linear tensors keep their decay. No column of results/all_runs.csv carries the mask, so "
                 "the inventory writes k01NWD with plain k01's cell key (granularity scalar) and kLNWD with plain kL's "
                 "(layerwise). They are NOT plain measurements of those cells. The 6 rows are listed in "
                 "results/CORPUS-EXCLUSIONS.tsv under the DECAY_MASK witness kind added at CORRECTIONS 269; drop them before "
                 "pooling runs by cell. The k01 anchor prints DECAY_MASK: off and is an ordinary measurement of its cell."),
    },
    "MT234": {
        "batch": "csv1", "arms": {"INERT": 1, "SHADOWLOW": 4, "NAIVELOW": 4, "MUTE": 3},
        "title": "csv1's four intervened arms separate one tensor's applied step size from the vote it casts",
        "source": f"{INTERVENTIONS_TSV} at {MECH4_COMMIT[:7]}; CORRECTIONS 262, 269 and 272",
        "note": ("INERT, SHADOWLOW and NAIVELOW ran PATCH_SHADOWVOTE (SHADOW_VOTE=<shadow or natural>:<floor or shared>:<name>), "
                 "which separates layer4.1.bn2.weight's APPLIED step size from the term it casts into the shared "
                 "meta-gradient sum: applied=floor pins idx 50's applied step at float32(exp(float32(-15))) from init, "
                 "vote=shadow rebuilds 50's slot of the shared sum at the SHARED step size from 50's real momentum and "
                 "weights -- a COUNTERFACTUAL vote, not MetaOptimize's own -- and vote=natural leaves the harness's own "
                 "floor-built trace. INERT (shadow:shared) is the identity control, inside its registered +/-2.0 bar. MUTE ran "
                 "PATCH_VOTEWEIGHT (VOTE_W=layer4.1.bn2.weight:0), cvt1's string. All 12 rows carry plain k01's cell key "
                 "(granularity scalar): seed for seed they differ from it only in run, job_id, node, wallclock_min and the "
                 "accuracy columns, and they are NOT plain scalar measurements. They are listed in "
                 "results/CORPUS-EXCLUSIONS.tsv -- 9 under the SHADOW_VOTE witness kind added at CORRECTIONS 269 and 3 under "
                 "the existing VOTE_W kind; drop them before pooling runs by cell. Track F proposed the same 12 rows "
                 "independently (CORRECTIONS 269.7) and the ingest regenerated them without reading that file, finding the "
                 "same 12 keys with 0 differing cells across all 9 columns. The k01 and HEAD arms are ordinary measurements."),
    },
    "MT235": {
        "batch": "cwd2", "arms": {"k01WD0": 3, "HIGHHEADPATH": 3, "HIGHWD0": 3, "LOWWD0": 3},
        "title": "cwd2's twelve intervened arms combine a step-size hold, a forced complement replay and a decay mask",
        "source": f"{INTERVENTIONS_TSV} at {MECH4_COMMIT[:7]}; CORRECTIONS 261, 269 and 273",
        "note": ("k01WD0 ran PATCH_DECAYMASK alone (DECAY_MASK=layer4.1.bn2.weight), masking the base optimiser's coupled "
                 "weight decay on ONE tensor of 53 -- 512 parameters, idx 50 -- in the weight update and in the meta trace. "
                 "HIGHHEADPATH is cvt6 / cvt9's stall arm replicated in batch: BETA_HOLD=layer4.1.bn2.weight:tri:9428 with "
                 "the complement forced onto cvt6's recorded HEAD path (COMP_HOLD=rec:cvt6_headpath). HIGHWD0 and LOWWD0 add "
                 "the mask to that pair, so those six runs carry THREE interventions at once; LOWWD0 freezes 50 at the -15 "
                 "floor instead, as the control under the same mask. In every held arm the applied step sizes are exogenous "
                 "(open loop), and the complement is FORCED onto a replay, so nothing here says a FREE complement behaves the "
                 "same way. No column of results/all_runs.csv carries any of this, so k01WD0 takes plain k01's cell key and "
                 "the three held arms take HEAD's. The 12 rows are listed in results/CORPUS-EXCLUSIONS.tsv -- the nine "
                 "multi-kind rows by their BETA_HOLD line, with the further holds verified by the three MULTI_KIND entries "
                 "registered at CORRECTIONS 269 and by each run's own log; drop them before pooling runs by cell."),
    },
}
MECH4_INGEST_COMMIT = "66a19fbc9b29976d44ef74c03cd9b388ca742007"
MECH4_INGEST_CORRECTIONS_SHA256 = "00d042175616d5bb8ae4016e48f7740564cdcde1b54d85f09947e37978e545ac"
MECH4_CORRECTIONS_SHA256 = "8d1ff79bd11fec5a38bed44db9645e5be822d87d4d2ad13b2bc43c31c82272dc"
MECH4_ENTRIES = (270, 271, 272, 273)
MECH4_INGEST_ENTRIES = (267, 268, 269)
# Row 229 (cst1, MT229), amended in place. CORRECTIONS 268 registered and PUSHED a frozen successor scorer before it was run
# on a single cst1 record; 273 carries its reading into the row, keeping the predecessor's verdict verbatim in a SUPERSEDED
# bracket. The registered predecessor is UNEDITED and STILL FROZEN (RULE 16), and 265.3's reading stands as the correct
# output of a defective check, so nothing earlier is corrected and the Corrected badge is NOT set -- an outcome moved by
# later data is not a corrected earlier claim (the CORRECTIONS 229 precedent). The outcome DOES move, because the verdict
# column moved and the successor's registered clauses SPLIT: the isolation rescue transfers to a second meta step size and
# the count-matched control stays null, but the vote-dominance nomination misses its frozen 750-record bar by 41 records of
# 1,500 (TOP3_C by six), so the batch may not write the NOMINATED+ISO-RESCUES+CTL-NULL headline. That is VERDICT_RULES
# "mixed", never "met": under the MT218-MT231 precedent a verdict whose registered account misses a band is not Goal met.
MECH4_AMENDMENT_MARK = "[AMENDED at cycle 155, CORRECTIONS 273"
MECH4_AMENDMENT_TAG = "CORRECTIONS 273"
MECH4_AMENDMENTS = {
    "MT229": {
        "line": 229, "number": 273, "entry": "273", "previous": "unresolved", "outcome": "mixed",
        # cells 3 and 5 gain one appended bracket that keeps everything before it; cell 4 is rewritten with the predecessor's
        # verdict kept verbatim inside a SUPERSEDED bracket; cell 6 only gains citations. Cells 0-2 may not change at all.
        "appended_cells": (3, 5), "verdict_cell": 4, "ref_cell": 6,
        "superseded": "UNRESOLVED-DECOMPOSITION",
        "branch": "NOMINATION-PARTIAL+ISO-RESCUES+CTL-NULL",
        "citations": ("268 (the frozen successor registered and scored)", "273 (this amendment)"),
        "reason": ("Outcome moved: Open -> Mixed. Amended at CORRECTIONS 273, carrying the supersession CORRECTIONS 268.11 "
                   "left owed. The REGISTERED predecessor cST1_carrier_contrast_score.py is UNEDITED and STILL FROZEN under "
                   "RULE 16, and 265.3's reading stands as the correct output of a defective check; a FROZEN SUCCESSOR, "
                   "cST2_carriervote_score.py, was registered and pushed as commit 9581897 BEFORE it was run on a single "
                   "cst1 record, with the ms-aware G-DECOMP tolerance 1e-3 + 4*ulp32(beta)/ms as its ONE change and every "
                   "other bar the literal frozen at 256.5 -- proved mechanically over 27 scalar literals, 17 maps and licence "
                   "tables, all 29 branches and licence strings and decide() on 4,000 random points, 0 differences, with the "
                   "two scorers' stdouts differing on 11 of 115 lines and the old and new bars agreeing at cct1's ms 1e-3 on "
                   "all 85,912 coordinates. Reading the same nine runs it PASSES G-DECOMP with 0 failing coordinates of "
                   "6,876, so the numbers 265 could print only as DESCRIPTIVE are now the batch's REGISTERED levels: the "
                   "isolation rescue TRANSFERS to meta step 3e-4 (D_ISO +41.1567 pp = +74.00 SE) and the count-matched "
                   "control stays at k01's level (D_CTL +0.1807 pp), but the vote-dominance nomination does NOT -- DOM_C "
                   "0.4727 is 709 records against a 750-record bar, MISSING BY 41, and TOP3_C 0.4960 is 744 against 750, "
                   "MISSING BY SIX. The registered clauses therefore split between met and missed, which is Mixed and not "
                   "Goal met, and the batch may not write the NOMINATED+ISO-RESCUES+CTL-NULL headline sentence. No earlier "
                   "published claim is corrected, so the Corrected badge is not set and the superseded predecessor verdict "
                   "is kept verbatim in the record."),
    },
}


def mech4_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cvt10/cwd1/csv1/cwd2 landing; only header line 3, row 229 and four appended rows may differ."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH4_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != MECH4_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {MECH4_COMMIT[:12]} does not match the pinned cvt10/cwd1/csv1/cwd2-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = must_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != MECH4_LAST_ROW or len(before) != MECH4_FIRST_ROW - 1 or changed != MECH4_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {MECH4_COMMIT[:7]} moved a line or edited lines other than {sorted(MECH4_EDITED_LINES)}: {sorted(changed)}")
    for spec in MECH4_AMENDMENTS.values():
        check_row229_amendment(before[spec["line"] - 1], lines[spec["line"] - 1], spec)
    return lines


def check_row229_amendment(before: str, after: str, spec: dict) -> None:
    """``after`` is ``before`` with the CORRECTIONS 273 amendment applied, or a ValueError.

    The question, comparison and scale cells may not change; the result and so-what cells may only gain one appended
    bracket carrying the mark; the verdict cell is rewritten but must keep the predecessor's verdict verbatim inside a
    SUPERSEDED bracket; and the ref cell may only gain citations."""
    old, new = table_cells(before), table_cells(after)
    changed = [i for i in range(len(old)) if old[i] != new[i]]
    expected = sorted({*spec["appended_cells"], spec["verdict_cell"], spec["ref_cell"]})
    if len(old) != 7 or len(new) != 7 or changed != expected:
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the {MECH4_AMENDMENT_TAG} amendment must change exactly cells {expected}")
    for cell in spec["appended_cells"]:
        if not new[cell].startswith(old[cell]) or new[cell].count(MECH4_AMENDMENT_MARK) != 1:
            raise ValueError(f"MASTER-TABLE line {spec['line']} cell {cell}: the amendment must be appended and keep every earlier word")
        if strip_inserted_brackets(new[cell], MECH4_AMENDMENT_TAG).rstrip() != old[cell].rstrip():
            raise ValueError(f"MASTER-TABLE line {spec['line']} cell {cell} changed more than the inserted {MECH4_AMENDMENT_TAG} bracket")
    verdict = new[spec["verdict_cell"]]
    if verdict.count(MECH4_AMENDMENT_MARK) != 1 or not clean(verdict).startswith(f"[{MECH4_AMENDMENT_MARK[1:]}"):
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the amended verdict must open with the {MECH4_AMENDMENT_TAG} bracket")
    kept = clean(verdict).replace(" ", "")
    if "[SUPERSEDED" not in clean(verdict) or clean(old[spec["verdict_cell"]]).replace(" ", "") not in kept:
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the amended verdict must keep the superseded wording verbatim")
    if spec["branch"] not in clean(strip_inserted_brackets(verdict, MECH4_AMENDMENT_TAG)):
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the amended verdict does not carry {spec['branch']}")
    if not new[spec["ref_cell"]].startswith(old[spec["ref_cell"]]):
        raise ValueError(f"MASTER-TABLE line {spec['line']}: the ref cell may only gain citations")
    for citation in spec["citations"]:
        if citation not in clean(new[spec["ref_cell"]]):
            raise ValueError(f"MASTER-TABLE line {spec['line']}: the ref cell does not cite {citation}")


def mech4_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the landing and at its ingest; each file's closing trailer is the only line that may go."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH4_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != MECH4_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH4_COMMIT[:12]} does not match the pinned bytes")
    previous = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH4_INGEST_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(previous).hexdigest() != MECH4_INGEST_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH4_INGEST_COMMIT[:12]} does not match the pinned bytes")
    lines, before = raw.decode("utf-8").splitlines(), previous.decode("utf-8").splitlines()
    earlier, _ = must_corrections(repo)
    for name, later, older, entries in [(MECH4_INGEST_COMMIT, before, earlier, MECH4_INGEST_ENTRIES), (MECH4_COMMIT, lines, before, MECH4_ENTRIES)]:
        if len(later) <= len(older) or not older[-1].startswith(MUST_TRAILER) or not later[-1].startswith(MUST_TRAILER):
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} does not append entries below the previous closing trailer")
        if later[:len(older) - 1] != older[:-1]:
            changed = [n for n in range(1, len(older)) if later[n - 1] != older[n - 1]]
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} changed an earlier line: {changed[:8]}")
        appended = [line for line in later[len(older) - 1:] if line.startswith("## ")]
        if [line.split(".")[0] for line in appended] != [f"## {number}" for number in entries]:
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} does not append exactly entries {entries}: {appended[:4]}")
    return lines, before


def mech4_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 232-235 (cvt10, cwd1, csv1, cwd2) and attach their intervention notes."""
    return with_interventions(appended_rows(lines, MECH4_ROWS, MECH4_FIRST_ROW, MECH4_LAST_ROW, MECH4_COMMIT), MECH4_INTERVENTIONS)


# ---------------------------------------------------------------------------
# 15. The cwd3 landing (CORRECTIONS 278): the ResNet carrier-only decay mask.
# ---------------------------------------------------------------------------
MECH5_COMMIT = "97eb0499e8c41bdf510768d61a8651df5fe302c3"
MECH5_MASTER_TABLE_SHA256 = "4f8bee03475b27eb475d6515f1cd7f466442f80715b365f3c15425c6f63c0d15"
MECH5_FIRST_ROW, MECH5_LAST_ROW = 236, 236
MECH5_EDITED_LINES = {3}  # the run / GPU-hour header only: this landing amended NO earlier row and did not touch line 5.
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
# MT236 (cwd3) is Goal met, as MT232-MT235 and MT230: it returns its registered branch AND its registered state word in the
# registered direction -- the best of the accounts CORRECTIONS 275.4 wrote down before any run existed -- the scorer exits 0
# with every gate passing, the positive reference NWD replicates cwd1 in batch, G-BITE excludes the batch's own broken-mask
# null on records, and NO registered account misses a band and NO control fails, which is the test the MT218-MT235 precedent
# applies before Goal met is granted. What the row bounds -- two floor readings that are BOUNDS and not measured zeros, a
# residual that is a bound in the other direction and whose sign is horizon-dependent, a descriptive ratio that drifts
# through 1, position class and count/dose both unseparated, one switch changing both routes, one network, one cell, one
# horizon, three seeds -- is SCOPE that the reason and scope carry, exactly as MT235's bounds are. It corrects no earlier
# published notebook claim, so it carries no Corrected badge: the wording the landing fixed was its own report's, fixed
# inside the same entry (CORRECTIONS 278.6) before any of it reached the register.
MECH5_ROWS = {
    236: ("met", 9, ["cwd3"], ["page-19"], None, "CARRIER-DECAY-SUFFICES + CAR-REC+CTL-NULL+CTL2-NULL: on ResNet18_c100 at ciso1's cell, with every arm SCALAR, removing the base optimiser's coupled weight decay from the THREE ctd1 carrier BatchNorm scales ALONE -- layer4.0.bn2.weight, layer4.0.shortcut.1.weight and layer4.1.bn2.weight, 1,536 of 11,220,132 parameters, in the weight update AND in the meta trace -- removes the scalar collapse: CARWD0 70.2640 against k01 22.9853, P_CAR +47.2787 pp = +89.34 SE, and CARWD0 lands INSIDE the network-wide 20-scale arm's recovery band, 4.5413 pp above the 65.7227 bar. It is SPECIFIC against both matched non-carrier sets -- P_SPEC = CARWD0 - CTLWD0 = +47.2627 pp = +89.31 SE -- while cdep1's count-, numel-, width- and depth-matched DEPTH triple {47,48,56} (CTLWD0 23.0013, P_CTL +0.0160 pp) and its class-pure DEPTH2 pair {47,56} (CTL2WD0 22.9333, P_CTL2 -0.0520 pp) both stay on k01's floor. This CLOSES the gap CORRECTIONS 271 and 274 named and that the campaign's own write-up conceded unreservedly, and the anchors replicate between batch (k01 against cwd1's 22.9513 and the corpus census 22.96; NWD 70.7227 against cwd1's k01NWD 70.7760, delta 0.0533 pp, inside MATCH). Every gate passes, the registered scorer exits 0 on both hosts and again after the ingest, an independent parser rebuilds all 259 printed lines byte-identically on two hosts, and G-BITE passes 15/15 with a positive decay term on 6,000 of 6,000 masked probe records and a cross-read that refuses every wrong mask size -- which excludes on records the name-list mask that never bit, the null whose levels would have forged exactly the opposite verdict. Bounded, and led with: the ONLY measured positives are CARWD0's. Both control readings are LOCATIONS at k01's floor, not measured zeros, so what is established about them is that their effects are below the registered 2.0 pp null bar and below 1.0744 pp (CTLWD0) and 1.1104 pp (CTL2WD0) at plus or minus 2 SE -- 2 SE is 1.058364 pp, the HALF-WIDTH of the interval and not itself the bound -- a ratio of 42.6 to 1 at the 2-SE bound or 23.6 to 1 at the null bar, and the phrase is 'does not lift the run off the floor', never 'has no effect'. P_SET = NWD - CARWD0 = +0.4587 pp = +0.87 SE is a bound in the OTHER direction, not distinguishable from zero at the 5.0 pp match bar and below +1.5171 pp at 2 SE, AND ITS SIGN IS HORIZON-DEPENDENT: read in successive five-epoch test windows it runs -0.9153 at epochs 55-59 through -0.1640 at 75-79 to +0.1573 at 80-84 and +0.4587 at 95-99, crossing zero near epoch 80 and still moving at epoch 99, with CARWD0 plateaued at +0.0040 pp per epoch while NWD still climbs at +0.0276, a twenty-epoch slope difference of +0.4726 pp -- the same size as P_SET itself. F_CAR 0.9904 is DESCRIPTIVE and drifts through 1 across the horizon, from 1.0202 at epochs 55-59, so no share may be read off it and 'the carriers carry 99 per cent of the effect' may not be written. Two confounds travel with the sentence and neither is excluded: ResNet18_c100 has exactly FIVE 512-wide BatchNorm scales, indices 47, 50, 53, 56 and 59, and THREE of them are the carriers, so a class-pure, count-matched, carrier-free triple cannot exist at that depth and the batch cannot tell these three tensors from this position class (Kim et al. arXiv:2205.07260, registered at 275.1 before any run and re-derived from the architecture three independent ways here); and, separately, CTLWD0's index 48 is a BatchNorm SHIFT whose weight norm is 1.3e-10 at the first probe record, so coupled decay on it does essentially nothing and CTLWD0's effective intervention is TWO genuine scales rather than three, which leaves a count-or-dose account fitting every number in this batch exactly as well as the carrier account -- a two-carrier arm would separate them and was not run. One switch changes the weight update and the meta trace together, so no route is isolated; the three carriers are masked together, so which ONE of them matters is never asked; one network, one cell, one horizon of 100 epochs, one weight-decay value, no layerwise arm, three seeds; sigma is the frozen prior 0.648113 rather than the in-batch 0.408720, so every SE quoted is conservative; and a RULE 16 defect is reported and NOT fixed -- the scorer's descriptive weight-norm readout takes the upper middle of an even-length list as its median, printing 16 where the true median is 13.6569, which no bar, level, contrast, state, branch word or stamp reads. A landing-entry ordering slip is disclosed in the record rather than papered over: two full-coverage RULE 20 passes had already passed on these fifteen finished files, but the landing's own third pass came after its ingest check had read plateau5, not before."),
}
# Earlier records this row bears on. The import does not rewrite them; the relationship is listed so it stays reviewable.
MECH5_BEARS_ON = {
    236: ["MT233", "MT235"],  # cwd1's network-wide 20-scale mask, decomposed to the three carriers, and cwd2's one-tensor PlainNet result
}
# The 12 intervened rows of this ingest, all under the single DECAY_MASK kind (CORRECTIONS 269). No run of this batch carries
# a second kind, so it adds no multi-kind row and no new kind. The file is byte-identical at the ingest 064dff6, where the
# rows were appended, and at this landing, which did not touch it.
MECH5_INTERVENTIONS_TSV_SHA256 = "50f75cde4cabc23af84318131e98c5aa3397ab848b18cff4cdfc6ed40706aa6b"
MECH5_INTERVENTIONS = {
    "MT236": {
        "batch": "cwd3", "arms": {"CARWD0": 3, "CTLWD0": 3, "CTL2WD0": 3, "NWD": 3},
        "title": "cwd3's four masked arms run with the coupled weight decay switched off on a named set of normalisation tensors",
        "source": f"{INTERVENTIONS_TSV} at {MECH5_COMMIT[:7]}; CORRECTIONS 269, 275 and 278",
        "note": "CARWD0, CTLWD0, CTL2WD0 and NWD ran PATCH_DECAYMASK, which sets the BASE optimiser's coupled weight decay to 0 on a named tensor set -- in the weight update AND in the meta trace h <- gamma(1 - wd*a)h - delta. The batch needed NO new harness code: the patch's registered grammar already admits a name list, so it runs from cwd1's tree and cwd1's runner unchanged, and a proof job on the real GPU path checked all four name-list strings against the live model at every one of 300 real steps before any arm was launched. CARWD0 masks the three ctd1 carriers (indices 50, 53 and 59, 1,536 parameters); CTLWD0 masks cdep1's count-matched DEPTH triple (47, 48 and 56, 1,536 parameters, one of them a BatchNorm SHIFT rather than a scale); CTL2WD0 masks cdep1's class-pure DEPTH2 pair (47 and 56, 1,024 parameters); NWD masks all 20 BatchNorm scales, which is cwd1's own mask re-run in this batch as the recovery reference. No CSV column records a weight-decay mask, so the run inventory writes all four arms with the plain scalar cell key: they are NOT plain measurements of that cell. The 12 rows are listed in results/CORPUS-EXCLUSIONS.tsv under the DECAY_MASK witness kind added at CORRECTIONS 269 -- one kind only, no run of this batch carries a second -- and every witness was generated from the run's own log rather than typed. Drop them before pooling runs by cell. The three k01 runs print DECAY_MASK: off and are ordinary measurements of their cell.",
    },
}
MECH5_INGEST_COMMIT = "064dff6fcb70971e58593929fae84de61319e814"
MECH5_PREVIOUS_COMMIT = "80bffe5ecf1c3064fceec3903ec1590287f82545"
MECH5_PREVIOUS_CORRECTIONS_SHA256 = "07cf4770219ea12852c1da264b835576d08e742b322d0ea1a6443d03b4d126b5"
MECH5_CORRECTIONS_SHA256 = "9206d3ad473bf8e363c710b5c0c5ab851f08d81697adcefa2f4b7822420807b4"
MECH5_ENTRIES = (278,)
MECH5_PREVIOUS_ENTRIES = (274, 275, 276, 277)


def mech5_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cwd3 landing; only header line 3 and the one appended row may differ, and no line may move."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH5_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != MECH5_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {MECH5_COMMIT[:12]} does not match the pinned cwd3-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    before = mech4_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != MECH5_LAST_ROW or len(before) != MECH5_FIRST_ROW - 1 or changed != MECH5_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {MECH5_COMMIT[:7]} moved a line or edited lines other than {sorted(MECH5_EDITED_LINES)}: {sorted(changed)}")
    mech5_corrections(repo)  # the landing amends no row, so the append-only CORRECTIONS check rides here
    return lines


def mech5_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the cwd3 landing and at the write-up revision before it; append-only across both steps."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH5_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != MECH5_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH5_COMMIT[:12]} does not match the pinned bytes")
    previous = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH5_PREVIOUS_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(previous).hexdigest() != MECH5_PREVIOUS_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH5_PREVIOUS_COMMIT[:12]} does not match the pinned bytes")
    lines, before = raw.decode("utf-8").splitlines(), previous.decode("utf-8").splitlines()
    earlier, _ = mech4_corrections(repo)
    for name, later, older, entries in [(MECH5_PREVIOUS_COMMIT, before, earlier, MECH5_PREVIOUS_ENTRIES), (MECH5_COMMIT, lines, before, MECH5_ENTRIES)]:
        if len(later) <= len(older) or not older[-1].startswith(MUST_TRAILER) or not later[-1].startswith(MUST_TRAILER):
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} does not append entries below the previous closing trailer")
        if later[:len(older) - 1] != older[:-1]:
            changed = [n for n in range(1, len(older)) if later[n - 1] != older[n - 1]]
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} changed an earlier line: {changed[:8]}")
        appended = [line for line in later[len(older) - 1:] if line.startswith("## ")]
        if [line.split(".")[0] for line in appended] != [f"## {number}" for number in entries]:
            raise ValueError(f"{CORRECTIONS_DOC} at {name[:7]} does not append exactly entries {entries}: {appended[:4]}")
    return lines, before


def mech5_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE line 236 (cwd3) and attach its intervention note.  The landing amends no earlier row."""
    return with_interventions(appended_rows(lines, MECH5_ROWS, MECH5_FIRST_ROW, MECH5_LAST_ROW, MECH5_COMMIT), MECH5_INTERVENTIONS)


# ---------------------------------------------------------------------------
# 16. The cwd4 and cwd5 landings (CORRECTIONS 283 and 285): two appended rows, in two steps.
# ---------------------------------------------------------------------------
# Campaign commit 26baf4a (cycle 158, CORRECTIONS 283) appended the cwd4 landing as MASTER-TABLE line 237 and recounted
# header line 3; campaign commit 8554afa (cycle 159, CORRECTIONS 285) appended the cwd5 landing as line 238 and recounted
# header line 3 again. Neither step amended any earlier row, neither touched line 5, and no line moved. The two rows import
# together because the cwd4 landing said so in its own commit message ("THE SITE IS NOT TOUCHED -- rows 237 and 238 import
# together"), not because they answer one question: they get a phase each.
#
# BOTH ROWS ARE MIXED, and neither is Goal met by default. The MT218-MT236 precedent grants Goal met only when the fired
# account returns its registered branch AND its registered state word in the registered direction, NO registered account
# misses a band, and NO control fails (MT230's note). Each row fails a different half of that test:
#   * MT237 (cwd4) answers its question and refutes the count / dose rival, but the fired account's OWN registered level
#     band for ONE53 is K01 (18-28) and ONE53 landed at 58.5780. A registered expectation is defied, exactly as at MT231
#     (cmg1), where ACCOUNT-A3-MERGE-HARMLESS was ruled "a BRANCH match only" because no account's levels were reproduced.
#     The verdict's second token additionally turns on a near bar (ONE50's REC by +0.40 SE), which the row itself labels
#     DESCRIPTIVE / UNSURE -- MT231's other reason for Mixed.
#   * MT238 (cwd5) hits every registered band on the ladder (all four scalar rungs and all four layerwise references) and
#     no control fails, but the batch's SECOND registered question -- whether the carrier account still holds wherever the
#     collapse still exists (281.3) -- is unanswerable at this cell: CARW2 was pre-registered to be read ONLY if rung W2
#     collapses, W2 is NOGAP, so CAR-UNREADABLE fires and nothing is read from 71.8327. That is MT228's (cmo1's)
#     ISO-UNREADABLE at wd 0 repeating one rung higher, and the registered scorer says so in those words.
# Neither row corrects an earlier PUBLISHED notebook claim, so neither carries a Corrected badge: every wording the two
# refute passes fixed was the landing reports' own, fixed inside the same entries before any of it reached the register.
MECH6_STEP_COMMIT = "26baf4a154bebd1332f2454cd67d700ec4af6e1a"
MECH6_STEP_MASTER_TABLE_SHA256 = "a34406c0b0d4f414792186ab15d70bd49c8b8f22a05aeea69351aac4b3fcc674"
MECH6_COMMIT = "8554afae4008a92f612d67f194db4ec186630fb9"
MECH6_MASTER_TABLE_SHA256 = "eac919a3c4ef45c60c49c673eb8f6f3851fe26d3e33bf3e2b2ed141879e30f86"
MECH6_FIRST_ROW, MECH6_STEP_ROW, MECH6_LAST_ROW = 237, 237, 238
MECH6_EDITED_LINES = {3}  # each step recounted the run / GPU-hour header and nothing else; line 5 was left alone twice.
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
MECH6_ROWS = {
    237: ("mixed", 9, ["cwd4"], ["page-19"], None, "ONE-SUFFICES-PARTIAL + TWO-REC+CTL2-NULL+ONE50-REC+ONE53-PART+ONE59-REC: the question is answered and 278's COUNT / DOSE rival is REFUTED at this cell -- at MATCHED count two P_2SPEC = TWOWD0 - CTL2WD0 = 67.0713 - 22.8547 = +44.2167 pp = +83.94 SE, the class-pure, count-, width-, depth- and numel-matched carrier-free pair {47, 56} stays on k01's floor while the carrier pair {50, 53} recovers, and one carrier alone reaches REC (ONE59 67.9567, P_ONE59 +45.1353 pp = +85.69 SE), so the effect is not a function of the number of masked scales alone and WRITEUP-mechanism's O-12 closes; at probe record 0 the removed decay dm_wdterm is a pure function of k and bit-identical across seeds, so TWOWD0 and CTL2WD0 remove NUMERICALLY IDENTICAL weight decay at initialisation and still differ by +44.2167 pp. Every gate passes, the registered scorer exits 0 UNEDITED on both hosts and again on the post-ingest corpus, an independent parser rebuilds all 359 printed lines with 0 mismatches and 0 violations, G-BITE passes 21/21 with a positive dm_wdterm on 9,000 of 9,000 masked records and a cross-read that refuses every wrong-k read, and the anchors replicate cwd3 in batch. BUT A REGISTERED EXPECTATION IS DEFIED: the fired account's own registered LEVEL BAND for ONE53 is K01 (18-28), and ONE53 landed at 58.5780 -- +35.7567 pp = +67.88 SE above k01, outside K01, outside PART (32-55) and outside FREE (62-78) -- so ONE-SUFFICES-PARTIAL is a BRANCH and STATE-WORD match whose level prediction for the non-sufficing carrier is missed by about 30 pp, as MT231's ACCOUNT-A3-MERGE-HARMLESS was a branch match only. The verdict's second token also turns on a NEAR BAR: ONE50's REC clears the in-batch 65.0100 bar by +0.2133 pp = +0.40 SE, far inside the 1.053511 pp half-width, so the ONE50-REC token and any 'two of the three singles suffice' phrasing are DESCRIPTIVE / UNSURE by the row's own words, while ONE59's REC, ONE53's PART and TWOWD0's REC are resolved. Bounded, and led with: CTL2WD0's NULL is a LOCATION at k01's floor and a BOUND, never a measured zero -- P_CTL2 +0.0333 pp = +0.06 SE with a +/-2 SE interval [-1.0202, +1.0868] that SPANS ZERO and per-seed signs that flip -- so the licensed form is 'leaves the run at k01's floor' and never 'does nothing'. The MAGNITUDE rival is NOT refuted and this batch makes it MORE attractive, not less: the three singles' levels are in exactly the mean-|L| rank order with both gaps resolved, re-derived IN BATCH on cwd4's own k01 pinned records at 140.1x between the smallest carrier and the largest admissible control, so MAGNITUDE-NOT-SEPARATED and CTL-NOT-MAGNITUDE-MATCHED stand unconditionally and no sentence may read the result as tensor identity. Position class now bites at count ONE -- the two REC singles {50, 59} are both Kim gamma_last and the PART single {53} is gamma_down -- and the dose ladder is NOT cleanly monotone at the point estimates, ONE59 exceeding the carrier pair by +0.8853 pp = +1.68 SE, not resolved. One switch changes the weight update and the meta trace together, so no route is isolated; one network, one cell, ONE weight-decay value (coupled 0.1), every arm scalar with no layerwise arm, one 100-epoch horizon, three seeds, and sigma is the frozen prior 0.645141, so every SE is conservative. A refute pass could not refute the verdict and produced six corrections and two hardenings, three of them floor-as-magnitude wording of the class 164.6 forbids; one RULE 16 defect is REPORTED and NOT fixed, a literal '100 %%' printed by two bare print() calls in the scorer's not-licensed block, read by no bar, level, contrast, branch or stamp."),
    238: ("mixed", 9, ["cwd5"], ["page-19"], None, "THRESHOLD-W1-W2 + W1-COLLAPSE+W2-NOGAP+W3-NOGAP+W4-NOGAP+CAR-UNREADABLE: the ladder's own question is answered in a registered direction, adversely, and the answer was written out in advance as a reachable one -- the shared-step-size collapse is PRESENT at the campaign's coupled weight decay 0.1 and ABSENT at 1e-2, 1e-3 and 5e-4. CO-PRIMARY G_W4 = kLW4 - k01W4 = 67.8813 - 72.4080 = -4.5267 pp = -8.59 SE, +/-2 SE [-5.5802, -3.4732], negative on all three seeds: at the STANDARD CIFAR weight decay the layerwise arm is BELOW the scalar arm. G_W1 = +46.0700 pp = +87.46 SE, G_W2 = -1.0893 pp = -2.07 SE, G_W3 = -4.1607 pp = -7.90 SE. No state is near-bar (W1 collapses by 11.4230 pp; W2 / W3 / W4 sit 35.2203 / 38.3293 / 38.4673 pp above their own rungs' collapse bars), the account replicates on every individual seed, the anchor replicates cwd3's k01 in batch at +0.2387 pp so the threshold is not a failed-anchor artefact, both grains ran AT EVERY RUNG so no gap is cross-batch, and every registered LEVEL BAND is hit -- all four scalar rungs and all four layerwise references land where THRESHOLD-W1-W2 said they would. No control fails: every rung's layerwise reference is healthy, the mask bit on 1,500 of 1,500 masked records, and the CAR-NULL null is excluded by evidence. BUT THE BATCH'S SECOND REGISTERED QUESTION IS UNANSWERABLE AT THIS CELL: the carrier companion CARW2, registered to say whether the carrier account still holds wherever the collapse still exists, was pre-registered to be read ONLY if rung W2 collapses; W2 is NOGAP, so CAR-UNREADABLE fires, P_CARW2 and D_CARW2 were never computed, and 71.8327 is a LEVEL from which nothing is read -- not that the carrier account holds at 1e-2, not that it fails. That is MT228's ISO-UNREADABLE at weight decay 0 repeating one rung higher, named in those words by the registered scorer, and the risk was registered rather than hidden. Because the mask DID bite, it is a design-scope limit and not a patch failure. Bounded, and led with: FOUR rungs can only BRACKET a transition and never locate one, so the change lies somewhere in the UNRUN interval between 1e-2 and 0.1 and nothing inside that decade may be named -- neither a threshold value nor 'the collapse exists only at 0.1' (LADDER-IS-FOUR-POINTS). G_W1 is a floor LOCATION, so its SIZE is where the arm landed and not a measured effect; each NOGAP is a bound in the other direction, 'below the 10 pp bar', never 'the two grains are equal'. G_W2 is NOT resolved, its +/-2 SE interval [-2.1428, -0.0358] ending 0.04 pp from zero, and may not be pooled with W3 and W4. What is identified is the CONJUNCTION of scalar grouping with a coupled decay at or near 0.1, not grouping alone and not decay alone; DECOUPLED decay is untested in either direction and descoped with reasons, and which route the decay acts through is not separated. One network, one cell otherwise, one 100-epoch horizon, three seeds, sigma the frozen prior 0.645141. The consequence the registration wrote in advance: this is the area chair's CORNER-CASE charge LANDING, WRITEUP-mechanism's O-14 closes against the generality of the result, and the mechanism line must be written as a diagnostic of ONE extreme configuration -- a practitioner at 5e-4 will never meet this failure -- while wd 0.1 remains the parent paper's own value, so nothing is retracted and only the scope moves from assumed to measured. A refute pass could not refute the verdict; it refuted one supporting claim, the scorer stage's provenance, which touches no scored quantity, and applied eight further wording fixes. One RULE 16 defect is REPORTED and NOT fixed: the scorer's DESCRIPTIVE gap-monotonicity stamp is suppressed by an unintended substring match on CAR-UNREADABLE, so the monotonicity is re-derived from the four G values and no stamp is quoted for it."),
}
# Earlier records these rows bear on. The import does not rewrite them; the relationship is listed so it stays reviewable.
MECH6_BEARS_ON = {
    237: ["MT236", "MT233"],           # cwd3's three-carrier mask decomposed to a matched pair and three singles; cwd1's network-wide mask
    238: ["MT236", "MT233", "MT228"],  # the wd-0.1 mechanism results whose SCOPE the ladder measures, and cmo1's wd-0 endpoint
}
# The 18 intervened rows of the cwd4 ingest, all under the single DECAY_MASK kind (CORRECTIONS 269): 6 masked arms x 3
# seeds. The 3 k01 runs print DECAY_MASK: off and own no row.
MECH6_CWD4_INGEST_COMMIT = "8b9fbd2d83304b419a4193d5cb4283478a0e3d5a"
MECH6_CWD4_INTERVENTIONS_TSV_SHA256 = "29327a9622170513214d2fcadd80b413575a047f6b440c483b1019eaa5a1ce55"
MECH6_INTERVENTIONS = {
    "MT237": {
        "batch": "cwd4", "arms": {"CARWD0": 3, "TWOWD0": 3, "CTL2WD0": 3, "ONE50": 3, "ONE53": 3, "ONE59": 3},
        "title": "cwd4's six masked arms run with the coupled weight decay switched off on a named set of BatchNorm scales",
        "source": f"{INTERVENTIONS_TSV} at {MECH6_CWD4_INGEST_COMMIT[:7]}; CORRECTIONS 269, 280 and 283",
        "note": "CARWD0, TWOWD0, CTL2WD0, ONE50, ONE53 and ONE59 ran PATCH_DECAYMASK, which sets the BASE optimiser's coupled weight decay to 0 on a named tensor set -- in the weight update AND in the meta trace. The batch needed NO new harness code: it runs from cwd1's tree and cwd1's runner unchanged, with the patch's registered name-list grammar used at new places, and a proof job on the real GPU path checked every name-list string against the live model before any arm was launched. CARWD0 masks the three ctd1 carriers (indices 50, 53 and 59, 1,536 parameters) and is the batch's IN-BATCH recovery reference, so no cwd3 level enters any bar; TWOWD0 masks the carrier pair {50, 53} and CTL2WD0 the class-pure, count-, width-, depth- and numel-matched carrier-FREE pair {47, 56}, both 1,024 parameters, which is the matched-count-two contrast cwd3 could not build; ONE50, ONE53 and ONE59 mask one carrier each, 512 parameters, turning the contrast into a dose ladder 1 / 2 / 3 on the carrier side. No CSV column records a weight-decay mask, so the run inventory writes all six arms with the plain scalar cell key: they are NOT plain measurements of that cell. The 18 rows are listed in results/CORPUS-EXCLUSIONS.tsv under the DECAY_MASK witness kind added at CORRECTIONS 269 -- one kind only, no run of this batch carries a second -- and every witness was generated from the run's own log rather than typed, then cross-checked against the run tree's own PARTITION-MANIFEST.txt with 0 mismatches. Drop them before pooling runs by cell. The three k01 runs print DECAY_MASK: off and are ordinary measurements of their cell.",
    },
}
# The 21 ARGS-deviating rows of the cwd5 ingest: 7 non-anchor arms x 3 seeds under the ARGS_WD_BASE kind (CORRECTIONS 263).
# CARW2's three rows are the corpus's first TWO-AXIS rows (CORRECTIONS 284): a run that BOTH deviates on an ARGS value and
# prints an ON <KIND> line. They are listed with the ARGS witness, and their DECAY_MASK line is held in the intervention
# cell's second clause and read back from the run's own log.
MECH6_INGEST_COMMIT = "91fcd57b700962e7b5aaa75508f102dad430bd89"
MECH6_INTERVENTIONS_TSV_SHA256 = "4e6d938745fade2eb02dbbda79bc1bc096d5f077c9092b5dd4ef2b10cfbfc7db"
MECH6_ARGS_DEVIATIONS = {
    "MT238": {
        "batch": "cwd5", "arms": {"k01W2": 3, "kLW2": 3, "k01W3": 3, "kLW3": 3, "k01W4": 3, "kLW4": 3, "CARW2": 3},
        "title": "cwd5's three lower rungs differ from the standard cell in the base weight-decay flag, and its carrier companion deviates on TWO axes at once",
        "source": f"{INTERVENTIONS_TSV} at {MECH6_INGEST_COMMIT[:7]}; CORRECTIONS 263, 281, 284 and 285",
        "note": "cwd5 varies ONE axis, --weight-decay-base, across four rungs: 0.1 (the campaign's own value, the anchor), 1e-2, 1e-3 and 5e-4, with BOTH grains run in batch at every rung. results/all_runs.csv has no column for a base-optimiser CLI flag, so the 18 runs of the three lower rungs carry the plain 0.1 cell key of their own grain: seed for seed they differ from their anchor arm only in run, job_id, node, wallclock_min and the accuracy columns. They are NOT measurements of the standard cell. There is no PATCH ON line to read for them, because no patch ran: the witness is the run's OWN ARGS: line, the one prefix every run prints, read with argparse last-wins semantics, and each row is listed under the ARGS-value witness kind ARGS_WD_BASE added at CORRECTIONS 263. The three CARW2 runs are different and are the corpus's FIRST TWO-AXIS rows: they deviate on --weight-decay-base 1e-2 AND run PATCH_DECAYMASK on the three ctd1 carriers, so a single witness column cannot describe them. CORRECTIONS 284 registered the shape they use -- listed with the ARGS witness, the DECAY_MASK: on line held beside it and checked against the run's own log -- and the campaign's checker verifies both axes. The six anchor runs at the standard 0.1 deviate on nothing and own no row. Drop every listed row before pooling runs by cell.",
    },
}
# docs/CORRECTIONS.md is append-only across the whole cycle: only the closing "Next free number" trailer is ever replaced.
# 279-282 were appended between the cwd3 landing and the cwd4 registration (7ff4685), 283 by the cwd4 landing itself
# (26baf4a), 284 by the two-axis code gap (5045875), 285 by the cwd5 landing (8554afa) and 286 by the post-landing
# verification (72c4902). The two ingest commits 8b9fbd2 and 91fcd57 touched results/ alone and no entry at all.
MECH6_PREVIOUS_COMMIT = "7ff46858ad4dfd5bcbac604706c4108b9426f085"
MECH6_PREVIOUS_CORRECTIONS_SHA256 = "c8226da5aef49f88d4af1214870d62ce77489fb7e99f27aba0bee0afe44b79a6"
MECH6_STEP_CORRECTIONS_SHA256 = "c0b243624a18eb41ec44e24d1c30fb366ed8c540e479db94096adff175032380"
MECH6_GAP_COMMIT = "504587580f2343e7003b012ae9fff22afb94d584"
MECH6_GAP_CORRECTIONS_SHA256 = "06dea902b5bcf1f3356a7e6c2c89f18a952a619189fac97def6e9dd81ea79c01"
MECH6_CORRECTIONS_SHA256 = "c5cb62e4036f2b9b9dc22b429a226ff5a46509a906dc4f27762d88011e7eef12"
MECH6_PREVIOUS_ENTRIES = (279, 280, 281, 282)
MECH6_STEP_ENTRIES = (283,)
MECH6_GAP_ENTRIES = (284,)
MECH6_ENTRIES = (285,)


def mech6_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the cwd4 and cwd5 landings; each step may edit header line 3 and append ONE row, and nothing else."""
    before = mech5_master_table(repo)
    for commit, expected, last in [(MECH6_STEP_COMMIT, MECH6_STEP_MASTER_TABLE_SHA256, MECH6_STEP_ROW),
                                   (MECH6_COMMIT, MECH6_MASTER_TABLE_SHA256, MECH6_LAST_ROW)]:
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{MASTER_TABLE}"])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{MASTER_TABLE} at {commit[:12]} does not match the pinned cwd4/cwd5-landing bytes")
        lines = raw.decode("utf-8").splitlines()
        changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
        if len(lines) != last or len(before) != last - 1 or changed != MECH6_EDITED_LINES:
            raise ValueError(f"MASTER-TABLE at {commit[:7]} moved a line or edited lines other than {sorted(MECH6_EDITED_LINES)}: {sorted(changed)}")
        before = lines
    mech6_corrections(repo)  # neither landing amends a row, so the append-only CORRECTIONS check rides here
    return before


def mech6_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at each step of the cwd4 / cwd5 cycle; append-only from the cwd3 landing through all four."""
    steps = [(MECH6_PREVIOUS_COMMIT, MECH6_PREVIOUS_CORRECTIONS_SHA256, MECH6_PREVIOUS_ENTRIES),
             (MECH6_STEP_COMMIT, MECH6_STEP_CORRECTIONS_SHA256, MECH6_STEP_ENTRIES),
             (MECH6_GAP_COMMIT, MECH6_GAP_CORRECTIONS_SHA256, MECH6_GAP_ENTRIES),
             (MECH6_COMMIT, MECH6_CORRECTIONS_SHA256, MECH6_ENTRIES)]
    older = mech5_corrections(repo)[0]
    previous = None
    for commit, expected, entries in steps:
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{CORRECTIONS_DOC}"])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{CORRECTIONS_DOC} at {commit[:12]} does not match the pinned bytes")
        later = raw.decode("utf-8").splitlines()
        if len(later) <= len(older) or not older[-1].startswith(MUST_TRAILER) or not later[-1].startswith(MUST_TRAILER):
            raise ValueError(f"{CORRECTIONS_DOC} at {commit[:7]} does not append entries below the previous closing trailer")
        if later[:len(older) - 1] != older[:-1]:
            changed = [n for n in range(1, len(older)) if later[n - 1] != older[n - 1]]
            raise ValueError(f"{CORRECTIONS_DOC} at {commit[:7]} changed an earlier line: {changed[:8]}")
        appended = [line for line in later[len(older) - 1:] if line.startswith("## ")]
        if [line.split(".")[0] for line in appended] != [f"## {number}" for number in entries]:
            raise ValueError(f"{CORRECTIONS_DOC} at {commit[:7]} does not append exactly entries {entries}: {appended[:4]}")
        previous, older = older, later
    return older, previous


def mech6_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 237-238 (cwd4, cwd5) and attach their exclusion notes.  Neither landing amends a row."""
    rows = with_interventions(appended_rows(lines, MECH6_ROWS, MECH6_FIRST_ROW, MECH6_LAST_ROW, MECH6_COMMIT), MECH6_INTERVENTIONS)
    for row in rows:
        spec = MECH6_ARGS_DEVIATIONS.get(row["id"])
        if spec:
            row["args_deviations"] = json.dumps({"title": spec["title"], "note": spec["note"], "batch": spec["batch"],
                                                 "arms": spec["arms"], "source": spec["source"]}, ensure_ascii=False)
    return rows


# ---------------------------------------------------------------------------
# 17. The caw2 and cgw1 landings (CORRECTIONS 295 and 296): two appended rows, in ONE step.
# ---------------------------------------------------------------------------
# Campaign commit 8a99001 (cycle 160, CORRECTIONS 295-297) appended the caw2 landing as MASTER-TABLE line 239 and the cgw1
# landing as line 240 and recounted header line 3. It amended no earlier row, did not touch line 5, and no line moved. The
# joint ingest 477a853 touched results/ alone. CORRECTIONS 297 (the path decision) is a planning entry and owns no row.
#
# MT239 (caw2) is MIXED, as MT238. The co-primary is answered in its registered direction -- the harness's standard AdamW +
# Adam does not collapse at wd 0.1 (G_A -5.1053 pp, NOGAP) -- the registered prediction (A-NOGAP with ATTR-BASE-PROTECTS) is
# returned, every registered level band of the fired account is hit and the control reproduces. But the batch's dose half,
# the clause that would make the no-collapse a statement about the recipe rather than about its realised dose, returned
# CONTROL-DOSE-NOT-REACHED: the dose arm reached RHO_X 0.0591 against a 0.5 bar, so immunity at the control's dose is
# UNDECIDED, and the attribution word is read beside a dose word that confounds it. That is MT238's shape: the first question
# answered, the second unanswerable at this cell.
# MT240 (cgw1) is OPEN, as MT183 (bn1): the PRIMARY returned AUDIT-UNDECIDED (D4 +0.2410 pp between the +0.15 and +0.30
# bars), whose registered licence is "report the interval; no survive / vanish sentence". The co-reported scalar reading
# (SCALAR-BEATS-BEST) IS resolved and is carried in the reason, and every level the registration predicted at 5e-4 was missed
# (the partitions landed 2.7-2.9 pp above HEALTH_MIN where >= 5.3 was predicted), which is scope, not a verdict.
# Neither row corrects an earlier PUBLISHED notebook claim, so neither carries a Corrected badge.
MECH7_COMMIT = "8a99001db99bf8de13a598b3295fe873f2f35778"
MECH7_MASTER_TABLE_SHA256 = "6cf741514987831fc9d62c532b1c67653776ea7c459093f5e0f9f07d29614fbd"
MECH7_FIRST_ROW, MECH7_LAST_ROW = 239, 240
MECH7_EDITED_LINES = {3}  # the run / GPU-hour header and the tally, recounted in place; no row amended, line 5 untouched.
# line -> (rule, section, batches, figure pages, corrected note or None, one-line reason)
MECH7_ROWS = {
    239: ("mixed", 9, ["caw2"], ["page-19"], None, "NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED: the co-primary is answered in its registered direction -- with the harness's standard AdamW + Adam at wd 0.1 on ResNet18_c100 the scalar arm lands at 72.9493 against its in-batch layerwise 67.8440, G_A = -5.1053 pp = -9.75 SE, far from its 33.922 collapse bar (NOGAP), and at the dose arm's wd 1.0 at 73.2507 against 68.6840 (NOGAP). The registered prediction (A-NOGAP with ATTR-BASE-PROTECTS) is returned: swapping only the meta optimiser still collapses (MS 13.2440 against its own layerwise 68.2060, +54.9620 pp = +104.92 SE, 3/3 seeds, at a realised peak shrink 3.4078x the control's) and swapping only the base does not (LS 72.5673 against 68.1093). Every registered level band of the fired account is hit, the SGDm + Lion control reproduces its collapse in batch (K01 22.5240) with DOM_C 0.7360, every gate passes, the registered scorer exits 0 UNEDITED on both hosts and again post-ingest, and an independent parser rebuilds all 267 printed lines on each host. BUT THE BATCH'S DOSE HALF IS UNDECIDED: the dose arm's realised peak shrink reached only RHO_X = 0.0591 of the control's (bar 0.5; the worst seed pairing 0.0605), so the branch is NOT-REACHED and not IMMUNE -- whether the standard recipe would survive the control's dose is not decided -- and the attribution word is read beside DOSE-L-BELOW: the base swap removed the collapse AND about 98 % of the realised shrink AND changed base momentum 0.99 -> 0.9, so 'the base protects' is not separated from 'the base never reaches the dangerous dose'. That is MT238's shape -- the first question answered, the second unanswerable at this cell -- so the record is Mixed, not Goal met. Bounded, and led with: every NOGAP is a bound, 'below the 10 pp bar', never 'the grains are equal'; K01's and MS's late shrinks are the BETA_CLIP floor; 'the standard recipe' is the HARNESS's AdamW + Adam, whose unnormalised first moment makes its step up to ten times torch's at the same step size, with alpha-scaled decay; wd 1.0 is a dose arm, not a standard value; one cell, three seeds, 100 epochs, and wd 1e-2, a Lion base and alpha-independent decay were not run. Consequence recorded by the campaign (CORRECTIONS 295.7, 297): at this cell the collapse is a corner case of the SGDm base and not a hazard of the standard recipe, and the ICML plan's Step-1 gate does not fire. A refute pass could not refute the verdict and applied eight wording fixes; no RULE 16 defect is new."),
    240: ("open", 10, ["cgw1"], ["page-8"], None, "AUDIT-UNDECIDED: the primary is undecided -- at the standard decay VALUE 5e-4 on the audit's core cell (ResNet18 / CIFAR-10, SGDm 0.99 + Lion, ms 1e-4, alpha0 1e-3) the uniform-minus-aligned difference D4 = chunk777 - nodewise = 87.9060 - 87.6650 = +0.2410 pp = +1.90 SE, +/-2 SE [-0.0120, +0.4940], between the +0.15 VANISHES and +0.30 SURVIVES bars, and the registered licence is 'report the interval; no survive / vanish sentence'. The in-batch anchor reproduces the audit at 0.1 (D1 +0.3693 pp = +2.53 SE, SURVIVES, within 0.19 of the +0.5556 pool) and the rung 1e-2 SURVIVES too (+0.4533 pp = +3.10 SE), so the count-matched sign is never reversed. The co-reported scalar reading IS resolved: at 5e-4 plain scalar (90.4650) is ABOVE both audited partitions, by +2.5590 pp = +20.23 SE and +2.8000 pp = +22.13 SE (SCALAR-BEATS-BEST), and layerwise (89.8110) is below scalar there (-5.17 SE, descriptive), so the registered TMLR consequence binds: the audit's practical significance must be qualified whatever D4 says. The prior missed every W4 level (all arms predicted at 90.3 or above; the partitions landed at 87.7-87.9), which is scope and not a verdict. Every gate passes, the registered scorer exits 0 UNEDITED on both hosts and again post-ingest, and an independent parser passes 15 of 15 checks, byte-identical on both hosts. Bounded, and led with: ONE cell (nothing about CIFAR-100); three rungs; every rung alpha-scaled decay; and W4 is far less decayed than a standard SGD recipe -- its realised plateau shrink is 1/11 to 1/17 of lr*wd and about 1/115 to 1/883 of a standard recipe's momentum-amplified shrink -- so no reading settles whether the audit effect is an artefact of non-standard decay, and 'at standard decay' in the licence means the VALUE 5e-4 applied alpha-scaled. Leave-one-seed-out on D4 runs +0.1793 to +0.3000 (descriptive), and no seed may be added after reading. A refute pass could not refute the verdict; it fixed three prose slips (among them the landing's weakening of the verifier's caveat, restored) and four qualifications, and one LATENT RULE 16 defect is REPORTED and NOT fixed (a later duplicate epoch line would overwrite an earlier one; none exists in this batch)."),
}
# Earlier records these rows bear on. The import does not rewrite them; the relationship is listed so it stays reviewable.
MECH7_BEARS_ON = {
    239: ["MT238", "MT228", "MT236"],  # the wd ladder whose scope the base axis extends; cmo1's momentum leg; the carrier result at wd 0.1
    240: ["MT175", "MT181", "MT238"],  # the audit's headline and its core cell's first measured D; the cwd5 grain reversal
}
MECH7_INTERVENTIONS: dict[str, dict] = {}  # neither batch ran a patch; every exclusion row of the ingest is an ARGS-value row
# The 46 ARGS-deviating rows of the joint ingest 477a853 (CORRECTIONS 296.9): caw2 18 (12 ARGS_MOMENTUM_BASE, 6 TWO-ARGS listed
# by ARGS_WD_BASE) and cgw1 28 (ARGS_WD_BASE at 1e-2 and 5e-4). The landing commit 8a99001 left the list byte-identical.
MECH7_INGEST_COMMIT = "477a85328a32f5145bc4f5f786914d6b1939df62"
MECH7_INTERVENTIONS_TSV_SHA256 = "7b034d3efeba5bbe4f9e356a129e0812c863bc520745f56203d3505d199da642"
# CORRECTIONS 294's registry, re-typed (not imported): the (batch, arm)s that may deviate on two ARGS kinds, with each value.
MECH7_MULTI_ARGS = {
    ("caw2", "XS"): {"ARGS_MOMENTUM_BASE": "0.9", "ARGS_WD_BASE": "1.0"},
    ("caw2", "XL"): {"ARGS_MOMENTUM_BASE": "0.9", "ARGS_WD_BASE": "1.0"},
}
MECH7_ARGS_DEVIATIONS = {
    "MT239": {
        "batch": "caw2", "arms": {"LS": 3, "LL": 3, "AS": 3, "AL": 3, "XS": 3, "XL": 3},
        "title": "caw2's AdamW-base arms differ from the standard cell in the base momentum flag, and its dose arms deviate on TWO ARGS kinds at once",
        "source": f"{INTERVENTIONS_TSV} at {MECH7_INGEST_COMMIT[:7]}; CORRECTIONS 263, 290, 294 and 296",
        "note": "caw2 runs the mechanism cell with the base optimiser, the meta optimiser and the weight decay each changed in its own cell. results/all_runs.csv carries the base and meta optimiser as columns but has no column for a base-optimiser CLI flag, so the twelve AdamW-base runs of cells L and A carry --momentum-param-base 0.9 -- AdamW's own beta1, flagged only because the registered ARGS standard (0.99) is set per flag at the SGDm mechanism cell -- under a cell key that does not show it, and are listed under the ARGS-value witness kind ARGS_MOMENTUM_BASE added at CORRECTIONS 263. The six dose-arm runs XS and XL deviate on TWO ARGS kinds at once: that momentum AND --weight-decay-base 1.0. They are the corpus's FIRST TWO-ARGS rows (CORRECTIONS 294): each is listed ONCE, by its weight-decay witness, with both flags written in the intervention cell, and the momentum kind is held to a registry of the two arms that may do this and read back from the run's own ARGS line. The control K01 and the meta-swap arms MS and ML run the standard 0.99 / 0.1 and own no row. Drop every listed row before pooling runs by cell.",
    },
    "MT240": {
        "batch": "cgw1", "arms": {"chW2": 3, "ndW2": 3, "k01W2": 3, "kLW2": 3, "chW4": 4, "ndW4": 4, "k01W4": 4, "kLW4": 4},
        "title": "cgw1's two lower rungs differ from the audit cell in the base weight-decay flag",
        "source": f"{INTERVENTIONS_TSV} at {MECH7_INGEST_COMMIT[:7]}; CORRECTIONS 263, 291 and 296",
        "note": "cgw1 varies ONE axis, --weight-decay-base, across three rungs of the audit's core cell: 0.1 (the audit's own value, the in-batch anchor), 1e-2 and the standard value 5e-4, with the uniform and aligned partitions and plain scalar and layerwise run at every rung. results/all_runs.csv has no column for a base-optimiser CLI flag, so the 28 runs of the two lower rungs carry the plain 0.1 cell key of their own grain and are NOT measurements of the audit cell. No patch ran: the witness is the run's OWN ARGS: line, read with argparse last-wins semantics, and each row is listed under the ARGS-value witness kind ARGS_WD_BASE added at CORRECTIONS 263. The twelve anchor runs at the standard 0.1 deviate on nothing and own no row. Drop every listed row before pooling runs by cell.",
    },
}
# docs/CORRECTIONS.md at the landing is the cwd5 landing's file with every line above its closing trailer unchanged and entries
# 286-297 below it, closed by a fresh trailer. (289-291 were written by concurrent tracks and inserted before 292 while they
# ran; the pinned landing file shows them in numeric order, which is the only state this import reads.)
MECH7_CORRECTIONS_SHA256 = "c198a0f314daaf42ac7e856eb38b5bc3ed0f63c4167a81bdb66a7195a11b221b"
MECH7_ENTRIES = tuple(range(286, 298))


def mech7_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the caw2 / cgw1 landing; ONE step that edits header line 3 and appends TWO rows, and nothing else."""
    before = mech6_master_table(repo)
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH7_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != MECH7_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {MECH7_COMMIT[:12]} does not match the pinned caw2/cgw1-landing bytes")
    lines = raw.decode("utf-8").splitlines()
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != MECH7_LAST_ROW or len(before) != MECH7_FIRST_ROW - 1 or changed != MECH7_EDITED_LINES:
        raise ValueError(f"MASTER-TABLE at {MECH7_COMMIT[:7]} moved a line or edited lines other than {sorted(MECH7_EDITED_LINES)}: {sorted(changed)}")
    mech7_corrections(repo)  # neither landing amends a row, so the append-only CORRECTIONS check rides here
    return lines


def mech7_corrections(repo: Path) -> tuple[list[str], list[str]]:
    """docs/CORRECTIONS.md at the caw2 / cgw1 landing; append-only below the cwd5 landing's closing trailer."""
    older = mech6_corrections(repo)[0]
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{MECH7_COMMIT}:{CORRECTIONS_DOC}"])
    if hashlib.sha256(raw).hexdigest() != MECH7_CORRECTIONS_SHA256:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH7_COMMIT[:12]} does not match the pinned bytes")
    later = raw.decode("utf-8").splitlines()
    if len(later) <= len(older) or not older[-1].startswith(MUST_TRAILER) or not later[-1].startswith(MUST_TRAILER):
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH7_COMMIT[:7]} does not append entries below the previous closing trailer")
    if later[:len(older) - 1] != older[:-1]:
        changed = [n for n in range(1, len(older)) if later[n - 1] != older[n - 1]]
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH7_COMMIT[:7]} changed an earlier line: {changed[:8]}")
    appended = [line for line in later[len(older) - 1:] if line.startswith("## ")]
    if [line.split(".")[0] for line in appended] != [f"## {number}" for number in MECH7_ENTRIES]:
        raise ValueError(f"{CORRECTIONS_DOC} at {MECH7_COMMIT[:7]} does not append exactly entries {MECH7_ENTRIES}: {appended[:4]}")
    return later, older


def mech7_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 239-240 (caw2, cgw1) and attach their ARGS-value exclusion notes.  Neither amends a row."""
    rows = with_interventions(appended_rows(lines, MECH7_ROWS, MECH7_FIRST_ROW, MECH7_LAST_ROW, MECH7_COMMIT), MECH7_INTERVENTIONS)
    for row in rows:
        spec = MECH7_ARGS_DEVIATIONS.get(row["id"])
        if spec:
            row["args_deviations"] = json.dumps({"title": spec["title"], "note": spec["note"], "batch": spec["batch"],
                                                 "arms": spec["arms"], "source": spec["source"]}, ensure_ascii=False)
    return rows


def apply_mech4_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply CORRECTIONS 273's in-place amendment of row 229 (cst1, MT229) to its record, moving its outcome."""
    lines, before = mech4_master_table(repo), must_master_table(repo)
    mech4_corrections(repo)
    if {spec["line"] for spec in MECH4_AMENDMENTS.values()} != MECH4_EDITED_LINES - {3}:
        raise ValueError("Every row amended at CORRECTIONS 273 needs exactly one record amendment")
    missing = set(MECH4_AMENDMENTS) - {row["id"] for row in rows}
    if missing:
        raise ValueError(f"CORRECTIONS 273 amendments name absent records: {sorted(missing)}")
    amended = []
    for row in rows:
        spec = MECH4_AMENDMENTS.get(row["id"])
        if not spec:
            amended.append(row)
            continue
        line, number = spec["line"], spec["number"]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if row["master_table_line"] != str(line):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != spec["previous"]:
            raise ValueError(f"{row['id']} outcome drifted before its CORRECTIONS {number} amendment: {row['outcome']} (expected {spec['previous']})")
        row = dict(row)
        # The record's result and scope take the amended cells, superseded wording and all; the verdict is re-read from the
        # amended cell, with the amendment bracket and the SUPERSEDED bracket dropped before the tokens are split.
        row["result"] = clean(new[3])
        row["scope"] = f"{clean(new[2])}. {clean(new[5])}"
        verdict = clean(strip_inserted_brackets(new[spec["verdict_cell"]], MECH4_AMENDMENT_TAG))
        verdict = re.sub(r"\s*\[SUPERSEDED.*$", "", verdict)
        tokens = [token.strip() for token in verdict.split(" + ") if token.strip()]
        if tokens[0] != spec["branch"]:
            raise ValueError(f"{row['id']}: the amended verdict does not open with {spec['branch']}")
        rule = {value: key for key, value in RULE_OUTCOME.items() if value}[spec["outcome"]]
        row["reason"] = (f"Verdict: {' + '.join(tokens)} {VERDICT_RULES[rule].split(':')[0]}: {spec['branch']}: "
                         f"{spec['reason']} The predecessor's verdict, kept verbatim: {clean(old[spec['verdict_cell']])}.")
        row["mapping_rule"] = rule
        row["outcome"] = spec["outcome"]
        sources = json.loads(row.get("sources") or "[]")
        # The anchor that pinned the row's text at the MUST-tier landing now points at the amended row, and the citations
        # the amendment added to the ref cell join the record's sources.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        added = [clean(part) for part in split_references(new[spec["ref_cell"]]) if clean(part) not in sources
                 and re.search(r"\b(?:CORRECTIONS|FINDINGS|CLOSEOUT)\b|[\w/]+\.(?:py|sh|md)", part)]
        row["sources"] = json.dumps([*sources, *added, f"{CORRECTIONS_DOC} [CORRECTIONS {number}]"], ensure_ascii=False)
        further = json.loads(row.get("further_amendments") or "[]")
        further.append({"number": number, "line": line, "commit": MECH4_COMMIT, "previousOutcome": spec["previous"], "outcome": spec["outcome"],
                        "reason": spec["reason"], "source": f"{MASTER_TABLE} line {line} at {MECH4_COMMIT[:7]}; CORRECTIONS {number}"})
        row["further_amendments"] = json.dumps(further, ensure_ascii=False)
        amended.append(row)
    return amended

def amended_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the amendment commit; only the listed lines may differ from the appended-rows pin."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{AMENDMENT_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != AMENDMENT_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {AMENDMENT_COMMIT[:12]} does not match the pinned amendment bytes")
    lines = raw.decode("utf-8").splitlines()
    before = appended_master_table(repo)
    changed = {n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}
    if len(lines) != len(before) or changed != AMENDED_LINES:
        raise ValueError(f"MASTER-TABLE amendments moved a line or edited lines other than {sorted(AMENDED_LINES)}: {sorted(changed)}")
    return lines


def apply_row_amendments(rows: list[dict], repo: Path) -> list[dict]:
    """Apply ROW_AMENDMENTS to the combined register, checking each against the pinned rows."""
    lines, before = amended_master_table(repo), appended_master_table(repo)
    by_id = {row["id"]: row for row in rows}
    if {line for line, *_ in ROW_AMENDMENTS.values()} != AMENDED_LINES - {3}:
        raise ValueError("Every amended MASTER-TABLE row needs exactly one record amendment")
    amended = []
    for row in rows:
        if row["id"] not in ROW_AMENDMENTS:
            amended.append(row)
            continue
        line, previous, outcome, token, batches, reason, replacement = ROW_AMENDMENTS[row["id"]]
        old, new = table_cells(before[line - 1]), table_cells(lines[line - 1])
        if len(old) != 7 or len(new) != 7 or old[0] != new[0]:
            raise ValueError(f"MASTER-TABLE line {line} changed its question cell or column count")
        question = re.sub(r"^\[[^\]]*\]\s*", "", clean(new[0]))
        if norm(row["original_question"])[:60] not in norm(question):
            raise ValueError(f"{row['id']} is not the record of MASTER-TABLE line {line}")
        if row["outcome"] != previous:
            raise ValueError(f"{row['id']} outcome drifted before its amendment: {row['outcome']} (expected {previous})")
        if token is None:
            if new[4] != old[4] or outcome != previous:
                raise ValueError(f"MASTER-TABLE line {line}: an unchanged verdict cannot change the outcome")
        elif not clean(new[4]).startswith(token) or outcome == previous:
            raise ValueError(f"MASTER-TABLE line {line} verdict does not start with {token}")
        if AMENDMENT_MARK not in new[5] or clean(old[5]).replace(" ", "") not in clean(new[5]).replace(" ", ""):
            raise ValueError(f"MASTER-TABLE line {line}: the amendment must be bracketed and keep the superseded wording")
        row = dict(row)
        history = {key: row[key] for key in ["result", "reason", "scope"]}
        if replacement:
            row.update(replacement)
        else:
            row["scope"] = f"{row['scope']} Amended at CORRECTIONS 229: {reason}"
        sources = json.loads(row.get("sources") or "[]")
        # Anchors that pinned the pre-amendment row text now point at the amended row.
        sources = [source | {"rowText": lines[line - 1]} if isinstance(source, dict) and source.get("rowText") == before[line - 1] else source for source in sources]
        row["sources"] = json.dumps([*sources, AMENDMENT_SOURCE], ensure_ascii=False)
        row["batches"] = json.dumps(list(dict.fromkeys([*json.loads(row["batches"]), *batches])))
        row["outcome"] = outcome
        row["amendment"] = json.dumps({"line": line, "commit": AMENDMENT_COMMIT, "previousOutcome": previous, "outcome": outcome,
                                       "reason": reason, "previous": history if replacement else None,
                                       "source": f"{MASTER_TABLE} line {line} at {AMENDMENT_COMMIT[:7]}; CORRECTIONS 229"}, ensure_ascii=False)
        amended.append(row)
    missing = set(ROW_AMENDMENTS) - by_id.keys()
    if missing:
        raise ValueError(f"Row amendments name absent records: {sorted(missing)}")
    return amended


def partition_id(line: int) -> str:
    return f"MT{line:03d}"


def clean(text: str) -> str:
    """Markdown emphasis and code marks off, backslash escapes made literal; wording and numbers unchanged."""
    # A backslash-escaped punctuation mark (S\*, a\_b, \|) is the literal character. Hold it out of the
    # emphasis and code rules below, which would otherwise pair its star with another one, then drop the backslash.
    text = re.sub(r"\\([!-/:-@\[-`{-~])", lambda match: f"\0{ord(match[1])}\0", text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", text)
    text = re.sub(r"\0(\d+)\0", lambda match: chr(int(match[1])), text)
    return re.sub(r"\s+", " ", text).strip()


def split_references(ref: str) -> list[str]:
    """Split a MASTER-TABLE ref cell into citations on ';' outside brackets; a ';' inside (), [] or {} belongs to its citation.

    Rows 224-225 write 'CORRECTIONS 242 (registration and launch; ... corrected in place at 246), 246 (landing)'. A cell whose
    brackets do not balance is kept as one citation rather than cut at a guessed position.
    """
    pairs, stack, parts, start = {")": "(", "]": "[", "}": "{"}, [], [], 0
    for index, char in enumerate(ref):
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                stack = ["unbalanced"]
                break
        elif char == ";" and not stack:
            parts.append(ref[start:index])
            start = index + 1
    if stack:
        parts, start = [], 0
    parts.append(ref[start:])
    return [part.strip() for part in parts if part.strip()]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", clean(text).lower())


def table_cells(line: str) -> list[str]:
    if not line.startswith("|") or not line.rstrip().endswith("|"):
        raise ValueError("Not a Markdown table row")
    # A Markdown-escaped pipe (\\|) belongs to its cell, not to the column split.
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]


def master_table_at_commit(repo: Path, commit: str = PARTITION_AUDIT_COMMIT) -> list[str]:
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {commit[:12]} does not match the pinned section-10 bytes")
    return raw.decode("utf-8").splitlines()


def partition_rows(lines: list[str]) -> list[dict]:
    """Parse the 37 section-10 rows and apply the explicit verdict mapping."""
    heading = lines[SECTION_HEADING_LINE - 1]
    if not heading.startswith("## 10.") or "count-matched partition audit" not in heading:
        raise ValueError("MASTER-TABLE section 10 heading moved")
    header = table_cells(lines[FIRST_ROW - 3])
    if header != ["tested", "varied", "scale", "result", "verdict", "so what", "ref"]:
        raise ValueError("MASTER-TABLE section 10 columns changed")
    if set(PARTITION_AUDIT) != set(range(FIRST_ROW, LAST_ROW + 1)) or len(lines) != LAST_ROW:
        raise ValueError("The partition-audit mapping must cover exactly lines 175-211")
    rows = []
    for line in range(FIRST_ROW, LAST_ROW + 1):
        cells = table_cells(lines[line - 1])
        if len(cells) != 7:
            raise ValueError(f"MASTER-TABLE line {line} is not a seven-column row")
        tested, varied, scale, result, verdict, so_what, ref = cells
        rule, batches, corrected, reason = PARTITION_AUDIT[line]
        outcome = RULE_OUTCOME[rule]
        kind = "method-check" if rule == "method" else "research"
        references = [{"path": MASTER_TABLE, "line": line, "rowText": lines[line - 1]}]
        references += [part for part in split_references(ref) if re.search(r"\b(?:CORRECTIONS|FINDINGS|CLOSEOUT)\b|[\w/]+\.(?:py|sh|md)", part)]
        rows.append({
            "id": partition_id(line), "section": str(SECTION), "area": AREA,
            "goal": clean(tested), "comparison": clean(varied), "why": WHY,
            "result": clean(result), "outcome": outcome or "",
            "reason": f"Verdict: {clean(verdict)} {VERDICT_RULES[rule].split(':')[0]}: {reason}",
            "scope": f"{clean(scale)}. {clean(so_what)}",
            "batches": json.dumps(batches), "sources": json.dumps(references, ensure_ascii=False),
            "original_question": clean(tested), "register_page": "",
            "kind": kind, "corrected": "1" if corrected else "",
            "correction": json.dumps({"note": corrected, "source": f"{MASTER_TABLE} line {line} at {PARTITION_AUDIT_COMMIT[:7]}"}, ensure_ascii=False) if corrected else "",
            "mapping_rule": rule, "master_table_line": str(line),
        })
    return rows


def appended_master_table(repo: Path) -> list[str]:
    """MASTER-TABLE at the appended-rows commit, checked against the partition-audit pin."""
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{APPENDED_COMMIT}:{MASTER_TABLE}"])
    if hashlib.sha256(raw).hexdigest() != APPENDED_MASTER_TABLE_SHA256:
        raise ValueError(f"{MASTER_TABLE} at {APPENDED_COMMIT[:12]} does not match the pinned appended-row bytes")
    lines = raw.decode("utf-8").splitlines()
    pinned = master_table_at_commit(repo)
    drift = [n for n in range(1, len(pinned) + 1) if n not in APPENDED_HEADER_LINES and lines[n - 1] != pinned[n - 1]]
    if drift or len(lines) != APPENDED_LAST_ROW:
        raise ValueError(f"MASTER-TABLE lines 1-{len(pinned)} changed outside the header, or rows were added past line {APPENDED_LAST_ROW}: {drift}")
    return lines


def landed_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE line 218 (cvt1) and attach its intervention note."""
    return with_interventions(appended_rows(lines, LANDED_ROWS, LANDED_FIRST_ROW, LANDED_LAST_ROW, LANDED_COMMIT), LANDED_INTERVENTIONS)


def with_interventions(rows: list[dict], interventions: dict) -> list[dict]:
    """Attach each record's intervention note (the arms whose CSV cell key hides a vote-weight or step-size-hold patch)."""
    for row in rows:
        spec = interventions.get(row["id"])
        if spec:
            source = spec.get("source") or f"{INTERVENTIONS_TSV} at {LANDED_COMMIT[:7]}; CORRECTIONS 230"
            row["intervention"] = json.dumps({"title": spec["title"], "note": spec["note"], "batch": spec["batch"], "arms": spec["arms"],
                                              "source": source}, ensure_ascii=False)
    return rows


def appended_rows(lines: list[str], mapping: dict | None = None, first: int = APPENDED_FIRST_ROW, last: int = APPENDED_LAST_ROW,
                  commit: str = APPENDED_COMMIT) -> list[dict]:
    """Parse appended MASTER-TABLE rows (default lines 212-217) and apply the explicit verdict mapping."""
    mapping = APPENDED_ROWS if mapping is None else mapping
    if set(mapping) != set(range(first, last + 1)) or len(lines) != last:
        raise ValueError(f"The appended-row mapping must cover exactly lines {first}-{last}")
    rows = []
    for line in range(first, last + 1):
        cells = table_cells(lines[line - 1])
        if len(cells) != 7:
            raise ValueError(f"MASTER-TABLE line {line} is not a seven-column row")
        tested, varied, scale, result, verdict, so_what, ref = cells
        rule, section, batches, figures, corrected, reason = mapping[line]
        # The row opens with a bracketed provenance label; the question follows it.
        label = re.match(r"^\[([^\]]*)\]\s*", clean(tested))
        if not label:
            raise ValueError(f"MASTER-TABLE line {line} lost its appended-row label")
        question = clean(tested)[label.end():]
        tokens = [token.strip() for token in clean(verdict).split(" + ")]
        if not reason.startswith(tokens[0]):
            raise ValueError(f"MASTER-TABLE line {line}: the mapping reason must start with the row's first verdict token {tokens[0]}")
        outcome = RULE_OUTCOME[rule]
        references = [{"path": MASTER_TABLE, "line": line, "rowText": lines[line - 1]}]
        # Record text is plain text, and a citation is record text: MASTER-TABLE line 229 writes a scorer gate in backticks
        # inside its ref cell. clean() is applied to each citation, not to the cell, so the bracket-aware split is unchanged.
        # Only the appended rows are cleaned here; the section-10 rows keep the citations they were published with.
        references += [clean(part) for part in split_references(ref) if re.search(r"\b(?:CORRECTIONS|FINDINGS|CLOSEOUT)\b|[\w/]+\.(?:py|sh|md)", part)]
        rows.append({
            "id": partition_id(line), "section": str(section), "area": AREAS[section],
            "goal": question, "comparison": clean(varied), "why": label[1] + ".",
            "result": clean(result), "outcome": outcome or "",
            "reason": f"Verdict: {' + '.join(tokens)} {VERDICT_RULES[rule].split(':')[0]}: {reason}",
            "scope": f"{clean(scale)}. {clean(so_what)}",
            "batches": json.dumps(batches), "sources": json.dumps(references, ensure_ascii=False),
            "original_question": question, "register_page": "",
            "kind": "method-check" if rule == "method" else "research", "corrected": "1" if corrected else "",
            "correction": json.dumps({"note": corrected, "source": f"{MASTER_TABLE} line {line} at {commit[:7]}"}, ensure_ascii=False) if corrected else "",
            "mapping_rule": rule, "master_table_line": str(line), "figure_ids": json.dumps(figures),
        })
    return rows


BASE_TEXT_FIELDS = ["goal", "comparison", "why", "result", "reason", "scope", "original_question"]


def remap_base_row(row: dict) -> dict:
    """Four outcomes plus the corrected badge for one row of the base register, its text cleaned like a MASTER-TABLE cell."""
    row = dict(row)
    # The campaign's register export copies MASTER-TABLE cells with their Markdown (**bold**, `code`); 39 of its 111 rows
    # carry marks. Rows parsed from MASTER-TABLE here go through clean(), so the base rows must too, or their titles,
    # scopes and verdict warnings show literal ** and backticks. clean() changes nothing else in these rows.
    for key in BASE_TEXT_FIELDS:
        if key in row:
            row[key] = clean(row[key])
    if row["outcome"] == "correction":
        if row["id"] not in CORRECTION_REMAP:
            raise ValueError(f"Register outcome 'correction' has no approved mapping: {row['id']}")
        kind, outcome, reason = CORRECTION_REMAP[row["id"]]
        row.update(kind=kind, outcome=outcome or "", corrected="1", mapping_rule="approved-correction-split",
                   correction=json.dumps({"previousOutcome": "correction", "previousLabel": "Corrected", "note": reason,
                                          "source": REGISTER_EXPORT}, ensure_ascii=False))
    else:
        if row["id"] in CORRECTION_REMAP:
            raise ValueError(f"Approved mapping names a row that is no longer 'correction': {row['id']}")
        if row["outcome"] not in OUTCOMES:
            raise ValueError(f"Unknown register outcome for {row['id']}: {row['outcome']}")
        row.update(kind="research", corrected="", correction="", mapping_rule="register")
    return row


def load_register(workspace: Path, repo: Path) -> list[dict]:
    """The published register: every row below, with the pinned in-place row amendments applied (229, then 231, then 234/236).

    The cvt4 / cvt5 landing (CORRECTIONS 240-241) amended no row; CORRECTIONS 244 then amended rows 222 and 223 in place.
    The cvt6 / cvt7 landing (CORRECTIONS 246-247) amended no row either (only line 5, which feeds no record), nor did the
    cvt8 / cvt9 landing (CORRECTIONS 252-253); its final audit (CORRECTIONS 253.15) then corrected row 227 in place.
    The four MUST-tier landings (CORRECTIONS 264-266) amended no row and did not touch line 5 either. The cvt10 / cwd1 /
    csv1 / cwd2 landing (CORRECTIONS 270-273) then amended row 229 in place, which moves MT229's outcome."""
    register = apply_cgn3_amendments(apply_row_amendments(load_unamended_register(workspace, repo), Path(repo)), Path(repo))
    register = apply_cvt89_amendments(apply_c244_amendments(apply_cvt23_amendments(register, Path(repo)), Path(repo)), Path(repo))
    return apply_mech4_amendments(register, Path(repo))


def load_unamended_register(workspace: Path, repo: Path) -> list[dict]:
    """Base register (four-outcome model) plus the pinned partition-audit and appended rows."""
    path = Path(workspace) / "outputs/tables/complete_experiment_register.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        base = [remap_base_row(row) for row in csv.DictReader(handle)]
    missing = set(CORRECTION_REMAP) - {row["id"] for row in base}
    if missing:
        raise ValueError(f"Approved correction mapping names absent IDs: {sorted(missing)}")
    added = (partition_rows(master_table_at_commit(Path(repo))) + appended_rows(appended_master_table(Path(repo)))
             + landed_rows(landed_master_table(Path(repo))) + cgn3_rows(cgn3_master_table(Path(repo)))
             + cvt23_rows(cvt23_master_table(Path(repo))) + cvt45_rows(cvt45_master_table(Path(repo)))
             + cvt67_rows(cvt67_master_table(Path(repo))) + cvt89_rows(cvt89_master_table(Path(repo)))
             + must_rows(must_master_table(Path(repo))) + mech4_rows(mech4_master_table(Path(repo)))
             + mech5_rows(mech5_master_table(Path(repo)))
             + mech6_rows(mech6_master_table(Path(repo)))
             + mech7_rows(mech7_master_table(Path(repo))))
    existing = {row["id"] for row in base}
    collisions = existing & {row["id"] for row in added}
    high = sorted(i for i in existing if re.fullmatch(r"MT\d{3}", i) and int(i[2:]) >= NEW_ID_FLOOR)
    if collisions or high:
        raise ValueError(f"New partition-audit IDs would collide with existing IDs: {sorted(collisions) or high}")
    return base + added


def anchor_master_table(rows: list[dict], lines: list[str]) -> list[dict]:
    """Resolve every MASTER-TABLE source anchor by row content in ``lines``.

    The register's line numbers come from an uncommitted MASTER-TABLE snapshot
    with an extra row at line 20; anchoring by content keeps each experiment on
    its own row in whichever committed revision is published.
    """
    questions = {}
    for number, text in enumerate(lines, 1):
        # Only the first cell identifies a row; some rows carry prose after the
        # closing pipe (e.g. line 15), so do not require a strict table row here.
        if text.startswith("|") and text.count("|") >= 2:
            questions.setdefault(norm(text.split("|")[1]), []).append(number)
    by_text = {}
    for number, text in enumerate(lines, 1):
        by_text.setdefault(text, []).append(number)

    def resolve(row, requested, row_text=None):
        if row_text is not None:
            hits = by_text.get(row_text, [])
        else:
            key = norm(row["original_question"])
            exact = [n for q, ns in questions.items() if key and q == key for n in ns]
            # The register keeps the question text; the table cell may add a bold
            # label or a trailing clause, so fall back to containment of a
            # distinctive prefix. Uniqueness is still required below.
            hits = exact or [n for q, ns in questions.items() if len(key) >= 20 and key[:60] in q for n in ns]
        if len(hits) != 1:
            raise ValueError(f"MASTER-TABLE anchor for {row['id']} is not unique by content (requested line {requested}): {hits}")
        return hits[0]

    anchored = []
    for row in rows:
        row = dict(row)
        sources = json.loads(row.get("sources") or "[]")
        updated = []
        for source in sources:
            if isinstance(source, dict) and str(source.get("path", "")).endswith(MASTER_TABLE):
                source = {key: value for key, value in source.items() if key != "rowText"} | {"line": resolve(row, source.get("line"), source.get("rowText"))}
            elif isinstance(source, str) and re.search(re.escape(MASTER_TABLE) + r":\d+", source):
                requested = int(re.search(re.escape(MASTER_TABLE) + r":(\d+)", source)[1])
                source = re.sub(re.escape(MASTER_TABLE) + r":\d+", f"{MASTER_TABLE}:{resolve(row, requested)}", source)
            updated.append(source)
        row["sources"] = json.dumps(updated, ensure_ascii=False)
        anchored.append(row)
    return anchored
