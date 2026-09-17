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

Record text is plain text: ``clean()`` removes Markdown bold, emphasis and code marks from every MASTER-TABLE cell and
from the text fields of the campaign's register export (``BASE_TEXT_FIELDS``), which copies the cells with their marks.

IDs: existing IDs are never renumbered. New MASTER-TABLE rows are keyed
``MT<line>`` on their line in the pinned commit (MT175-MT211, then MT212-MT217, then MT218, then MT219, then MT220-MT221,
then MT222-MT223, then MT224-MT225);
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
AREAS = {1: "Baseline comparisons", 9: "Mechanism and isolation"}
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
    "experimentIds": [f"MT{line}" for line in range(APPENDED_FIRST_ROW, 225 + 1)],
    "source": "docs/CORRECTIONS.md 216-226 at 64e4f47",
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
                             (CVT67_COMMIT, CVT67_INTERVENTIONS_TSV_SHA256)]:
        pinned = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{INTERVENTIONS_TSV}"])
        if hashlib.sha256(pinned).hexdigest() != expected:
            raise ValueError(f"{INTERVENTIONS_TSV} at {commit[:12]} does not match the pinned bytes")
        if not pinned.startswith(raw):
            raise ValueError(f"{INTERVENTIONS_TSV} changed rows listed at {previous[:7]}; it may only grow by appended rows")
        raw, previous = pinned, commit
    body = [line for line in raw.decode("utf-8").splitlines() if line and not line.startswith("#")]
    rows = list(csv.DictReader(body, delimiter="\t"))
    runs = {}
    for eid, spec in {**LANDED_INTERVENTIONS, **CVT23_INTERVENTIONS, **CVT45_INTERVENTIONS, **CVT67_INTERVENTIONS}.items():
        mine = [row for row in rows if row["batch"] == spec["batch"]]
        arms = {arm: sum(row["arm"] == arm for row in mine) for arm in spec["arms"]}
        if arms != spec["arms"] or len(mine) != sum(spec["arms"].values()):
            raise ValueError(f"{INTERVENTIONS_TSV} does not list the registered intervened arms of {eid}: {arms}")
        for row in mine:
            intervention_kinds(row["intervention"])  # every listed hold must be a known kind
            runs[row["job_id"]] = {**row, "experimentId": eid}
    if len(runs) != len(rows):
        raise ValueError(f"{INTERVENTIONS_TSV} lists runs with no registered intervention record")
    return runs


# The exclusion list's intervention column is free text: one PATCH=value per hold, separated by spaces (CORRECTIONS 245:
# cvt6's forced arms carry BETA_HOLD and COMP_HOLD at once). Each kind is named on the run page; an unknown patch stops
# the export rather than being guessed.
INTERVENTION_KINDS = {"VOTE_W": "vote-weight", "BETA_HOLD": "step-size hold", "COMP_HOLD": "complement step-size hold",
                      "GROUP_HOLD": "group step-size hold"}


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


def intervention_phrase(intervention: str) -> str:
    """'step-size hold intervention', or 'step-size hold and complement step-size hold interventions' for a two-kind run."""
    names = [INTERVENTION_KINDS[patch] for patch, _ in intervention_kinds(intervention)]
    return f"{names[0]} intervention" if len(names) == 1 else " and ".join(names) + " interventions"


def additional_witness(lines: list[str], patch: str, value: str) -> str:
    """The run log's one '<patch>: on' line, checked against the listed value (floor, tri:<P> or rec:<id>).

    The exclusion list carries the witness of a run's first hold only; a second hold is witnessed by the run's own log."""
    found = [line for line in lines if line.startswith(f"{patch}: on ")]
    if len(found) != 1:
        raise ValueError(f"A run log must print exactly one '{patch}: on' line; found {len(found)}")
    mode = value.rsplit(":", 2)
    if value.endswith(":floor") or value == "floor":
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
        "note": ("The five held arms ran cvt4's tree plus the opt-in PATCH_COMPHOLD (COMP_HOLD=tri:<P> | rec:<id>), on HEAD's grouping "
                 "({50} [52,1]). PATCH_BETAHOLD holds tensor 50's beta (layer4.1.bn2.weight's log step size); PATCH_COMPHOLD overwrites "
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
        references += [part.strip() for part in ref.split(";") if re.search(r"\b(?:CORRECTIONS|FINDINGS|CLOSEOUT)\b|[\w/]+\.(?:py|sh|md)", part)]
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
        references += [part.strip() for part in ref.split(";") if re.search(r"\b(?:CORRECTIONS|FINDINGS|CLOSEOUT)\b|[\w/]+\.(?:py|sh|md)", part)]
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
    The cvt6 / cvt7 landing (CORRECTIONS 246-247) amended no row either (only line 5, which feeds no record)."""
    register = apply_cgn3_amendments(apply_row_amendments(load_unamended_register(workspace, repo), Path(repo)), Path(repo))
    return apply_c244_amendments(apply_cvt23_amendments(register, Path(repo)), Path(repo))


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
             + cvt67_rows(cvt67_master_table(Path(repo))))
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
