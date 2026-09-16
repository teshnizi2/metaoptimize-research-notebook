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

IDs: existing IDs are never renumbered. New MASTER-TABLE rows are keyed
``MT<line>`` on their line in the pinned commit (MT175-MT211, then MT212-MT217);
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
    "id": "phase-09", "date": "2026-09-15", "period": "15-16 Sep", "title": "Off BatchNorm, off residuals, long horizons",
    "test": "GroupNorm and residual-free ResNet-18 isolation; VGG rescue at 328 epochs; unaugmented CIFAR-100 baseline",
    "observed_result": "Gap and carrier set transfer to GroupNorm; one BN scale rescues without residuals; VGG rescue holds at 328 epochs; unaugmented deficit +12.18 pp",
    "next_question": "Identity versus magnitude remains unresolved on every network",
    "experimentIds": [f"MT{line}" for line in range(APPENDED_FIRST_ROW, APPENDED_LAST_ROW + 1)],
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


def partition_id(line: int) -> str:
    return f"MT{line:03d}"


def clean(text: str) -> str:
    """Markdown emphasis and code marks off; wording and numbers unchanged."""
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", text)
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


def appended_rows(lines: list[str]) -> list[dict]:
    """Parse MASTER-TABLE lines 212-217 and apply the explicit verdict mapping."""
    if set(APPENDED_ROWS) != set(range(APPENDED_FIRST_ROW, APPENDED_LAST_ROW + 1)) or len(lines) != APPENDED_LAST_ROW:
        raise ValueError(f"The appended-row mapping must cover exactly lines {APPENDED_FIRST_ROW}-{APPENDED_LAST_ROW}")
    rows = []
    for line in range(APPENDED_FIRST_ROW, APPENDED_LAST_ROW + 1):
        cells = table_cells(lines[line - 1])
        if len(cells) != 7:
            raise ValueError(f"MASTER-TABLE line {line} is not a seven-column row")
        tested, varied, scale, result, verdict, so_what, ref = cells
        rule, section, batches, figures, corrected, reason = APPENDED_ROWS[line]
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
            "correction": json.dumps({"note": corrected, "source": f"{MASTER_TABLE} line {line} at {APPENDED_COMMIT[:7]}"}, ensure_ascii=False) if corrected else "",
            "mapping_rule": rule, "master_table_line": str(line), "figure_ids": json.dumps(figures),
        })
    return rows


def remap_base_row(row: dict) -> dict:
    """Four outcomes plus the corrected badge for one row of the base register."""
    row = dict(row)
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
    """Base register (four-outcome model) plus the pinned partition-audit rows."""
    path = Path(workspace) / "outputs/tables/complete_experiment_register.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        base = [remap_base_row(row) for row in csv.DictReader(handle)]
    missing = set(CORRECTION_REMAP) - {row["id"] for row in base}
    if missing:
        raise ValueError(f"Approved correction mapping names absent IDs: {sorted(missing)}")
    added = partition_rows(master_table_at_commit(Path(repo))) + appended_rows(appended_master_table(Path(repo)))
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
