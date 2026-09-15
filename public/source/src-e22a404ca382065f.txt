#!/usr/bin/env python3
# =============================================================================
# cU1_alpha0_granularity_score.py
#
#   IS THERE AN alpha0 x GRANULARITY INTERACTION ON CIFAR-100 AT 100 EPOCHS,
#   OR ONLY A MAIN EFFECT OF alpha0 -- AND DOES THE ARM ORDERING DEPEND ON
#   THE INITIALISATION CONSTANT?
#
# Cycle 135.  ZERO GPU.  This scorer has NO RUNS OF ITS OWN: it reads rows that
# already exist in results/all_runs.csv.
#
# -----------------------------------------------------------------------------
# RULE 21 STATUS -- STATED FIRST, BECAUSE IT LIMITS EVERYTHING BELOW
# -----------------------------------------------------------------------------
# THIS FILE DOES NOT CARRY THE RULE 21 LABEL.  Precedent: cQ1 (CORRECTIONS 149)
# and cS1 (CORRECTIONS 159).  RULE 21's proof is "the scorer's commit precedes
# the earliest sacct Submit of its own batch".  This scorer has no batch, so
# that proof DOES NOT EXIST and is not claimed.
#
# WHAT IS CLAIMED, AND ONLY THIS:
#   * COMMIT-BEFORE-FIRST-EXECUTION.  Every filter, literal, bar, gate and
#     verdict string below is fixed in this one committed file, and the file is
#     run UNEDITED afterwards.  The margin is recorded in docs/CORRECTIONS.md
#     and docs/STATUS.md as (commit timestamp) -> (first execution timestamp).
#
# WHAT IS EXPLICITLY *NOT* CLAIMED -- DISCLOSED, NOT LIFTED:
#   * THIS IS NOT A BLIND TEST.  Building the census below required listing the
#     stratum, and that listing PRINTED the individual `plateau5` values of the
#     primary cells.  The author therefore saw the primary rows before this file
#     was committed.  Additionally, the cycle briefing quoted four alleged
#     figures (layerwise 69.532 -> 44.968, scalar 22.749 -> 22.208) BEFORE any
#     work started.  This scorer's value is REPRODUCIBILITY and a fixed verdict
#     map, NOT pre-registration.  Any reader who wants a blind test of this
#     question must run a new batch.
#
# STANDING RULE 16: run this file UNEDITED and quote its verdict.  Passing its
# documented arguments (--csv, --selftest) is NOT editing it.
#
# -----------------------------------------------------------------------------
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# A four-lens viability assessment reported that at fixed meta_stepsize 1e-3,
# moving alpha0 from 1e-6 to 1e-3 takes CIFAR-100 `layerwise` from 69.532 (n=20)
# to 44.968 (n=39) -- a 24.6 pp swing -- while `scalar` barely moves
# (22.749 -> 22.208).  If that were true, the ARM ORDERING of the cell every
# recent headline comes from would depend on an initialisation constant that has
# never been mapped, and it would be a second confound of the same shape as the
# open RULE 11 meta-stepsize confound.
#
# It also stands against a RECORDED verdict.  docs/MASTER-TABLE.md row
# "Does granularity transfer to a second dataset (CIFAR-100) ..." is CONFIRMED
# and reads, verbatim: "a0=1e-6: scalar 22.78 -> blk6 52.48 -> layerwise 69.31
# (gain +46.52).  a0=1e-3: scalar 22.57 -> blk6 51.32 -> layerwise 69.59
# (gain +47.02).  Essentially insensitive to alpha0."  Both cannot be right.
# This file decides which, on a count-matched, WITHIN-BATCH stratum.
#
# -----------------------------------------------------------------------------
# THE STRATUM AND EVERY FILTER -- FROZEN
# -----------------------------------------------------------------------------
#   dataset          == CIFAR100
#   network          == ResNet18_c100
#   base             == SGDm          meta == Lion
#   gamma            == 1             augment == 1
#   beta_clip        == -15:-2.3026   batch_size == 100
#   hier             == ''            (NO hierarchical operator: `additive` and
#                                      `shrink` rows are EXCLUDED)
#   collapsed        == 0    complete == 1    superseded == 0
#   epochs_done      == 100           (the 20-epoch table is a SEPARATE stratum,
#                                      censused but never pooled with this one)
#   meta_stepsize    == 1e-3          (1e-4 is censused; see NOTE MS below)
#   granularity      in {scalar, resnet18_blocks, layerwise}
#   alpha0           in {1e-6, 1e-3}
#
# PRIMARY COLUMN: `plateau5`.  The CSV `plateau` column is BANNED as primary.
# `best_test` is NOT a plateau and is printed only under a NOT-A-PLATEAU label.
# TRAIN COMPANION: `final_train`.  The campaign's `train5` is computed from the
# .out files and is NOT in the CSV; `final_train` is the CSV's train column and
# is labelled as such wherever it appears.  It is reported alongside every TEST
# number, never instead of one.
#
# NOTE MS -- WHY meta_stepsize 1e-4 CANNOT ENTER THE PRIMARY.  In the whole
# corpus, CIFAR-100 has 20 rows at ms=1e-4 and ALL 20 sit at alpha0=1e-3, and
# NONE of them is `scalar`, `layerwise` or `resnet18_blocks` (they are
# `nodewise`, `nodewise1d`, `chunk771`, `chunk2293`).  ms and alpha0 are
# therefore PERFECTLY ALIASED on CIFAR-100 outside ms=1e-3, and no ms contrast
# at fixed alpha0 exists for any arm in the primary set.  The scorer prints this
# census and refuses any ms claim.
#
# -----------------------------------------------------------------------------
# THE DESIGN -- WHY *WITHIN-BATCH* AND COUNT-MATCHED
# -----------------------------------------------------------------------------
# BATCH is the unit of replication in this campaign (F(62,85)=5.47, p 6.9e-13).
# In the frozen stratum exactly TWO batches carry BOTH alpha0 levels for ALL
# THREE granularities: `c100` (n=2 per cell) and `c100b` (n=3 per cell).  That
# 2x3x2 block is perfectly balanced -- 5 rows per (granularity, alpha0), the
# same 5 seeds on both sides -- and is the PRIMARY.  `c1b` carries both alpha0
# levels for `layerwise` only and is the SECONDARY extension, reported and never
# pooled into the primary.
#
#   d(batch, g)  = mean plateau5[alpha0=1e-3] - mean plateau5[alpha0=1e-6]
#   DELTA(g)     = mean over the PRIMARY batches of d(batch, g)      (batch = unit)
#
# MAIN EFFECT OF alpha0 ALONE:  every DELTA(g) resolvably non-zero (|DELTA| >
#   MAIN_BAR) AND all of the same sign AND the spread across g is within noise
#   (every pairwise |DELTA(g1) - DELTA(g2)| <= INTERACTION_BAR).
#
# INTERACTION alpha0 x GRANULARITY: at least one pairwise
#   |DELTA(g1) - DELTA(g2)| > INTERACTION_BAR.  The PRIMARY pair, named in
#   advance because it is the pair the assessment names, is
#   (scalar, layerwise).  The other two pairs are reported as SECONDARY.
#
# ORDERING: the three arms are ranked by their own alpha0-level means within the
#   primary batches.  ORDER-FLIPS only if the rank order differs between the two
#   alpha0 levels AND the flipped pair separates by more than ARGMAX_BAR at both
#   levels.  Otherwise ORDER-PRESERVED (or ORDER-UNRESOLVED if a pair is inside
#   the bar at either level).
#
# THE ASSESSMENT'S OWN CLAIM is scored as a separate, pre-stated arithmetic test
# so that it cannot be softened after the fact:
#   ASSESSMENT-REPRODUCES  iff  DELTA(layerwise) <= -20.0 pp  AND
#                               |DELTA(scalar)|  <=   1.0 pp
#   otherwise ASSESSMENT-DOES-NOT-REPRODUCE.
#
# -----------------------------------------------------------------------------
# THE NOISE FLOOR -- RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME,
# NOT COPIED FROM ANY EARLIER SCORER
# -----------------------------------------------------------------------------
# CORRECTIONS 156.9 recorded that cts3's SIGMA_W had drifted by the time scl1's
# rows landed.  No literal is imported here.  SIGMA_W below is the pooled
# within-(batch x granularity x alpha0) SD of `plateau5` over the FULL frozen
# 100-epoch / ms=1e-3 stratum (all batches, both alpha0 levels, the three
# primary granularities), computed from results/all_runs.csv at 2,573 rows:
#
#   SIGMA_W      = 0.552209   df 40, 23 cells, 63 members, 0 singleton cells
#   SIGMA_TRAIN  = 0.385535   (same cells, column `final_train`)
#
# Bars, each an arithmetic consequence of the frozen census (c100 n=2,
# c100b n=3, both alpha0 levels, all three granularities):
#
#   var(DELTA(g)) = [ SIGMA^2*(1/2+1/2) + SIGMA^2*(1/3+1/3) ] / 4 = 0.4166667*SIGMA^2
#   SE_DELTA      = SIGMA * sqrt(0.4166667)
#   MAIN_BAR      = 2 * SE_DELTA
#   SE_DID        = SIGMA * sqrt(2*0.4166667)
#   INTERACTION_BAR = 2 * SE_DID
#   var(arm mean at one alpha0) = [SIGMA^2/2 + SIGMA^2/3]/4 = 0.2083333*SIGMA^2
#   ARGMAX_BAR    = 2 * SIGMA * sqrt(2*0.2083333)
#
# Every bar derives from the FROZEN literal, never from the live corpus.  A
# later ingest may move the live value; the --selftest will then FAIL that
# check, and -- exactly as cts3 established -- NO VERDICT MOVES WITH IT.
#
# -----------------------------------------------------------------------------
# THE PREMISE CHECK IS CORPUS-CONDITIONAL (cR1's pattern), SO IT DOES NOT
# BECOME A KNOWN-FALSE ASSERTION LATER
# -----------------------------------------------------------------------------
# The one live batch at registration time is `cpk3` (21 jobs, launched
# CORRECTIONS 159, NOT ours to score or ingest here).  Its rows are granularity
# strings of the form `[k,62-k]` at `epochs_done=772`.  BOTH of those are
# excluded by the frozen stratum -- granularity is restricted to three literal
# names and epochs_done to 100 -- so cpk3's landing CANNOT move this census.
# The premise check is therefore an equality that stays TRUE after that ingest,
# rather than one that must fail.  It is stated as: the per-(batch, granularity,
# alpha0) census of the frozen stratum equals CENSUS_100 exactly.  If a FUTURE
# batch does add named-granularity 100-epoch ms=1e-3 CIFAR-100 rows, this check
# FAILS loudly and that FAIL is correct behaviour: the stratum changed and this
# file is frozen.
# =============================================================================

import argparse
import collections
import csv
import math
import os
import sys

CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "results", "all_runs.csv")

# ---------------------------------------------------------------- frozen ----
PRIMARY_GRANS = ("scalar", "resnet18_blocks", "layerwise")
PRIMARY_BATCHES = ("c100", "c100b")
SECONDARY_BATCH = "c1b"          # layerwise only, both alpha0 levels
A_LO, A_HI = "1e-6", "1e-3"

SIGMA_W = 0.552209
SIGMA_W_DF = 40
SIGMA_W_CELLS = 23
SIGMA_W_MEMBERS = 63
SIGMA_TRAIN = 0.385535

VAR_DELTA_COEF = (1.0 / 2 + 1.0 / 2) / 4.0 + (1.0 / 3 + 1.0 / 3) / 4.0   # 0.4166667
VAR_ARM_COEF = (1.0 / 2 + 1.0 / 3) / 4.0                                 # 0.2083333

SE_DELTA = SIGMA_W * math.sqrt(VAR_DELTA_COEF)
MAIN_BAR = 2.0 * SE_DELTA
SE_DID = SIGMA_W * math.sqrt(2.0 * VAR_DELTA_COEF)
INTERACTION_BAR = 2.0 * SE_DID
ARGMAX_BAR = 2.0 * SIGMA_W * math.sqrt(2.0 * VAR_ARM_COEF)

SE_DELTA_TR = SIGMA_TRAIN * math.sqrt(VAR_DELTA_COEF)
MAIN_BAR_TR = 2.0 * SE_DELTA_TR
INTERACTION_BAR_TR = 2.0 * SIGMA_TRAIN * math.sqrt(2.0 * VAR_DELTA_COEF)

ASSESS_LAYER_BAR = -20.0     # DELTA(layerwise) must be <= this to reproduce
ASSESS_SCALAR_BAR = 1.0      # |DELTA(scalar)| must be <= this to reproduce

# the frozen census of the primary stratum: (batch, granularity, alpha0) -> n
CENSUS_100 = {
    ("c100",  "scalar",          "1e-6"): 2, ("c100",  "scalar",          "1e-3"): 2,
    ("c100b", "scalar",          "1e-6"): 3, ("c100b", "scalar",          "1e-3"): 3,
    ("cbl1",  "scalar",          "1e-6"): 3,
    ("cpk1",  "scalar",          "1e-6"): 3,
    ("cts1",  "scalar",          "1e-6"): 3,
    ("hb1",   "scalar",          "1e-6"): 3,
    ("c100",  "resnet18_blocks", "1e-6"): 2, ("c100",  "resnet18_blocks", "1e-3"): 2,
    ("c100b", "resnet18_blocks", "1e-6"): 3, ("c100b", "resnet18_blocks", "1e-3"): 3,
    ("hb1",   "resnet18_blocks", "1e-6"): 3,
    ("c100",  "layerwise",       "1e-6"): 2, ("c100",  "layerwise",       "1e-3"): 2,
    ("c100b", "layerwise",       "1e-6"): 3, ("c100b", "layerwise",       "1e-3"): 3,
    ("c1b",   "layerwise",       "1e-6"): 3, ("c1b",   "layerwise",       "1e-3"): 3,
    ("cbl1",  "layerwise",       "1e-6"): 3,
    ("cpk1",  "layerwise",       "1e-6"): 3,
    ("cts1",  "layerwise",       "1e-6"): 3,
    ("hb1",   "layerwise",       "1e-6"): 3,
}

# CIFAR-100 corpus-level facts frozen at registration, each re-derived by --selftest
N_C100_SCALAR_ROWS = 33          # every CIFAR-100 scalar row in the corpus ...
N_C100_SCALAR_MS1E3 = 33         # ... and every one of them sits at ms=1e-3
N_C100_MS1E4_ROWS = 20           # the whole CIFAR-100 ms=1e-4 stratum ...
N_C100_MS1E4_AT_A0_1E3 = 20      # ... sits entirely at alpha0=1e-3
MS1E4_GRANS = ("chunk2293", "chunk771", "nodewise", "nodewise1d")


# ------------------------------------------------------------------ data ----
def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def batch_of(row):
    return row["run"].split("-")[0]


def in_stratum(r):
    return (r["dataset"] == "CIFAR100"
            and r["network"] == "ResNet18_c100"
            and r["base"] == "SGDm"
            and r["meta"] == "Lion"
            and r["gamma"] == "1"
            and r["augment"] == "1"
            and r["beta_clip"] == "-15:-2.3026"
            and r["batch_size"] == "100"
            and r["hier"] == ""
            and r["collapsed"] == "0"
            and r["complete"] == "1"
            and r["superseded"] == "0"
            and r["epochs_done"] == "100"
            and r["meta_stepsize"] == "1e-3"
            and r["granularity"] in PRIMARY_GRANS
            and r["alpha0"] in (A_LO, A_HI))


def census(rows):
    c = collections.Counter()
    for r in rows:
        if in_stratum(r):
            c[(batch_of(r), r["granularity"], r["alpha0"])] += 1
    return dict(c)


def cells(rows, col):
    g = collections.defaultdict(list)
    for r in rows:
        if in_stratum(r):
            g[(batch_of(r), r["granularity"], r["alpha0"])].append(float(r[col]))
    return g


def pooled_sd(g):
    ss, df, n_cells, mem, single = 0.0, 0, 0, 0, 0
    for v in g.values():
        mem += len(v)
        n_cells += 1
        if len(v) < 2:
            single += 1
            continue
        m = sum(v) / len(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
    return math.sqrt(ss / df), df, n_cells, mem, single


def mean(v):
    return sum(v) / len(v)


# ----------------------------------------------------------------- score ----
def deltas(g, batches):
    """DELTA(gran) over the given batch set, batch = unit of replication."""
    out = {}
    for gran in PRIMARY_GRANS:
        per = []
        for b in batches:
            lo = g.get((b, gran, A_LO))
            hi = g.get((b, gran, A_HI))
            if lo and hi:
                per.append((b, mean(hi) - mean(lo)))
        if per:
            out[gran] = (mean([d for _, d in per]), per)
    return out


def arm_means(g, batches):
    out = {}
    for gran in PRIMARY_GRANS:
        for a0 in (A_LO, A_HI):
            per = [mean(g[(b, gran, a0)]) for b in batches if (b, gran, a0) in g]
            if per:
                out[(gran, a0)] = (mean(per), len(per))
    return out


def verdict(dl, bar_main, bar_int):
    names = [gr for gr in PRIMARY_GRANS if gr in dl]
    if len(names) < 3:
        return "UNRESOLVED-THIN", []
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairs.append((a, b, dl[a][0] - dl[b][0]))
    if any(abs(d) > bar_int for _, _, d in pairs):
        return "INTERACTION-ALPHA0-x-GRANULARITY", pairs
    resolved = [gr for gr in names if abs(dl[gr][0]) > bar_main]
    if len(resolved) == len(names) and len(set(dl[gr][0] > 0 for gr in names)) == 1:
        return "MAIN-EFFECT-ALPHA0-ONLY", pairs
    if not resolved:
        return "NO-RESOLVED-ALPHA0-EFFECT", pairs
    return "PARTIAL-ALPHA0-EFFECT-NO-INTERACTION", pairs


def order_verdict(am):
    ranks = {}
    for a0 in (A_LO, A_HI):
        ranks[a0] = sorted(PRIMARY_GRANS, key=lambda gr: -am[(gr, a0)][0])
    if ranks[A_LO] == ranks[A_HI]:
        gaps = []
        for a0 in (A_LO, A_HI):
            r = ranks[a0]
            for i in range(len(r) - 1):
                gaps.append(am[(r[i], a0)][0] - am[(r[i + 1], a0)][0])
        if min(gaps) > ARGMAX_BAR:
            return "ORDER-PRESERVED", ranks
        return "ORDER-PRESERVED-BUT-A-GAP-IS-INSIDE-THE-BAR", ranks
    return "ORDER-FLIPS", ranks


def contaminated_pools(rows):
    """The pools an UNMATCHED query would produce, so the reader can see which
    one manufactures the assessment's figure and exactly what it mixes in."""
    base = [r for r in rows
            if r["dataset"] == "CIFAR100" and r["network"] == "ResNet18_c100"
            and r["granularity"] == "layerwise" and r["meta_stepsize"] == "1e-3"
            and r["alpha0"] == A_HI and r["collapsed"] == "0"]
    defs = [
        ("P0  matched primary stratum (this file)",
         lambda r: in_stratum(r) and r["granularity"] == "layerwise"
         and r["alpha0"] == A_HI and batch_of(r) in PRIMARY_BATCHES),
        ("P1  matched stratum, all batches",
         lambda r: in_stratum(r) and r["granularity"] == "layerwise"
         and r["alpha0"] == A_HI),
        ("P2  + 20/5-epoch rows, hier still excluded",
         lambda r: r in base and r["hier"] == ""),
        ("P3  + hier additive/shrink, box C only, epochs >= 20",
         lambda r: r in base and r["beta_clip"] == "-15:-2.3026"
         and r["hier"] != "shrink" and int(r["epochs_done"]) >= 20),
        ("P4  everything: any hier, any box, any horizon",
         lambda r: r in base),
    ]
    out = []
    for label, fn in defs:
        v = [float(r["plateau5"]) for r in rows if fn(r) and r["plateau5"] not in ("", "?")]
        out.append((label, len(v), mean(v) if v else float("nan")))
    return out


def score(csv_path):
    rows = load(csv_path)
    print("=" * 78)
    print("cU1_alpha0_granularity_score -- alpha0 x GRANULARITY on CIFAR-100, 100 ep")
    print("=" * 78)
    print("corpus: %s   %d rows" % (os.path.relpath(csv_path), len(rows)))
    print("PRIMARY COLUMN plateau5.  The CSV `plateau` column is BANNED as primary;")
    print("`best_test` is NOT a plateau.  TRAIN column reported = `final_train`.")
    print("RULE 21 LABEL: NOT CARRIED.  See this file's header.")

    # -- G0 PREMISE (corpus-conditional) --------------------------------------
    print("\n-- G0 PREMISE: the frozen stratum census ------------------------------")
    live = census(rows)
    if live == CENSUS_100:
        print("   census MATCHES the frozen CENSUS_100 exactly (%d cells)." % len(live))
        premise = True
    else:
        premise = False
        print("   CENSUS CHANGED.  The stratum this file was frozen against no longer")
        print("   matches the corpus.  Every bar below still derives from the FROZEN")
        print("   literals, so no verdict moves -- but the reader must be told:")
        for k in sorted(set(live) | set(CENSUS_100)):
            a, b = CENSUS_100.get(k, 0), live.get(k, 0)
            if a != b:
                print("      %-34s frozen %d -> live %d" % (str(k), a, b))

    print("\n   cells that exist, with n  (batch, granularity, alpha0):")
    for k in sorted(live, key=lambda k: (PRIMARY_GRANS.index(k[1]), k[2], k[0])):
        print("      %-8s %-16s a0=%-5s n=%d" % (k[0], k[1], k[2], live[k]))

    # -- the noise floor, re-derived ------------------------------------------
    print("\n-- THE NOISE FLOOR: re-derived from the live corpus now ---------------")
    g5 = cells(rows, "plateau5")
    gtr = cells(rows, "final_train")
    s5, df5, c5, m5, sg5 = pooled_sd(g5)
    st_, dft, ct, mt, sgt = pooled_sd(gtr)
    print("   live  SIGMA_W(plateau5)    = %.6f  df %d, %d cells, %d members, %d singleton"
          % (s5, df5, c5, m5, sg5))
    print("   frozen SIGMA_W             = %.6f  df %d, %d cells, %d members"
          % (SIGMA_W, SIGMA_W_DF, SIGMA_W_CELLS, SIGMA_W_MEMBERS))
    print("   live  SIGMA_TRAIN(final_train) = %.6f   frozen = %.6f" % (st_, SIGMA_TRAIN))
    print("   bars (TEST, from the FROZEN literal):")
    print("      SE_DELTA        %.6f      MAIN_BAR        %.6f" % (SE_DELTA, MAIN_BAR))
    print("      SE_DID          %.6f      INTERACTION_BAR %.6f" % (SE_DID, INTERACTION_BAR))
    print("      ARGMAX_BAR      %.6f" % ARGMAX_BAR)
    print("   bars (TRAIN): MAIN_BAR_TR %.6f  INTERACTION_BAR_TR %.6f"
          % (MAIN_BAR_TR, INTERACTION_BAR_TR))

    # -- the primary block -----------------------------------------------------
    for col, gg, bar_m, bar_i, sig, tag in (
            ("plateau5",    g5,  MAIN_BAR,    INTERACTION_BAR,    SIGMA_W,     "TEST "),
            ("final_train", gtr, MAIN_BAR_TR, INTERACTION_BAR_TR, SIGMA_TRAIN, "TRAIN")):
        print("\n" + "-" * 78)
        print("%s -- column `%s`, PRIMARY batches %s (count-matched 2+3 per cell)"
              % (tag, col, list(PRIMARY_BATCHES)))
        print("-" * 78)
        am = arm_means(gg, PRIMARY_BATCHES)
        print("   %-16s %10s %10s %12s" % ("granularity", "a0=1e-6", "a0=1e-3", "DELTA"))
        dl = deltas(gg, PRIMARY_BATCHES)
        for gr in PRIMARY_GRANS:
            lo = am.get((gr, A_LO), (float("nan"), 0))[0]
            hi = am.get((gr, A_HI), (float("nan"), 0))[0]
            d = dl.get(gr, (float("nan"), []))[0]
            print("   %-16s %10.4f %10.4f %+12.4f" % (gr, lo, hi, d))
        print("   per-batch d(batch, gran):")
        for gr in PRIMARY_GRANS:
            if gr in dl:
                print("      %-16s %s" % (gr, "  ".join(
                    "%s %+.4f" % (b, d) for b, d in dl[gr][1])))
        v, pairs = verdict(dl, bar_m, bar_i)
        print("   pairwise DELTA differences (bar %.4f):" % bar_i)
        for a, b, d in pairs:
            print("      %-16s - %-16s = %+8.4f   %s"
                  % (a, b, d, "OVER BAR" if abs(d) > bar_i else "within bar"))
        print("   VERDICT (%s): %s" % (tag.strip(), v))
        if col == "plateau5":
            primary_verdict, primary_dl, primary_am = v, dl, am

    # -- ordering --------------------------------------------------------------
    print("\n-- ARM ORDERING at each alpha0 (bar %.4f) -----------------------------"
          % ARGMAX_BAR)
    ov, ranks = order_verdict(primary_am)
    for a0 in (A_LO, A_HI):
        print("   a0=%-5s  %s" % (a0, "  >  ".join(
            "%s %.3f" % (gr, primary_am[(gr, a0)][0]) for gr in ranks[a0])))
    print("   VERDICT (ORDER): %s" % ov)

    # -- the assessment's own claim -------------------------------------------
    print("\n-- THE ASSESSMENT'S CLAIM, scored against its pre-stated arithmetic ----")
    dlay = primary_dl.get("layerwise", (float("nan"), []))[0]
    dsca = primary_dl.get("scalar", (float("nan"), []))[0]
    print("   claim: layerwise 69.532 -> 44.968 (DELTA about -24.6) with scalar flat")
    print("   test : DELTA(layerwise) <= %.1f  AND  |DELTA(scalar)| <= %.1f"
          % (ASSESS_LAYER_BAR, ASSESS_SCALAR_BAR))
    print("   got  : DELTA(layerwise) = %+.4f   DELTA(scalar) = %+.4f" % (dlay, dsca))
    av = ("ASSESSMENT-REPRODUCES"
          if (dlay <= ASSESS_LAYER_BAR and abs(dsca) <= ASSESS_SCALAR_BAR)
          else "ASSESSMENT-DOES-NOT-REPRODUCE")
    print("   VERDICT (ASSESSMENT): %s" % av)

    # -- secondary: c1b extension for layerwise -------------------------------
    print("\n-- SECONDARY: layerwise with `%s` added (never pooled into the primary)"
          % SECONDARY_BATCH)
    ext = list(PRIMARY_BATCHES) + [SECONDARY_BATCH]
    dl3 = deltas(g5, ext)
    if "layerwise" in dl3:
        print("   DELTA(layerwise) over %s = %+.4f  [%s]"
              % (ext, dl3["layerwise"][0],
                 "  ".join("%s %+.4f" % (b, d) for b, d in dl3["layerwise"][1])))

    # -- where 44.968 comes from ----------------------------------------------
    print("\n-- WHERE AN UNMATCHED QUERY LANDS: layerwise at alpha0=1e-3 ------------")
    print("   %-52s %5s %10s" % ("pool", "n", "mean p5"))
    for label, n, m in contaminated_pools(rows):
        print("   %-52s %5d %10.4f" % (label, n, m))
    print("   (the assessment reported n=39, mean 44.968)")

    # -- the ms aliasing ------------------------------------------------------
    print("\n-- THE meta_stepsize AXIS ON CIFAR-100: ALIASED, NOT MAPPED ------------")
    c100 = [r for r in rows if r["dataset"] == "CIFAR100"]
    sc = [r for r in c100 if r["granularity"] == "scalar"]
    ms4 = [r for r in c100 if r["meta_stepsize"] == "1e-4"]
    print("   CIFAR-100 scalar rows: %d, of which ms=1e-3: %d"
          % (len(sc), sum(1 for r in sc if r["meta_stepsize"] == "1e-3")))
    print("   CIFAR-100 ms=1e-4 rows: %d, of which alpha0=1e-3: %d"
          % (len(ms4), sum(1 for r in ms4 if r["alpha0"] == A_HI)))
    print("   ms=1e-4 granularities: %s"
          % sorted(set(r["granularity"] for r in ms4)))
    print("   => no `scalar`, no `layerwise`, no `resnet18_blocks`, no cut-position")
    print("      arm exists at ms=1e-4 on CIFAR-100, and every ms=1e-4 row sits at")
    print("      alpha0=1e-3.  ms and alpha0 are PERFECTLY ALIASED off ms=1e-3.")
    print("      NO ms CLAIM IS ISSUED BY THIS FILE.")
    if ms4:
        best = max(float(r["plateau5"]) for r in ms4 if r["plateau5"] not in ("", "?"))
        print("   best plateau5 anywhere in the CIFAR-100 ms=1e-4 stratum: %.4f" % best)

    # -- NOT-A-PLATEAU ---------------------------------------------------------
    print("\n-- NOT-A-PLATEAU (printed for completeness, gates nothing) -------------")
    bt = collections.defaultdict(list)
    for r in rows:
        if in_stratum(r) and batch_of(r) in PRIMARY_BATCHES:
            bt[(r["granularity"], r["alpha0"])].append(float(r["best_test"]))
    for gr in PRIMARY_GRANS:
        print("   %-16s best_test a0=1e-6 %8.4f   a0=1e-3 %8.4f"
              % (gr, mean(bt[(gr, A_LO)]), mean(bt[(gr, A_HI)])))

    # -- scope -----------------------------------------------------------------
    print("\n-- SCOPE, mandatory with every number above ---------------------------")
    print("   CIFAR-100 / ResNet18_c100 / SGDm+Lion / ms 1e-3 / 100 epochs / batch 100")
    print("   / AUGMENT=1 / box -15:-2.3026 / HIER unset.  alpha0 in {1e-6, 1e-3}")
    print("   ONLY -- 1e-4 does not exist for these three arms on CIFAR-100.  Says")
    print("   NOTHING about CIFAR-10, Tiny-ImageNet or ImageNet-489; nothing about")
    print("   any hierarchical or shrinkage operator; nothing about cut position")
    print("   (the `[k,62-k]` arms are a different axis and are EXCLUDED); nothing")
    print("   about the 20-epoch budget, where the alpha0 x granularity table is")
    print("   n=1/cell and is already recorded CONFIRMED in docs/MASTER-TABLE.md.")
    print("   NOT A BLIND TEST -- see the RULE 21 block in this file's header.")
    print("")
    print("   FINAL: %s | ORDER %s | %s | PREMISE %s"
          % (primary_verdict, ov, av, "PASS" if premise else "CENSUS-CHANGED"))
    return 0


# -------------------------------------------------------------- selftest ----
def selftest(csv_path):
    fails = []

    def ok(label, cond):
        print("   [%s] %s" % ("PASS" if cond else "FAIL", label))
        if not cond:
            fails.append(label)

    rows = load(csv_path)
    print("cU1 selftest -- every constant re-derived from the corpus, none quoted")
    print("corpus rows: %d" % len(rows))

    print("\n-- the frozen census ---------------------------------------------------")
    live = census(rows)
    ok("stratum census equals CENSUS_100 (corpus-conditional premise)",
       live == CENSUS_100)
    ok("the primary block is balanced: c100 n=2 and c100b n=3 in all 6 cells",
       all(live.get(("c100", gr, a0)) == 2 and live.get(("c100b", gr, a0)) == 3
           for gr in PRIMARY_GRANS for a0 in (A_LO, A_HI)))
    ok("`cpk3` cannot enter this stratum (772 epochs, granularity `[k,62-k]`)",
       not any(in_stratum(r) for r in rows if batch_of(r) == "cpk3"))

    print("\n-- the noise floor -----------------------------------------------------")
    s5, df5, c5, m5, sg5 = pooled_sd(cells(rows, "plateau5"))
    st_, dft, ct, mt, sgt = pooled_sd(cells(rows, "final_train"))
    ok("SIGMA_W re-derives to the frozen literal %.6f (got %.6f)" % (SIGMA_W, s5),
       abs(s5 - SIGMA_W) < 5e-6)
    ok("SIGMA_W df=%d cells=%d members=%d, 0 singleton cells" % (df5, c5, m5),
       (df5, c5, m5, sg5) == (SIGMA_W_DF, SIGMA_W_CELLS, SIGMA_W_MEMBERS, 0))
    ok("SIGMA_TRAIN re-derives to %.6f (got %.6f)" % (SIGMA_TRAIN, st_),
       abs(st_ - SIGMA_TRAIN) < 5e-6)

    print("\n-- the bars are arithmetic, not literals -------------------------------")
    ok("MAIN_BAR = 2*SIGMA_W*sqrt(0.4166667) = %.6f" % MAIN_BAR,
       abs(MAIN_BAR - 2 * SIGMA_W * math.sqrt(5.0 / 12)) < 1e-9)
    ok("INTERACTION_BAR = sqrt(2)*MAIN_BAR = %.6f" % INTERACTION_BAR,
       abs(INTERACTION_BAR - math.sqrt(2) * MAIN_BAR) < 1e-9)
    ok("ARGMAX_BAR = 2*SIGMA_W*sqrt(2*0.2083333) = %.6f" % ARGMAX_BAR,
       abs(ARGMAX_BAR - 2 * SIGMA_W * math.sqrt(5.0 / 12)) < 1e-9)

    print("\n-- the CIFAR-100 ms aliasing, re-derived -------------------------------")
    c100 = [r for r in rows if r["dataset"] == "CIFAR100"]
    sc = [r for r in c100 if r["granularity"] == "scalar"]
    ms4 = [r for r in c100 if r["meta_stepsize"] == "1e-4"]
    ok("CIFAR-100 has exactly %d scalar rows" % N_C100_SCALAR_ROWS,
       len(sc) == N_C100_SCALAR_ROWS)
    ok("every CIFAR-100 scalar row sits at ms=1e-3",
       sum(1 for r in sc if r["meta_stepsize"] == "1e-3") == N_C100_SCALAR_MS1E3)
    ok("CIFAR-100 has exactly %d ms=1e-4 rows" % N_C100_MS1E4_ROWS,
       len(ms4) == N_C100_MS1E4_ROWS)
    ok("every CIFAR-100 ms=1e-4 row sits at alpha0=1e-3",
       sum(1 for r in ms4 if r["alpha0"] == A_HI) == N_C100_MS1E4_AT_A0_1E3)
    ok("ms=1e-4 granularities are exactly %s" % (MS1E4_GRANS,),
       tuple(sorted(set(r["granularity"] for r in ms4))) == MS1E4_GRANS)
    ok("no primary-set arm exists at ms=1e-4 on CIFAR-100",
       not any(r["granularity"] in PRIMARY_GRANS for r in ms4))

    print("\n-- the verdict map is total ---------------------------------------------")
    fake = {"scalar": (0.0, []), "resnet18_blocks": (0.0, []), "layerwise": (-30.0, [])}
    ok("a 30 pp layerwise-only swing returns INTERACTION",
       verdict(fake, MAIN_BAR, INTERACTION_BAR)[0] == "INTERACTION-ALPHA0-x-GRANULARITY")
    flat = {gr: (-5.0, []) for gr in PRIMARY_GRANS}
    ok("a common -5 pp shift returns MAIN-EFFECT-ALPHA0-ONLY",
       verdict(flat, MAIN_BAR, INTERACTION_BAR)[0] == "MAIN-EFFECT-ALPHA0-ONLY")
    null = {gr: (0.01, []) for gr in PRIMARY_GRANS}
    ok("an all-null table returns NO-RESOLVED-ALPHA0-EFFECT",
       verdict(null, MAIN_BAR, INTERACTION_BAR)[0] == "NO-RESOLVED-ALPHA0-EFFECT")
    ok("a two-arm table returns UNRESOLVED-THIN",
       verdict({"scalar": (0.0, [])}, MAIN_BAR, INTERACTION_BAR)[0] == "UNRESOLVED-THIN")
    am_flip = {("scalar", A_LO): (10.0, 2), ("resnet18_blocks", A_LO): (20.0, 2),
               ("layerwise", A_LO): (30.0, 2),
               ("scalar", A_HI): (30.0, 2), ("resnet18_blocks", A_HI): (20.0, 2),
               ("layerwise", A_HI): (10.0, 2)}
    ok("a reversed ranking returns ORDER-FLIPS",
       order_verdict(am_flip)[0] == "ORDER-FLIPS")

    print("\n-- the assessment test is arithmetic ------------------------------------")
    ok("the reproduce bars are the frozen literals -20.0 / 1.0",
       (ASSESS_LAYER_BAR, ASSESS_SCALAR_BAR) == (-20.0, 1.0))

    print("=" * 74)
    if fails:
        print("SELFTEST FAIL -- %d checks failed" % len(fails))
        for f in fails:
            print("   %s" % f)
        return 1
    print("SELFTEST PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest(a.csv) if a.selftest else score(a.csv))


if __name__ == "__main__":
    main()
