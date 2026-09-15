#!/usr/bin/env python3
# =============================================================================
# cY1_cru1_score.py -- THE REGISTERED SCORER FOR BATCH `cru1`,
#     RULE 11 ON THE HEADLINE CIFAR-100 CELL: THE granularity x meta-stepsize
#     x alpha0 CROSSING THAT BREAKS THE ALIAS CORRECTIONS 160 MEASURED.
#
# STANDING RULE 21: this file is git-committed BEFORE any `cru1` run exists.
# STANDING RULE 16: once committed it is NEVER edited.  If it is wrong it is
# wrong in public and a SUCCESSOR file says so.
#
# RUN
#   python3 analysis/cY1_cru1_score.py --selftest
#   python3 analysis/cY1_cru1_score.py <runsdir>      <-- THE FULL, DOCUMENTED
#                                                         INVOCATION.  There is
#                                                         NO second argument to
#                                                         forget.  Everything
#                                                         optional is DEFAULTED
#                                                         to this batch's own
#                                                         paths (CORRECTIONS
#                                                         164.2's defect, closed
#                                                         by construction).
#   python3 analysis/cY1_cru1_score.py <runsdir> --probe-dir <path>   (override)
#
# -----------------------------------------------------------------------------
# WHY THIS BATCH EXISTS -- THE THREE FACTS THAT MOTIVATE IT, EACH RE-DERIVED
# FROM THE LIVE CORPUS BY --selftest AND NONE QUOTED FROM PROSE
# -----------------------------------------------------------------------------
# (1) RULE 11 -- compare tuned arms at their own optima -- is OPEN on the exact
#     cell every recent headline comes from.  On CIFAR-100 the corpus carries
#     EXACTLY TWO meta-stepsize levels, `1e-3` and `1e-4`, and the axis is
#     PERFECTLY ALIASED with `alpha0`: every one of the 42 CIFAR-100 `scalar`
#     rows, all 101 `layerwise` rows and all 38 `resnet18_blocks` rows sit at
#     `ms=1e-3`, and the ENTIRE `ms=1e-4` stratum is 20 rows, ALL at
#     `alpha0=1e-3`, whose granularities are ONLY {chunk771, chunk2293,
#     nodewise, nodewise1d} -- no scalar, no layerwise, no blk6, no
#     cut-position arm.  Section A re-derives all of that.
#
# (2) THE ALIAS IS MECHANICALLY FORCED, NOT A SCHEDULING ACCIDENT, and this
#     scorer says so with arithmetic taken off the live source.  In
#     `HF.Lion_meta_update` the meta step is
#
#         self.beta[i] = (1-ms*wd_meta)*self.beta[i] - ms * torch.sign(...)
#
#     and every `cru1` job passes `--weight-decay-meta 0`, so |dbeta| = ms
#     EXACTLY, once per minibatch.  100 epochs x 500 minibatches = 50,000
#     meta-steps, so the TOTAL distance beta can travel is TRAVEL = 50000*ms,
#     whatever the loss surface does.  With `BETA_CLIP=-15:-2.3026` the ceiling
#     sits D_up = -2.3026 - ln(alpha0) above the start.  At alpha0=1e-6,
#     D_up = 11.513 and ms=1e-4 gives TRAVEL = 5.0: alpha can NEVER exceed
#     exp(-13.8155+5) = 1.484e-4 in the whole run.  That is why the corpus has
#     no ms=1e-4 row at alpha0=1e-6.  A LADDER THAT VARIES ONLY ms CANNOT BREAK
#     THIS ALIAS; `cru1` crosses both axes.  Section B re-derives every rung.
#
# (3) THE HEADLINE CELL IS NOT AT ANYONE'S OPTIMUM, AND THE CORPUS ALREADY SAYS
#     SO.  The best CIFAR-100 plateau5 anywhere in the corpus is 72.408
#     (`gm2-ch-s1`, chunk771) and the whole top of the table sits at
#     **ms=1e-4 / alpha0=1e-3** -- the same cell in every other respect
#     (SGDm/Lion, gamma 1, AUGMENT=1, clip -15:-2.3026, hier unset, batch 100,
#     100 epochs, ResNet18_c100).  At that rung chunk771 = 71.995,
#     chunk2293 = 72.000, nodewise1d = 71.932, nodewise = 70.422 -- EVERY ONE
#     above the headline `layerwise` 69.5945 / 69.5321 at ms=1e-3.  The
#     headline arms have never been run there.  Section C re-derives it.
#
#     And on CIFAR-10, where the same contrast HAS been laddered, the
#     granularity gap COLLAPSES under per-arm tuning: layerwise - scalar is
#     +3.371 pp at the shared ms=1e-3 (a0=1e-3) and +0.624 pp at ms=1e-4 where
#     BOTH arms peak -- a 5.4x shrink.  `hz9` reported the same phenomenon
#     reversing sign on a different contrast.  Nobody has done this on
#     CIFAR-100, where the gap is +47 pp.
#
# -----------------------------------------------------------------------------
# THE DESIGN.  2 arms x 10 (ms, alpha0) cells x 3 seeds = 60 jobs, ONE
# submission, CIFAR-100 / ResNet18_c100, 100 epochs, PROBE=100.
# -----------------------------------------------------------------------------
#   arms   `sc`  = scalar     (m = 1)
#          `lay` = layerwise  (m = 62)
#   LADDER A, alpha0 = 1e-3 : ms in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}
#   LADDER B, alpha0 = 1e-6 : ms in {      1e-4, 3e-4, 1e-3, 3e-3}
#   seeds  {15, 16, 17} -- ZERO rows anywhere in the corpus carry them.
#
#   THE ALIAS IS BROKEN BY CONSTRUCTION: FOUR ms levels {1e-4, 3e-4, 1e-3,
#   3e-3} appear at BOTH alpha0 levels for BOTH arms.  Gate G2 proves the
#   crossing is actually realised in the landed runs rather than asserted here.
#
# WHAT WAS DROPPED, AND WHAT THAT COSTS.  `resnet18_blocks` (m=6, the parent
# paper's own partition) and the `k49` cut-position arm are NOT in this batch.
# Adding either at the full ladder is 30 more jobs and ~22 more GPU-hours, which
# does not fit a three-way-shared 24 h window; the standing instruction is to
# prioritise BRACKETING each arm's optimum over adding arms.  CONSEQUENCE,
# REGISTERED IN ADVANCE AND BINDING: `cru1` closes RULE 11 for the
# scalar-vs-layerwise contrast ONLY.  It does NOT close it for blk6, for cut
# position, for class count or for ImageNet, and NO verdict produced by this
# file may be quoted as if it did.
#
# -----------------------------------------------------------------------------
# WHY PROBE=100.  The central confound of this design is that lowering ms does
# two things at once: it tunes the meta-optimiser AND it mechanically limits how
# far beta can descend.  CORRECTIONS 155 measured, on CIFAR-10 at ms=3e-4, that
# the scalar arm's single beta reaches the -15 floor at meta-step ~34,100 and
# sits pinned 31.8% of the time while layerwise pins ~5.5% of coordinates.  With
# PROBE=100 this batch records 500 beta snapshots per run, so a GAP-SHRINKS
# result at a low rung can be ATTRIBUTED rather than merely reported.  The probe
# analysis is a SKIP, never a FAIL, if the files are absent: the verdict does not
# depend on it.
# =============================================================================

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "all_runs.csv")

# =============================================================================
# THE FROZEN DESIGN
# =============================================================================
BATCH = "cru1"
PREFIX = "cru1-"
NET, DSET = "ResNet18_c100", "CIFAR100"
BASE_ALG, META_ALG = "SGDm", "Lion"
GAMMA, AUG = "1", "1"
CLIP_C = "-15:-2.3026"
BATCH_SIZE = 100
EPOCHS = 100
SEEDS = (15, 16, 17)
PROBE_EVERY = 100

ARMS = ("sc", "lay")
GRAN = {"sc": "scalar", "lay": "layerwise"}

# LADDER A is alpha0=1e-3; LADDER B is alpha0=1e-6.  Ordered LOW ms -> HIGH ms.
LADDER = {
    "1e-3": ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3"),
    "1e-6": ("1e-4", "3e-4", "1e-3", "3e-3"),
}
ALPHAS = ("1e-3", "1e-6")
SHARED_MS = "1e-3"                       # the rung EVERY headline is quoted at
CROSSED_MS = ("1e-4", "3e-4", "1e-3", "3e-3")   # present at BOTH alpha0 levels

CELLS = tuple((ms, a0) for a0 in ALPHAS for ms in LADDER[a0])
N_JOBS = len(CELLS) * len(ARMS) * len(SEEDS)     # 60

# meta-steps: 50,000 train images / batch 100 = 500 per epoch.
STEPS_PER_EPOCH = 50000 // BATCH_SIZE
META_STEPS = EPOCHS * STEPS_PER_EPOCH            # 50,000
BETA_LO, BETA_HI = -15.0, -2.3026

# =============================================================================
# THE NOISE FLOOR -- RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME,
# NEVER COPIED FROM A LITERAL IN PROSE.  (sigma_w has drifted twice after a
# registration in this campaign: CORRECTIONS 156 and 159.)  Every reader below
# EXCLUDES cru1's own rows, so each of these is invariant under this batch's
# own ingest -- the property cW1 (CORRECTIONS 165) achieved and cS2 only
# claimed (161.9).
#
# Pooled within (batch x granularity x ms x alpha0) across-seed SD of plateau5,
# over the standard CIFAR-100 cell at 100 epochs.  THREE derivations:
#   AB   scalar + layerwise            0.528555   df 36, 20 cells, 56 members
#   SC   scalar only                   0.613640   df 18, 10 cells, 28 members
#   LAY  layerwise only                0.426833   df 18, 10 cells, 28 members
# THE FROZEN RULE: SIGMA_W = max of the three = 0.613640 (the scalar-only
# derivation).  Taking the max is the conservative choice and it is the arm
# whose level this batch actually moves.
# =============================================================================
SIGMA_AB, SIGMA_AB_DF, SIGMA_AB_CELLS, SIGMA_AB_MEMBERS = 0.528555, 36, 20, 56
SIGMA_SC, SIGMA_SC_DF, SIGMA_SC_CELLS, SIGMA_SC_MEMBERS = 0.613640, 18, 10, 28
SIGMA_LAY, SIGMA_LAY_DF, SIGMA_LAY_CELLS, SIGMA_LAY_MEMBERS = 0.426833, 18, 10, 28
SIGMA_W = 0.613640

# THE GATE USES max(SIGMA_W, sigma_in_batch).  Registered here, in advance, so
# that a batch noisier than the corpus cannot be scored against a bar the corpus
# earned.  sigma_in_batch is the pooled across-seed SD of the 20 landed cells.
NSEED = len(SEEDS)
SE_CELL = SIGMA_W / math.sqrt(NSEED)              # 0.354285
SE_GAP = SIGMA_W * math.sqrt(2.0 / NSEED)         # 0.501035  (lay - sc, one cell)
SE_DGAP = SIGMA_W * math.sqrt(4.0 / NSEED)        # 0.708570  (gap - gap, 4 means)
BAR_GAP = 2.0 * SE_GAP
BAR_DGAP = 2.0 * SE_DGAP

# =============================================================================
# THE CORPUS ANCHORS.  Frozen here; --selftest RE-DERIVES every one from the
# live CSV with cru1's rows excluded, and FAILS on any drift beyond 5e-4.
# =============================================================================
# CIFAR-100, the standard cell, the SHARED headline rung ms=1e-3.
C100_MS1E3 = {
    ("sc", "1e-3"): (22.560000, 5), ("lay", "1e-3"): (69.594500, 8),
    ("sc", "1e-6"): (22.795652, 23), ("lay", "1e-6"): (69.532100, 20),
}
# THE NUMBER EVERY HEADLINE USES.  Re-derived, not quoted.
G_SHARED_CORPUS = {"1e-3": 47.034500, "1e-6": 46.736448}

# CIFAR-10 / ResNet18, same cell otherwise, 100 epochs: the ONLY place in the
# corpus where this contrast has ever been laddered in ms.  (mean, n)
C10 = {
    ("sc", "1e-5", "1e-3"): (90.837600, 5), ("lay", "1e-5", "1e-3"): (91.056800, 5),
    ("sc", "3e-5", "1e-3"): (91.672000, 5), ("lay", "3e-5", "1e-3"): (91.869600, 5),
    ("sc", "1e-4", "1e-3"): (92.262400, 5), ("lay", "1e-4", "1e-3"): (92.886545, 11),
    ("sc", "3e-4", "1e-3"): (88.753250, 8), ("lay", "3e-4", "1e-3"): (91.978250, 8),
    ("sc", "1e-3", "1e-3"): (87.840500, 20), ("lay", "1e-3", "1e-3"): (91.211600, 20),
    ("sc", "1e-4", "1e-6"): (45.594000, 5), ("lay", "1e-4", "1e-6"): (46.362400, 5),
    ("sc", "3e-4", "1e-6"): (87.649200, 5), ("lay", "3e-4", "1e-6"): (90.227600, 5),
    ("sc", "1e-3", "1e-6"): (87.912800, 15), ("lay", "1e-3", "1e-6"): (90.919000, 26),
}
# NO flat-cell CIFAR-10 anchor exists at ms=3e-3 for either arm: every CIFAR-10
# 3e-3 row in the corpus carries hier=additive.  Section D proves that, and the
# 3e-3 rung is therefore registered as an EXTRAPOLATED TOP BRACKET with no point
# prediction -- its only job is to show the argmax is not at the 1e-3 endpoint.
C10_NO_ANCHOR = (("3e-3", "1e-3"), ("3e-3", "1e-6"))

# The same-cell CIFAR-100 granularities that DO exist at ms=1e-4 / alpha0=1e-3.
C100_MS1E4_OTHER = {
    "chunk771": (71.995429, 7), "chunk2293": (72.000000, 3),
    "nodewise1d": (71.932000, 3), "nodewise": (70.422000, 7),
}
CORPUS_BEST_C100_P5 = 72.408          # gm2-ch-s1, chunk771, ms=1e-4, a0=1e-3
CORPUS_BEST_C100_BEST_TEST = 72.80    # gc1-ch-s3, chunk771, ms=1e-4, a0=1e-3

# =============================================================================
# THE REGISTERED ACCOUNTS.  Three, and they make DIFFERENT predictions.
# =============================================================================
# ACCOUNT T -- TUNING-IRRELEVANT.  The granularity gap is a property of the
#   partition and survives per-arm tuning: at each arm's own optimal ms the gap
#   is still at least half what it is at the shared ms=1e-3.
#       T predicts  RATIO = G_tuned / G_shared >= 0.5  at BOTH alpha0.
#
# ACCOUNT D -- DESCENT-LIMIT.  The gap at ms=1e-3 is produced by the scalar
#   arm's single beta descending to the clamp floor while layerwise's 62
#   coordinates do not (CORRECTIONS 155 measured exactly that on CIFAR-10:
#   31.8% coordinate-occupancy at the floor for scalar, 5.5% for layerwise).
#   Because |dbeta| = ms exactly, the descent is TRAVEL-limited, so the gap
#   should be a MONOTONE INCREASING function of TRAVEL = 50000*ms and should
#   fall towards zero once TRAVEL is too small for beta to cross the clip.
#       D predicts  RATIO <= 0.5, gap monotone non-decreasing in ms within each
#       alpha0 ladder, and gap <= 2*SE_GAP at every FROZEN rung.
#
# ACCOUNT F -- GRANULARITY-INOPERATIVE-AT-FROZEN.  A NEGATIVE CONTROL that both
#   T and D must respect.  At TRAVEL/D_up < 0.5 (rungs 1e-5 and 3e-5 at
#   alpha0=1e-3) beta is confined to a narrow interval around beta0 and BOTH
#   arms are within a factor of a few of a FIXED step size alpha0=1e-3.  The
#   granularity variable then has almost nothing to act on.
#       F predicts  |gap| <= 2*SE_GAP at ms in {1e-5, 3e-5} / alpha0=1e-3.
#   On CIFAR-10 at the two comparable frozen rungs at alpha0=1e-6 the measured
#   gaps are +0.004 pp (ms=1e-5) and +0.027 pp (ms=3e-5): F has a precedent.
#   IF F FAILS -- if a large gap survives where beta cannot move -- then the gap
#   is NOT a step-size-adaptation phenomenon at all and BOTH T and D are
#   incomplete.  That is the most consequential single outcome available here.
#
# THE TWO POINT-PREDICTORS, registered separately because they DISAGREE and the
# disagreement is itself the interesting quantity.
#   PREDICTOR X -- CIFAR-10 ERROR-RATIO TRANSFER.
#       R(arm) = mean over alpha0 of (100 - C100(arm, ms=1e-3, a0))
#                                  / (100 -  C10(arm, ms=1e-3, a0))
#       pred(arm, rung) = 100 - R(arm) * (100 - C10(arm, rung))
#     Declared VALID only where the CIFAR-10 anchor is >= 85 pp; outside that
#     the ratio form is out of range and returns nonsense (it predicts a
#     NEGATIVE accuracy at ms=1e-4/alpha0=1e-6), so those rungs get NO point
#     prediction and carry NO branch weight.  Section E re-derives R and checks
#     the model reproduces its own calibration rung to within 0.5 pp.
#   PREDICTOR Y -- CORPUS-INTERNAL, SAME CELL.  At ms=1e-4 / alpha0=1e-3 the
#     corpus already has four FINER partitions in the identical cell, spanning
#     70.422 .. 72.000.  Y predicts layerwise lands in [69.594, 72.408] there --
#     at least its own ms=1e-3 level, at most the corpus best.
#   X and Y disagree on that one cell by ~3.4 pp (X says 75.761, Y caps at
#   72.408).  BOTH are reported.  Neither adjudicates a branch on its own.
R_SC = 6.37798216504928      # re-derived, full precision; [E1] re-checks at 1e-9
R_LAY = 3.4074285482463496
X_VALID_MIN_C10 = 85.0
Y_LAY_MS1E4_LO, Y_LAY_MS1E4_HI = 69.594500, 72.408

# =============================================================================
# THE BRANCH MAP.  Evaluated SEPARATELY at each alpha0, then jointly.
# Precedence is top to bottom: the first branch whose condition holds wins.
# =============================================================================
#  E0  ADAPTATION-NOT-NEEDED     the arm's argmax rung is FROZEN
#                                (TRAVEL/D_up < 0.5).  TERMINAL, not
#                                unresolved: the rungs below a frozen rung are
#                                mechanically identical (alpha is confined to a
#                                shrinking interval around alpha0), so there is
#                                nothing left to bracket.  It means a nearly
#                                FIXED step size beats the meta-learned one.
#  E1  UNRESOLVED-OPTIMUM-AT-LADDER-EDGE
#                                the arm's argmax is a NON-frozen endpoint of
#                                its own ladder.  NO tuned comparison may be
#                                claimed for that alpha0.
#  B1  GAP-REVERSES              G_tuned < -BAR_GAP        (hz9's outcome)
#  B2  GAP-CLOSES                |G_tuned| <= BAR_GAP
#  B3  GAP-SHRINKS               G_tuned > BAR_GAP and
#                                G_shared - G_tuned > BAR_DGAP
#  B4  GAP-SURVIVES-TUNING       G_tuned > BAR_GAP and
#                                G_shared - G_tuned <= BAR_DGAP
#
# STAMPS, reported ALONGSIDE the branch and never as the branch:
#  TRAVEL-CONFOUNDED   the winning rung for either arm has TRAVEL/D_up < 1
#                      (beta cannot reach the clamp ceiling), so "the arm was
#                      tuned" and "the arm was mechanically prevented from
#                      collapsing" are not separated by the level alone.
#  MONOTONE-IN-TRAVEL / NOT-MONOTONE   Account D's shape test.
#  F-HOLDS / F-FAILS   the frozen-rung negative control.
#  ALIAS-BROKEN / ALIAS-NOT-BROKEN     G2's crossing check.
# =============================================================================
FROZEN_RATIO = 0.5      # TRAVEL/D_up below this is FROZEN
T_RATIO_MIN = 0.5       # Account T's registered threshold

# =============================================================================
# .out parsing.  Identical regexes to cW1 (CORRECTIONS 165), unchanged.
# =============================================================================
EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
NAME_RE = re.compile(
    r"^cru1-(sc|lay)-m(1e-5|3e-5|1e-4|3e-4|1e-3|3e-3)-a(1e-3|1e-6)-s(15|16|17)-(\d+)\.out$")

# Every non-axis flag must be identical across all 60 runs.  The AXES are
# exactly these three; everything else is invariant.
AXIS_FLAGS = ("stepsize-groups", "meta-stepsize", "alpha0", "seed", "run-name")


# =============================================================================
# helpers
# =============================================================================
def travel(ms):
    return META_STEPS * float(ms)


def d_up(a0):
    return BETA_HI - math.log(float(a0))


def d_down(a0):
    return math.log(float(a0)) - BETA_LO


def reach(ms, a0):
    """(TRAVEL, D_up, D_down, alpha_max, alpha_min, status) -- pure arithmetic
    off |dbeta| = ms exactly, the clip, and the meta-step count."""
    b0 = math.log(float(a0))
    T = travel(ms)
    du, dd = d_up(a0), d_down(a0)
    amax = math.exp(min(BETA_HI, b0 + T))
    amin = math.exp(max(BETA_LO, b0 - T))
    r = T / du
    if r < FROZEN_RATIO:
        st = "FROZEN"
    elif r < 1.0:
        st = "STARVED"
    elif r < 3.0:
        st = "ADEQUATE"
    else:
        st = "FREE"
    return T, du, dd, amax, amin, st


def pred_x(arm, ms, a0):
    """Predictor X.  Returns (value, 'VALID'|reason)."""
    key = (arm, ms, a0)
    if key not in C10:
        return None, "NO-CIFAR-10-ANCHOR"
    c10 = C10[key][0]
    if c10 < X_VALID_MIN_C10:
        return None, "OUT-OF-RANGE (C10 %.3f < %.1f)" % (c10, X_VALID_MIN_C10)
    R = R_SC if arm == "sc" else R_LAY
    return 100.0 - R * (100.0 - c10), "VALID"


def tail5(d, budget):
    v = [d[e] for e in range(budget - 5, budget) if e in d]
    return (sum(v) / 5.0) if len(v) == 5 else None


def parse_argsline(line):
    """[(flag, value)] in order.  Values may be quoted; flags are --x."""
    toks = line.split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            f = t[2:]
            if "=" in f:
                k, v = f.split("=", 1)
                out.append((k, v.strip("'\"")))
                i += 1
                continue
            v = ""
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v = toks[i + 1].strip("'\"")
                i += 1
            out.append((f, v))
        i += 1
    return out


def parse_envline(line):
    out = {}
    for m in re.finditer(r"([A-Z_][A-Z0-9_]*)=(\S*)", line):
        out[m.group(1)] = m.group(2)
    return out


def load_runs(runsdir):
    recs = []
    pats = [os.path.join(runsdir, PREFIX + "*.out"),
            os.path.join(runsdir, BATCH, PREFIX + "*.out")]
    files = sorted(set(sum([glob.glob(p) for p in pats], [])))
    for f in files:
        m = NAME_RE.match(os.path.basename(f))
        if not m:
            recs.append({"file": f, "bad_name": True})
            continue
        arm, ms, a0, seed, job = (m.group(1), m.group(2), m.group(3),
                                  int(m.group(4)), m.group(5))
        te, tr, args, env, done, tb = {}, {}, None, None, False, 0
        with open(f, errors="ignore") as fh:
            for line in fh:
                a = ARGS_RE.match(line)
                if a and args is None:
                    args = a.group(1)
                e = ENV_RE.match(line)
                if e and env is None:
                    env = e.group(1).strip()
                if line.strip() == "RUN_DONE":
                    done = True
                if "Traceback (most recent call last)" in line:
                    tb += 1
                x = EP_RE.search(line)
                if x:
                    te[int(x.group(1))] = float(x.group(2))
                y = EPTR_RE.search(line)
                if y:
                    tr[int(y.group(1))] = float(y.group(2))
        occ = parse_argsline(args) if args else []
        cnt = collections.Counter(k for k, _ in occ)
        recs.append({
            "file": f, "bad_name": False, "arm": arm, "ms": ms, "a0": a0,
            "seed": seed, "job": job, "args": args, "env": env, "occ": occ,
            "eff": dict(occ), "envd": parse_envline(env) if env else {},
            "repeated": sorted([k for k, v in cnt.items() if v > 1]),
            "run_done": done, "traceback": tb, "test": te, "train": tr,
            "n_ep": len(te), "p5": tail5(te, EPOCHS), "t5": tail5(tr, EPOCHS),
        })
    return recs


# =============================================================================
# the corpus.  EVERY reader excludes cru1's own rows, so every frozen premise
# is invariant under this batch's own ingest.
# =============================================================================
def _csv_rows(path=CSV, exclude_own=True):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r.get("superseded") or "0") == "1":
                continue
            if exclude_own and str(r.get("run") or "").startswith(PREFIX):
                continue
            out.append(r)
    return out


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _std_cell(r, ds, net):
    return (r["network"] == net and r["dataset"] == ds
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH_SIZE)
            and r["epochs_done"] == str(EPOCHS) and r["collapsed"] == "0")


def corpus_level(ds, net, gran, ms, a0, col="plateau5", path=CSV):
    v = []
    for r in _csv_rows(path):
        if (_std_cell(r, ds, net) and r["granularity"] == gran
                and r["meta_stepsize"] == ms and r["alpha0"] == a0):
            try:
                v.append(float(r[col]))
            except (TypeError, ValueError):
                pass
    return (statistics.mean(v) if v else None), len(v)


def corpus_sigma(grans, path=CSV):
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _std_cell(r, DSET, NET) or r["granularity"] not in grans:
            continue
        try:
            cells[(_batch_of(r["run"]), r["granularity"],
                   r["meta_stepsize"], r["alpha0"])].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    ss = df = nc = nm = 0
    for v in cells.values():
        nm += len(v)
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
            nc += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, nc, nm


def corpus_ms_alias(path=CSV):
    """CORRECTIONS 160's alias, re-derived.  Returns
    (ms census, ms=1e-4 stratum: alpha0 census + granularity census,
     per-granularity ms census for the three headline arms)."""
    c = [r for r in _csv_rows(path) if r["dataset"] == DSET]
    ms_c = collections.Counter(r["meta_stepsize"] for r in c)
    lo = [r for r in c if r["meta_stepsize"] == "1e-4"]
    lo_a0 = collections.Counter(r["alpha0"] for r in lo)
    lo_g = collections.Counter(r["granularity"] for r in lo)
    per = {}
    for g in ("scalar", "layerwise", "resnet18_blocks"):
        per[g] = collections.Counter(r["meta_stepsize"] for r in c
                                     if r["granularity"] == g)
    return ms_c, lo_a0, lo_g, per


def corpus_c10_3e3_hier(path=CSV):
    """Every CIFAR-10 ms in {3e-3} row for the two arms, with its hier value.
    Registration claims: the flat cell has NONE."""
    out = []
    for r in _csv_rows(path):
        if (r["dataset"] == "CIFAR10" and r["network"] == "ResNet18"
                and r["granularity"] in ("scalar", "layerwise")
                and r["meta_stepsize"] == "3e-3"):
            out.append((r["run"], r["granularity"], r.get("hier") or "",
                        r["alpha0"]))
    return out


def corpus_best_c100(path=CSV):
    best_p5 = best_bt = None
    for r in _csv_rows(path):
        if r["dataset"] != DSET:
            continue
        try:
            p = float(r["plateau5"])
            if best_p5 is None or p > best_p5[0]:
                best_p5 = (p, r["run"], r["granularity"], r["meta_stepsize"],
                           r["alpha0"])
        except (TypeError, ValueError):
            pass
        try:
            b = float(r["best_test"])
            if best_bt is None or b > best_bt[0]:
                best_bt = (b, r["run"], r["granularity"], r["meta_stepsize"],
                           r["alpha0"])
        except (TypeError, ValueError):
            pass
    return best_p5, best_bt


def corpus_seed_collisions(path=CSV):
    return sum(1 for r in _csv_rows(path, exclude_own=False)
               if r["network"] == NET and str(r.get("seed")) in
               tuple(str(s) for s in SEEDS))


def count_own_rows(path=CSV):
    with open(path) as fh:
        return sum(1 for r in csv.DictReader(fh)
                   if str(r.get("run") or "").startswith(PREFIX))


# =============================================================================
# probe -- OPTIONAL.  Absent files are a SKIP, never a FAIL.
# =============================================================================
def resolve_probe_dir(runsdir, explicit=None):
    if explicit:
        return explicit, "given explicitly with --probe-dir"
    if not runsdir:
        return None, "no runsdir"
    up = os.path.dirname(os.path.abspath(runsdir.rstrip("/")))
    for p, how in ((os.path.join(runsdir, BATCH),
                    "DEFAULT <runsdir>/%s" % BATCH),
                   (os.path.join(up, BATCH), "DEFAULT <runsdir>/../%s" % BATCH),
                   (runsdir, "DEFAULT <runsdir>")):
        if os.path.isdir(p) and glob.glob(os.path.join(p, "probe_*")):
            return p, how
    return None, "no probe_* directory found under the defaults"


def probe_summary(probe_root):
    """{run: (min_beta, frac_records_with_a_pinned_low_coord, n_records)}."""
    out = {}
    if not probe_root:
        return out
    for d in sorted(glob.glob(os.path.join(probe_root, "probe_*"))):
        run = os.path.basename(d)[len("probe_"):]
        f = os.path.join(d, "probe.jsonl")
        if not os.path.isfile(f):
            continue
        mn, n, pin = None, 0, 0
        with open(f, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                n += 1
                b = j.get("beta_min", j.get("min_beta"))
                lo = j.get("n_at_lo", j.get("n_lo"))
                if b is not None:
                    b = float(b)
                    mn = b if mn is None else min(mn, b)
                if lo is not None and float(lo) >= 1:
                    pin += 1
        if n:
            out[run] = (mn, pin / float(n), n)
    return out


# =============================================================================
# formatting
# =============================================================================
def fmt(x, w=9, p=4):
    return ("%*.*f" % (w, p, x)) if isinstance(x, float) else ("%*s" % (w, x))


def se_str(x, se):
    return "%+.4f pp (%+.2f SE)" % (x, x / se) if se else "%+.4f pp" % x


def close(a, b, tol=5e-4):
    return a is not None and b is not None and abs(a - b) <= tol


# =============================================================================
# --selftest.  Re-derives EVERY frozen number from the live corpus, from
# arithmetic, or from the design, with cru1's own rows EXCLUDED.
# =============================================================================
def selftest(path=CSV):
    P = F = 0

    def chk(ok, label, detail=""):
        nonlocal P, F
        if ok:
            P += 1
            print("  PASS  %s %s" % (label, detail))
        else:
            F += 1
            print("  FAIL  %s %s" % (label, detail))

    print("=" * 78)
    print("cY1_cru1_score.py --selftest")
    print("=" * 78)

    # ---- A: CORRECTIONS 160's alias, RE-DERIVED --------------------------
    print("\n[A] THE ALIAS ON CIFAR-100, RE-DERIVED FROM THE LIVE CORPUS")
    ms_c, lo_a0, lo_g, per = corpus_ms_alias(path)
    print("      CIFAR-100 meta_stepsize census: %s" % dict(ms_c))
    chk(set(ms_c) == {"1e-3", "1e-4"},
        "A1 CIFAR-100 carries EXACTLY two ms levels",
        "-> %s" % sorted(ms_c))
    print("      the ms=1e-4 stratum: %d rows, alpha0 %s"
          % (sum(lo_a0.values()), dict(lo_a0)))
    chk(set(lo_a0) == {"1e-3"},
        "A2 the ENTIRE ms=1e-4 stratum is at alpha0=1e-3",
        "-> %s" % dict(lo_a0))
    print("      its granularities: %s" % dict(lo_g))
    chk(not ({"scalar", "layerwise", "resnet18_blocks"} & set(lo_g)),
        "A3 no scalar / layerwise / blk6 anywhere at ms=1e-4",
        "-> %s" % sorted(lo_g))
    chk(not any(g.startswith("sets:") or re.match(r"^\[\d+,\d+\]$", g)
                for g in lo_g),
        "A4 no cut-position arm anywhere at ms=1e-4", "-> %s" % sorted(lo_g))
    for g in ("scalar", "layerwise", "resnet18_blocks"):
        chk(set(per[g]) == {"1e-3"},
            "A5 every CIFAR-100 %-15s row is at ms=1e-3" % g,
            "(n=%d) -> %s" % (sum(per[g].values()), dict(per[g])))

    # ---- B: the reachability arithmetic -----------------------------------
    print("\n[B] REACHABILITY -- |dbeta| = ms EXACTLY, %d meta-steps, clip "
          "[%g, %g]" % (META_STEPS, BETA_LO, BETA_HI))
    chk(META_STEPS == 50000, "B0 meta-step count",
        "= %d epochs x %d steps = %d" % (EPOCHS, STEPS_PER_EPOCH, META_STEPS))
    print("      %-6s %-6s %9s %8s %8s %11s %11s  %s"
          % ("ms", "a0", "TRAVEL", "D_up", "T/D_up", "alpha_max", "alpha_min",
             "status"))
    for a0 in ALPHAS:
        for ms in LADDER[a0]:
            T, du, dd, amx, amn, st = reach(ms, a0)
            print("      %-6s %-6s %9.2f %8.3f %8.3f %11.4e %11.4e  %s"
                  % (ms, a0, T, du, T / du, amx, amn, st))
    T, du, _, amx, _, st = reach("1e-4", "1e-6")
    chk(st in ("FROZEN", "STARVED") and amx < 2.0e-4,
        "B1 the alias is MECHANICALLY FORCED: ms=1e-4 at alpha0=1e-6 can never",
        "reach alpha > %.4e (T/D_up = %.3f, %s)" % (amx, T / du, st))
    fr = [(ms, a0) for a0 in ALPHAS for ms in LADDER[a0]
          if reach(ms, a0)[5] == "FROZEN"]
    chk(set(fr) == {("1e-5", "1e-3"), ("3e-5", "1e-3"), ("1e-4", "1e-6")},
        "B2 the FROZEN rungs are exactly the registered three", "-> %s" % fr)
    fr3 = [(ms, a0) for a0 in ALPHAS for ms in LADDER[a0]
           if reach(ms, a0)[5] == "FREE"]
    chk(len(fr3) == 5, "B3 five FREE rungs (beta unconstrained)", "-> %s" % fr3)

    # ---- C: the corpus's own best CIFAR-100 number ------------------------
    print("\n[C] THE CORPUS'S OWN BEST CIFAR-100 NUMBER, RE-DERIVED")
    bp, bb = corpus_best_c100(path)
    print("      best plateau5 : %.4f  %s  (%s, ms=%s, a0=%s)" % bp)
    print("      best best_test: %.4f  %s  (%s, ms=%s, a0=%s)" % bb)
    chk(close(bp[0], CORPUS_BEST_C100_P5, 1e-3),
        "C1 best CIFAR-100 plateau5 = %.3f" % CORPUS_BEST_C100_P5,
        "-> %.4f (%s)" % (bp[0], bp[1]))
    chk(bp[3] == "1e-4" and bp[4] == "1e-3",
        "C2 THE CORPUS BEST IS AT ms=1e-4 / alpha0=1e-3, NOT AT THE HEADLINE",
        "rung -> ms=%s a0=%s" % (bp[3], bp[4]))
    for g, (v, n) in C100_MS1E4_OTHER.items():
        got, gn = corpus_level(DSET, NET, g, "1e-4", "1e-3", path=path)
        chk(close(got, v) and gn == n,
            "C3 same-cell %-11s at ms=1e-4/a0=1e-3" % g,
            "-> %.4f (n=%d), frozen %.4f (n=%d)" % (got, gn, v, n))
    hi_lay = max(v for v, _ in C100_MS1E4_OTHER.values())
    chk(hi_lay > C100_MS1E3[("lay", "1e-3")][0],
        "C4 the ms=1e-4 stratum BEATS the headline layerwise level",
        "%.4f > %.4f" % (hi_lay, C100_MS1E3[("lay", "1e-3")][0]))

    # ---- D: the headline gap, the CIFAR-10 ladder, and the 3e-3 gap ------
    print("\n[D] THE HEADLINE GAP AND THE ONLY EXISTING ms LADDER FOR THIS "
          "CONTRAST")
    for (arm, a0), (v, n) in sorted(C100_MS1E3.items()):
        got, gn = corpus_level(DSET, NET, GRAN[arm], "1e-3", a0, path=path)
        chk(close(got, v) and gn == n,
            "D1 CIFAR-100 %-3s a0=%-4s at the shared ms=1e-3" % (arm, a0),
            "-> %.6f (n=%d)" % (got, gn))
    for a0 in ALPHAS:
        g = (corpus_level(DSET, NET, "layerwise", "1e-3", a0, path=path)[0]
             - corpus_level(DSET, NET, "scalar", "1e-3", a0, path=path)[0])
        chk(close(g, G_SHARED_CORPUS[a0]),
            "D2 G_shared(a0=%s) -- THE NUMBER EVERY HEADLINE USES" % a0,
            "-> %+.6f pp (%.2f SE_GAP)" % (g, g / SE_GAP))
    for k, (v, n) in sorted(C10.items()):
        arm, ms, a0 = k
        got, gn = corpus_level("CIFAR10", "ResNet18", GRAN[arm], ms, a0,
                               path=path)
        chk(close(got, v) and gn == n,
            "D3 CIFAR-10 %-3s ms=%-5s a0=%-4s" % (arm, ms, a0),
            "-> %.6f (n=%d)" % (got, gn))
    g_sh10 = C10[("lay", "1e-3", "1e-3")][0] - C10[("sc", "1e-3", "1e-3")][0]
    g_tu10 = C10[("lay", "1e-4", "1e-3")][0] - C10[("sc", "1e-4", "1e-3")][0]
    print("      CIFAR-10 a0=1e-3: gap at the shared ms=1e-3 = %+.4f pp; at "
          "ms=1e-4," % g_sh10)
    print("      where BOTH arms peak, = %+.4f pp -- a %.2fx SHRINK under "
          "per-arm tuning." % (g_tu10, g_sh10 / g_tu10))
    chk(g_tu10 < 0.5 * g_sh10,
        "D4 on CIFAR-10 this very contrast ALREADY collapses under tuning",
        "%.4f -> %.4f" % (g_sh10, g_tu10))
    sc_arg = max(LADDER["1e-3"][:-1],
                 key=lambda m: C10[("sc", m, "1e-3")][0])
    lay_arg = max(LADDER["1e-3"][:-1],
                  key=lambda m: C10[("lay", m, "1e-3")][0])
    chk(sc_arg == "1e-4" and lay_arg == "1e-4",
        "D5 on CIFAR-10 BOTH arms peak at ms=1e-4, INTERIOR to this ladder",
        "-> sc %s, lay %s" % (sc_arg, lay_arg))
    rows3e3 = corpus_c10_3e3_hier(path)
    flat = [r for r in rows3e3 if not r[2].strip()]
    chk(not flat,
        "D6 NO flat-cell CIFAR-10 anchor exists at ms=3e-3 -- the top rung is",
        "an EXTRAPOLATED BRACKET with no point prediction (%d rows, all hier=%s)"
        % (len(rows3e3), sorted(set(r[2] for r in rows3e3))))

    # ---- E: the two predictors -------------------------------------------
    print("\n[E] THE TWO POINT-PREDICTORS")
    for arm, R in (("sc", R_SC), ("lay", R_LAY)):
        rs = []
        for a0 in ALPHAS:
            c1 = corpus_level("CIFAR10", "ResNet18", GRAN[arm], "1e-3", a0,
                              path=path)[0]
            c1h = corpus_level(DSET, NET, GRAN[arm], "1e-3", a0, path=path)[0]
            rs.append((100.0 - c1h) / (100.0 - c1))
        chk(close(statistics.mean(rs), R, 1e-9),
            "E1 R_%-3s re-derived" % arm,
            "-> %.6f  (per-alpha0 %s)" % (statistics.mean(rs),
                                          ["%.4f" % x for x in rs]))
    ok = True
    for arm in ARMS:
        for a0 in ALPHAS:
            v, why = pred_x(arm, "1e-3", a0)
            if v is None or abs(v - C100_MS1E3[(arm, a0)][0]) > 0.5:
                ok = False
    chk(ok, "E2 X reproduces its OWN calibration rung to within 0.5 pp",
        "(a model that cannot do that is not a model)")
    print("      %-6s %-6s %10s %10s %10s   validity" %
          ("ms", "a0", "X(sc)", "X(lay)", "X(gap)"))
    for a0 in ALPHAS:
        for ms in LADDER[a0]:
            xs, ws = pred_x("sc", ms, a0)
            xl, wl = pred_x("lay", ms, a0)
            g = ("%10.3f" % (xl - xs)) if (xs is not None and xl is not None) \
                else "%10s" % "--"
            print("      %-6s %-6s %10s %10s %s   %s"
                  % (ms, a0,
                     ("%.3f" % xs) if xs is not None else "--",
                     ("%.3f" % xl) if xl is not None else "--", g,
                     ws if ws == wl else "%s / %s" % (ws, wl)))
    xs, _ = pred_x("sc", "1e-4", "1e-3")
    xl, _ = pred_x("lay", "1e-4", "1e-3")
    print("      PREDICTOR X at the pivotal rung ms=1e-4/a0=1e-3: sc %.3f, "
          "lay %.3f, gap %+.3f" % (xs, xl, xl - xs))
    print("      PREDICTOR Y at the same rung: lay in [%.3f, %.3f] -- at least "
          "its own" % (Y_LAY_MS1E4_LO, Y_LAY_MS1E4_HI))
    print("      ms=1e-3 level, at most the corpus best.  X and Y DISAGREE by "
          "%.3f pp; both" % (xl - Y_LAY_MS1E4_HI))
    print("      are reported and NEITHER adjudicates a branch.")
    chk(xl > Y_LAY_MS1E4_HI,
        "E3 X and Y disagree at the pivotal rung -- registered as such",
        "X %.3f vs Y cap %.3f" % (xl, Y_LAY_MS1E4_HI))

    # ---- F: the noise floor ----------------------------------------------
    print("\n[F] THE NOISE FLOOR, RE-DERIVED (cru1's own rows EXCLUDED)")
    for label, grans, frozen, fdf, fc, fm in (
            ("AB  scalar+layerwise", {"scalar", "layerwise"},
             SIGMA_AB, SIGMA_AB_DF, SIGMA_AB_CELLS, SIGMA_AB_MEMBERS),
            ("SC  scalar only", {"scalar"},
             SIGMA_SC, SIGMA_SC_DF, SIGMA_SC_CELLS, SIGMA_SC_MEMBERS),
            ("LAY layerwise only", {"layerwise"},
             SIGMA_LAY, SIGMA_LAY_DF, SIGMA_LAY_CELLS, SIGMA_LAY_MEMBERS)):
        s, df, nc, nm = corpus_sigma(grans, path)
        chk(close(s, frozen, 1e-5) and df == fdf and nc == fc and nm == fm,
            "F1 sigma %s" % label,
            "-> %.6f df %d cells %d members %d" % (s, df, nc, nm))
    smax = max(corpus_sigma({"scalar", "layerwise"}, path)[0],
               corpus_sigma({"scalar"}, path)[0],
               corpus_sigma({"layerwise"}, path)[0])
    chk(close(smax, SIGMA_W, 1e-5),
        "F2 SIGMA_W = max of the three (the frozen rule)", "-> %.6f" % smax)
    print("      SE_CELL %.6f   SE_GAP %.6f   SE_DGAP %.6f"
          % (SE_CELL, SE_GAP, SE_DGAP))
    print("      BAR_GAP %.6f   BAR_DGAP %.6f  (2 SE each)"
          % (BAR_GAP, BAR_DGAP))
    print("      THE GATE USES max(SIGMA_W, sigma_in_batch) -- registered here,")
    print("      in advance, so a noisier batch cannot borrow the corpus's bar.")

    # ---- G: NO ARM AT OR NEAR THE FLOOR ----------------------------------
    print("\n[G] NO ARM IS PREDICTED AT OR NEAR A FLOOR")
    print("      Floors: CHANCE = 1.00 pp (100 classes).  DEGENERATE-SATURATION")
    print("      = the collapsed-alpha level, empirically the scalar arm at the")
    print("      FREE rungs, %.4f pp (n=%d, the corpus's largest single cell)."
          % (C100_MS1E3[("sc", "1e-6")][0], C100_MS1E3[("sc", "1e-6")][1]))
    CHANCE = 1.0
    SAT = C100_MS1E3[("sc", "1e-6")][0]
    worst = None
    print("      %-6s %-6s %-4s %10s %10s %10s" %
          ("ms", "a0", "arm", "X pred", "SE over", "SE over"))
    print("      %-6s %-6s %-4s %10s %10s %10s" %
          ("", "", "", "", "CHANCE", "SATURATION"))
    for a0 in ALPHAS:
        for ms in LADDER[a0]:
            for arm in ARMS:
                v, why = pred_x(arm, ms, a0)
                if v is None:
                    print("      %-6s %-6s %-4s %10s   %s" % (ms, a0, arm, "--", why))
                    continue
                dc = (v - CHANCE) / SE_CELL
                ds = (v - SAT) / SE_CELL
                if worst is None or dc < worst[0]:
                    worst = (dc, ms, a0, arm, v)
                print("      %-6s %-6s %-4s %10.3f %10.1f %10.1f"
                      % (ms, a0, arm, v, dc, ds))
    chk(worst is not None and worst[0] > 30.0,
        "G1 EVERY model-valid cell clears the CHANCE floor by > 30 SE",
        "worst %.1f SE (%s ms=%s a0=%s pred %.3f)"
        % (worst[0], worst[3], worst[1], worst[2], worst[4]))
    disc = [("1e-5", "1e-3"), ("3e-5", "1e-3"), ("1e-4", "1e-3")]
    ok, mn = True, None
    for ms, a0 in disc:
        v, _ = pred_x("sc", ms, a0)
        d = (v - SAT) / SE_CELL
        mn = d if mn is None else min(mn, d)
        if d < 30.0:
            ok = False
    chk(ok, "G2 the BRANCH-DECIDING scalar cells (the low a0=1e-3 rungs) are",
        "predicted >= %.1f SE ABOVE the saturation level -- a 'the arm just "
        "died'" % mn)
    print("           model cannot produce them, which is exactly what cpr1's")
    print("           headline (CORRECTIONS 164) could not rule out.")
    print("      The three cells with NO point prediction "
          "(ms=3e-3 x2, ms=1e-4/a0=1e-6)")
    print("      are declared EXTRAPOLATED / MECHANISM-CONTROL and carry NO")
    print("      branch weight.  Their CIFAR-10 counterparts at ms=1e-4/a0=1e-6")
    print("      are %.3f (sc) and %.3f (lay) -- far above chance, so no arm is"
          % (C10[("sc", "1e-4", "1e-6")][0], C10[("lay", "1e-4", "1e-6")][0]))
    print("      predicted at the chance floor anywhere in this batch.")

    # ---- H: the design ----------------------------------------------------
    print("\n[H] THE DESIGN")
    chk(len(CELLS) == 10 and N_JOBS == 60,
        "H1 10 cells x 2 arms x 3 seeds", "-> %d jobs" % N_JOBS)
    chk(all(ms in LADDER["1e-3"] and ms in LADDER["1e-6"]
            for ms in CROSSED_MS) and len(CROSSED_MS) == 4,
        "H2 FOUR ms levels appear at BOTH alpha0 -- the alias is broken by",
        "construction -> %s" % list(CROSSED_MS))
    chk(SHARED_MS in LADDER["1e-3"] and SHARED_MS in LADDER["1e-6"],
        "H3 the shared headline rung ms=1e-3 is IN BATCH at both alpha0",
        "(RULE: every comparison WITHIN batch)")
    ncol = corpus_seed_collisions(path)
    chk(ncol == 0, "H4 ZERO %s rows anywhere carry seed 15, 16 or 17" % NET,
        "-> %d" % ncol)
    own = count_own_rows(path)
    if own:
        print("  [INFO] RULE 21 own-row gate NOT APPLICABLE -- %d %s rows are "
              "already in the corpus (post-ingest run)." % (own, PREFIX))
    else:
        chk(True, "H5 no %s row exists in the corpus yet (RULE 21 premise)"
            % PREFIX, "")

    print("\n" + "=" * 78)
    print("SELFTEST: %d PASS, %d FAIL" % (P, F))
    print("=" * 78)
    return F


# =============================================================================
# score
# =============================================================================
def score(runsdir, probe_dir=None):
    print("=" * 78)
    print("cY1_cru1_score.py -- RULE 11 ON THE HEADLINE CIFAR-100 CELL")
    print("runsdir: %s" % runsdir)
    pdir, phow = resolve_probe_dir(runsdir, probe_dir)
    print("probe  : %s  (%s)" % (pdir or "NOT FOUND -- probe section will SKIP",
                                 phow))
    print("=" * 78)

    recs = load_runs(runsdir)
    good = [r for r in recs if not r.get("bad_name")]
    bad = [r for r in recs if r.get("bad_name")]

    # ---- G1: integrity ----------------------------------------------------
    print("\n[G1] COMPLETENESS AND INTEGRITY")
    print("  .out files matching %s* : %d  (%d unparseable names)"
          % (PREFIX, len(recs), len(bad)))
    for r in bad:
        print("      BAD NAME: %s" % os.path.basename(r["file"]))
    done = [r for r in good if r["run_done"]]
    full = [r for r in good if r["n_ep"] >= EPOCHS]
    p5ok = [r for r in good if r["p5"] is not None]
    tb = [r for r in good if r["traceback"]]
    rep = [r for r in good if r["repeated"]]
    print("  RUN_DONE %d/%d | %d epochs %d/%d | plateau5 computable %d/%d"
          % (len(done), len(good), EPOCHS, len(full), len(good),
             len(p5ok), len(good)))
    print("  tracebacks %d | runs with a REPEATED flag %d" % (len(tb), len(rep)))
    for r in rep:
        print("      REPEATED FLAG in %s: %s"
              % (os.path.basename(r["file"]), r["repeated"]))

    # ENV audit -- BETA_CLIP and PROBE cannot ride the ARGS line.
    envc = collections.Counter(
        (r["envd"].get("BETA_CLIP"), r["envd"].get("PROBE"),
         r["envd"].get("AUGMENT"), r["envd"].get("HIER")) for r in good)
    print("  ENV audit (BETA_CLIP, PROBE, AUGMENT, HIER): %s" % dict(envc))
    env_ok = (len(envc) == 1
              and list(envc)[0][:3] == (CLIP_C, str(PROBE_EVERY), AUG))
    print("  ENV audit %s" % ("PASS" if env_ok else "FAIL"))

    # every non-axis ARGS flag identical across all runs
    novar = collections.defaultdict(set)
    for r in good:
        for k, v in r["eff"].items():
            if k not in AXIS_FLAGS:
                novar[k].add(v)
    varying = {k: sorted(v) for k, v in novar.items() if len(v) > 1}
    print("  non-axis ARGS flags that VARY: %s" % (varying or "NONE"))

    # the axes must be exactly what the file name says
    mism = []
    for r in good:
        if r["eff"].get("meta-stepsize") != r["ms"]:
            mism.append((r["file"], "meta-stepsize", r["eff"].get("meta-stepsize"), r["ms"]))
        if r["eff"].get("alpha0") != r["a0"]:
            mism.append((r["file"], "alpha0", r["eff"].get("alpha0"), r["a0"]))
        if r["eff"].get("stepsize-groups") != GRAN[r["arm"]]:
            mism.append((r["file"], "stepsize-groups",
                         r["eff"].get("stepsize-groups"), GRAN[r["arm"]]))
        if r["eff"].get("seed") != str(r["seed"]):
            mism.append((r["file"], "seed", r["eff"].get("seed"), str(r["seed"])))
    print("  name-vs-ARGS mismatches: %d" % len(mism))
    for m in mism[:10]:
        print("      %s: %s ARGS=%r NAME=%r" % (os.path.basename(m[0]), m[1], m[2], m[3]))

    complete = (len(good) == N_JOBS and len(done) == N_JOBS
                and len(p5ok) == N_JOBS and not bad and not tb and not rep
                and not mism and not varying and env_ok)
    print("  G1 %s (%d/%d usable runs)"
          % ("PASS" if complete else "INCOMPLETE", len(p5ok), N_JOBS))

    # ---- cell means -------------------------------------------------------
    cells = collections.defaultdict(list)
    cellt = collections.defaultdict(list)
    for r in p5ok:
        cells[(r["arm"], r["ms"], r["a0"])].append(r["p5"])
        if r["t5"] is not None:
            cellt[(r["arm"], r["ms"], r["a0"])].append(r["t5"])
    M = {k: statistics.mean(v) for k, v in cells.items()}
    MT = {k: statistics.mean(v) for k, v in cellt.items()}
    NC = {k: len(v) for k, v in cells.items()}

    # ---- G2: the crossing is REALISED, not just designed -------------------
    print("\n[G2] THE ALIAS -- BROKEN BY CONSTRUCTION, AND VERIFIED IN THE "
          "LANDED RUNS")
    realised = []
    for ms in CROSSED_MS:
        both = all(NC.get((arm, ms, a0), 0) == len(SEEDS)
                   for a0 in ALPHAS for arm in ARMS)
        realised.append((ms, both))
        print("  ms=%-5s present at BOTH alpha0 for BOTH arms, 3/3 seeds: %s"
              % (ms, "YES" if both else "NO"))
    alias_ok = all(b for _, b in realised)
    per_a0_ms = {a0: sorted(set(ms for (a, ms, b) in M if b == a0))
                 for a0 in ALPHAS}
    print("  ms levels realised: %s" % per_a0_ms)
    print("  G2 %s" % ("ALIAS-BROKEN" if alias_ok else "ALIAS-NOT-BROKEN"))

    # ---- in-batch sigma ---------------------------------------------------
    ss = df = 0
    for v in cells.values():
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
    sig_in = math.sqrt(ss / df) if df else float("nan")
    sig_use = max(SIGMA_W, sig_in) if df else SIGMA_W
    se_cell = sig_use / math.sqrt(len(SEEDS))
    se_gap = sig_use * math.sqrt(2.0 / len(SEEDS))
    se_dgap = sig_use * math.sqrt(4.0 / len(SEEDS))
    print("\n[SIGMA] registered %.6f | in-batch %.6f (df %d) | USED %.6f "
          "(the registered max rule)" % (SIGMA_W, sig_in, df, sig_use))
    print("        SE_CELL %.6f  SE_GAP %.6f  SE_DGAP %.6f  BARS %.6f / %.6f"
          % (se_cell, se_gap, se_dgap, 2 * se_gap, 2 * se_dgap))

    # ---- the ladder table -------------------------------------------------
    print("\n[LADDER] plateau5 PRIMARY, train5 alongside, per alpha0")
    for a0 in ALPHAS:
        print("\n  alpha0 = %s" % a0)
        print("  %-6s %8s %9s %9s %9s %9s %11s  %s"
              % ("ms", "TRAVEL", "sc p5", "sc tr5", "lay p5", "lay tr5",
                 "gap (SE)", "reach"))
        for ms in LADDER[a0]:
            T, du, _, amx, amn, st = reach(ms, a0)
            s = M.get(("sc", ms, a0))
            l = M.get(("lay", ms, a0))
            st5 = MT.get(("sc", ms, a0))
            lt5 = MT.get(("lay", ms, a0))
            g = (l - s) if (s is not None and l is not None) else None
            print("  %-6s %8.1f %9s %9s %9s %9s %11s  %s"
                  % (ms, T,
                     ("%.4f" % s) if s is not None else "--",
                     ("%.2f" % st5) if st5 is not None else "--",
                     ("%.4f" % l) if l is not None else "--",
                     ("%.2f" % lt5) if lt5 is not None else "--",
                     ("%+.3f/%+.1f" % (g, g / se_gap)) if g is not None else "--",
                     st))

    # ---- the verdict, per alpha0 -----------------------------------------
    print("\n[VERDICT] RULE 11, evaluated SEPARATELY at each alpha0")
    verdicts, stamps = {}, {}
    for a0 in ALPHAS:
        lad = LADDER[a0]
        have = [ms for ms in lad
                if M.get(("sc", ms, a0)) is not None
                and M.get(("lay", ms, a0)) is not None]
        if SHARED_MS not in have or len(have) < 2:
            verdicts[a0] = "UNSCORABLE-MISSING-CELLS"
            stamps[a0] = ["have=%s" % have]
            print("  a0=%s  UNSCORABLE -- landed rungs %s" % (a0, have))
            continue
        g_sh = M[("lay", SHARED_MS, a0)] - M[("sc", SHARED_MS, a0)]
        arg = {arm: max(have, key=lambda m: M[(arm, m, a0)]) for arm in ARMS}
        g_tu = M[("lay", arg["lay"], a0)] - M[("sc", arg["sc"], a0)]
        d = g_sh - g_tu
        ratio = (g_tu / g_sh) if g_sh else float("nan")
        st_arg = {arm: reach(arg[arm], a0)[5] for arm in ARMS}
        edges = (lad[0], lad[-1])
        sm = []
        # stamps
        if any(reach(arg[arm], a0)[0] / d_up(a0) < 1.0 for arm in ARMS):
            sm.append("TRAVEL-CONFOUNDED")
        gaps = [M[("lay", m, a0)] - M[("sc", m, a0)] for m in have]
        sm.append("MONOTONE-IN-TRAVEL"
                  if all(gaps[i] <= gaps[i + 1] + 2 * se_gap
                         for i in range(len(gaps) - 1))
                  else "NOT-MONOTONE-IN-TRAVEL")
        frz = [m for m in have if reach(m, a0)[5] == "FROZEN"]
        if frz:
            fg = [abs(M[("lay", m, a0)] - M[("sc", m, a0)]) for m in frz]
            sm.append("F-HOLDS" if max(fg) <= 2 * se_gap else "F-FAILS")
        # branch
        if any(st_arg[arm] == "FROZEN" for arm in ARMS):
            v = "ADAPTATION-NOT-NEEDED"
        elif any(arg[arm] in edges for arm in ARMS):
            v = "UNRESOLVED-OPTIMUM-AT-LADDER-EDGE"
        elif g_tu < -2 * se_gap:
            v = "GAP-REVERSES"
        elif abs(g_tu) <= 2 * se_gap:
            v = "GAP-CLOSES"
        elif d > 2 * se_dgap:
            v = "GAP-SHRINKS"
        else:
            v = "GAP-SURVIVES-TUNING"
        verdicts[a0], stamps[a0] = v, sm
        print("\n  a0=%s" % a0)
        print("    G_shared (ms=%s, the number every headline uses) = %s"
              % (SHARED_MS, se_str(g_sh, se_gap)))
        print("    corpus G_shared for reference (cru1 excluded)     = %+.4f pp"
              % G_SHARED_CORPUS[a0])
        for arm in ARMS:
            print("    argmax_ms(%-3s) = %-5s  level %.4f  reach %s"
                  % (arm, arg[arm], M[(arm, arg[arm], a0)], st_arg[arm]))
        print("    G_tuned (each arm at its OWN optimum)             = %s"
              % se_str(g_tu, se_gap))
        print("    G_shared - G_tuned                               = %s"
              % se_str(d, se_dgap))
        print("    RATIO = G_tuned / G_shared                       = %.4f "
              "(Account T needs >= %.2f)" % (ratio, T_RATIO_MIN))
        print("    ACCOUNT T %s | ACCOUNT D %s"
              % ("SUPPORTED" if ratio >= T_RATIO_MIN else "REFUTED",
                 "SUPPORTED" if ratio < T_RATIO_MIN else "REFUTED"))
        print("    BRANCH: %s" % v)
        print("    STAMPS: %s" % ", ".join(sm))

    # ---- predictors, scored ----------------------------------------------
    print("\n[PREDICTORS] X (CIFAR-10 error-ratio transfer) and Y "
          "(corpus-internal)")
    print("  %-6s %-6s %-4s %10s %10s %9s" %
          ("ms", "a0", "arm", "observed", "X pred", "resid"))
    resid = []
    for a0 in ALPHAS:
        for ms in LADDER[a0]:
            for arm in ARMS:
                o = M.get((arm, ms, a0))
                x, why = pred_x(arm, ms, a0)
                if o is None:
                    continue
                if x is None:
                    print("  %-6s %-6s %-4s %10.4f %10s %9s   %s"
                          % (ms, a0, arm, o, "--", "--", why))
                    continue
                resid.append(o - x)
                print("  %-6s %-6s %-4s %10.4f %10.3f %+9.3f"
                      % (ms, a0, arm, o, x, o - x))
    if resid:
        print("  X residuals: mean %+.3f pp, |max| %.3f pp over %d cells"
              % (statistics.mean(resid), max(abs(r) for r in resid), len(resid)))
    ly = M.get(("lay", "1e-4", "1e-3"))
    if ly is not None:
        xv, _ = pred_x("lay", "1e-4", "1e-3")
        print("  PIVOTAL CELL layerwise @ ms=1e-4/a0=1e-3: observed %.4f" % ly)
        print("    X said %.3f; Y said [%.3f, %.3f].  %s"
              % (xv, Y_LAY_MS1E4_LO, Y_LAY_MS1E4_HI,
                 "Y" if Y_LAY_MS1E4_LO <= ly <= Y_LAY_MS1E4_HI else
                 ("X" if abs(ly - xv) < abs(ly - Y_LAY_MS1E4_HI) else "NEITHER")))
        print("    corpus best CIFAR-100 plateau5 anywhere = %.3f; this cell %s"
              % (CORPUS_BEST_C100_P5,
                 "EXCEEDS it" if ly > CORPUS_BEST_C100_P5 else "does not exceed it"))

    # ---- probe -----------------------------------------------------------
    print("\n[PROBE] beta descent -- Account D's mechanism, SKIP if absent")
    ps = probe_summary(pdir)
    if not ps:
        print("  SKIP -- no probe records found.  The verdict does not depend "
              "on this section.")
    else:
        agg = collections.defaultdict(list)
        for run, (mn, pin, n) in ps.items():
            m = re.match(r"^cru1-(sc|lay)-m(\S+?)-a(\S+?)-s\d+$", run)
            if m and mn is not None:
                agg[(m.group(1), m.group(2), m.group(3))].append((mn, pin))
        print("  %-4s %-6s %-6s %10s %12s" %
              ("arm", "ms", "a0", "min beta", "pinned frac"))
        for k in sorted(agg):
            mns = [x[0] for x in agg[k]]
            pns = [x[1] for x in agg[k]]
            print("  %-4s %-6s %-6s %10.4f %12.4f"
                  % (k[0], k[1], k[2], statistics.mean(mns),
                     statistics.mean(pns)))

    # ---- FINAL -----------------------------------------------------------
    print("\n" + "=" * 78)
    parts = ["%s:%s" % (a0, verdicts.get(a0, "UNSCORABLE")) for a0 in ALPHAS]
    parts.append("ALIAS-BROKEN" if alias_ok else "ALIAS-NOT-BROKEN")
    if not complete:
        parts.append("G1-INCOMPLETE")
    print("FINAL: " + " | ".join(parts))
    for a0 in ALPHAS:
        if stamps.get(a0):
            print("  stamps a0=%s: %s" % (a0, ", ".join(stamps[a0])))
    print("SCOPE, REGISTERED IN ADVANCE AND BINDING: this closes RULE 11 for the")
    print("scalar-vs-layerwise contrast on CIFAR-100/ResNet18_c100 at 100 epochs")
    print("ONLY.  NOT for resnet18_blocks, NOT for cut position, NOT for class")
    print("count, NOT for ImageNet.")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runsdir", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe-dir", default=None,
                    help="OPTIONAL -- defaulted to <runsdir>/%s" % BATCH)
    a = ap.parse_args()
    if a.selftest:
        return 1 if selftest() else 0
    if not a.runsdir:
        ap.error("give a runsdir, or --selftest")
    return score(a.runsdir, a.probe_dir)


if __name__ == "__main__":
    sys.exit(main())
